import torch
import torch.nn as nn
import numpy as np
import time
from config import (
    config,
    E_MAC_32BIT_J,
    E_MAC_8BIT_J,
    E_AC_8BIT_J,
    E_MEM_J_PER_BYTE,
)
from torch.ao.quantization import DeQuantStub, QuantStub
from torch.ao.nn.quantized.modules.linear import Linear as QuantizedLinear
import snntorch
import pandas as pd

# --- Theoretical Energy Constants ---
# Imported directly from config.py (single source of truth). The previous
# config.get("E_MAC_32BIT_J", ...) form always fell through to the hardcoded
# default, because these are module-level constants in config.py, NOT keys of
# the `config` dict. Editing config.py now actually takes effect here.

# Default firing-rate gain. 1.0 => spike probability == pixel intensity (no
# saturation). With gain=2.0 every pixel above 0.5 fires on EVERY timestep, which
# saturates the code and inflates spike counts.
DEFAULT_SPIKE_GAIN = 1.0


def encode_poisson_rate(x, num_steps, gain=DEFAULT_SPIKE_GAIN):
    """Canonical Poisson RATE encoding.

    This is the single source of truth used by training, evaluation, the models'
    internal encoders, and the op/energy counters. Keeping one implementation is
    deliberate: four separate copies previously drifted apart, so the models were
    trained on one spike density and their energy measured on another.

    Per-timestep spike probability p = clamp(x_norm * gain, 0, 1), where x_norm is
    min-max normalized to [0, 1]. Expected spikes per pixel = T * p, so more
    timesteps give a better rate estimate (standard rate coding).

    NOTE: do NOT divide p by num_steps. That makes the expected spike count
    clamp(x*gain,0,1) <= 1 per pixel regardless of T, which destroys the rate code
    and makes T meaningless.

    Args:
        x: input tensor, [batch, ...] (flattened internally)
        num_steps: number of timesteps T
        gain: firing-rate scale
    Returns:
        Binary spike train [num_steps, batch, features]
    """
    x = x.view(x.size(0), -1)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    p = torch.clamp(x_norm * gain, 0, 1)
    p = p.unsqueeze(0).expand(num_steps, -1, -1)
    return (torch.rand_like(p) < p).float()


def compute_accuracy(outputs, labels):
    """Compute accuracy from model outputs and labels."""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def calibrate_input_gain(model, calib_loader, device):
    """Dynamic gain based on 99th percentile of input activations."""
    activations = []
    for x, _ in calib_loader:
        # Flatten for MLP input
        x = x.view(x.size(0), -1)
        activations.append(x.abs().max().item())
    if activations:
        perc = np.percentile(activations, 99)
        gain = 1.5 / perc if perc > 0 else 1.0
    else:
        gain = 1.0
    return gain

def count_mac_operations(model, input_data):
    """Count MAC operations for MLP models during forward pass."""
    if isinstance(input_data, tuple):
        # For MLP, input should be flattened (batch_size, input_features)
        input_size = (1, np.prod(input_data))  # Flatten to (batch, features)
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size, device=device)
    elif isinstance(input_data, torch.Tensor):
        dummy_input = input_data.clone()
        if dummy_input.dim() >= 3:  # If image tensor, flatten
            dummy_input = dummy_input.view(dummy_input.size(0), -1)
        if dummy_input.dim() == 1:  # Add batch dimension
            dummy_input = dummy_input.unsqueeze(0)
    else:
        raise TypeError("input_data must be a tuple or torch.Tensor")

    macs = 0
    hooks = []

    def linear_hook(module, input_val, output_val):
        nonlocal macs
        inp = input_val[0]
        batch_size = inp.size(0)
        
        # For Linear layers: MACs = input_features * output_features (per sample)
        # Total MACs = MACs_per_sample * batch_size
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            macs_per_sample = module.in_features * module.out_features
            total_macs = macs_per_sample * batch_size
            macs += total_macs

    # Register hooks only for Linear layers (including quantized)
    for m in model.modules():
        if isinstance(m, (nn.Linear, QuantizedLinear)):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        # Check if this is an SNN model and handle accordingly
        if hasattr(model, 'forward') and 'snn' in model.__class__.__name__.lower():
            try:
                # For SNN models, we need to account for time steps
                if hasattr(model, 'time_steps'):
                    time_steps = model.time_steps
                else:
                    time_steps = 32  # Default
                
                spike_data = encode_poisson_rate(dummy_input, time_steps)
                _ = model(spike_data)
                # For SNN models, divide by time_steps to get per-timestep MACs
                # since the hooks count MACs for all timesteps
                macs = macs // time_steps if time_steps > 0 else macs
            except Exception as e:
                # Fallback to direct input if spike encoding fails
                try:
                    _ = model(dummy_input)
                except:
                    pass  # Skip MAC counting if model fails
        else:
            _ = model(dummy_input)

    for h in hooks:
        h.remove()
    
    return int(macs)

def calculate_energy(model, input_data, hardware_profile=None):
    """
    Calculate energy consumption for MLP models.
    
    For standard MLPs: Uses MAC operations
    For SNNs: Uses AC operations 
    """
    # Check if this is an SNN model
    is_snn = any(isinstance(m, (snntorch.Leaky, snntorch.Synaptic)) 
                 for m in model.modules())
    
    if is_snn:
        # For SNN models, use AC operations energy calculation
        time_steps = getattr(model, 'time_steps', config['num_steps'])
        ac_ops, spike_count = count_ac_operations(model, input_data, time_steps)
        energy = ac_ops * E_AC_8BIT_J
        print(f"SNN MLP detected. AC ops: {ac_ops}, Spikes: {spike_count}, Energy: {energy:.4e} J")
    else:
        # For standard MLPs, use MAC operations
        macs = count_mac_operations(model, input_data)
        energy = macs * E_MAC_32BIT_J
        print(f"Standard MLP detected. MACs: {macs}, Energy: {energy:.4e} J")

    return energy

def calculate_inference_time(model, input_data, hardware_profile=None, warmup=5, repeats=10):
    """Calculate inference time for MLP models."""
    # Prepare input tensor
    if isinstance(input_data, tuple):
        input_size = (1, np.prod(input_data))  # Flatten for MLP
        device = next(model.parameters()).device
        input_tensor = torch.randn(*input_size, device=device)
    else:
        input_tensor = input_data
        if input_tensor.dim() >= 3:  # Flatten if needed
            input_tensor = input_tensor.view(input_tensor.size(0), -1)
        if input_tensor.dim() == 1:  # Add batch dim
            input_tensor = input_tensor.unsqueeze(0)

    # Calculate theoretical time based on operations
    macs = count_mac_operations(model, input_data)
    
    # Use hardware profile from config
    if hardware_profile is None:
        hardware_profile = config.get("hardware_profile", "default")
    
    if hardware_profile in config.get("hardware_constants", {}):
        # Config defines the throughput under the key "gpu_flops"; fall back to a
        # CPU-ish default if it is ever missing (avoids KeyError on "cpu_flops").
        cpu_flops = config["hardware_constants"][hardware_profile].get("gpu_flops", 100e9)
    else:
        cpu_flops = 100e9  # 100 GFLOPS default for CPU
    
    flops = macs * 2  # Approximate FLOPS as 2 * MACs
    theoretical_time = flops / cpu_flops if cpu_flops > 0 else 0

    # Measure actual inference time
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    actual_time = (end_time - start_time) / repeats

    return actual_time, theoretical_time

def calculate_power(energy, inference_time):
    """Calculate power consumption from energy and time."""
    if inference_time > 0:
        return energy / inference_time
    return 0

def count_ac_operations(model, input_data, time_steps):
    """Count Accumulate operations for SNN MLP models - following AlexNet pattern."""
    print("DEBUG_CM_AC_OPS: Entered count_ac_operations for MLP.")
    device = None
    try:
        params_list = list(model.parameters())
        device = params_list[0].device if params_list else torch.device('cpu') 
        print(f"DEBUG_CM_AC_OPS: Model device: {device}")
    except Exception as e:
        print(f"DEBUG_CM_AC_OPS: Error getting model device: {type(e).__name__}: {e}. Defaulting to CPU.")
        device = torch.device('cpu')

    if isinstance(input_data, tuple):
        # For MLP: tuple should be (input_features,)
        input_size = (1, np.prod(input_data))  # Flatten for MLP
        sample_input = torch.randn(*input_size, device=device)
        print(f"DEBUG_CM_AC_OPS: Created sample_input from tuple, shape: {sample_input.shape}")
    elif isinstance(input_data, torch.Tensor):
        sample_input = input_data.to(device)
        # Flatten for MLP if needed
        if sample_input.dim() >= 3:
            sample_input = sample_input.view(sample_input.size(0), -1)
        if sample_input.dim() == 1:
            sample_input = sample_input.unsqueeze(0)
        print(f"DEBUG_CM_AC_OPS: Using input_data tensor, shape: {sample_input.shape}, device: {sample_input.device}")
    else:
        raise TypeError("input_data must be tuple or torch.Tensor")

    print(f"DEBUG_CM_AC_OPS: Final sample_input shape for MLP SNN: {sample_input.shape}")

    observed_spikes_current_frame = []
    def spike_counting_hook(module, inp, out):
        spikes_in_frame = 0
        if isinstance(module, snntorch.Leaky):
            # snntorch Leaky returns (spk, mem) when output=True, else a bare spk tensor.
            # Count spikes (out[0]), NOT membrane potential.
            spk = out[0] if isinstance(out, tuple) else out
            spikes_in_frame = (spk > 0).sum().item()
            # else: print(f"DEBUG Leaky hook: output not tuple or wrong len: {type(out)}")
        elif isinstance(module, (nn.Linear, QuantizedLinear)):
            act_tensor = out.dequantize() if hasattr(out, 'is_quantized') and out.is_quantized else out
            spikes_in_frame = (act_tensor > 0).sum().item()
        # else: print(f"DEBUG hook: not Leaky or Linear: {type(module)}")
        
        if spikes_in_frame > 0:
            # print(f"DEBUG hook on {type(module).__name__}: observed {spikes_in_frame} spikes.")
            observed_spikes_current_frame.append(spikes_in_frame)
        # else: print(f"DEBUG hook on {type(module).__name__}: observed 0 spikes.")

    hooks = []
    try:
        print("DEBUG_CM_AC_OPS: Registering hooks...")
        has_snn_neurons = any(isinstance(m, snntorch.Leaky) for m in model.modules())
        if has_snn_neurons:
            print("DEBUG_CM_AC_OPS: SNN Leaky neurons found, hooking them.")
            for m_name, m in model.named_modules(): 
                if isinstance(m, snntorch.Leaky):
                    hooks.append(m.register_forward_hook(spike_counting_hook))
        else:
            print("DEBUG_CM_AC_OPS: No SNN Leaky neurons, hooking Linear layers.")
            for m_name, m in model.named_modules():
                if isinstance(m, (nn.Linear, QuantizedLinear)):
                    hooks.append(m.register_forward_hook(spike_counting_hook))
        print(f"DEBUG_CM_AC_OPS: Registered {len(hooks)} hooks.")
    except StopIteration:
        print("DEBUG_CM_AC_OPS: StopIteration during hook registration!")
        return 0,0
    except Exception as e:
        print(f"DEBUG_CM_AC_OPS: Exception during hook registration! Type: {type(e).__name__}, Error: {e}")
        return 0,0

    if not hooks:
        print("DEBUG_CM_AC_OPS: No hooks were registered. Returning 0, 0.")
        return 0, 0

    model.eval()
    total_ac_ops = 0
    total_spike_ops = 0
    with torch.no_grad():
        try:
            observed_spikes_current_frame.clear()
            if has_snn_neurons: 
                print(f"DEBUG_CM_AC_OPS: About to call snntorch-style MLP model with spike trains of type {type(model).__name__}")
                # Same canonical encoder used for training/eval, so the spikes we
                # count are the spikes the model actually runs on.
                spike_data = encode_poisson_rate(sample_input, time_steps)
                _ = model(spike_data)
                print("DEBUG_CM_AC_OPS: snntorch-style MLP model call completed.")
            else: 
                print(f"DEBUG_CM_AC_OPS: About to loop {time_steps} times for non-snntorch MLP model {type(model).__name__} with 2D input.")
                for t_step in range(time_steps):
                    _ = model(sample_input) 
                    if (t_step + 1) % 10 == 0 or (t_step + 1) == time_steps:
                        print(f"DEBUG_CM_AC_OPS: Looped model call, step {t_step + 1}/{time_steps} completed.")
                print("DEBUG_CM_AC_OPS: Looped model calls completed.")

        except StopIteration as si:
            print(f"DEBUG_CM_AC_OPS: StopIteration caught during model call! Error: {si}")
            for h in hooks: h.remove()
            return 0,0 
        except Exception as e:
            print(f"DEBUG_CM_AC_OPS: Exception caught during model call! Type: {type(e).__name__}, Error: {e}")
            for h in hooks: h.remove()
            return 0,0
            
        all_observed_spikes = list(observed_spikes_current_frame) 
        total_spike_ops = sum(all_observed_spikes)
        total_ac_ops = total_spike_ops # As per current definition
        print(f"DEBUG_CM_AC_OPS: Spikes collected: {len(all_observed_spikes)} entries, Sum: {total_spike_ops}.")
        
    for h in hooks:
        h.remove()
    print("DEBUG_CM_AC_OPS: Hooks removed.")
    
    print(f"DEBUG_CM_AC_OPS: Returning ac_ops={total_ac_ops}, spike_ops={total_spike_ops}")
    return total_ac_ops, total_spike_ops

def count_memory_access(model, sample_input_or_shape, is_snn=False, time_steps=None, batch_size=1):
    """
    Count memory accesses for MLP models.
    
    Args:
        model: The MLP model
        sample_input_or_shape: Input tensor or shape tuple  
        is_snn: Whether this is an SNN model
        time_steps: Number of time steps for SNN
        batch_size: Batch size
    
    Returns:
        Total memory accesses in bytes
    """
    if isinstance(sample_input_or_shape, tuple):
        input_features = np.prod(sample_input_or_shape)
        device = next(model.parameters()).device
        sample_input = torch.randn(batch_size, input_features, device=device)
    else:
        sample_input = sample_input_or_shape
        if sample_input.dim() >= 3:  # Flatten if needed
            sample_input = sample_input.view(sample_input.size(0), -1)

    total_memory_bytes = 0
    hooks = []

    def hook_fn(module, input, output):
        nonlocal total_memory_bytes

        if isinstance(module, (nn.Linear, QuantizedLinear)):
            # Byte width per element. An INT8 quantized layer stores weights and
            # activations at 1 byte; an FP32 layer at 4 bytes. Detect per-module
            # (not per-model) so a mixed-precision net -- e.g. the QDSNN with an
            # INT8 backbone but FP32 exit heads -- is counted correctly layer by
            # layer. Bias is left at 4 bytes (kept as int32/fp32 in quantized
            # inference). Previously every layer was counted at 4 bytes, which
            # overstated INT8 memory (and hence the memory-dominated energy) ~4x.
            is_int8 = isinstance(module, QuantizedLinear)
            elem_bytes = 1 if is_int8 else 4

            # Weight access
            weight_elements = module.in_features * module.out_features
            weight_bytes = weight_elements * elem_bytes

            # Bias access (if exists)
            bias_bytes = 0
            if module.bias is not None:
                bias_bytes = module.out_features * 4

            # Input activation access
            input_bytes = input[0].numel() * elem_bytes

            # Output activation storage - handle tuple outputs
            if isinstance(output, tuple):
                # SNN neurons return (spikes, membrane_potential) - use spikes
                output_tensor = output[0] if len(output) > 0 else output[0]
            else:
                output_tensor = output

            output_bytes = output_tensor.numel() * elem_bytes

            layer_memory = weight_bytes + bias_bytes + input_bytes + output_bytes

            # NOTE: For SNN models the forward pass loops over time internally, so this
            # hook already fires once per timestep. Do NOT multiply by time_steps again
            # (that would double-count by a factor of T).

            total_memory_bytes += layer_memory

    # Register hooks
    for module in model.modules():
        if isinstance(module, (nn.Linear, QuantizedLinear)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        if is_snn and time_steps:
            try:
                spike_data = encode_poisson_rate(sample_input, time_steps)
                _ = model(spike_data)
            except Exception as e:
                # Fallback to simple forward pass
                try:
                    _ = model(sample_input)
                except:
                    pass  # Skip if model fails
        else:
            _ = model(sample_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_memory_bytes

def count_syops(model, input_data, time_steps):
    """Count synaptic operations for SNN models - same as AC operations for MLPs."""
    ac_ops, spike_count = count_ac_operations(model, input_data, time_steps)
    return ac_ops

def calculate_theoretical_energy(mac_count, ac_count, spike_count, ann_mem_bytes, snn_mem_bytes):
    """
    Calculate theoretical energy consumption based on operation counts and memory access.
    
    Args:
        mac_count: Number of MAC operations (for ANNs)
        ac_count: Number of AC operations (for SNNs)
        spike_count: Number of spikes generated
        ann_mem_bytes: Memory bytes accessed by ANN
        snn_mem_bytes: Memory bytes accessed by SNN
    
    Returns:
        Dictionary with energy breakdown
    """
    # Compute energy components
    mac_energy = mac_count * E_MAC_32BIT_J
    ac_energy = ac_count * E_AC_8BIT_J
    ann_mem_energy = ann_mem_bytes * E_MEM_J_PER_BYTE
    snn_mem_energy = snn_mem_bytes * E_MEM_J_PER_BYTE
    
    # Neuromorphic energy calculation
    neuromorphic_ac_energy = ac_count * config.get("E_AC_NEUROMORPHIC_J", 0.01e-12)
    neuromorphic_mem_energy = snn_mem_bytes * config.get("E_MEM_NEUROMORPHIC_J_PER_BYTE", 2e-12)
    
    return {
        'mac_energy': mac_energy,
        'ac_energy': ac_energy,
        'ann_mem_energy': ann_mem_energy,
        'snn_mem_energy': snn_mem_energy,
        'total_ann_energy': mac_energy + ann_mem_energy,
        'total_snn_energy': ac_energy + snn_mem_energy,
        'neuromorphic_energy': neuromorphic_ac_energy + neuromorphic_mem_energy,
        'spike_count': spike_count
    }

def normalize_weights(model, calib_loader, time_steps=20):
    """
    Normalize weights for SNN models based on maximum activations.
    
    Args:
        model: The SNN model to normalize
        calib_loader: DataLoader for calibration data
        time_steps: Number of time steps for SNN simulation
    
    Returns:
        The model with normalized weights
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Dictionary to store maximum activations for each layer
    max_activations = {}
    
    def record_max_activation(name):
        def hook(module, input, output):
            if name not in max_activations:
                max_activations[name] = 0
            
            # Handle different output types
            if isinstance(output, tuple):
                activation = output[0]  # Usually spikes are first element
            else:
                activation = output
            
            if hasattr(activation, 'abs'):
                current_max = activation.abs().max().item()
                max_activations[name] = max(max_activations[name], current_max)
        return hook
    
    # Register hooks for Linear layers
    hooks = []
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(record_max_activation(name))
            hooks.append(hook)
            layer_names.append(name)
    
    # Run calibration to find maximum activations
    print("Running calibration to find maximum activations...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calib_loader):
            data = data.view(data.size(0), -1).to(device)  # Flatten for MLP
            
            # Run SNN simulation
            mem = None
            for t in range(time_steps):
                if hasattr(model, 'forward_step'):
                    spk, mem = model.forward_step(data, mem)
                else:
                    spk = model(data)
            
            if batch_idx >= 10:  # Use limited batches for calibration
                break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Normalize weights based on maximum activations
    scaling_factors = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in max_activations:
                max_act = max_activations[name]
                if max_act > 1.0:  # Only scale if activation exceeds 1
                    scale_factor = 1.0 / max_act
                    module.weight.data *= scale_factor
                    if module.bias is not None:
                        module.bias.data *= scale_factor
                    scaling_factors[name] = scale_factor
                    print(f"Scaled {name} by factor {scale_factor:.4f} (max_act: {max_act:.4f})")
    
    print(f"Weight normalization complete. Scaled {len(scaling_factors)} layers.")
    return model

def encode_direct(x, num_steps=16):
    """Direct encoding - repeat input for each time step."""
    batch_size = x.size(0)
    feature_size = x.view(batch_size, -1).size(1)  # Flatten
    
    # Normalize to [0, 1] range
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    x_flat = x_norm.view(batch_size, feature_size)
    
    # Repeat for each time step
    encoded = x_flat.unsqueeze(1).repeat(1, num_steps, 1)
    return encoded

def encode_rate(x, num_steps, gain=1.0):
    """Rate encoding - convert pixel intensities to spike rates."""
    batch_size = x.size(0)
    feature_size = x.view(batch_size, -1).size(1)
    
    # Normalize and apply gain
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    x_flat = x_norm.view(batch_size, feature_size) * gain
    
    # Generate spikes based on probability
    spikes = torch.rand(batch_size, num_steps, feature_size, device=x.device) < x_flat.unsqueeze(1)
    return spikes.float()

def encode_temporal(x, num_steps=4):
    """Temporal encoding - earlier spikes for higher intensities."""
    batch_size = x.size(0)
    feature_size = x.view(batch_size, -1).size(1)
    
    # Normalize input
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    x_flat = x_norm.view(batch_size, feature_size)
    
    # Create temporal encoding
    spikes = torch.zeros(batch_size, num_steps, feature_size, device=x.device)
    
    # Spike timing based on intensity (higher intensity = earlier spike)
    spike_times = ((1.0 - x_flat) * (num_steps - 1)).long()
    
    for b in range(batch_size):
        for f in range(feature_size):
            t = spike_times[b, f].item()
            if t < num_steps:
                spikes[b, t, f] = 1.0
    
    return spikes

def calculate_snn_energy_paper_method(total_spikes_per_sample, e_ac_paper_j):
    """Calculate SNN energy using the paper's method."""
    return total_spikes_per_sample * e_ac_paper_j

def calculate_fair_theoretical_energy_from_dict(res, model_key):
    """
    Calculate fair theoretical energy from a metrics dictionary.
    This function mirrors the AlexNet and LeNet implementation exactly.
    
    Args:
        res (dict): Dictionary containing results for a model (e.g., loaded from CSV).
                    Expected keys include 'ac_ops', 'snn_mem_accesses', 'macs', 
                    'ann_mem_accesses', potentially 'ac_operations', 'syops', 'operation_count'.
        model_key (str): String identifier for the model (e.g., 'snn_mlp').

    Returns:
        tuple[float, str]: A tuple containing:
            - energy_j (float): Calculated theoretical energy in Joules (NaN if calculation fails).
            - source (str): A string describing how the energy was calculated or why it failed.
    """
    energy_j = np.nan # Default to NaN
    ops_cost = 0
    mem_cost = 0
    source = "Not Calculated"

    # Match any SNN variant, including threshold-suffixed eval names like
    # 'branchy_quantized_snn_mlp_cfg0.2_0.4'. Exact-equality checks silently
    # routed those (and quantized_snn_mlp) down the ANN/MAC branch.
    is_snn_type = model_key.startswith('snn_') or 'snn' in model_key
    is_quantized = 'quant' in model_key

    try:
        if is_snn_type:
            # SNN Energy: AC Ops + Memory
            # Prioritize 'ac_ops', fallback to 'ac_operations'
            ac_ops = res.get('ac_ops', res.get('ac_operations', np.nan))
            # Prioritize 'snn_mem_accesses', fallback to 'snn_memory_access' (if exists)
            mem_accesses = res.get('snn_mem_accesses', res.get('snn_memory_access', np.nan))
            # Assume 8-bit AC cost for fair comparison across SNNs (quantized or not)
            # This assumption might need refinement based on specific model/hardware
            precision_cost = E_AC_8BIT_J 
            
            if pd.notna(ac_ops) and ac_ops > 0: # Ensure ops are positive
                ops_cost = float(ac_ops) * precision_cost
                source = "AC Ops"
            else:
                # If AC ops missing/zero, try SyOps as a proxy (use with caution)
                syops = res.get('syops', np.nan)
                if pd.notna(syops) and syops > 0:
                     ops_cost = float(syops) * precision_cost # Treat SyOp cost similar to AC
                     source = "SyOps (proxy for AC)"
                else:
                    # Cannot calculate ops_cost if both AC and SyOps are missing/zero
                    # We might still proceed if memory cost is available, but flag it.
                    # Let's return NaN for energy if ops are fundamentally missing.
                    print(f"[Calc FairEnergy WARN] {model_key}: Missing or zero primary SNN Ops (AC Ops/SyOps). Cannot calculate energy.")
                    return np.nan, "Missing SNN Ops"

            if pd.notna(mem_accesses) and mem_accesses > 0:
                mem_cost = float(mem_accesses) * E_MEM_J_PER_BYTE
                source += " + SNN Mem"
            else:
                # Allow calculation even if memory is missing, just use ops cost
                print(f"[Calc FairEnergy WARN] {model_key}: Missing SNN memory accesses. Energy based only on Ops.")
                source += " (Mem Missing)"
            
            energy_j = ops_cost + mem_cost
            
        else: # ANN models
            # ANN Energy: MAC Ops + Memory
            # Prioritize 'macs', fallback to 'operation_count'
            mac_ops = res.get('macs', res.get('operation_count', np.nan))
             # Prioritize 'ann_mem_accesses', fallback to 'ann_memory_access' (if exists)
            mem_accesses = res.get('ann_mem_accesses', res.get('ann_memory_access', np.nan))
            # Determine precision cost based on model key or potentially a 'precision' field
            # Defaulting to 32-bit if not explicitly quantized
            precision_field = res.get('precision', None)
            if precision_field and 'int8' in str(precision_field).lower():
                 is_quantized = True
            elif 'quant' in model_key: # Fallback check in name
                 is_quantized = True
            
            precision_cost = E_MAC_8BIT_J if is_quantized else E_MAC_32BIT_J
            precision_label = "INT8" if is_quantized else "FP32"

            if pd.notna(mac_ops) and mac_ops > 0:
                ops_cost = float(mac_ops) * precision_cost
                source = f"MAC Ops ({precision_label})"
            else:
                print(f"[Calc FairEnergy WARN] {model_key}: Missing or zero MAC Ops. Cannot calculate energy.")
                return np.nan, "Missing ANN Ops"

            if pd.notna(mem_accesses) and mem_accesses > 0:
                mem_cost = float(mem_accesses) * E_MEM_J_PER_BYTE
                source += " + ANN Mem"
            else:
                print(f"[Calc FairEnergy WARN] {model_key}: Missing ANN memory accesses. Energy based only on Ops.")
                source += " (Mem Missing)"

            energy_j = ops_cost + mem_cost

        # Final check for validity
        if pd.notna(energy_j) and energy_j >= 0: # Allow zero energy if calculated as such
            return energy_j, source
        else:
             # This path is taken if energy_j remains NaN or becomes negative
             print(f"[Calc FairEnergy WARN] {model_key}: Final energy calculation resulted in NaN or negative value.")
             return np.nan, "Calculation Failed"
            
    except Exception as e:
        import traceback
        print(f"[Calc FairEnergy ERROR] {model_key}: Error during calculation: {e}")
        # traceback.print_exc() # Optional: print full traceback for debugging
        return np.nan, "Calculation Error"
