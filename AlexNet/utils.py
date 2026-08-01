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
from torch.ao.nn.quantized.modules.conv import Conv2d as QuantizedConv2d
from torch.ao.nn.quantized.modules.linear import Linear as QuantizedLinear
import snntorch
import pandas as pd


# --- Theoretical energy constants ---
# Imported from config.py (single source of truth). They used to be redefined
# here with values that CONTRADICTED config.py (E_MEM was 160e-12 here vs 4e-12
# there) and a third set was hardcoded again inside calculate_theoretical_energy.

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
        activations.append(x.abs().max().item())
    if activations:
        perc = np.percentile(activations, 99)
        gain = 1.5 / perc if perc > 0 else 1.0
    else:
        gain = 1.0
    return gain

def count_mac_operations(model, input_data):
    """Count MAC operations for a model during a forward pass.
    Handles standard and quantized Conv/Linear layers."""
    if isinstance(input_data, tuple):
        input_size = (1, *input_data)
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size, device=device)
    elif isinstance(input_data, torch.Tensor):
        dummy_input = input_data
        if dummy_input.dim() == 3:
            dummy_input = dummy_input.unsqueeze(0)

    else:
        raise TypeError("input_data must be a tuple (C, H, W) or a torch.Tensor")

    macs = 0
    hooks = []

    def conv_hook(module, input_val, output_val):
        nonlocal macs
        inp = input_val[0] 
        inp_shape = inp.size()
        output_shape = output_val.size()

        if len(inp_shape) == 4: 
            in_channels = inp_shape[1]
        elif len(inp_shape) == 3: 
            in_channels = inp_shape[0]
        else:
            return

        out_channels = output_shape[1]
        output_height = output_shape[2]
        output_width = output_shape[3]
        
        kernel_h, kernel_w = module.kernel_size
        
        macs_per_output_pixel = in_channels * kernel_h * kernel_w
        total_output_pixels = out_channels * output_height * output_width
        macs_layer = macs_per_output_pixel * total_output_pixels
        macs += macs_layer


    def linear_hook(module, input_val, output_val):
        nonlocal macs
        inp = input_val[0]
        macs_layer = module.in_features * module.out_features
        macs += macs_layer

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, QuantizedConv2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, (nn.Linear, QuantizedLinear)):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()
    return int(macs)

def calculate_energy(model, input_data, hardware_profile=None):
    """Energy for one inference.

    SNNs are charged per ACCUMULATE (one per spike); ANNs per MAC. The old
    version charged every model -- including the SNNs -- per MAC, which is the
    whole point of the SNN energy argument and made the SNN numbers wrong."""
    if hardware_profile is None:
        hardware_profile = config.get("hardware_profile", "default")

    profile = config.get("hardware_constants", {}).get(hardware_profile, {})
    energy_per_mac = profile.get("energy_per_mac", E_MAC_32BIT_J)
    energy_per_ac = profile.get("energy_per_ac", E_AC_8BIT_J)

    is_snn = any(isinstance(m, snntorch.Leaky) for m in model.modules())

    if is_snn:
        time_steps = getattr(model, 'time_steps', config["num_steps"])
        ac_ops, spike_count = count_ac_operations(model, input_data, time_steps)
        energy = ac_ops * energy_per_ac
        print(f"SNN detected. AC ops: {ac_ops}, Spikes: {spike_count}, "
              f"Energy: {energy:.4e} J (using {energy_per_ac:.2e} J/AC)")
    else:
        macs = count_mac_operations(model, input_data)
        energy = macs * energy_per_mac
        print(f"CNN detected. MACs: {macs}, Energy: {energy:.4e} J (using {energy_per_mac:.2e} J/MAC)")

    return energy

def calculate_inference_time(model, input_data, hardware_profile=None, warmup=5, repeats=10):
    
    if hardware_profile is None:
        hardware_profile = config.get("hardware_profile", "default")

    if hardware_profile in config.get("hardware_constants", {}):
        gpu_flops = config["hardware_constants"][hardware_profile]["gpu_flops"]
    else:
        gpu_flops = 10e12 

    if isinstance(input_data, tuple):
        input_size = (1, *input_data) 
        device = next(model.parameters()).device
        input_tensor = torch.randn(*input_size, device=device)
    else:
        input_tensor = input_data
        if input_tensor.dim() == 3: 
            input_tensor = input_tensor.unsqueeze(0)

    macs = count_mac_operations(model, input_data)
    flops = macs * 2 
    theoretical_time = flops / gpu_flops if gpu_flops > 0 else 0

    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            for _ in range(repeats):
                _ = model(input_tensor)
            end_event.record()

        torch.cuda.synchronize()
        measured_time = start_event.elapsed_time(end_event) / (1000 * repeats) 
    else:
        with torch.no_grad():
            start_time = time.time()
            for _ in range(repeats):
                _ = model(input_tensor)
            measured_time = (time.time() - start_time) / repeats

    print(f"CNN detected. MACs: {macs}, Measured Time: {measured_time:.4e} s, Theoretical: {theoretical_time:.4e} s")
    return measured_time 

def calculate_power(energy, inference_time):
    if inference_time is None or inference_time == 0:
        print("Warning: Cannot calculate power due to zero or None inference time.")
        return 0
    power = energy / inference_time
    print(f"Power: {power:.4f} W")
    return power

def count_ac_operations(model, input_data, time_steps):
    """Count accumulate (AC) operations for an SNN by counting emitted spikes.

    Two bugs used to make this return garbage:
      1. The hook read `out[1]` -- snntorch's Leaky returns (spk, mem), so this
         counted the MEMBRANE POTENTIAL, not spikes.
      2. The model was fed a 5-D tensor built by repeating a sample, which the
         model's forward rejected; the resulting exception was swallowed by a
         bare `except` that returned (0, 0). AC ops silently came back as zero.
    Both models now take raw images and Poisson-encode internally.
    """
    try:
        params_list = list(model.parameters())
        device = params_list[0].device if params_list else torch.device('cpu')
    except Exception:
        device = torch.device('cpu')

    if isinstance(input_data, tuple):
        sample_input = torch.randn(1, *input_data, device=device)
    elif isinstance(input_data, torch.Tensor):
        sample_input = input_data.to(device)
    else:
        raise TypeError("input_data must be tuple (C,H,W) or torch.Tensor (B,C,H,W or T,B,C,H,W)")

    if sample_input.dim() == 5:   # already a spike train -> take one frame
        sample_input = sample_input[0]
    elif sample_input.dim() == 3:  # single sample -> add batch dim
        sample_input = sample_input.unsqueeze(0)
    elif sample_input.dim() != 4:
        raise ValueError(f"count_ac_operations expects 4D (B,C,H,W), got {sample_input.shape}")

    observed_spikes = []

    def spike_counting_hook(module, inp, out):
        # snntorch Leaky returns (spk, mem) when it has state, else a bare spk
        # tensor. Count spikes (out[0]), NOT the membrane potential.
        spk = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(spk):
            observed_spikes.append((spk > 0).sum().item())

    has_snn_neurons = any(isinstance(m, snntorch.Leaky) for m in model.modules())
    if not has_snn_neurons:
        return 0, 0

    hooks = [m.register_forward_hook(spike_counting_hook)
             for m in model.modules() if isinstance(m, snntorch.Leaky)]
    if not hooks:
        return 0, 0

    model.eval()
    try:
        with torch.no_grad():
            # The model Poisson-encodes the 4-D image batch internally and loops
            # over its own time window, so the hooks fire once per timestep.
            _ = model(sample_input)
        total_spike_ops = sum(observed_spikes)
    finally:
        for h in hooks:
            h.remove()

    total_ac_ops = total_spike_ops  # one spike drives one accumulate
    return total_ac_ops, total_spike_ops


def count_memory_access(model, sample_input_or_shape, is_snn=False, time_steps=None, batch_size=1):
    """Count bytes moved (weights + biases + input/output activations) for one inference.

    The old version ran an SNN by calling the model once per timestep from the
    OUTSIDE while the model also looped over time INTERNALLY (T*T passes), then
    divided the total by T to "average" it. It also keyed SNN detection off a
    `time_steps` attribute the branchy model did not have. Now the hooks simply
    fire once per internal timestep and we sum them -- no external loop, no
    division.
    """
    if isinstance(sample_input_or_shape, torch.Tensor):
        sample_input = sample_input_or_shape
        if sample_input.dim() == 5:      # [T,B,C,H,W] -> one frame
            sample_input = sample_input[0]
        elif sample_input.dim() == 3:    # [C,H,W] -> add batch
            sample_input = sample_input.unsqueeze(0)
    else:
        C, H, W = sample_input_or_shape[-3:]
        device = next(model.parameters()).device
        sample_input = torch.randn(batch_size, C, H, W, device=device)

    total_memory_bytes = 0
    hooks = []

    def hook_fn(module, inp, output):
        nonlocal total_memory_bytes

        # INT8 backbone accounting: a real quantized Conv/Linear stores and moves
        # 1 byte per weight/activation element. A fake-quant (QAT) module keeps
        # FP32 tensors (element_size 4) only so gradients can flow, but it
        # represents an INT8 layer -- so count it at 1 byte, matching what the
        # converted-INT8 model would report. FP32 layers (e.g. the QDSNN exit
        # heads, which are intentionally not quantized) keep their real width.
        is_quantized = (hasattr(module, 'weight_fake_quant')
                        or type(module).__module__.startswith('torch.ao.nn.quantized'))

        def nbytes(t, quantized):
            if not torch.is_tensor(t):
                return 0
            return t.nelement() * (1 if quantized else t.element_size())

        # Quantized layers expose weight()/bias() as callables, not attributes.
        w = module.weight() if callable(getattr(module, 'weight', None)) else getattr(module, 'weight', None)
        b = module.bias() if callable(getattr(module, 'bias', None)) else getattr(module, 'bias', None)
        weight_bytes = nbytes(w, is_quantized)
        bias_bytes = nbytes(b, False)  # bias stays higher precision (int32/FP32)

        input_bytes = nbytes(inp[0], is_quantized)
        out_t = output[0] if isinstance(output, tuple) else output
        output_bytes = nbytes(out_t, is_quantized)

        total_memory_bytes += weight_bytes + bias_bytes + input_bytes + output_bytes

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, QuantizedConv2d, QuantizedLinear)):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    try:
        with torch.no_grad():
            # Both SNN models encode + loop over time internally, so a single
            # call already covers all timesteps.
            _ = model(sample_input)
    finally:
        for h in hooks:
            h.remove()

    return total_memory_bytes


def count_syops(model, input_data, time_steps):
    """Synaptic operations == AC operations for these spiking models."""
    return count_ac_operations(model, input_data, time_steps)[0]

def calculate_theoretical_energy(mac_count, ac_count, spike_count, ann_mem_bytes, snn_mem_bytes):
    """Theoretical ANN vs SNN energy, using the constants from config.py.

    This used to hardcode a THIRD set of constants (E_AC_J=0.01e-12,
    E_MEM=160e-12) that disagreed with both config.py and the module-level
    constants above, so the energy numbers depended on which function produced
    them."""
    ann_op_energy = mac_count * E_MAC_32BIT_J
    ann_mem_energy = ann_mem_bytes * E_MEM_J_PER_BYTE
    ann_total_energy = ann_op_energy + ann_mem_energy

    snn_op_energy = ac_count * E_AC_8BIT_J
    snn_mem_energy = snn_mem_bytes * E_MEM_J_PER_BYTE
    snn_total_energy = snn_op_energy + snn_mem_energy

    print(f"ANN Energy (Op:{ann_op_energy:.2e}, Mem:{ann_mem_energy:.2e}) = {ann_total_energy:.2e} J")
    print(f"SNN Energy (Op:{snn_op_energy:.2e}, Mem:{snn_mem_energy:.2e}) = {snn_total_energy:.2e} J")
    return ann_total_energy, snn_total_energy


def normalize_weights(model, calib_loader, time_steps=20):
    print("Starting single-pass weight normalization and threshold calculation...")
    device = next(model.parameters()).device
    model.eval()

    layers_info = [] 
    activation_layer_names = []
    module_list = list(model.named_modules())

    print("Identifying layers for normalization and activation hooking:")
    for i, (name, module) in enumerate(module_list):
        if module.__class__.__name__.lower().startswith("qcfs"):
            activation_layer_names.append(name)
            print(f"  Found activation layer: {name}")
            preceding_layer_module = None
            preceding_layer_name = "None"
            for j in range(i - 1, -1, -1):
                prev_name, prev_module = module_list[j]
                if isinstance(prev_module, (nn.Conv2d, nn.Linear)):
                    preceding_layer_module = prev_module
                    preceding_layer_name = prev_name
                    print(f"    Found preceding layer to normalize: {preceding_layer_name}")
                    break 

            if preceding_layer_module:
                layers_info.append((name, preceding_layer_module, module))
            else:
                 print(f"    Warning: No preceding Conv/Linear layer found for {name}")

    if not layers_info:
        print("Warning: No activation layers with preceding Conv/Linear layers found.")
        return {}

    max_activations = {}
    hooks = []

    def record_max_activation(name):
        def hook(module, input, output):
            current_max = output.detach().abs().max().item()
            if name not in max_activations or current_max > max_activations[name]:
                max_activations[name] = current_max
        return hook

    print("\nRegistering hooks for max activation recording...")
    for act_name, _, activation_module in layers_info:
        hooks.append(activation_module.register_forward_hook(record_max_activation(act_name)))
        print(f"  Registered hook on: {act_name}")

    print(f"\nRunning calibration ({len(calib_loader)} batches) to record max activations (single pass)...")
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            x = x.to(device)
            _ = model(x) 
            if (i + 1) % 10 == 0 or (i + 1) == len(calib_loader):
                 print(f"  Calibration batch {i+1}/{len(calib_loader)} processed.")

    for h in hooks:
        h.remove()
    print("Hooks removed.")

    print("\nMax activations recorded (SINGLE PASS):")
    for name, max_val in max_activations.items():
        print(f"  {name}: {max_val:.6f}")

    print("\nApplying normalization (single pass)...")
    normalized_count = 0
    skipped_layers = []
    epsilon = 1e-6 

    for act_name, layer_to_normalize, _ in layers_info:
        layer_name = [name for name, mod in model.named_modules() if mod is layer_to_normalize][0]
        if act_name in max_activations:
            max_act = max_activations[act_name]
            if max_act > epsilon:
                scale_factor = max_act
                layer_to_normalize.weight.data.div_(scale_factor)
                if layer_to_normalize.bias is not None:
                    layer_to_normalize.bias.data.div_(scale_factor)
                print(f"  Normalized layer '{layer_name}' weights/bias using max_act ({max_act:.4f}) from '{act_name}'")
                normalized_count += 1
            else:
                print(f"  Skipping normalization for layer '{layer_name}' (max activation from '{act_name}' is zero or too small: {max_act:.6f})")
                skipped_layers.append(layer_name)
        else:
            print(f"  Warning: No max activation recorded for '{act_name}'. Cannot normalize preceding layer '{layer_name}'.")
            skipped_layers.append(layer_name)

    print(f"\nSingle-pass weight normalization complete. Normalized {normalized_count} layers.")
    if skipped_layers:
        print(f"  Skipped normalization for layers: {list(set(skipped_layers))}") 

    print("\nCalculating SNN thresholds...")
    snn_thresholds = {}
    for act_name in activation_layer_names: 
        if act_name in max_activations:
            max_act = max_activations[act_name]
            if max_act > epsilon:
                threshold = time_steps * max_act
                snn_thresholds[act_name] = threshold
                print(f"  Threshold for SNN layer corresponding to '{act_name}': {threshold:.6f} (max_act={max_act:.4f}, T={time_steps})")
            else:
                print(f"  Skipping threshold calculation for '{act_name}' (max activation zero or too small), setting threshold to 0.0")
                snn_thresholds[act_name] = 0.0 
        else:
             print(f"  Warning: No max activation recorded for '{act_name}', cannot calculate threshold. Setting threshold to 1.0")
             snn_thresholds[act_name] = 1.0 

    print("SNN threshold calculation complete.")
    return snn_thresholds


def encode_direct(x, num_steps=16):
    if x.max() <= 0: 
        print("Warning: Input tensor max value is <= 0 in encode_direct. Returning zeros.")
        return torch.zeros(num_steps, *x.shape, device=x.device, dtype=torch.float)

    x_norm = x / x.max()  
    spike_train = (x_norm.unsqueeze(0).expand(num_steps, -1, -1, -1, -1) > torch.rand(num_steps, *x.shape, device=x.device)).float()
    return spike_train


def encode_poisson(x, num_steps, gain=None):
    """Canonical Poisson rate coding: P(spike per step) = clamp(x_norm*gain, 0, 1),
    x_norm min-max normalized (NO division by num_steps -- see the definition).

    Shared by the QSNN and QDSNN and matched by the AC/spike counters.
    Re-exported from the model module so there is exactly one definition."""
    from models.quantized_snn_alexnet import encode_poisson as _enc
    return _enc(x, num_steps, gain)


def encode_rate(x, num_steps, gain=1.0):
    """Rate coding without the 1/num_steps dt factor (kept for reference).

    Prefer encode_poisson. The old body had unreachable code after its `return`."""
    x_scaled = torch.clamp(x, 0, 1) * gain
    spike_prob = x_scaled.unsqueeze(0).expand(num_steps, *x.shape)
    return (torch.rand_like(spike_prob) < spike_prob).float()

def encode_temporal(x, num_steps=4):
    
    x_expanded = x.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)  # [T, B, C, H, W]
    x_cum = torch.cumsum(x_expanded, dim=0)
    return (x_cum > torch.rand_like(x_cum)).float()

def calculate_snn_energy_paper_method(total_spikes_per_sample, e_ac_paper_j):
    
    if total_spikes_per_sample < 0:
        print("Warning: total_spikes_per_sample is negative. Setting to 0 for energy calculation.")
        total_spikes_per_sample = 0
    return total_spikes_per_sample * e_ac_paper_j

def calculate_fair_theoretical_energy_from_dict(res, model_key):
    energy_j = np.nan 
    ops_cost = 0
    mem_cost = 0
    source = "Not Calculated"

    # Both quantized spiking models must be recognised as SNNs, or the QSNN gets
    # billed per-MAC while the QDSNN gets billed per-AC -- a 150x apples-to-oranges
    # gap that would flatter the QDSNN for the wrong reason.
    is_snn_type = (model_key.startswith('snn_')
                   or 'snn' in model_key)
    is_quantized = 'quant' in model_key

    try:
        if is_snn_type:
            ac_ops = res.get('ac_ops', res.get('ac_operations', np.nan))
            mem_accesses = res.get('snn_mem_accesses', res.get('snn_memory_access', np.nan))
            precision_cost = E_AC_8BIT_J 
            
            if pd.notna(ac_ops) and ac_ops > 0: 
                ops_cost = float(ac_ops) * precision_cost
                source = "AC Ops"
            else:
                syops = res.get('syops', np.nan)
                if pd.notna(syops) and syops > 0:
                     ops_cost = float(syops) * precision_cost 
                     source = "SyOps (proxy for AC)"
                else:
                    print(f"[Calc FairEnergy WARN] {model_key}: Missing or zero primary SNN Ops (AC Ops/SyOps). Cannot calculate energy.")
                    return np.nan, "Missing SNN Ops"

            if pd.notna(mem_accesses) and mem_accesses > 0:
                mem_cost = float(mem_accesses) * E_MEM_J_PER_BYTE
                source += " + SNN Mem"
            else:
                print(f"[Calc FairEnergy WARN] {model_key}: Missing SNN memory accesses. Energy based only on Ops.")
                source += " (Mem Missing)"
            
            energy_j = ops_cost + mem_cost
            
        else: # ANN models
            mac_ops = res.get('macs', res.get('operation_count', np.nan))
            
            mem_accesses = res.get('ann_mem_accesses', res.get('ann_memory_access', np.nan))
            
            precision_field = res.get('precision', None)
            if precision_field and 'int8' in str(precision_field).lower():
                 is_quantized = True
            elif 'quant' in model_key: 
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

        
        if pd.notna(energy_j) and energy_j >= 0: 
            return energy_j, source
        else:
             
             print(f"[Calc FairEnergy WARN] {model_key}: Final energy calculation resulted in NaN or negative value.")
             return np.nan, "Calculation Failed"
            
    except Exception as e:
        import traceback
        print(f"[Calc FairEnergy ERROR] {model_key}: Error during calculation: {e}")
        traceback.print_exc() 
        return np.nan, "Calculation Error"
