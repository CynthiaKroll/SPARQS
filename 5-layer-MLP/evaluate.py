import argparse
import torch
import time
import os
import csv
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import json

# Import MLP models
from models.mlp import MLP
from models.mlp_snn import MLPSNN
from models.branchy_mlp import BranchyMLP
from models.quantized_snn_mlp import QuantizedSNNMLP
from models.branchy_quantized_snn_mlp import BranchyQuantizedSNNMLP
from models.rl_environment import QLearningAgent

from config import config
from data_loader import get_data_loaders, get_validation_loader
from utils import (calculate_energy, calculate_inference_time, calculate_power, 
                  count_mac_operations, count_memory_access, calculate_theoretical_energy, 
                  count_ac_operations, count_syops, compute_accuracy,
                  calculate_fair_theoretical_energy_from_dict, E_MAC_32BIT_J, E_MAC_8BIT_J, E_AC_8BIT_J, E_MEM_J_PER_BYTE,
                  encode_poisson_rate, DEFAULT_SPIKE_GAIN)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def load_model(model_name, model_path, device, time_steps=None):
    """Load and return the specified model."""
    print(f"Loading model: {model_name} from {model_path}")
    
    # Determine model class
    if model_name == 'standard_mlp':
        model = MLP(num_classes=config['num_classes'])
    elif model_name == 'snn_mlp':
        model = MLPSNN(num_classes=config['num_classes'],
                       beta=config['snn']['beta'],
                       time_steps=time_steps or config['num_steps'])
    elif model_name == 'branchy_mlp':
        model = BranchyMLP(num_classes=config['num_classes'])
    elif model_name == 'quantized_snn_mlp':
        model = QuantizedSNNMLP(num_classes=config['num_classes'],
                                beta=config['snn']['beta'],
                                time_steps=time_steps or config['num_steps'])
    elif model_name == 'branchy_quantized_snn_mlp':
        model = BranchyQuantizedSNNMLP(num_classes=config['num_classes'],
                                       time_window=time_steps or config['num_steps'],
                                       beta=config['snn']['beta'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        try:
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model weights from {model_path}")
        except Exception as e:
            # A converted INT8 checkpoint cannot load into an FP32 skeleton;
            # rebuild the quantized skeleton and load the INT8 weights into that.
            # (This used to exist only for the QDSNN, so the INT8 QSNN silently
            # fell through to "randomly initialized weights".)
            if model_name in ('quantized_snn_mlp', 'branchy_quantized_snn_mlp'):
                print(f"[INFO] Rebuilding INT8 skeleton to load converted checkpoint ({e})")
                from models.quantized_snn_mlp import get_quant_engine
                torch.backends.quantized.engine = get_quant_engine()
                if model_name == 'quantized_snn_mlp':
                    from models.quantized_snn_mlp import prepare_quantized_snn_mlp_for_qat
                    model = prepare_quantized_snn_mlp_for_qat(model)
                    model.eval().cpu()
                    model = torch.quantization.convert(model, inplace=False)
                else:
                    model.fuse_model()
                    model.qconfig = torch.quantization.QConfig(
                        activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
                        weight=torch.quantization.MinMaxObserver.with_args(
                            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
                    torch.quantization.prepare_qat(model, inplace=True)
                    model.eval().cpu()
                    torch.quantization.convert(model, inplace=True)
                # Strict load: a mismatch here must FAIL LOUDLY, not silently
                # evaluate random weights.
                model.load_state_dict(state_dict)
                model.is_quantized = True
                print(f"Loaded converted INT8 {model_name} from {model_path}")
            else:
                print(f"Error loading model weights: {e}")
                print("Using randomly initialized weights")
    else:
        print(f"Model path {model_path} does not exist. Using randomly initialized weights")

    # Load the learned RL exit thresholds for the dynamic QDSNN, if present.
    if model_name == 'branchy_quantized_snn_mlp':
        agent_path = model_path.replace('.pth', '_agent.pkl')
        if os.path.exists(agent_path):
            from models.rl_environment import QLearningAgent
            agent = QLearningAgent(n_exits=getattr(model, 'n_exits', 2))
            agent.load(agent_path)
            model.prepare_for_inference(rl_agent=agent)
            print(f"Loaded RL agent; exit thresholds = {model.confidence_thresholds}")

    # INT8 quantized models must run on CPU; others go to the requested device.
    model.to('cpu' if getattr(model, 'is_quantized', False) else device)
    model.eval()
    return model

def evaluate_model(model, model_name, test_loader, device, hardware_profile=None, force_exit_path=None, time_steps_from_args=None):
    import numpy as np # Ensure numpy is available in this scope
    import pandas as pd # Adding explicit import here
    # Import the centralized calculation function and constants
    from utils import (calculate_fair_theoretical_energy_from_dict, count_mac_operations,
                       count_memory_access, count_ac_operations, E_MAC_32BIT_J, E_MAC_8BIT_J, E_AC_8BIT_J, E_MEM_J_PER_BYTE)
    from collections import OrderedDict
    import time

    # Force CPU for quantized models (same as AlexNet)
    is_quantized_eval = 'quantized' in model_name.lower()
    precision_eval = 'INT8' if is_quantized_eval else 'FP32'

    def _model_device(m):
        # A fully-converted INT8 model can have NO FP32 parameters at all
        # (packed weights are not nn.Parameters), so next(m.parameters())
        # would raise StopIteration.
        for p in m.parameters():
            return p.device
        for b in m.buffers():
            return b.device
        return torch.device('cpu')

    # INT8 quantized kernels only exist on CPU. Moving a converted model to CUDA
    # (as this function used to do, claiming "running natively") segfaults the
    # process on the first forward.
    if getattr(model, 'is_quantized', False):
        device = torch.device('cpu')
        model.to(device)
        print("[INFO] Quantized model pinned to CPU (INT8 kernels are CPU-only)")

    # Use a consistent input shape for op/mem counting (adapted for MNIST MLP)
    # Using the actual input size from the loader if possible, else default
    try:
        sample_batch_for_shape, _ = next(iter(test_loader))
        # For MNIST MLP, we flatten to [batch_size, 784]
        if sample_batch_for_shape.dim() >= 2:
             input_features = sample_batch_for_shape.view(sample_batch_for_shape.size(0), -1).size(1)
             input_shape_for_counts = (input_features,)  # Tuple for MLP input
             # Ensure input_tensor_for_counts is on the same device as the model for op counting
             model_device_for_op_count = _model_device(model)
             input_tensor_for_counts = sample_batch_for_shape[0:1].view(1, -1).to(model_device_for_op_count)
             print(f"[Eval] Using input shape {input_shape_for_counts} from loader for op/mem counts (tensor on {input_tensor_for_counts.device}).")
        else:
             raise ValueError("Loader data has unexpected shape.")
    except Exception as e_shape:
        print(f"[Eval Warning] Could not get input shape from loader ({e_shape}). Using default (784,).")
        input_shape_for_counts = (784,) # Default MNIST flattened shape
        # Create a dummy tensor based on default shape for counting, on model's device
        model_device_for_op_count = _model_device(model)
        input_tensor_for_counts = torch.randn(1, 784, device=model_device_for_op_count)

    # Operation and Memory Counts (Run once before timing loop) - following AlexNet pattern
    mac_to_log_final = 0
    ann_mem_access_final = 0.0
    ac_ops_to_log_final = 0 # Explicitly 0 for non-SNN models
    snn_mem_accesses_val = 0.0 # Explicitly 0 for non-SNN models
    spike_count_val = np.nan
    time_steps_eval = np.nan # No time steps for ANNs

    if time_steps_from_args is not None:
        time_steps_eval = time_steps_from_args
        print(f"[Eval] Using time_steps from args: {time_steps_eval}")
    elif "snn" in model_name.lower(): # Fallback for SNNs if not passed directly
        time_steps_eval = getattr(model, 'time_steps', config.get('num_steps', 32)) # Default SNN steps
        print(f"[Eval] Using model's time_steps or config num_steps for SNN: {time_steps_eval}")

    # Dynamic branchy models dispatch on a custom flag; force inference mode BEFORE
    # counting ops so the early-exit path (not the all-exits training path) is measured.
    if hasattr(model, 'training_mode'):
        model.training_mode = False

    # --- Op / memory profiling ------------------------------------------------
    # For early-exit models the cost of a forward pass depends on which exit a
    # sample takes, so profiling one image reports that image's cost rather than
    # the model's average. Profile each exit path separately here, then weight the
    # paths by the exit distribution measured over the test set after the loop.
    def _run_counts():
        """Count MACs / memory / AC ops for whichever path the model takes now."""
        is_snn_c = "snn" in model_name.lower()
        c = {'macs': np.nan, 'ann_mem': 0.0, 'snn_mem': 0.0, 'ac_ops': 0, 'spikes': np.nan}
        try:
            c['macs'] = count_mac_operations(model, input_tensor_for_counts)
        except Exception as e_mac:
            print(f"[Eval Error] Failed to count MACs for {model_name}: {e_mac}")
        try:
            if is_snn_c:
                c['snn_mem'] = count_memory_access(model, input_tensor_for_counts, is_snn=True, time_steps=time_steps_eval)
            else:
                # For ANN (is_snn=False), count_memory_access expects a single sample tensor
                c['ann_mem'] = count_memory_access(model, input_tensor_for_counts, is_snn=False)
        except Exception as e_mem:
            print(f"[Eval Error] Failed to count Memory Accesses for {model_name}: {e_mem}")
            if is_snn_c:
                c['snn_mem'] = np.nan
            else:
                c['ann_mem'] = np.nan
        if is_snn_c:
            try:
                c['ac_ops'], c['spikes'] = count_ac_operations(model, input_tensor_for_counts, time_steps_eval)
            except Exception as e_ac:
                print(f"[Eval Error] Failed to count AC operations for SNN {model_name}: {e_ac}")
                c['ac_ops'] = np.nan
                c['spikes'] = np.nan
        return c

    # Stays empty for static models, whose cost is data-independent.
    per_path_counts = {}
    if hasattr(model, 'n_exits') and hasattr(model, 'confidence_thresholds'):
        n_ex = model.n_exits
        saved_thresholds = list(model.confidence_thresholds)
        try:
            for path in range(1, n_ex + 2):
                # Confidences are softmax outputs in [0, 1] compared with > or >=,
                # so -1.0 always fires an exit and 2.0 never does. Paths 1..n_ex
                # force that branch; path n_ex+1 forces the final exit.
                forced = [2.0] * n_ex
                if path <= n_ex:
                    forced[path - 1] = -1.0
                model.confidence_thresholds = forced
                print(f"[Eval] Profiling exit path {path}/{n_ex + 1} for {model_name}...")
                per_path_counts[path] = _run_counts()
                p = per_path_counts[path]
                print(f"[Eval]   path {path}: macs={p['macs']}, ac_ops={p['ac_ops']}, "
                      f"snn_mem={p['snn_mem']}, ann_mem={p['ann_mem']}")
        finally:
            model.confidence_thresholds = saved_thresholds
        # Provisional; replaced by the exit-weighted mix once the loop has run.
        base_counts = per_path_counts[1]
    else:
        base_counts = _run_counts()

    mac_to_log_final = base_counts['macs']
    ann_mem_access_final = base_counts['ann_mem']
    snn_mem_accesses_val = base_counts['snn_mem']
    ac_ops_to_log_final = base_counts['ac_ops']
    spike_count_val = base_counts['spikes']
    if not per_path_counts:
        print(f"[Eval] MACs counted: {mac_to_log_final}")
        if "snn" in model_name.lower():
            print(f"[Eval] SNN Memory Accesses counted: {snn_mem_accesses_val}")
            print(f"[Eval] AC operations counted: {ac_ops_to_log_final}, Spikes: {spike_count_val}")
        else:
            print(f"[Eval] ANN Memory Accesses counted: {ann_mem_access_final}")

    # Calculate Fair Theoretical Energy using centralized function - following AlexNet pattern
    fair_theoretical_energy_j_val = np.nan
    calc_source = "Not Calculated"
    temp_res_dict = {
        'macs': mac_to_log_final,
        'ac_ops': ac_ops_to_log_final,
        'ann_mem_accesses': ann_mem_access_final,
        'snn_mem_accesses': snn_mem_accesses_val,
        'precision': precision_eval
    }
    try:
        print(f"[Eval] Calculating Fair Theoretical Energy for {model_name}...")
        fair_theoretical_energy_j_val, calc_source = calculate_fair_theoretical_energy_from_dict(temp_res_dict, model_name)
        if pd.notna(fair_theoretical_energy_j_val):
            print(f"[Eval] Fair Theoretical Energy calculated: {fair_theoretical_energy_j_val:.4g} J (Source: {calc_source})")
        else:
             print(f"[Eval Warning] Fair Theoretical Energy calculation failed. Reason: {calc_source}")
    except Exception as e_fair_eng:
        print(f"[Eval Error] Exception during Fair Theoretical Energy calculation for {model_name}: {e_fair_eng}")
        fair_theoretical_energy_j_val = np.nan
        calc_source = "Calculation Exception"

    # Projected Neuromorphic Energy calculation - following AlexNet pattern
    projected_neuromorphic_energy_val = 0.0
    if "snn" in model_name.lower() and pd.notna(ac_ops_to_log_final) and pd.notna(snn_mem_accesses_val):
        try:
            E_AC_NEUROMORPHIC_J = config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_ac", 0.01e-12)
            E_MEM_NEUROMORPHIC_J_PER_BYTE = config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_mem_access", 2e-12)
            
            neuromorphic_ac_energy = ac_ops_to_log_final * E_AC_NEUROMORPHIC_J
            neuromorphic_mem_energy = snn_mem_accesses_val * E_MEM_NEUROMORPHIC_J_PER_BYTE
            projected_neuromorphic_energy_val = neuromorphic_ac_energy + neuromorphic_mem_energy
            print(f"[Eval] Projected Neuromorphic Energy calculated: {projected_neuromorphic_energy_val:.4e} J")
        except Exception as e:
            print(f"[Eval Error] Exception during Projected Neuromorphic Energy calculation: {e}")

    # Timing and evaluation - following AlexNet pattern but adapted for MLP
    model.eval()

    # Make sure dynamic models are in inference mode regardless of thresholds
    if hasattr(model, 'training_mode'):
        model.training_mode = False
        
    # Disable debug mode during benchmarking to avoid overhead
    if hasattr(model, 'debug_confidence'):
        model.debug_confidence = False
    
    # Perform warmup passes for all models to ensure fair timing comparison
    print(f"Performing warmup inference passes for {model_name}...")
    dummy_input = torch.randn(8, 784, device=device)  # Batch of 8 MNIST flattened images
    with torch.no_grad():
        for _ in range(5):  # 5 warmup passes
            if "snn" in model_name.lower():
                # Create spike train for SNN models during warmup (canonical encoder)
                spike_warmup = encode_poisson_rate(dummy_input, int(time_steps_eval))
                _ = model(spike_warmup)
            else:
                _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None  # Ensure GPU operations complete
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()  # Ensure MPS operations complete
            
    total_correct = 0
    total_samples = 0
    avg_time = 0
    
    # Track exit distribution for dynamic models. Exit points are 1-based
    # (1..n_exits, plus the final exit = n_exits+1), so key the counts the same way.
    exit_counts = {}
    if hasattr(model, 'n_exits'):
        exit_counts = {i: 0 for i in range(1, model.n_exits + 2)}
    # Exit-aware delay tracking (average exit depth and timesteps-to-exit per sample)
    exit_depth_sum = 0.0
    exit_ts_sum = 0.0
    exit_delay_count = 0
    
    print(f"Evaluating {model_name}...")
    
    # Timing measurement
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.view(images.size(0), -1).to(device)  # Flatten for MLP
            labels = labels.to(device)
            
            # Handle SNN models with proper spike encoding
            if "snn" in model_name.lower():
                # Canonical Poisson rate encoding, identical to training and to the
                # op/energy counters (see utils.encode_poisson_rate).
                spike_data = encode_poisson_rate(images, int(time_steps_eval))
                
                if "branchy" in model_name.lower():
                    # Branchy SNN models
                    result = model(spike_data)
                    if isinstance(result, tuple) and len(result) == 3:
                        exit_outputs, final_output, exit_points = result
                    elif isinstance(result, tuple) and len(result) == 2:
                        exit_outputs, final_output = result
                        exit_points = None
                    else:
                        final_output = result
                        exit_outputs = []
                        exit_points = None
                    
                    # Sum over time for each exit
                    if exit_outputs:
                        exit_outputs = [out.sum(dim=0) if out.dim() == 3 else out for out in exit_outputs]
                    if isinstance(final_output, torch.Tensor) and final_output.dim() == 3:
                        final_output = final_output.sum(dim=0)

                    # Score each sample by the exit it actually took. final_output
                    # is FROZEN (near-zero) for early-exited samples, so using it
                    # for everyone reported chance-level accuracy whenever the
                    # early exits engaged.
                    outputs = final_output
                    if exit_points is not None and exit_outputs:
                        outputs = final_output.clone()
                        pts = exit_points.to(outputs.device)
                        for i, eo in enumerate(exit_outputs):
                            mask = pts == (i + 1)
                            if mask.any():
                                outputs[mask] = eo[mask]
                else:
                    # Standard SNN models
                    outputs = model(spike_data)
                    if isinstance(outputs, torch.Tensor) and outputs.dim() == 3:  # [time_steps, batch_size, num_classes]
                        outputs = outputs.sum(dim=0)  # Sum over time
            else:
                # Handle non-SNN models
                if "branchy" in model_name.lower():
                    result = model(images)
                    if isinstance(result, tuple) and len(result) == 3:
                        exit_outputs, final_output, exit_points = result
                    elif isinstance(result, tuple) and len(result) == 2:
                        exit_outputs, final_output = result
                        exit_points = None
                    else:
                        final_output = result
                        exit_outputs = []
                        exit_points = None

                    # BranchyMLP composes exit outputs into final_output itself
                    # (exit_outputs is empty), so this is a no-op there; kept for
                    # symmetry with the SNN branch above.
                    outputs = final_output
                    if exit_points is not None and exit_outputs:
                        outputs = final_output.clone()
                        pts = exit_points.to(outputs.device)
                        for i, eo in enumerate(exit_outputs):
                            mask = pts == (i + 1)
                            if mask.any():
                                outputs[mask] = eo[mask]
                else:
                    outputs = model(images)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # Track exit statistics for branchy models
            if "branchy" in model_name.lower() and exit_points is not None:
                batch_size = images.size(0)
                for i in range(batch_size):
                    exit_pt = exit_points[i].item() if hasattr(exit_points[i], 'item') else 2
                    if exit_pt in exit_counts:
                        exit_counts[exit_pt] += 1
                # Exit-aware delay: average exit depth and timesteps used per sample
                exit_depth_sum += float(exit_points.float().sum().item())
                exit_delay_count += batch_size
                last_ts = getattr(model, 'last_exit_timesteps', None)
                if last_ts is not None:
                    exit_ts_sum += float(last_ts.float().sum().item())
    
    avg_time = (time.time() - start_time) / total_samples if total_samples > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Calculate per-step timing for SNNs
    avg_inference_time_per_step_ms = np.nan
    if "snn" in model_name.lower() and pd.notna(time_steps_eval):
        avg_inference_time_per_step_ms = (avg_time * 1000) / time_steps_eval
    
    # Exit-weighted cost for early-exit models: X = sum_k p_k * X_k, combining the
    # per-path counts profiled above with the exit distribution just measured over
    # the whole test set. Without this the reported cost is whichever path the
    # single profiling image happened to take.
    if per_path_counts:
        total_exits = sum(exit_counts.values())
        if total_exits > 0:
            weights = {k: exit_counts.get(k, 0) / total_exits for k in per_path_counts}

            def _mix(field):
                return sum(weights[k] * float(per_path_counts[k][field]) for k in per_path_counts)

            mac_to_log_final = _mix('macs')
            ann_mem_access_final = _mix('ann_mem')
            snn_mem_accesses_val = _mix('snn_mem')
            ac_ops_to_log_final = _mix('ac_ops')
            spike_count_val = _mix('spikes') if pd.notna(base_counts['spikes']) else np.nan

            w_str = ", ".join(f"{k}:{v:.4f}" for k, v in weights.items())
            print(f"[Eval] Exit-weighted over {total_exits} samples (p = {{{w_str}}}): "
                  f"macs={mac_to_log_final:.1f}, ac_ops={ac_ops_to_log_final:.1f}, "
                  f"snn_mem={snn_mem_accesses_val:.1f}, ann_mem={ann_mem_access_final:.1f}")

            temp_res_dict = {
                'macs': mac_to_log_final,
                'ac_ops': ac_ops_to_log_final,
                'ann_mem_accesses': ann_mem_access_final,
                'snn_mem_accesses': snn_mem_accesses_val,
                'precision': precision_eval,
            }
            fair_theoretical_energy_j_val, calc_source = calculate_fair_theoretical_energy_from_dict(
                temp_res_dict, model_name)
            if pd.notna(fair_theoretical_energy_j_val):
                print(f"[Eval] Exit-weighted Fair Theoretical Energy: "
                      f"{fair_theoretical_energy_j_val:.4g} J (Source: {calc_source})")
            else:
                print(f"[Eval Warning] Exit-weighted energy calculation failed. Reason: {calc_source}")

            if "snn" in model_name.lower() and pd.notna(ac_ops_to_log_final) and pd.notna(snn_mem_accesses_val):
                E_AC_N = config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_ac", 0.01e-12)
                E_MEM_N = config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_mem_access", 2e-12)
                projected_neuromorphic_energy_val = (ac_ops_to_log_final * E_AC_N
                                                     + snn_mem_accesses_val * E_MEM_N)
                print(f"[Eval] Exit-weighted Projected Neuromorphic Energy: "
                      f"{projected_neuromorphic_energy_val:.4e} J")
        else:
            print(f"[Eval Warning] No exit statistics collected for {model_name}; "
                  f"reported cost reflects the exit-1 path only.")

    # Energy calculation
    avg_energy = 0
    avg_power = 0
    
    # Use theoretical energy for ALL models now (AlexNet pattern)
    if pd.notna(fair_theoretical_energy_j_val):
        # Replace hardware-measured energy with theoretical energy
        avg_energy = fair_theoretical_energy_j_val
        # Calculate theoretical power using theoretical energy and measured time
        if avg_time > 0:
            avg_power = avg_energy / avg_time
        print(f"  Using Theoretical Energy: {avg_energy:.6e} J")
        print(f"  Using Theoretical Power: {avg_power:.4f} W (Energy/Time)")
    else:
        print(f"[WARN] Theoretical energy calculation failed for {model_name}. Energy and power will remain 0.")

    # Prepare results following AlexNet pattern exactly
    results_data = OrderedDict([
        ("model_name", model_name),
        ("device", str(device)),
        ("accuracy", avg_acc),
        ("energy", avg_energy), 
        ("inference_time", avg_time),
        ("power", avg_power),
        ("macs", mac_to_log_final),
        ("ann_mem_accesses", ann_mem_access_final if "snn" not in model_name.lower() else np.nan), # Bytes
        ("ann_theoretical_energy", avg_energy if "snn" not in model_name.lower() else np.nan), # Joules, MAC-based for ANNs
        ("dynamic_theoretical_energy", avg_energy), # Joules, same as energy for consistency with LeNet
        # SNN specific
        ("time_steps_configured", time_steps_eval if "snn" in model_name.lower() else np.nan), # Log the T used
        ("avg_inference_time_per_step_ms", avg_inference_time_per_step_ms if "snn" in model_name.lower() else np.nan), # Log per-step time for SNNs
        ("ac_ops", ac_ops_to_log_final), # AC operations for SNNs
        ("spike_count", spike_count_val), # Spike count for SNNs
        ("snn_mem_accesses", snn_mem_accesses_val), # Bytes, calculated for SNNs
        ("fair_theoretical_energy_j", fair_theoretical_energy_j_val), # Joules, AC-based for SNNs, MAC-based for ANNs
        ("projected_neuromorphic_energy", projected_neuromorphic_energy_val), # Joules
        # Exit-aware delay metrics (dynamic early-exit models); NaN for static models
        ("avg_exit_depth", (exit_depth_sum / exit_delay_count) if exit_delay_count else np.nan),
        ("avg_timesteps_to_exit", (exit_ts_sum / exit_delay_count) if exit_delay_count else np.nan),
    ])

    # Handle exit statistics for branchy models (like LeNet)
    if exit_counts:
        total_samples_eval = sum(exit_counts.values())
        results_data["total_samples"] = total_samples_eval
        
        # Convert to percentages like LeNet format. Exit counts are 1-based
        # (1..n_exits, final exit = n_exits+1 = the highest key); the old code
        # read key 0 for the final exit, which always reported 0%.
        if len(exit_counts) >= 3:  # Standard 3-exit branchy model
            final_exit_key = max(exit_counts)
            results_data["exit1_percentage"] = (exit_counts.get(1, 0) / total_samples_eval * 100) if total_samples_eval > 0 else 0
            results_data["exit2_percentage"] = (exit_counts.get(2, 0) / total_samples_eval * 100) if total_samples_eval > 0 else 0
            results_data["final_exit_percentage"] = (exit_counts.get(final_exit_key, 0) / total_samples_eval * 100) if total_samples_eval > 0 else 0
    
    # Convert OrderedDict to DataFrame for saving
    df_list = {k: [v] for k,v in results_data.items()}
    df = pd.DataFrame(df_list)

    log_path = f"results/logs/{model_name}_eval.csv"
    df.to_csv(log_path, index=False)
    print(f"Results saved to {log_path}")
    
    # Return comprehensive metrics dictionary (matching AlexNet/LeNet pattern)
    return dict(results_data)

def save_results(results, csv_path):
    """Save evaluation results to CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Handle exit distribution properly
    if "exit_distribution" in results and results["exit_distribution"]:
        # Convert exit_distribution to individual columns like LeNet
        exit_dist = results["exit_distribution"]
        total_samples = sum(exit_dist.values())
        
        if len(exit_dist) >= 3:  # Standard 3-exit branchy model
            # Exit points are 1-based; the final exit is the highest key.
            results["exit1_percentage"] = (exit_dist.get(1, 0) / total_samples * 100) if total_samples > 0 else 0
            results["exit2_percentage"] = (exit_dist.get(2, 0) / total_samples * 100) if total_samples > 0 else 0
            results["final_exit_percentage"] = (exit_dist.get(max(exit_dist), 0) / total_samples * 100) if total_samples > 0 else 0
        
        results["total_samples"] = total_samples
        # Remove the original exit_distribution to match LeNet format
        del results["exit_distribution"]
    
    # Convert results to DataFrame
    df = pd.DataFrame([results])
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP models on MNIST")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "standard_mlp", "snn_mlp", "branchy_mlp", 
                                "quantized_snn_mlp", "branchy_quantized_snn_mlp"],
                        help="Model to evaluate")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model weights")
    parser.add_argument("--batch_size", type=int, default=config["test_batch_size"],
                        help="Batch size for evaluation")
    parser.add_argument("--time_steps", type=int, default=config["num_steps"],
                        help="Time steps for SNN models")
    parser.add_argument("--thresholds", type=float, nargs=2, default=None,
                        help="Fixed exit confidence thresholds for branchy models "
                             "(e.g. --thresholds 0.2 0.4). Overrides the RL agent's "
                             "learned thresholds. NOTE: thresholds are NOT stored in "
                             "checkpoints, so the paper's Cfg rows must be produced "
                             "by evaluating the QDSNN checkpoint with this flag.")
    parser.add_argument("--output_dir", type=str, default="results/logs",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Load data
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    
    # Define model configurations
    model_configs = {
        "standard_mlp": {
            "path": "results/standard_mlp.pth",
            "snn": False,
            "branchy": False
        },
        "snn_mlp": {
            "path": "results/snn_mlp.pth", 
            "snn": True,
            "branchy": False
        },
        "branchy_mlp": {
            "path": "results/branchy_mlp.pth",
            "snn": False,
            "branchy": True
        },
        "quantized_snn_mlp": {
            "path": "results/quantized_snn_mlp.pth",
            "snn": True,
            "branchy": False
        },
        "branchy_quantized_snn_mlp": {
            "path": "results/branchy_quantized_snn_mlp.pth",
            "snn": True,
            "branchy": True
        }
    }
    
    # Determine which models to evaluate
    if args.model == "all":
        models_to_eval = list(model_configs.keys())
    else:
        models_to_eval = [args.model]
    
    all_results = []
    
    for model_name in models_to_eval:
        try:
            config_info = model_configs[model_name]
            model_path = args.model_path or config_info["path"]
            
            # Load model
            model = load_model(model_name, model_path, device, args.time_steps)

            # Optional fixed exit thresholds (paper "Cfg" rows). These override
            # the RL agent's learned thresholds for this evaluation only. The
            # eval name gets a cfg suffix so each threshold configuration writes
            # its own CSV instead of overwriting the RL row's.
            eval_name = model_name
            if args.thresholds and hasattr(model, 'prepare_for_inference'):
                model.prepare_for_inference(confidence_thresholds=list(args.thresholds))
                eval_name = f"{model_name}_cfg{args.thresholds[0]:g}_{args.thresholds[1]:g}"
                print(f"[Eval] Using fixed exit thresholds {list(args.thresholds)} -> {eval_name}")

            # Evaluate once per model (saves comprehensive CSV internally).
            # The old "exit strategy" sweep evaluated the SAME dynamic model
            # three times under different names -- force_exit_path was never
            # implemented -- producing three identical rows labelled as if they
            # were different exit policies.
            results = evaluate_model(model, eval_name, test_loader, device,
                                     time_steps_from_args=args.time_steps)
            all_results.append(results)
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_path = os.path.join(args.output_dir, "all_models_eval.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined results saved to {combined_path}")
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()