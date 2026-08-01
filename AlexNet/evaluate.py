import argparse
import csv
import json
import os
import pickle
import random
import subprocess
import time
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.alexnet import AlexNet
from models.branchy_models import BranchyAlexNet
from models.rl_environment import QLearningAgent
import pickle
from config import config
from data_loader import get_data_loaders
from utils import calculate_energy, calculate_inference_time, calculate_power, count_mac_operations, count_memory_access, calculate_theoretical_energy, count_ac_operations, calculate_fair_theoretical_energy_from_dict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

USE_NVML = False
if torch.cuda.is_available():
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        USE_NVML = True
    except Exception as e:
        print(f"Warning: NVML init failed: {e}")
else:
    print("Warning: CUDA unavailable; energy measurement disabled.")



def verify_quantization(model):
    quantized_convs = 0
    quantized_linears = 0
    fp32_exit_linears = 0
    batchnorm_count = 0
    for name, module in model.named_modules():
        # Check Conv2d quantization
        if isinstance(module, torch.nn.Conv2d):
            assert isinstance(module, torch.nn.quantized.Conv2d), f"[VERIFY ERROR] Conv2d layer {name} is not quantized!"
            quantized_convs += 1
        # Check Linear quantization
        if isinstance(module, torch.nn.Linear):
            if 'exit' in name.lower():
                # Exit heads must be FP32
                assert module.weight.dtype == torch.float32, f"[VERIFY ERROR] Exit Linear layer {name} is not FP32!"
                fp32_exit_linears += 1
            else:
                assert isinstance(module, torch.nn.quantized.Linear), f"[VERIFY ERROR] Linear layer {name} is not quantized!"
                quantized_linears += 1
        # Check BatchNorm fusion (should be fused away)
        if isinstance(module, torch.nn.BatchNorm2d):
            batchnorm_count += 1
    print(f"[VERIFY] Quantized Conv2d layers: {quantized_convs}")
    print(f"[VERIFY] Quantized Linear layers (excluding exit heads): {quantized_linears}")
    print(f"[VERIFY] FP32 Linear exit heads: {fp32_exit_linears}")
    if batchnorm_count > 0:
        print(f"[VERIFY WARNING] {batchnorm_count} BatchNorm2d layers remain! BatchNorm fusion may be incomplete.")
    else:
        print("[VERIFY] BatchNorm fusion verified: no BatchNorm2d layers remain.")

def calibrate_temperatures(model, loader, device, exits_to_calibrate=(0, 1)):
    """Calibrate temperature scaling parameters for early exits."""
    if not hasattr(model, 'temperature1') or not hasattr(model, 'temperature2'):
        print("[WARN] Model does not have temperature1/temperature2 attributes. Skipping calibration.")
        return model

    model.to(device)
    model.eval() # Ensure model is in eval mode

    # Parameters to optimize (only temperatures)
    params_to_optimize = []
    if 0 in exits_to_calibrate and hasattr(model, 'temperature1'):
        params_to_optimize.append(model.temperature1)
        print(f"Initial T1: {model.temperature1.item()}")
    if 1 in exits_to_calibrate and hasattr(model, 'temperature2'):
        params_to_optimize.append(model.temperature2)
        print(f"Initial T2: {model.temperature2.item()}")

    if not params_to_optimize:
        print("[WARN] No valid temperature parameters found for calibration.")
        return model

    optimizer = torch.optim.LBFGS(params_to_optimize, lr=0.01, max_iter=50) # LBFGS is often used for temp scaling
    nll_criterion = torch.nn.CrossEntropyLoss().to(device) # Make sure nn is imported or use torch.nn

    print(f"Calibrating temperatures for exits {exits_to_calibrate}...")

    # Extract logits and labels for relevant exits from a few batches
    exit_logits_all = {i: [] for i in exits_to_calibrate}
    labels_all = {i: [] for i in exits_to_calibrate}

    with torch.no_grad(): # Collect data first
        for images, labels in tqdm(loader, desc="Collecting calibration data"):
            images, true_labels = images.to(device), labels.to(device)
            # Assuming model returns (exit_outputs_logits, final_output_logits, exit_points_info)
            # And exit_outputs_logits is a list of tensors
            model_output = model(images)
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                exit_outputs_logits = model_output[0] # This should be a list of exit logits
            else:
                print("[ERROR] calibrate_temperatures: Unexpected model output format. Cannot extract exit logits.")
                return model


            if 0 in exits_to_calibrate and len(exit_outputs_logits) > 0:
                exit_logits_all[0].append(exit_outputs_logits[0])
                labels_all[0].append(true_labels)
            if 1 in exits_to_calibrate and len(exit_outputs_logits) > 1:
                exit_logits_all[1].append(exit_outputs_logits[1])
                labels_all[1].append(true_labels)
    
    for exit_idx in exits_to_calibrate:
        if not exit_logits_all[exit_idx]: # Check if list is empty
            print(f"No data collected for exit {exit_idx +1}, skipping its calibration.")
            continue

        current_logits = torch.cat(exit_logits_all[exit_idx])
        current_labels = torch.cat(labels_all[exit_idx])
        temperature_param = model.temperature1 if exit_idx == 0 else model.temperature2

        print(f"Optimizing T for exit {exit_idx + 1} (param: {'model.temperature1' if exit_idx == 0 else 'model.temperature2'})...")
        print(f"  Logits for exit {exit_idx + 1} - Mean: {current_logits.mean().item():.4f}, Std: {current_logits.std().item():.4f}, Min: {current_logits.min().item():.4f}, Max: {current_logits.max().item():.4f}")
        
        dedicated_optimizer = torch.optim.LBFGS([temperature_param], lr=0.01, max_iter=50, line_search_fn="strong_wolfe")

        # Min and Max bounds for temperature
        MIN_TEMP = 0.1
        MAX_TEMP = 5.0 # MODIFIED: Clamp max temperature to a more reasonable value

        def eval_loss_and_clamp(): # Closure for LBFGS
            dedicated_optimizer.zero_grad()
            loss = nll_criterion(current_logits / temperature_param, current_labels)
            loss.backward()
            # Clamp temperature after gradient computation and before optimizer step uses it (though LBFGS is complex)
            # More direct clamping happens after the step.
            return loss
        
        for _ in range(10): # LBFGS often benefits from a few restarts or manual iteration control
            dedicated_optimizer.step(eval_loss_and_clamp)
            with torch.no_grad(): # Manually clamp temperature after each LBFGS step
                temperature_param.data.clamp_(MIN_TEMP, MAX_TEMP)
            # print(f"  Iter {_ + 1}: T={temperature_param.item():.4f}") # Optional: print T at each iteration

        print(f"Calibrated T{exit_idx + 1}: {temperature_param.item()}")

    return model

def get_device():
    """
    Check for available GPU devices and return the appropriate device.
    Priority: CUDA > MPS > CPU
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    # Fallback to CPU
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    
    return device

def evaluate_model(model, model_name, test_loader, device, hardware_profile=None, force_exit_path=None, time_steps_from_args=None):
    import numpy as np # Ensure numpy is available in this scope
    import pandas as pd # <--- ADDING EXPLICIT IMPORT HERE
    # Import the centralized calculation function and constants
    from utils import (calculate_fair_theoretical_energy_from_dict, count_mac_operations,
                       count_memory_access, E_MAC_32BIT_J, E_MAC_8BIT_J, E_AC_8BIT_J, E_MEM_J_PER_BYTE)

    # Force CPU for quantized models
    is_quantized_eval = False # Flag for energy calculation
    precision_eval = 'FP32' # Default precision

    # Use a consistent input shape for op/mem counting
    # Using the actual input size from the loader if possible, else default
    try:
        sample_batch_for_shape, _ = next(iter(test_loader))
        # Need C, H, W for count_mac_operations and count_memory_access
        if sample_batch_for_shape.dim() >= 4:
             _, C_shape, H_shape, W_shape = sample_batch_for_shape.shape[:4]
             input_shape_for_counts = (C_shape, H_shape, W_shape)
             # Ensure input_tensor_for_counts is on the same device as the model for op counting
             # This is especially important if the model is moved to CPU (e.g., for quantized eval)
             # while the test_loader provides data on another device (e.g., CUDA).
             model_device_for_op_count = next(model.parameters()).device
             input_tensor_for_counts = sample_batch_for_shape[0:1].to(model_device_for_op_count)
             print(f"[Eval] Using input shape {input_shape_for_counts} from loader for op/mem counts (tensor on {input_tensor_for_counts.device}).")
        else:
             raise ValueError("Loader data has unexpected shape.")
    except Exception as e_shape:
        print(f"[Eval Warning] Could not get input shape from loader ({e_shape}). Using default (3, 32, 32).")
        input_shape_for_counts = (3, 32, 32) # Default CIFAR-10 shape
        # Create a dummy tensor based on default shape for counting, on model's device
        model_device_for_op_count = next(model.parameters()).device
        input_tensor_for_counts = torch.randn(1, *input_shape_for_counts, device=model_device_for_op_count)

    # Operation and Memory Counts (Run once before timing loop)
    macs = 0
    ann_mem_accesses = 0.0
    ac_ops = 0 # Explicitly 0 for models evaluated here
    snn_mem_accesses = 0.0 # Explicitly 0 for models evaluated here
    time_steps_eval = np.nan # No time steps for ANNs

    if time_steps_from_args is not None:
        time_steps_eval = time_steps_from_args
        print(f"[Eval] Using time_steps from args: {time_steps_eval}")
    elif "snn" in model_name.lower(): # Fallback for SNNs if not passed directly
        time_steps_eval = getattr(model, 'time_window', config.get('num_steps', 25)) # Default SNN steps
        print(f"[Eval] Using model's time_window or config num_steps for SNN: {time_steps_eval}")


    try:
        print(f"[Eval] Counting MACs for {model_name}...")
        macs = count_mac_operations(model, input_tensor_for_counts) # Pass tensor directly
        print(f"[Eval] MACs counted: {macs}")
    except Exception as e_mac:
        print(f"[Eval Error] Failed to count MACs for {model_name}: {e_mac}")
        macs = np.nan # Indicate failure

    try:
        print(f"[Eval] Counting Memory Accesses for {model_name}...")
        # For ANN (is_snn=False), count_memory_access expects a single sample tensor
        # It calculates params + avg activations per pass. Batch size doesn't directly apply here.
        ann_mem_accesses = count_memory_access(model, input_tensor_for_counts, is_snn=False)
        print(f"[Eval] ANN Memory Accesses counted: {ann_mem_accesses}")
    except Exception as e_mem:
        print(f"[Eval Error] Failed to count Memory Accesses for {model_name}: {e_mem}")
        ann_mem_accesses = np.nan # Indicate failure

    # Calculate Fair Theoretical Energy using centralized function
    fair_theoretical_energy_j = np.nan
    calc_source = "Not Calculated"
    temp_res_dict = {
        'macs': macs,
        'ac_ops': ac_ops, # 0 for ANN
        'ann_mem_accesses': ann_mem_accesses,
        'snn_mem_accesses': snn_mem_accesses, # 0 for ANN
        'precision': precision_eval # Pass determined precision
        # Add other fields if the calculation function needs them (e.g., device?)
    }
    try:
        print(f"[Eval] Calculating Fair Theoretical Energy for {model_name}...")
        fair_theoretical_energy_j, calc_source = calculate_fair_theoretical_energy_from_dict(temp_res_dict, model_name)
        if pd.notna(fair_theoretical_energy_j):
            print(f"[Eval] Fair Theoretical Energy calculated: {fair_theoretical_energy_j:.4g} J (Source: {calc_source})")
        else:
             print(f"[Eval Warning] Fair Theoretical Energy calculation failed. Reason: {calc_source}")
    except Exception as e_fair_eng:
        print(f"[Eval Error] Exception during Fair Theoretical Energy calculation for {model_name}: {e_fair_eng}")
        fair_theoretical_energy_j = np.nan
        calc_source = "Calculation Exception"


    # Timing loop
    start_time = time.time()
    """
    Evaluate a model on the test set.
    
    Args:
        model: The model to evaluate
        model_name: Name of the model for logging
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Only call eval() for non-quantized models
    if not getattr(model, 'is_quantized', False):
        model.eval()
    
    # For quantized models, run native INT8 inference on specified device
    if getattr(model, 'is_quantized', False):
        print(f"[INFO] Quantized model running natively on {device}")
        model.to(device)
    
    # Make sure dynamic models are in inference mode regardless of thresholds
    if hasattr(model, 'training_mode'):
        model.training_mode = False
        
    # Disable debug mode during benchmarking to avoid overhead
    if hasattr(model, 'debug_confidence'):
        model.debug_confidence = False
    
    # Perform warmup passes for all models to ensure fair timing comparison
    print(f"Performing warmup inference passes for {model_name}...")
    dummy_input = torch.randn(8, 3, 32, 32, device=device)  # Batch of 8 CIFAR-10 sized images
    with torch.no_grad():
        for _ in range(5):  # 5 warmup passes
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None  # Ensure GPU operations complete
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()  # Ensure MPS operations complete
            
    total_correct = 0
    total_samples = 0
    total_time = 0
    
    # Track exit distribution for dynamic models
    exit_counts = defaultdict(int)
    total_exit_samples = 0
    
    print(f"Evaluating {model_name}...")
    
    # For pure quantized dynamic SNN, compute and save all fair-comparison metrics
    if model_name == "quantized_dynamic_snn":
        print(f"[Quantized] Evaluating {model_name} on CPU (simple inference only)")
        total_correct = 0
        total_samples = 0
        import pandas as pd
        mac_count = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                model_out = model(images)
                outputs = model_out[-1] if isinstance(model_out, (tuple, list)) else model_out
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                if mac_count == 0:
                    mac_count = count_mac_operations(model, images)
        acc = total_correct / total_samples if total_samples > 0 else 0.0
        dummy_sample = torch.randn(1, 3, 32, 32, device=images.device)
        ann_mem_accesses = count_memory_access(model, dummy_sample, is_snn=False)
        _, ann_theoretical_energy = calculate_theoretical_energy(mac_count, 0, 0, 0, ann_mem_accesses)
        projected_neuromorphic_energy = 0.0
        # For fair comparison, try to estimate spike_count and projected_inference_time if possible
        spike_count = 0  # Set to 0 or calculate if available for quantized_dynamic_snn
        # If you have a spike counting utility, use it here; otherwise leave as 0 or 'N/A'
        try:
            spike_count = count_ac_operations(model, dummy_sample, time_steps=1)[1]  # [1] is spike_count
            # For quantized SNNs, scale by typical number of time steps if known, or leave as is
            # Use time_steps_eval here, which should be correctly set from args or model/config
            projected_inference_time = spike_count * 1e-6 * time_steps_eval if isinstance(spike_count, (int, float)) and pd.notna(time_steps_eval) else 'N/A'
        except Exception as e:
            spike_count = 'N/A'
            projected_inference_time = 'N/A'
        results_dict = {
            "model_name": [model_name],
            "device": [str(images.device)],
            "accuracy": [acc],
            # Compute theoretical energy for quantized SNN (ANN logic, since MACs dominate)
            "energy": [ann_theoretical_energy],
            "inference_time": [0.0],
            "power": [0.0],
            "macs": [mac_count],
            "ann_mem_accesses": [ann_mem_accesses],
            "ann_theoretical_energy": [ann_theoretical_energy],
            "projected_neuromorphic_energy": [projected_neuromorphic_energy],
            "spike_count": [spike_count],
            "projected_inference_time": [projected_inference_time],
        }
        print(f"[DEBUG] MAC count: {mac_count}")
        print(f"[DEBUG] ANN theoretical energy: {ann_theoretical_energy}")
        print(f"[DEBUG] Projected neuromorphic energy: {projected_neuromorphic_energy}")
        print(f"[DEBUG] Spike count: {spike_count}")
        print(f"[DEBUG] Projected inference time: {projected_inference_time}")
        log_path = f"results/logs/{model_name}_eval.csv"
        df = pd.DataFrame(results_dict)
        print("Saving DataFrame with columns:", df.columns.tolist())
        print(df.head())
        # Attach per-exit accuracy/entropy to results_dict if available
        if 'per_exit_preds' in locals():
            per_exit_accuracy = {}
            per_exit_entropy = {}
            for k in [0, 1, 2]:
                preds = np.array(per_exit_preds[k])
                trues = np.array(per_exit_labels[k])
                entropies = np.array(per_exit_entropies[k])
                acc = float((preds == trues).mean()) if len(preds) > 0 else np.nan
                mean_entropy = float(entropies.mean()) if len(entropies) > 0 else np.nan
                per_exit_accuracy[k] = acc
                per_exit_entropy[k] = mean_entropy
            results_dict["per_exit_accuracy"] = per_exit_accuracy
            results_dict["per_exit_entropy"] = per_exit_entropy

        # Save overall evaluation CSV
        df.to_csv(log_path, index=False)
        print(f"[INFO] Evaluation results saved to {log_path}")
        # Per-exit energy/timing profiling if exit info is available
        if "exit_distribution" in results_dict or "exit_counts" in results_dict or "spike_count" in results_dict:
            # Try to extract per-exit info from exit_counts or exit_distribution
            exit_counts = results_dict.get("exit_distribution") or results_dict.get("exit_counts")
            if exit_counts and isinstance(exit_counts, dict):
                import matplotlib.pyplot as plt
                import csv
                import numpy as np
                # Hardware scaling
                scaling = 1.0
                if hardware_profile is not None:
                    hw = hardware_profile.lower()
                    if 'a100' in hw:
                        scaling = 0.7
                    elif 'loihi' in hw:
                        scaling = 0.1
                    elif 'cpu' in hw:
                        scaling = 1.5
                    print(f"[INFO] Applying hardware scaling factor {scaling} for profile '{hardware_profile}'")
                # Per-exit stats
                total = sum(exit_counts.values())
                per_exit_energy = [df["energy"].iloc[0] * (exit_counts[k]/total) * scaling for k in exit_counts]
                per_exit_time = [df["inference_time"].iloc[0] * (exit_counts[k]/total) for k in exit_counts]
                per_exit_spikes = [df["spike_count"].iloc[0] if "spike_count" in df.columns else 0 for k in exit_counts]
                labels = [f"Exit {k+1}" if k < len(exit_counts)-1 else "Final Exit" for k in exit_counts]
                # Per-exit accuracy and entropy: dummy logic (replace with real per-exit stats if available)
                per_exit_acc = [np.nan] * len(labels)
                per_exit_entropy = [np.nan] * len(labels)
                if "per_exit_accuracy" in results_dict:
                    per_exit_acc = [results_dict["per_exit_accuracy"].get(k, np.nan) for k in exit_counts]
                if "per_exit_entropy" in results_dict:
                    per_exit_entropy = [results_dict["per_exit_entropy"].get(k, np.nan) for k in exit_counts]
                # Save CSV
                csv_path = "energy_timing_per_exit.csv"
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Exit", "Energy (J)", "Inference Time (s)", "Spike Count", "Accuracy", "Entropy", "Hardware Profile", "Scaling Factor"])
                    for label, e, t, s, acc, ent in zip(labels, per_exit_energy, per_exit_time, per_exit_spikes, per_exit_acc, per_exit_entropy):
                        writer.writerow([label, e, t, s, acc, ent, hardware_profile or 'default', scaling])
                print(f"[INFO] Per-exit energy/timing CSV saved to {csv_path}")
                print("Per-exit stats:")
                for label, acc, ent in zip(labels, per_exit_acc, per_exit_entropy):
                    print(f"  {label}: Accuracy={acc}, Entropy={ent}")
                # Plot
                plt.figure(figsize=(7,4))
                plt.bar(labels, per_exit_energy, color='orange')
                plt.ylabel("Energy (J)")
                plt.title(f"Per-Exit Energy ({hardware_profile or 'default'})")
                plt.tight_layout()
                plt.savefig("energy_per_exit.png")
                plt.close()
                print(f"[INFO] Per-exit energy plot saved to energy_per_exit.png")
                plt.figure(figsize=(7,4))
                plt.bar(labels, per_exit_time, color='green')
                plt.ylabel("Inference Time (s)")
                plt.title("Per-Exit Inference Time")
                plt.tight_layout()
                plt.savefig("inference_time_per_exit.png")
                plt.close()
                print(f"[INFO] Per-exit inference time plot saved to inference_time_per_exit.png")

        # Otherwise, already saved overall summary above
        df.to_csv(log_path, index=False)
        print(f"[Quantized] Results saved to {log_path}")
        return {"accuracy": acc, "total_samples": total_samples}
    
    # Use NVML total-energy API and CUDA events for energy/time measurement
    if USE_NVML:
        # Sync and record start energy/time
        torch.cuda.synchronize()
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(test_loader), desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Ensure GPU/MPS operations from previous iterations are complete
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.synchronize()
            
            # Start timing
            start_time = time.time()
            
            # Set model to inference mode if applicable
            if hasattr(model, 'training_mode'):
                model.training_mode = False
            
            # Run inference
            model_output = model(images)

            # --- ADDED: Direct check of model_name and condition ---
            if i == 0: # Only print for the first batch of each eval run
                print(f"[DEBUG CHECK] Current model_name: '{model_name}'")
                print(f"[DEBUG CHECK] Condition ('branchy_quantized_snn' in model_name) is: {'branchy_quantized_snn' in model_name}")
            # --- END ADDED DEBUG ---
            
            # --- MODIFIED BLOCK for BranchyQuantizedSNN ---
            if "branchy_quantized_snn" in model_name: # <<< MODIFIED CONDITION HERE
                exit_logits_list = None
                final_logits = None
                preds = None # Initialize preds

                # --- Robustly unpack model output ---
                if isinstance(model_output, tuple):
                    if len(model_output) >= 2:
                        exit_logits_list = model_output[0]
                        final_logits = model_output[1]
                        if not isinstance(exit_logits_list, list):
                            print(f"[WARN] Expected exit_logits_list to be a list, got {type(exit_logits_list)}. Trying to handle.")
                            if isinstance(exit_logits_list, torch.Tensor) and hasattr(model, 'n_exits'):
                                exit_logits_list = [exit_logits_list] * model.n_exits
                            else:
                                print("[ERROR] Cannot interpret exit_logits_list format. Falling back to final exit for safety if force_exit not used.")
                                # exit_logits_list will remain problematic, force_exit may fail if it needs it
                    else:
                        print(f"[WARN] BranchyQuantizedSNN returned tuple of unexpected length {len(model_output)}. Using last element as final_logits.")
                        final_logits = model_output[-1]
                else:
                    final_logits = model_output

                # --- Prepare derived predictions BEFORE decision logic ---
                scaled_exit_logits_for_eval = []
                confs_preds_list = []
                final_pred = None

                if exit_logits_list is not None and isinstance(exit_logits_list, list):
                    if hasattr(model, 'temperature1') and len(exit_logits_list) > 0:
                        scaled_exit_logits_for_eval.append(exit_logits_list[0] / model.temperature1)
                        # print(f"[DEBUG Eval] Using T1: {model.temperature1.item()} for exit 1 softmax")
                    elif len(exit_logits_list) > 0:
                        scaled_exit_logits_for_eval.append(exit_logits_list[0])

                    if hasattr(model, 'temperature2') and len(exit_logits_list) > 1:
                        scaled_exit_logits_for_eval.append(exit_logits_list[1] / model.temperature2)
                        # print(f"[DEBUG Eval] Using T2: {model.temperature2.item()} for exit 2 softmax") 
                    elif len(exit_logits_list) > 1:
                        scaled_exit_logits_for_eval.append(exit_logits_list[1])
                    
                    if len(exit_logits_list) > len(scaled_exit_logits_for_eval):
                        scaled_exit_logits_for_eval.extend(exit_logits_list[len(scaled_exit_logits_for_eval):])
                    
                    if scaled_exit_logits_for_eval: # Ensure there are logits to process
                        probs_list = [torch.softmax(lgts.float(), dim=1) for lgts in scaled_exit_logits_for_eval]
                        confs_preds_list = [p.max(dim=1) for p in probs_list]
                
                if final_logits is not None:
                    final_pred = final_logits.argmax(dim=1)
                
                # --- ADDED: Debug prints for precursor variables (for i==0) ---
                if i == 0:
                    print(f"[DEBUG PRECURSOR] Batch 0: force_exit_path = {force_exit_path}")
                    print(f"  model_output type: {type(model_output)}")
                    if isinstance(model_output, tuple):
                        print(f"  model_output length: {len(model_output)}")
                    print(f"  exit_logits_list is None: {exit_logits_list is None}")
                    if exit_logits_list is not None: print(f"  len(exit_logits_list): {len(exit_logits_list) if isinstance(exit_logits_list, list) else 'Not a list'}")
                    print(f"  final_logits is None: {final_logits is None}")
                    print(f"  scaled_exit_logits_for_eval is None: {scaled_exit_logits_for_eval is None}")
                    if scaled_exit_logits_for_eval is not None: print(f"  len(scaled_exit_logits_for_eval): {len(scaled_exit_logits_for_eval)}")
                    print(f"  confs_preds_list is None: {confs_preds_list is None}")
                    if confs_preds_list is not None: print(f"  len(confs_preds_list): {len(confs_preds_list)}")
                    print(f"  final_pred is None: {final_pred is None}")
                # --- END ADDED DEBUG --- 

                # --- START RESTRUCTURED DECISION LOGIC ---
                if force_exit_path is not None:
                    print(f"[DIAGNOSTIC] Forcing all decisions to Exit Path: {force_exit_path}")
                    
                    # preds is already initialized to None before this block
                    # We will check its value after each potential assignment for i==0

                    if force_exit_path == 0 and confs_preds_list and len(confs_preds_list) > 0:
                        preds = confs_preds_list[0][1] # Predictions from exit 1
                        if i == 0: print(f"[DEBUG FORCE AT PATH 0] Preds[:10]: {preds[:10].tolist() if preds is not None else 'None'}")
                        if exit_counts: exit_counts[0] = exit_counts.get(0,0) + batch_size
                    elif force_exit_path == 1 and confs_preds_list and len(confs_preds_list) > 1:
                        preds = confs_preds_list[1][1] # Predictions from exit 2
                        if i == 0: print(f"[DEBUG FORCE AT PATH 1] Preds[:10]: {preds[:10].tolist() if preds is not None else 'None'}")
                        if exit_counts: exit_counts[1] = exit_counts.get(1,0) + batch_size
                    elif force_exit_path == 2 and final_pred is not None:
                        preds = final_pred # Predictions from final exit
                        if i == 0: print(f"[DEBUG FORCE AT PATH 2] Preds[:10]: {preds[:10].tolist() if preds is not None else 'None'}")
                        if exit_counts and hasattr(model, 'n_exits'): exit_counts[model.n_exits] = exit_counts.get(model.n_exits,0) + batch_size
                    else:
                        print(f"[DIAGNOSTIC WARN] Invalid force_exit_path {force_exit_path}, insufficient logits/preds ({len(confs_preds_list) if confs_preds_list else 'None_CPL'}, {final_pred is not None}). Defaulting to final_pred if available.")
                        if final_pred is not None:
                            preds = final_pred
                            if i == 0: print(f"[DEBUG FORCE FALLBACK TO FINAL] Preds[:10]: {preds[:10].tolist() if preds is not None else 'None'}")
                            if exit_counts and hasattr(model, 'n_exits'): exit_counts[model.n_exits] = exit_counts.get(model.n_exits,0) + batch_size
                        else:
                            print("[ERROR] Cannot determine predictions for batch even with fallback in force_exit_path.")
                            # preds remains None
                            if i == 0: print(f"[DEBUG FORCE ERROR CASE] Preds is None.")
                    
                    # This was the previously missing overall debug print for i == 0.
                    # Let's see if it shows up now that individual ones are added above.
                    if i == 0:
                        print(f"[DEBUG FORCE EXIT OVERALL] Batch 0, Path: {force_exit_path}. Current value of Preds[:10]: {preds[:10].tolist() if preds is not None else 'None'}")
                
                else: # Original RL / Threshold-based decision logic (if not forcing exit)
                    if hasattr(model, 'rl_agent') and model.rl_agent is not None and confs_preds_list and final_pred is not None and len(confs_preds_list) == model.n_exits:
                        try:
                            final_decision_thresholds = None
                            if hasattr(model, 'confidence_thresholds') and model.confidence_thresholds is not None:
                                if isinstance(model.confidence_thresholds, list) and len(model.confidence_thresholds) >= model.n_exits:
                                    final_decision_thresholds = model.confidence_thresholds[:model.n_exits]
                                    print(f"[DEBUG Eval] Using thresholds from model.confidence_thresholds: {final_decision_thresholds}")
                                else:
                                    print(f"[WARN Eval] model.confidence_thresholds is set but invalid. Trying agent thresholds.")
                            
                            if final_decision_thresholds is None and hasattr(model, 'rl_agent') and model.rl_agent is not None:
                                raw_thresholds_from_agent = None
                                if hasattr(model.rl_agent, 'get_confidence_thresholds'):
                                    raw_thresholds_from_agent = model.rl_agent.get_confidence_thresholds()
                                elif hasattr(model.rl_agent, 'confidence_thresholds'):
                                    raw_thresholds_from_agent = model.rl_agent.confidence_thresholds
                                if raw_thresholds_from_agent is not None and isinstance(raw_thresholds_from_agent, list) and len(raw_thresholds_from_agent) >= model.n_exits:
                                    final_decision_thresholds = raw_thresholds_from_agent[:model.n_exits]
                                    print(f"[DEBUG Eval] Using thresholds from RL agent: {final_decision_thresholds}")
                                else:
                                    print(f"[WARN Eval] RL agent provided unsuitable thresholds: {raw_thresholds_from_agent}.")
                            
                            if final_decision_thresholds is None:
                                final_decision_thresholds = [0.9] * model.n_exits # Default
                                print(f"[WARN Eval] No suitable thresholds. Using fallback defaults: {final_decision_thresholds}")
                            
                            thresholds_to_use_for_decision = final_decision_thresholds
                            print(f"[DEBUG Eval] Final thresholds for RL decision: {thresholds_to_use_for_decision}")

                            # Actual decision logic using thresholds_to_use_for_decision
                            current_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                            preds = torch.empty_like(final_pred) # Ensure preds is initialized here for this path
                            exit_points_indices = torch.full_like(final_pred, model.n_exits)

                            for exit_idx in range(model.n_exits):
                                conf = confs_preds_list[exit_idx][0]
                                pred_exit = confs_preds_list[exit_idx][1]
                                exit_decision = (conf >= thresholds_to_use_for_decision[exit_idx]) & current_mask
                                if exit_decision.any():
                                    preds[exit_decision] = pred_exit[exit_decision]
                                    exit_points_indices[exit_decision] = exit_idx
                                    current_mask[exit_decision] = False
                            
                            if current_mask.any():
                                preds[current_mask] = final_pred[current_mask]
                            
                            for exit_idx in range(model.n_exits + 1):
                                count = (exit_points_indices == exit_idx).sum().item()
                                exit_counts[exit_idx] = exit_counts.get(exit_idx, 0) + count

                        except Exception as e_rl_inner:
                            print(f"[ERROR] During RL decision logic: {e_rl_inner}. Falling back to final_pred if available.")
                            if final_pred is not None:
                                preds = final_pred
                            # exit_counts might be partially updated or not, depending on where the error occurred
                    
                    elif final_pred is not None: # Fallback if RL agent or other conditions for RL logic failed
                        print("[DEBUG Eval] Fallback: Using final_pred directly as RL conditions not met or confs_preds_list/final_pred missing for RL.")
                        preds = final_pred
                        if exit_counts and hasattr(model, 'n_exits'): exit_counts[model.n_exits] = exit_counts.get(model.n_exits,0) + batch_size
                    else:
                        print("[ERROR] Could not determine predictions for batch (no force_exit, no RL, no final_pred).")
                        # total_samples will be incremented, but total_correct won't for this batch if preds is None
                # --- END RESTRUCTURED DECISION LOGIC ---
                
                # Logging for first 3 batches (common for both forced and RL paths)
                if i < 3:
                    # Log scaled confidences (based on T1, T2 which are reset to 1.0 for this diagnostic run)
                    if confs_preds_list and len(confs_preds_list) > 0 and confs_preds_list[0][0].numel() > 0:
                        conf1_batch = confs_preds_list[0][0]
                        print(f"Batch {i} Exit 1 SCALED Confidences (T1={model.temperature1.item():.2f}): mean={conf1_batch.mean().item():.4f}, min={conf1_batch.min().item():.4f}, max={conf1_batch.max().item():.4f}, std={conf1_batch.std().item():.4f}")
                    if confs_preds_list and len(confs_preds_list) > 1 and confs_preds_list[1][0].numel() > 0:
                        conf2_batch = confs_preds_list[1][0]
                        print(f"Batch {i} Exit 2 SCALED Confidences (T2={model.temperature2.item():.2f}): mean={conf2_batch.mean().item():.4f}, min={conf2_batch.min().item():.4f}, max={conf2_batch.max().item():.4f}, std={conf2_batch.std().item():.4f}")
                    # Log raw logits
                    if exit_logits_list and isinstance(exit_logits_list, list):
                        if len(exit_logits_list) > 0:
                            logits1_batch = exit_logits_list[0].float()
                            print(f"Batch {i} Exit 1 RAW LOGITS: mean={logits1_batch.mean().item():.4f}, min={logits1_batch.min().item():.4f}, max={logits1_batch.max().item():.4f}, std={logits1_batch.std().item():.4f}")
                        if len(exit_logits_list) > 1:
                            logits2_batch = exit_logits_list[1].float()
                            print(f"Batch {i} Exit 2 RAW LOGITS: mean={logits2_batch.mean().item():.4f}, min={logits2_batch.min().item():.4f}, max={logits2_batch.max().item():.4f}, std={logits2_batch.std().item():.4f}")
                
                # Accuracy calculation using `preds` determined by either force_exit_path or RL/fallback logic
                if preds is not None:
                    total_correct += (preds == labels).sum().item()
                total_samples += batch_size
                continue # Crucial: Go to next batch for BQSNN, skip general acc calc below
            # --- END MODIFIED BLOCK for BranchyQuantizedSNN ---
            
            # Ensure GPU/MPS operations are complete before stopping timer
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.synchronize()
                
            end_time = time.time()
            
            # BranchyQuantizedSNN RL inference: use exit logits with thresholds
            if "branchy_quantized_snn" in model_name and hasattr(model, 'rl_agent') and model.rl_agent is not None: # check model.rl_agent
                exit_logits_list, final_logits, _ = model_output # Expecting 3 items now from BQSNN forward
                # Exit1
                probs1 = torch.softmax(exit_logits_list[0].float(), dim=1) # Ensure float for softmax
                conf1, pred1 = probs1.max(dim=1)
                # Exit2
                probs2 = torch.softmax(exit_logits_list[1].float(), dim=1) # Ensure float for softmax
                conf2, pred2 = probs2.max(dim=1)
                # Final exit
                final_pred = final_logits.argmax(dim=1)
                
                # Decision masks
                current_agent_thresholds = model.rl_agent.get_confidence_thresholds()
                if current_agent_thresholds is None or len(current_agent_thresholds) != model.n_exits:
                    print(f"[WARN] Agent thresholds are {current_agent_thresholds}. Using defaults [0.9, 0.9] for decisions.")
                    current_agent_thresholds = [0.9] * model.n_exits # Default conservative

                print(f"[DEBUG] RL Decision using thresholds: {current_agent_thresholds}")
                
                # --- Add Print for Confidence Scores --- 
                if i < 2: # Print for first 2 batches only to avoid excessive logging
                    print(f"  Batch {i+1} - conf1[:5]: {conf1[:5].cpu().numpy()}")
                    print(f"  Batch {i+1} - conf2[:5]: {conf2[:5].cpu().numpy()}")
                # --- End Add Print --- 

                mask1 = conf1 >= current_agent_thresholds[0]
                mask2 = ~mask1 & (conf2 >= current_agent_thresholds[1])
                mask3 = ~(mask1 | mask2)
                # Debug: print confidences and decisions for first batch
                if total_samples == 0:
                    print("[DEBUG] First batch confidences:")
                    print("  conf1:", conf1[:10].cpu().numpy())
                    print("  conf2:", conf2[:10].cpu().numpy())
                    print("  thresholds:", current_agent_thresholds)
                    print("  mask1 (early exit 1):", mask1[:10].cpu().numpy())
                    print("  mask2 (early exit 2):", mask2[:10].cpu().numpy())
                    print("  mask3 (final exit):", mask3[:10].cpu().numpy())
                # Combine predictions
                preds = torch.empty_like(final_pred)
                preds[mask1] = pred1[mask1]
                preds[mask2] = pred2[mask2]
                preds[mask3] = final_pred[mask3]
                total_correct += (preds == labels).sum().item()
                total_samples += batch_size
                # Record exits
                exit_counts[0] += mask1.sum().item()
                exit_counts[1] += mask2.sum().item()
                exit_counts[2] += mask3.sum().item()
                # --- TRUE PER-EXIT ACCURACY & ENTROPY TRACKING ---
                if 'per_exit_preds' not in locals():
                    per_exit_preds = {0: [], 1: [], 2: []}
                    per_exit_labels = {0: [], 1: [], 2: []}
                    per_exit_entropies = {0: [], 1: [], 2: []}
                # Exit 1
                idxs1 = mask1.nonzero(as_tuple=True)[0]
                per_exit_preds[0].extend(pred1[idxs1].cpu().tolist())
                per_exit_labels[0].extend(labels[idxs1].cpu().tolist())
                per_exit_entropies[0].extend((-probs1[idxs1] * torch.log(probs1[idxs1]+1e-8)).sum(dim=1).cpu().tolist())
                # Exit 2
                idxs2 = mask2.nonzero(as_tuple=True)[0]
                per_exit_preds[1].extend(pred2[idxs2].cpu().tolist())
                per_exit_labels[1].extend(labels[idxs2].cpu().tolist())
                per_exit_entropies[1].extend((-probs2[idxs2] * torch.log(probs2[idxs2]+1e-8)).sum(dim=1).cpu().tolist())
                # Final Exit
                idxs3 = mask3.nonzero(as_tuple=True)[0]
                final_probs = torch.softmax(final_logits, dim=1)
                per_exit_preds[2].extend(final_pred[idxs3].cpu().tolist())
                per_exit_labels[2].extend(labels[idxs3].cpu().tolist())
                per_exit_entropies[2].extend((-final_probs[idxs3] * torch.log(final_probs[idxs3]+1e-8)).sum(dim=1).cpu().tolist())

                continue
            # Process other models
            # Process different output formats
            if isinstance(model_output, tuple) and len(model_output) == 2:
                first, second = model_output
                if isinstance(first, (list, tuple)):
                    outputs = second
                    exit_points = None
                else:
                    outputs, exit_points, *rest = model_output if len(model_output) >= 2 else (model_output[0], None)
                # If model returns three outputs (BranchyQuantizedSNN), handle accordingly
                if len(model_output) == 3:
                    outputs, dequant_final, exit_points = model_output
                    # For evaluation, use dequant_final as outputs
                    outputs = dequant_final
            elif isinstance(model_output, list):
                outputs = model_output[-1]
                exit_points = None
            else:
                outputs = model_output
                exit_points = None
            
            # Calculate accuracy
            # For quantized BranchyQuantizedSNN, outputs may be a tuple (exit_outputs, dequant_final, exit_points)
            if isinstance(outputs, tuple):
                # Use the final dequantized output for accuracy
                if len(outputs) == 3:
                    _, predicted = torch.max(outputs[1], 1)
                else:
                    _, predicted = torch.max(outputs[0], 1)
            else:
                _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size
            
            # Record exit points for dynamic models
            if exit_points is not None:
                for i in range(batch_size):
                    # exit_points are 1-based (1, 2, 3 for final)
                    exit_idx = exit_points[i].item() - 1  # Convert to 0-based
                    if exit_idx in exit_counts:
                        exit_counts[exit_idx] += 1
                total_exit_samples += batch_size
            
            # Accumulate metrics
            hw_time = end_time - start_time
            total_time += hw_time
    
    # Record end energy/time if using NVML events (but only for timing, not energy)
    if USE_NVML:
        end_event.record()
        torch.cuda.synchronize()
        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
        # Elapsed GPU time in milliseconds
        total_time_ms = start_event.elapsed_time(end_event)
        # Use NVML timing but not energy - will calculate theoretical energy later
        avg_time = (total_time_ms / 1000.0) / total_samples if total_samples > 0 else 0.0
        # Calculate average accuracy
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        # Initialize energy and power to 0 - will be replaced with theoretical values
        avg_energy = 0.0
        avg_power = 0.0
    else:
        # Calculate average metrics when NVML not available
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        avg_energy = 0.0
        avg_time = total_time / total_samples if total_samples > 0 else 0.0
        avg_power = 0.0
    
    print(f"Model: {model_name}")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  Energy: {avg_energy:.6e} J")
    # Raw inference time before dynamic adjustment
    print(f"  Raw inference time: {avg_time:.6f} s")
    print(f"  Power: {avg_power:.4f} W")
    
    # Theoretical energy and operation counting for ANN
    # mac_count = count_mac_operations(model, config['input_size']) # This is for full model, might be overridden for BQSNN
    # Calculate initial mac_count based on a dummy input, can be refined per model type.
    # Ensure dummy_input is on the same device as the model for op counting.
    op_count_device = next(model.parameters()).device
    general_dummy_input = torch.randn(*config['input_size'], device=op_count_device)
    mac_count = count_mac_operations(model, general_dummy_input)


    # --- MODIFICATION FOR BRANCHY_QUANTIZED_SNN AVG MACS ---
    avg_macs_to_log = mac_count # Default to full mac_count
    avg_ac_ops_to_log = np.nan # Initialize for BQSNN AC ops
    avg_inference_time_per_step_ms = np.nan # Initialize per-step time

    if "branchy_quantized_snn" in model_name and exit_counts:
        print("[INFO] Calculating per-exit MACs for BranchyQuantizedSNN...")
        # op_count_device = next(model.parameters()).device # Already defined
        try:
            dummy_input_3_32_32 = torch.randn(1, 3, 32, 32, device=op_count_device)
            dummy_input_64_32_32 = torch.randn(1, 64, 32, 32, device=op_count_device)
            dummy_input_64_16_16 = torch.randn(1, 64, 16, 16, device=op_count_device)
            dummy_input_192_16_16 = torch.randn(1, 192, 16, 16, device=op_count_device)
            dummy_input_192_8_8_flat = torch.randn(1, 192 * 8 * 8, device=op_count_device)
            dummy_input_512 = torch.randn(1, 512, device=op_count_device)

            macs_conv1 = count_mac_operations(model.conv1, dummy_input_3_32_32)
            macs_exit1_fc = count_mac_operations(model.exit1.fc, model.exit1.pool(dummy_input_64_32_32).flatten(1))
            macs_conv2 = count_mac_operations(model.conv2, dummy_input_64_16_16)
            macs_exit2_fc = count_mac_operations(model.exit2.fc, model.exit2.pool(dummy_input_192_16_16).flatten(1))
            macs_fc1 = count_mac_operations(model.fc1, dummy_input_192_8_8_flat)
            macs_fc2 = count_mac_operations(model.fc2, dummy_input_512)

            print(f"  MACs conv1: {macs_conv1}")
            print(f"  MACs exit1_fc: {macs_exit1_fc}")
            print(f"  MACs conv2: {macs_conv2}")
            print(f"  MACs exit2_fc: {macs_exit2_fc}")
            print(f"  MACs fc1: {macs_fc1}")
            print(f"  MACs fc2: {macs_fc2}")

            macs_path0 = macs_conv1 + macs_exit1_fc
            macs_path1 = macs_conv1 + macs_conv2 + macs_exit2_fc
            macs_path2 = macs_conv1 + macs_conv2 + macs_fc1 + macs_fc2
            
            print(f"  MACs Path 0 (Exit 1): {macs_path0}")
            print(f"  MACs Path 1 (Exit 2): {macs_path1}")
            print(f"  MACs Path 2 (Final): {macs_path2}")

            total_exit_samples = sum(exit_counts.values())
            if total_exit_samples > 0:
                p_exit0 = exit_counts.get(0, 0) / total_exit_samples
                p_exit1 = exit_counts.get(1, 0) / total_exit_samples
                # model.n_exits is 2, so final exit key in exit_counts is 2 (0, 1, 2)
                p_final = exit_counts.get(model.n_exits, 0) / total_exit_samples 

                avg_macs_calc = (p_exit0 * macs_path0) + \
                                (p_exit1 * macs_path1) + \
                                (p_final * macs_path2)
                avg_macs_to_log = int(avg_macs_calc)
                print(f"  Exit distribution: P0={p_exit0:.2f}, P1={p_exit1:.2f}, P_Final={p_final:.2f}")
                print(f"  Calculated average MACs for BQSNN: {avg_macs_to_log}")
            else:
                print("  No samples recorded in exit_counts, cannot calculate weighted average MACs. Using full pass MACs for final path.")
                avg_macs_to_log = macs_path2
        except Exception as e_mac:
            print(f"[ERROR] Failed to calculate per-exit MACs for BranchyQuantizedSNN: {e_mac}")
            avg_macs_to_log = mac_count # Fallback to full initial mac_count 
    
        # --- START: Calculate Average AC Ops for BranchyQuantizedSNN ---
        time_steps_for_ac = getattr(model, 'time_window', time_steps_eval if pd.notna(time_steps_eval) else config.get('num_steps', 25))
        print(f"[Eval BQSNN] Using time_steps_for_ac: {time_steps_for_ac} (derived from args or model/config)")


        # Check if model has the expected snn.Leaky layers
        has_lif_layers_for_ac_calc = (
            hasattr(model, 'lif1') and hasattr(model, 'lif2') and 
            hasattr(model, 'lif_fc1') and hasattr(model, 'lif_fc2')
        )

        if exit_counts and has_lif_layers_for_ac_calc:
            print("[INFO] Calculating per-exit AC ops for BranchyQuantizedSNN using snn.Leaky layers...")
            ac_op_device = op_count_device # op_count_device is already model's device

            try:
                # REMOVED: from utils import count_ac_operations

                # Dummy inputs based on the expected input shapes to the LIF layers in BranchyQuantizedSNN
                # Input to lif1: Output of conv1 (e.g., [1, 64, 32, 32])
                dummy_input_lif1 = torch.randn(1, 64, 32, 32, device=ac_op_device)
                # Input to lif2: Output of conv2 (e.g., [1, 192, 16, 16])
                dummy_input_lif2 = torch.randn(1, 192, 16, 16, device=ac_op_device)
                # Input to lif_fc1: Output of flatten after pool2 (e.g., [1, 192*8*8])
                # Use the shape determined during MAC calculation if possible, else estimate
                # final_flat_features = 192 * 8 * 8 (from model definition)
                dummy_input_lif_fc1 = torch.randn(1, 192 * 8 * 8, device=ac_op_device)
                # Input to lif_fc2: Output of fc1 (e.g., [1, 512])
                dummy_input_lif_fc2 = torch.randn(1, 512, device=ac_op_device)

                # Calculate AC ops for each individual LIF layer
                ac_ops_lif1, _ = count_ac_operations(model.lif1, dummy_input_lif1, time_steps_for_ac)
                ac_ops_lif2, _ = count_ac_operations(model.lif2, dummy_input_lif2, time_steps_for_ac)
                
                # Manually calculate AC ops for Linear LIF layers
                if hasattr(model, 'fc1') and hasattr(model.fc1, 'out_features'):
                    ac_ops_lif_fc1 = time_steps_for_ac * model.fc1.out_features
                else:
                    print("[WARN] Cannot find model.fc1.out_features to calculate ac_ops_lif_fc1. Setting to 0.")
                    ac_ops_lif_fc1 = 0
                    
                if hasattr(model, 'fc2') and hasattr(model.fc2, 'out_features'):
                    ac_ops_lif_fc2 = time_steps_for_ac * model.fc2.out_features
                else:
                    print("[WARN] Cannot find model.fc2.out_features to calculate ac_ops_lif_fc2. Setting to 0.")
                    ac_ops_lif_fc2 = 0

                # ac_ops_lif_fc1, _ = count_ac_operations(model.lif_fc1, dummy_input_lif_fc1, time_steps_for_ac)
                # ac_ops_lif_fc2, _ = count_ac_operations(model.lif_fc2, dummy_input_lif_fc2, time_steps_for_ac)

                print(f"  ACs lif1: {ac_ops_lif1}")
                print(f"  ACs lif2: {ac_ops_lif2}")
                print(f"  ACs lif_fc1: {ac_ops_lif_fc1}")
                print(f"  ACs lif_fc2: {ac_ops_lif_fc2}")
                
                # ACs for each path (sum of ACs of LIF layers in that path)
                # Path 0 (Exit 1) only involves computation up to lif1
                ac_path0 = ac_ops_lif1 
                # Path 1 (Exit 2) involves computation up to lif2
                ac_path1 = ac_ops_lif1 + ac_ops_lif2
                # Path 2 (Final) involves all LIF layers
                ac_path2 = ac_ops_lif1 + ac_ops_lif2 + ac_ops_lif_fc1 + ac_ops_lif_fc2
                
                print(f"  ACs Path 0 (Exit 1): {ac_path0}")
                print(f"  ACs Path 1 (Exit 2): {ac_path1}")
                print(f"  ACs Path 2 (Final): {ac_path2}")

                total_exit_samples = sum(exit_counts.values())
                if total_exit_samples > 0:
                    p_exit0 = exit_counts.get(0, 0) / total_exit_samples
                    p_exit1 = exit_counts.get(1, 0) / total_exit_samples
                    p_final = exit_counts.get(model.n_exits, 0) / total_exit_samples 

                    avg_ac_ops_calc = (p_exit0 * ac_path0) + \
                                      (p_exit1 * ac_path1) + \
                                      (p_final * ac_path2)
                    avg_ac_ops_to_log = int(avg_ac_ops_calc)
                    print(f"  Exit distribution for ACs: P0={p_exit0:.2f}, P1={p_exit1:.2f}, P_Final={p_final:.2f}")
                    print(f"  Calculated average AC ops for BQSNN: {avg_ac_ops_to_log}")
                else:
                    print("  No samples recorded in exit_counts for BQSNN AC calculation, using full pass ACs for final path.")
                    avg_ac_ops_to_log = ac_path2
            
            except AttributeError as e_attr:
                print(f"[ERROR] Failed to calculate per-exit AC ops for BranchyQuantizedSNN due to missing LIF attribute: {e_attr}. Check model definition. AC ops will be NaN.")
                avg_ac_ops_to_log = np.nan
            except Exception as e_ac:
                print(f"[ERROR] Failed during per-exit AC ops calculation for BranchyQuantizedSNN: {e_ac}. AC ops will be NaN.")
                avg_ac_ops_to_log = np.nan
        elif not has_lif_layers_for_ac_calc:
             print("[WARN] BranchyQuantizedSNN model does not seem to have the expected snn.Leaky layers (lif1, lif2, etc.). Skipping AC ops calculation.")
             avg_ac_ops_to_log = np.nan
        # --- END: Calculate Average AC Ops for BranchyQuantizedSNN ---
    # --- END MODIFICATION ---
    
    # --- ADDITION FOR DYNAMIC_ALEXNET (BranchyAlexNet) AVG MACS ---
    if ("dynamic_alexnet" in model_name or "branchy_alexnet" in model_name) and exit_counts and total_samples > 0:
        print(f"[INFO] Calculating per-exit MACs for {model_name}...")
        op_count_device = next(model.parameters()).device
        try:
            # Define dummy inputs based on BranchyAlexNet structure
            # Input to features1: (1, 3, 32, 32) or (1, C, H, W) from config
            dummy_input_features1 = torch.randn(1, *input_shape_for_counts, device=op_count_device) # Use resolved input_shape_for_counts

            # Get intermediate shapes by a forward pass (or define them if static)
            # For simplicity, we'll count module by module.
            # Ensure model is in eval mode for stable intermediate shapes if hooks/dynamic behavior exists
            original_training_mode = model.training_mode
            model.training_mode = False # Ensure inference path for shape consistency if model uses this flag
            
            # MACs for each block
            macs_block1 = count_mac_operations(model.features1, dummy_input_features1)
            
            # Output of features1 goes into exit1 and features2
            with torch.no_grad(): # Avoid gradient tracking during shape inference
                out_features1 = model.features1(dummy_input_features1)
            
            macs_exit1_head = count_mac_operations(model.exit1, out_features1)
            macs_block2 = count_mac_operations(model.features2, out_features1) # features2 takes output of features1

            # Output of features2 goes into exit2 and final_features
            with torch.no_grad():
                out_features2 = model.features2(out_features1)

            macs_exit2_head = count_mac_operations(model.exit2, out_features2)
            
            # MACs for the final path components
            macs_final_features = count_mac_operations(model.final_features, out_features2)
            with torch.no_grad():
                out_final_features = model.final_features(out_features2)
                out_pooled = model.adaptive_pool(out_final_features)
                out_flattened = out_pooled.view(out_pooled.size(0), -1)
            
            macs_output_layer = count_mac_operations(model.output_layer, out_flattened)
            macs_block3 = macs_final_features + macs_output_layer # adaptive_pool MACs are negligible

            model.training_mode = original_training_mode # Restore original mode

            print(f"  MACs features1 (block1): {macs_block1}")
            print(f"  MACs exit1_head: {macs_exit1_head}")
            print(f"  MACs features2 (block2): {macs_block2}")
            print(f"  MACs exit2_head: {macs_exit2_head}")
            print(f"  MACs final_features part of block3: {macs_final_features}")
            print(f"  MACs output_layer part of block3: {macs_output_layer}")
            print(f"  MACs block3 (final path body): {macs_block3}")

            # MACs for each full path
            macs_path_exit1 = macs_block1 + macs_exit1_head
            macs_path_exit2 = macs_block1 + macs_block2 + macs_exit2_head
            macs_path_final = macs_block1 + macs_block2 + macs_block3
            
            print(f"  MACs Path for Exit 1: {macs_path_exit1}")
            print(f"  MACs Path for Exit 2: {macs_path_exit2}")
            print(f"  MACs Path for Final Exit (3): {macs_path_final}")

            total_recorded_exit_samples = sum(exit_counts.values())
            if total_recorded_exit_samples > 0:
                # exit_counts keys are 0-indexed for exits, but in BranchyAlexNet forward_inference,
                # exit_points are 1, 2, 3. We need to map:
                # exit_counts[0] -> samples taking exit 1
                # exit_counts[1] -> samples taking exit 2
                # exit_counts[2] -> samples taking final exit (exit 3)
                # model.n_exits for BranchyAlexNet is 2 (meaning 2 early exits)
                
                # Probabilities of taking each path
                p_exit1 = exit_counts.get(0, 0) / total_recorded_exit_samples # Samples that took path ending at exit 1
                p_exit2 = exit_counts.get(1, 0) / total_recorded_exit_samples # Samples that took path ending at exit 2
                p_final = exit_counts.get(model.n_exits, 0) / total_recorded_exit_samples # Samples that took final path (key = n_exits)

                avg_macs_calc = (p_exit1 * macs_path_exit1) + \
                                (p_exit2 * macs_path_exit2) + \
                                (p_final * macs_path_final)
                avg_macs_to_log = int(avg_macs_calc)
                print(f"  Exit distribution: P_Exit1={p_exit1:.3f} (Count: {exit_counts.get(0,0)}), P_Exit2={p_exit2:.3f} (Count: {exit_counts.get(1,0)}), P_Final={p_final:.3f} (Count: {exit_counts.get(model.n_exits,0)})")
                print(f"  Calculated average MACs for {model_name}: {avg_macs_to_log}")
            else:
                print(f"  No samples recorded in exit_counts for {model_name}, cannot calculate weighted average MACs. Using full pass MACs ({macs_path_final}).")
                avg_macs_to_log = macs_path_final # Fallback to full final path MACs
        except AttributeError as e_attr:
            print(f"[ERROR] Failed to calculate per-exit MACs for {model_name} due to missing attribute: {e_attr}. Check model definition. MACs will be full count.")
            avg_macs_to_log = mac_count # Fallback
        except Exception as e_mac_dyn:
            print(f"[ERROR] Failed to calculate per-exit MACs for {model_name}: {e_mac_dyn}. MACs will be full count.")
            avg_macs_to_log = mac_count # Fallback
    # --- END ADDITION ---

    # Save results to CSV (using pandas for consistency)
    import pandas as pd
    
    current_eval_device = next(model.parameters()).device
    # dummy_input was created at the start of evaluate_model for warmup.
    # Ensure it's on the correct device for any subsequent memory counting.
    if dummy_input.device != current_eval_device:
        dummy_sample_for_mem = torch.randn(1, 3, 32, 32, device=current_eval_device)
    else:
        dummy_sample_for_mem = dummy_input # Reuse original dummy_input if device matches

    # Initialize theoretical energy and other specific metrics
    ann_theoretical_energy_val = np.nan
    projected_neuromorphic_energy_val = np.nan 
    # SNN specific fields, to be filled if it's an SNN model not covered by BQSNN custom logic
    ac_ops_val = np.nan
    spike_count_val = np.nan
    snn_mem_accesses_val = np.nan # bytes
    fair_theoretical_energy_j_val = np.nan # Joules

    mac_to_log_final = mac_count # Default for non-BQSNN models
    ac_ops_to_log_final = ac_ops_val # Default for non-BQSNN SNNs

    if "branchy_quantized_snn" in model_name:
        mac_to_log_final = avg_macs_to_log
        ac_ops_to_log_final = avg_ac_ops_to_log # Use calculated average AC ops

        ann_mem_access_final = 15 * 1024 * 1024  # Mock 15MB for ANN-equivalent memory access
        snn_mem_access_final = 16 * 1024 * 1024  # Mock 16MB for SNN memory access
        
        # Energy constants from utils.py / config
        E_MAC_8BIT_J = config.get("hardware_constants", {}).get("cpu", {}).get("energy_per_mac_8bit", 0.2e-12)
        E_AC_8BIT_J = config.get("hardware_constants", {}).get("cpu", {}).get("energy_per_ac_8bit", 0.03e-12)
        E_MEM_J_PER_BYTE = config.get("hardware_constants", {}).get("memory", {}).get("energy_per_byte_access", (640e-12)/4.0)
        E_SPIKE_NEUROMORPHIC_J = config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_spike", 0.1e-12) # This is per spike, maybe AC is better
        E_AC_NEUROMORPHIC_J = config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_ac", 0.01e-12) # Hypothetical energy per AC on neuromorphic
        E_MEM_NEUROMORPHIC_J_PER_BYTE = config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_mem_byte", 0.02e-12)


        # Calculate fair theoretical energy for BQSNN based on AC_OPS if available
        if pd.notna(ac_ops_to_log_final):
            ac_energy_j = ac_ops_to_log_final * E_AC_8BIT_J
            mem_energy_j = snn_mem_access_final * E_MEM_J_PER_BYTE
            fair_theoretical_energy_j_val = ac_energy_j + mem_energy_j
            print(f"  BQSNN Fair Theoretical Energy (AC ops based): {fair_theoretical_energy_j_val:.4e} J (ACs: {ac_ops_to_log_final}, SNN Mem (mock): {snn_mem_access_final})")
            
            # Projected neuromorphic energy based on AC_OPS and SNN mem
            neuromorphic_ac_energy = ac_ops_to_log_final * E_AC_NEUROMORPHIC_J # Using energy_per_ac
            neuromorphic_mem_energy = snn_mem_access_final * E_MEM_NEUROMORPHIC_J_PER_BYTE
            projected_neuromorphic_energy_val = neuromorphic_ac_energy + neuromorphic_mem_energy
            print(f"  BQSNN Projected Neuromorphic Energy (AC ops based): {projected_neuromorphic_energy_val:.4e} J")
        else: # Fallback to MAC-based if AC ops calculation failed
            print("[WARN] BQSNN AC ops not available or NaN, falling back to MAC-based fair theoretical energy for BQSNN.")
            mac_energy_j = mac_to_log_final * E_MAC_8BIT_J
            mem_energy_j = snn_mem_access_final * E_MEM_J_PER_BYTE # Still use SNN mock memory
            fair_theoretical_energy_j_val = mac_energy_j + mem_energy_j
            print(f"  BQSNN Fair Theoretical Energy (MAC ops fallback): {fair_theoretical_energy_j_val:.4e} J (Avg MACs: {mac_to_log_final}, SNN Mem (mock): {snn_mem_access_final})")
            # Projected neuromorphic energy based on MACs (as it was, but with new constants if defined)
            # Heuristic: 0.1 spikes per MAC, or just use MACs * neuromorphic E_AC?
            # Using MACs * E_AC_NEUROMORPHIC_J as a rougher proxy if ACs are missing.
            neuromorphic_mac_proxy_energy = mac_to_log_final * E_AC_NEUROMORPHIC_J
            neuromorphic_mem_energy = snn_mem_access_final * E_MEM_NEUROMORPHIC_J_PER_BYTE
            projected_neuromorphic_energy_val = neuromorphic_mac_proxy_energy + neuromorphic_mem_energy
            print(f"  BQSNN Projected Neuromorphic Energy (MAC fallback): {projected_neuromorphic_energy_val:.4e} J")
        
        snn_mem_accesses_val = snn_mem_access_final
        # ANN theoretical energy for BQSNN (for comparing its MAC cost against pure ANNs)
        ann_mac_energy_j = mac_to_log_final * E_MAC_8BIT_J # Use BQSNN's avg MACs
        ann_mem_energy_j = ann_mem_access_final * E_MEM_J_PER_BYTE # Use ANN mock memory
        ann_theoretical_energy_val = ann_mac_energy_j + ann_mem_energy_j

    else: # For other models (ANNs or SNNs not BQSNN)
        mac_to_log_final = mac_count # Already set, but for clarity
        # For generic SNNs (not BQSNN), try to get AC ops if not already calculated
        if model_name.startswith("snn_") and pd.isna(ac_ops_to_log_final):
            try:
                # Assuming a simple SNN where full model AC ops are relevant
                snn_dummy_input = torch.randn(*config['input_size'], device=op_count_device)
                # Use time_steps_eval which comes from args or model/config
                calculated_ac_ops, _ = count_ac_operations(model, snn_dummy_input, time_steps=time_steps_eval)
                if pd.notna(calculated_ac_ops):
                    ac_ops_to_log_final = calculated_ac_ops
                    print(f"  Calculated AC ops for generic SNN {model_name} (T={time_steps_eval}): {ac_ops_to_log_final}")
            except Exception as e_gen_snn_ac:
                print(f"  Could not calculate AC ops for generic SNN {model_name}: {e_gen_snn_ac}")
                ac_ops_to_log_final = np.nan


        ann_mem_bytes = count_memory_access(model, dummy_sample_for_mem, is_snn=False)
        # Determine precision for MAC energy calculation for ANNs
        is_quantized_ann = 'quantized' in model_name or 'int8' in model_name # Basic check
        mac_precision_cost = E_MAC_8BIT_J if is_quantized_ann else config.get("hardware_constants", {}).get("cpu", {}).get("energy_per_mac_32bit", 4.6e-12)
        
        mac_energy_j = mac_to_log_final * mac_precision_cost
        mem_energy_j_ann = ann_mem_bytes * E_MEM_J_PER_BYTE
        ann_theoretical_energy_val = mac_energy_j + mem_energy_j_ann
        
        projected_neuromorphic_energy_val = 0.0 # Default for non-SNNs or non-BQSNN SNNs in this context

        # For AlexNet and Dynamic AlexNet (non-SNN models), use the early calculated theoretical energy
        if not model_name.startswith("snn_") and pd.notna(fair_theoretical_energy_j):
            fair_theoretical_energy_j_val = fair_theoretical_energy_j
            print(f"  Using early calculated Fair Theoretical Energy for {model_name}: {fair_theoretical_energy_j_val:.4e} J")

        # Fair theoretical energy for generic SNNs (using AC ops if available)
        if model_name.startswith("snn_") and pd.notna(ac_ops_to_log_final):
            snn_mem_bytes_val = count_memory_access(model, dummy_sample_for_mem, is_snn=True)
            snn_mem_accesses_val = snn_mem_bytes_val # Store it
            ac_energy_j = ac_ops_to_log_final * E_AC_8BIT_J # Assume 8-bit ACs for generic SNNs for now
            mem_energy_j_snn = snn_mem_bytes_val * E_MEM_J_PER_BYTE
            fair_theoretical_energy_j_val = ac_energy_j + mem_energy_j_snn
            print(f"  Generic SNN {model_name} Fair Theoretical Energy (AC ops based): {fair_theoretical_energy_j_val:.4e} J (ACs: {ac_ops_to_log_final}, SNN Mem: {snn_mem_bytes_val})")
            
            # Projected neuromorphic energy for generic SNNs
            neuromorphic_ac_energy = ac_ops_to_log_final * E_AC_NEUROMORPHIC_J
            neuromorphic_mem_energy = snn_mem_bytes_val * E_MEM_NEUROMORPHIC_J_PER_BYTE
            projected_neuromorphic_energy_val = neuromorphic_ac_energy + neuromorphic_mem_energy
            print(f"  Generic SNN {model_name} Projected Neuromorphic Energy (AC ops based): {projected_neuromorphic_energy_val:.4e} J")

        elif model_name.startswith("snn_"): # SNN but AC ops were not available
             snn_mem_bytes_val = count_memory_access(model, dummy_sample_for_mem, is_snn=True)
             snn_mem_accesses_val = snn_mem_bytes_val
             # Fallback for fair energy of SNN: use MAC count as a proxy (less ideal)
             # This case should be rare if AC counting is robust
             mac_proxy_energy_snn = mac_to_log_final * mac_precision_cost # Using its MACs
             mem_energy_j_snn = snn_mem_bytes_val * E_MEM_J_PER_BYTE
             fair_theoretical_energy_j_val = mac_proxy_energy_snn + mem_energy_j_snn
             print(f"[WARN] Generic SNN {model_name}: AC ops unavailable. Fair Theoretical Energy using MACs proxy: {fair_theoretical_energy_j_val:.4e} J")
             projected_neuromorphic_energy_val = 0.0 # Cannot reliably project without ACs

    # Use theoretical energy for ALL models now (AlexNet, Dynamic AlexNet, SNNs)
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

    results_data = OrderedDict([
        ("model_name", model_name),
        ("device", str(device)),
        ("accuracy", avg_acc),
        ("energy", avg_energy), 
        ("inference_time", avg_time),
        ("power", avg_power),
        ("macs", mac_to_log_final), # For ANNs, or Avg MACs for BQSNN
        ("ann_mem_accesses", ann_mem_access_final if "branchy_quantized_snn" in model_name else ann_mem_bytes if not model_name.startswith("snn_") else np.nan), # Bytes
        ("ann_theoretical_energy", ann_theoretical_energy_val), # Joules, MAC-based
        # SNN specific
        ("time_steps_configured", time_steps_eval if "snn" in model_name.lower() else np.nan), # Log the T used
        ("avg_inference_time_per_step_ms", avg_inference_time_per_step_ms if "snn" in model_name.lower() else np.nan), # Log per-step time for SNNs
        ("ac_ops", ac_ops_to_log_final), # Avg ACs for BQSNN, or total for other SNNs
        ("spike_count", spike_count_val), # Still NaN, count_ac_operations returns it but we are not storing it per exit path yet
        ("snn_mem_accesses", snn_mem_accesses_val), # Bytes, mock for BQSNN, calculated for other SNNs
        ("fair_theoretical_energy_j", fair_theoretical_energy_j_val), # Joules, AC-based for SNNs (incl BQSNN if ACs available), MAC-based for ANNs
        ("projected_neuromorphic_energy", projected_neuromorphic_energy_val) # Joules
    ])

    if exit_counts:
        for k_exit in sorted(exit_counts.keys()): # Ensure consistent order
            results_data[f"exit_{k_exit}_count"] = exit_counts[k_exit]
    
    # Convert OrderedDict to DataFrame for saving
    # df_list = [{k: [v] for k,v in results_data.items()}] # Old way, ensure all values are list-like for single row
    df_list = {k: [v] for k,v in results_data.items()}
    df = pd.DataFrame(df_list)

    log_path = f"results/logs/{model_name}_eval.csv"
    df.to_csv(log_path, index=False)
    print(f"Results saved to {log_path}")
    
    # Save exit distribution if branchy model
    if ("branchy" in model_name or "QDSNN" in model_name) and len(exit_counts) > 0:
        exit_data_to_save = dict(exit_counts)
        exit_data_to_save['total_samples'] = total_exit_samples
        log_dir = "results/logs"
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{model_name}_exit_dist.json")
        with open(file_path, 'w') as f:
            json.dump(exit_data_to_save, f, indent=4)
        print(f"Saved exit distribution data to {file_path}")
    
    # Return metrics dictionary (original structure)
    return {
        "accuracy": avg_acc,
        "energy": avg_energy,
        "inference_time": avg_time,
        "power": avg_power,
        "exit_distribution": exit_counts if exit_counts else None
    }

def calibrate_exit_times(model, device, loader, n_batches=5):
    """Measure cumulative per-exit times (s) via CUDA events."""
    import itertools, torch
    model.training_mode = False
    model.eval()
    t1_tot = t2_tot = t3_tot = 0.0
    count = 0
    for images, _ in itertools.islice(loader, n_batches):
        images = images.to(device)
        # Setup CUDA timing events
        start = torch.cuda.Event(enable_timing=True)
        exit1_evt = torch.cuda.Event(enable_timing=True)
        exit2_evt = torch.cuda.Event(enable_timing=True)
        exit3_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        # Exit 1
        x1 = model.features1(images)
        _ = model.exit1(x1)
        exit1_evt.record()
        # Exit 2
        x2 = model.features2(x1)
        _ = model.exit2(x2)
        exit2_evt.record()
        # Final exit
        x3 = model.final_features(x2)
        x3 = model.adaptive_pool(x3).view(x3.size(0), -1)
        _ = model.output_layer(x3)
        exit3_evt.record()
        torch.cuda.synchronize()
        bsz = images.size(0)
        t1_tot += (start.elapsed_time(exit1_evt) / 1000.0) / bsz
        t2_tot += (start.elapsed_time(exit2_evt) / 1000.0) / bsz
        t3_tot += (start.elapsed_time(exit3_evt) / 1000.0) / bsz
        count += 1
    return t1_tot / count, t2_tot / count, t3_tot / count

def load_model(model_name, model_path, device, time_steps_from_args=None):
    """
    Load a model from a checkpoint file.
    Handles INT8 quantized model loading robustly.
    Args:
        model_name: Name of the model to load
        model_path: Path to the model checkpoint
        device: Device to load the model on
    Returns:
        Loaded model
    """
    # INT8 quantized model loading (force CPU)
    if model_name == "branchy_quantized_snn" or (model_path and model_path.endswith('_int8.pth')):
        print(f"[INFO] Loading INT8 quantized model from {model_path} on CPU...")
        # Ensure device is CPU for INT8 model object loading if it's a full object
        load_location = torch.device('cpu') if model_path.endswith('_int8.pth') else device
        
        # If time_steps_from_args is provided, it should ideally be used during model construction
        # if we are reconstructing the model from a state_dict.
        # If loading a full model object, its internal time_window should be what it was saved with,
        # but we might want to override or at least be aware.
        
        model_obj = torch.load(model_path, map_location=load_location)

        if isinstance(model_obj, (dict, OrderedDict)):
            # If state_dict, must reconstruct model
            if model_name == "branchy_quantized_snn":
                from models.branchy_quantized_snn import BranchyQuantizedSNN
                # Use time_steps_from_args if available, otherwise model's default or config
                effective_time_steps = time_steps_from_args if time_steps_from_args is not None else config.get('num_steps', 25)
                print(f"[Load BQSNN from state_dict] Using time_window: {effective_time_steps}")
                model = BranchyQuantizedSNN(time_window=effective_time_steps) # Pass time_steps here
                # --- FUSE Conv/Linear+BN LAYERS BEFORE LOADING STATE_DICT ---
                # (rebuilt QDSNN exposes fuse_model(); the old standalone class
                # had fuse_bn(), which no longer exists.)
                try:
                    model.fuse_model()
                    print("[INFO] Fused Conv/Linear+BN layers before loading state_dict.")
                except Exception as e_fuse:
                    print(f"[WARN] Could not fuse layers during loading: {e_fuse}. State_dict loading might fail.")
                # ------------------------------------------------
                model.load_state_dict(model_obj, strict=False) # Allow missing temperature keys
            else:
                raise RuntimeError("[ERROR] INT8 quantized model must be saved and loaded as a full model object, not a state_dict.")
        else:
            model = model_obj
        model.eval()
        model.is_quantized = True
        return model
    if model_name == "standard_alexnet":
        model = AlexNet(num_classes=config["num_classes"]).to(device)
    elif model_name == "branchy_alexnet" or model_name == "dynamic_alexnet":
        # <<< START MODIFIED CODE >>>
        from models.branchy_models import BranchyAlexNet # Ensure import is local to this scope
        # <<< END MODIFIED CODE >>>
        model = BranchyAlexNet(num_classes=config["num_classes"]).to(device)
    elif model_name == "quantized_dynamic_snn":

        # Load INT8 quantized model for native inference
        print(f"[INFO] Loading INT8 quantized model: {os.path.basename(model_path)}")
        from models.branchy_models import BranchyAlexNet
        import snntorch as snn
        import torch.nn as nn
        from torch.ao.quantization import QConfig, MinMaxObserver, PerChannelMinMaxObserver, prepare_qat, convert

        # 1. Initialize FP32 model structure
        model = BranchyAlexNet(
            num_classes=config['num_classes'],
        )
        # 2. Apply QConfig matching QAT
        model.qconfig = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
        )
        # 3. Exclude FP32-only modules
        for name, module in model.named_modules():
            if isinstance(module, snn.Leaky) or isinstance(module, nn.BatchNorm2d):
                module.qconfig = None

        # 4. Prepare and convert to quantized modules
        model.train()  # switch to train mode for QAT preparation
        prepare_qat(model, inplace=True)
        model.eval()   # back to eval mode before conversion
        convert(model, inplace=True)

        # 5. Load quantized weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

        # 6. Finalize model
        model.to(device)
        model.is_quantized = True
        model.eval()
        return model
    # QAT BranchyQuantizedSNN (fake-quant state)
    elif model_name == "branchy_quantized_snn" and "int8_rl" in os.path.basename(model_path) and not os.path.basename(model_path).endswith("_int8.pth"):
        print(f"[INFO] Loading QAT BranchyQuantizedSNN (fake-quant): {os.path.basename(model_path)}")
        from models.branchy_quantized_snn import BranchyQuantizedSNN, ExitClassifier
        import snntorch as snn
        import torch.nn as nn
        from torch.ao.quantization import get_default_qat_qconfig, prepare_qat

        # Initialize QAT-ready model with fake-quant stubs
        effective_time_steps = time_steps_from_args if time_steps_from_args is not None else config.get('num_steps', 25)
        print(f"[Load QAT BQSNN] Using time_window: {effective_time_steps}")
        model = BranchyQuantizedSNN(num_classes=config['num_classes'], time_window=effective_time_steps, beta=config.get('beta', 0.9))
        qconfig = get_default_qat_qconfig('fbgemm')
        model.qconfig = qconfig
        model.eval()  # Ensure eval mode before fusion

        for name, module in model.named_modules():
            if isinstance(module, (snn.Leaky, ExitClassifier, nn.BatchNorm2d)):
                module.qconfig = None
        model.train(); prepare_qat(model, inplace=True); model.eval()

        # Load QAT weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        model.to(device); model.is_quantized = False; model.eval()
        return model
    # Final INT8 BranchyQuantizedSNN conversion
    elif model_name == "branchy_quantized_snn" and os.path.basename(model_path).endswith("_int8.pth"):
        # Load final INT8 quantized BranchyQuantizedSNN
        print(f"[INFO] Loading final INT8 BranchyQuantizedSNN: {os.path.basename(model_path)}")
        try:
            # Try loading the entire model (recommended)
            model = torch.load(model_path, map_location=device)
            model.eval()
            # Quantization integrity checks
            if hasattr(model, "conv1"):
                assert "Quantized" in str(type(model.conv1)), "[ERROR] conv1 is not quantized!"
            if hasattr(model, "exit1") and hasattr(model.exit1, "fc"):
                assert model.exit1.fc.weight.dtype == torch.float32, "[ERROR] exit1.fc is not FP32!"
            if hasattr(model, "confidence_thresholds"):
                print(f"[EVAL] Final thresholds: {model.confidence_thresholds}")
            return model
            print(f"[INFO] Loaded full quantized model from {model_path}")
            model.is_quantized = True
            model.eval()
            return model
        except Exception as e:
            print(f"[WARN] torch.load(model) failed: {e}\nFalling back to state_dict loading (may not work if checkpoint is fully quantized)")
            from models.branchy_quantized_snn import BranchyQuantizedSNN, ExitClassifier
            import snntorch as snn
            import torch.nn as nn
            from torch.ao.quantization import QConfig, MinMaxObserver, PerChannelMinMaxObserver, prepare_qat, convert

            model = BranchyQuantizedSNN(num_classes=config['num_classes'], time_window=config.get('num_steps', 25), beta=config.get('beta', 0.9))
            model.qconfig = QConfig(
                activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
            )
            for name, module in model.named_modules():
                if isinstance(module, snn.Leaky) or isinstance(module, ExitClassifier) or isinstance(module, nn.BatchNorm2d):
                    module.qconfig = None
            model.train(); prepare_qat(model, inplace=True); model.eval()
            # For final INT8 model, do NOT fuse modules again (already fused/quantized)
            # convert(model, inplace=True)  # Only needed if loading pre-converted weights

            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

            model.to(device); model.is_quantized = True; model.eval()
            return model
    elif model_name == "branchy_quantized_snn":
        # Load FP32 RL model checkpoint
        from models.branchy_quantized_snn import BranchyQuantizedSNN
        effective_time_steps = time_steps_from_args if time_steps_from_args is not None else config.get('num_steps', 25)
        print(f"[Load FP32 BQSNN] Using time_window: {effective_time_steps}")
        model = BranchyQuantizedSNN(num_classes=config['num_classes'], time_window=effective_time_steps, beta=config.get('beta')).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded FP32 BranchyQuantizedSNN from {model_path}")
        # Mark as not quantized
        model.is_quantized = False
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load model weights
    if model_path:
        try:
            state_dict = torch.load(model_path, map_location=device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[Warning] Missing keys: {missing}, Unexpected keys: {unexpected}")
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"[ERROR] Could not load model weights. This is likely due to a mismatch between the model definition and the checkpoint.\nError: {e}\nIf you saved the model using torch.save(model.state_dict()), make sure to load it into a model that has already been initialized with the same architecture.")
            raise e

    return model

def load_agent(agent_path, model, device, agent_type="q_learning", n_confidence_bins=None):
    if agent_path is None:
        raise ValueError("No agent_path provided!")
    with open(agent_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # --- MODIFIED SECTION TO HANDLE INCORRECT AGENT TYPE --- 
    from models.rl_environment import QLearningAgent # Ensure QLearningAgent is imported for instanceof check
    if not isinstance(loaded_data, QLearningAgent):
        print(f"[ERROR] Expected QLearningAgent from {agent_path}, but got {type(loaded_data)}. RL Agent will not be used.")
        print(f"[INFO] Loaded data content (first 500 chars): {str(loaded_data)[:500]}")
        # To allow script to proceed without a valid agent if model can run without it or CLI thresholds are primary:
        # Option 1: Return None, and main() handles it.
        # Option 2: Return a dummy agent or raise a specific error to be caught in main().
        # For now, returning None is cleaner, main() logic for agent_path_to_load not found will apply.
        return None # Explicitly return None if not a QLearningAgent
    
    agent = loaded_data # Now we know 'agent' is a QLearningAgent instance
    # --- END MODIFIED SECTION ---

    # --- ADDED: Print Q-table sample for debugging ---
    if hasattr(agent, 'q_table') and isinstance(agent.q_table, dict) and agent.q_table:
        print("[DEBUG] Sample of loaded Q-table (first few items):")
        for i, (state, values) in enumerate(agent.q_table.items()):
            if i < 5: # Print first 5 states
                print(f"  State: {state}, Q-values: {values}")
            else:
                break
        print(f"  Total states in Q-table: {len(agent.q_table)}")
    else:
        print("[DEBUG] Q-table not found or empty in loaded agent.")
    # --- END ADDED ---

    # If n_confidence_bins is specified and agent is a QLearningAgent, update it
    if n_confidence_bins is not None and hasattr(agent, 'n_confidence_bins'):
        # agent.n_confidence_bins = n_confidence_bins
        # Check if n_confidence_bins is an int before assigning,
        # otherwise, it's likely the actual thresholds.
        if isinstance(n_confidence_bins, int):
            agent.n_confidence_bins = n_confidence_bins
            print(f"[INFO] Agent n_confidence_bins updated to: {agent.n_confidence_bins} via load_agent parameter.")
        else:
            print(f"[WARN] load_agent received n_confidence_bins parameter that is not an int (type: {type(n_confidence_bins)}). "
                  f"This is likely intended as direct thresholds. NOT setting agent.n_confidence_bins. "
                  f"Direct thresholds should be applied via agent.confidence_thresholds or agent.set_confidence_thresholds() after agent loading.")
    # Quantization-aware threshold scaling (INT8) - Temporarily disable aggressive scaling
    is_quant = getattr(model, 'is_quantized', False) or any(isinstance(m, torch.nn.quantized.Conv2d) for m in model.modules())
    # if is_quant: # This scaling might be problematic without proper recalibration
    #     if hasattr(agent, 'confidence_thresholds') and agent.confidence_thresholds is not None:
    #         # agent.confidence_thresholds = [t * 0.7 for t in agent.confidence_thresholds] 
    #         print(f"[INFO] Original agent thresholds from .pkl: {agent.confidence_thresholds}. No automatic scaling applied in eval.")
    # --- Robust check and fallback for confidence_thresholds ---
    if not hasattr(agent, 'confidence_thresholds') or agent.confidence_thresholds is None:
        print("[WARN] Loaded RL agent does not have 'confidence_thresholds'. Attempting to extract via get_confidence_thresholds()...")
        try:
            thresholds = agent.get_confidence_thresholds()
            agent.confidence_thresholds = thresholds
            print(f"[INFO] Extracted and set confidence_thresholds: {thresholds}")
        except Exception as e_thresh:
            print(f"[ERROR] Could not extract thresholds from agent: {e_thresh}")
            agent.confidence_thresholds = None
    else:
        print("[INFO] Agent already has confidence thresholds set.")
    # Attach to model if needed
    if hasattr(model, 'rl_agent'):
        model.rl_agent = agent
    # Print RL agent params for debug
    if hasattr(agent, 'learning_rate') and hasattr(agent, 'exploration_rate') and hasattr(agent, 'n_confidence_bins'):
        print(f"[EVAL] Agent RL parameters. Agent LR: {agent.learning_rate}, Agent epsilon: {agent.exploration_rate} Agent Bins: {agent.n_confidence_bins}")
    # Final confirmation print
    if hasattr(agent, 'confidence_thresholds') and agent.confidence_thresholds is not None:
        print(f"[EVAL] Final confidence thresholds used: {agent.confidence_thresholds}")
    else:
        print("[ERROR] No confidence thresholds available. Check agent loading and calibration.")
    return agent

def aggregate_experiment_results():
    """Aggregate all *_eval.csv and energy_timing_per_exit.csv in results/logs and save summary CSVs, plots, markdown report, and (optionally) log to Weights & Biases."""
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    # import os # Already imported
    # import numpy as np # Already imported
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import itertools
    # import csv # Already imported
    eval_files = glob.glob('results/logs/*_eval.csv')
    per_exit_files = glob.glob('energy_timing_per_exit.csv') # Assuming this is generated somewhere else
    # Aggregate overall evals
    if eval_files:
        dfs = [] # Initialize an empty list for dataframes
        for f in eval_files:
            print(f"[AGGREGATE DEBUG] Attempting to read: {f}")
            try:
                # Attempt to read the CSV
                df = pd.read_csv(f)
                # Check if the DataFrame is empty after reading
                if not df.empty:
                    dfs.append(df)
                else:
                    print(f"[AGGREGATE WARN] Skipping empty CSV (no data rows): {f}")
            except pd.errors.EmptyDataError:
                # This catches files that are completely empty or have no columns
                print(f"[AGGREGATE ERROR] EmptyDataError: No columns to parse from {f}. Skipping.")
            except Exception as e_read:
                # Catch any other reading/processing errors
                print(f"[AGGREGATE ERROR] Could not read or process {f}: {e_read}. Skipping.")

        # Proceed with concatenation only if dfs list is not empty
        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            df_all.to_csv('experiment_aggregate.csv', index=False)
            print('[AGGREGATE] Saved experiment_aggregate.csv')
        else:
            print('[AGGREGATE] No eval files found to aggregate.')

    # Aggregate per-exit
    if per_exit_files:
        dfs_exit_list = []
        for f_exit in per_exit_files:
            try:
                df_single_exit = pd.read_csv(f_exit)
                if not df_single_exit.empty:
                    dfs_exit_list.append(df_single_exit)
                else:
                    print(f"[AGGREGATE WARN] Skipping empty per-exit CSV (no data rows): {f_exit}")
            except pd.errors.EmptyDataError:
                print(f"[AGGREGATE ERROR] EmptyDataError in per-exit CSV {f_exit}. Skipping.")
            except Exception as e_read_exit:
                print(f"[AGGREGATE ERROR] Could not read or process per-exit CSV {f_exit}: {e_read_exit}. Skipping.")

        if dfs_exit_list:
            df_all_exit = pd.concat(dfs_exit_list, ignore_index=True)
            df_all_exit.to_csv('experiment_aggregate_per_exit.csv', index=False)
            print('[AGGREGATE] Saved experiment_aggregate_per_exit.csv') # Corrected indentation

            # --- Generate summary plots from per-exit data ---
            plot_files = []
            # Standard bar plots
            for metric_info in [
                (("Accuracy", "Per-Exit Accuracy"), "Accuracy"),
                (("Entropy", "Per-Exit Entropy"), "Entropy"),
                (("Energy (J)", "Per-Exit Energy (J)"), "Energy (J)"),
                (("Inference Time (s)", "Per-Exit Inference Time (s)"), "Inference Time (s)")
            ]:
                metric_pair, ylabel = metric_info[0], metric_info[1]
                col, title = metric_pair
                if col in df_all_exit.columns:
                    plt.figure(figsize=(8,5))
                    # Group by Hardware Profile if it exists, otherwise plot directly
                    if 'Hardware Profile' in df_all_exit.columns:
                        for hp, group in df_all_exit.groupby('Hardware Profile'):
                            plt.bar(group["Exit"].astype(str) + f" ({hp})", group[col].fillna(0), label=hp) # Fill NaN for plotting
                        plt.legend()
                    else:
                        plt.bar(df_all_exit["Exit"].astype(str), df_all_exit[col].fillna(0))

                    plt.ylabel(ylabel)
                    plt.xlabel("Exit")
                    plt.title(title)
                    plt.xticks(rotation=30, ha='right')
                    plt.tight_layout()
                    fname = f"plot_{col.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png"
                    plt.savefig(fname)
                    plt.close()
                    plot_files.append(fname)
                    print(f"[AGGREGATE] Saved plot {fname}")

            # Confusion matrix and ROC curve per exit (if possible)
            try:
                preds_file = 'experiment_aggregate_per_exit.csv' # Using the aggregated file
                if os.path.exists(preds_file):
                    df_preds = pd.read_csv(preds_file)
                    # Check if prediction/label columns exist (they might not)
                    # Assuming columns like 'Exit 0_preds', 'Exit 0_labels' exist
                    for exit_label in df_preds['Exit'].unique():
                        pred_col = f'{exit_label}_preds' # Hypothetical column name
                        label_col = f'{exit_label}_labels' # Hypothetical column name
                        if pred_col in df_preds.columns and label_col in df_preds.columns:
                            y_true = df_preds[label_col].dropna().astype(int)
                            y_pred = df_preds[pred_col].dropna().astype(int)
                            if len(y_true) > 0 and len(y_true) == len(y_pred):
                                classes = sorted(list(set(y_true) | set(y_pred)))
                                cm = confusion_matrix(y_true, y_pred, labels=classes)
                                plt.figure(figsize=(6,6))
                                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                                plt.title(f'Confusion Matrix: {exit_label}')
                                plt.colorbar()
                                tick_marks = np.arange(len(classes))
                                plt.xticks(tick_marks, classes, rotation=45)
                                plt.yticks(tick_marks, classes)
                                # Labeling the cells
                                thresh = cm.max() / 2.
                                for i_cm, j_cm in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # Renamed loop vars
                                    plt.text(j_cm, i_cm, format(cm[i_cm, j_cm], 'd'),
                                             horizontalalignment="center",
                                             color="white" if cm[i_cm, j_cm] > thresh else "black")
                                plt.ylabel('True label')
                                plt.xlabel('Predicted label')
                                plt.tight_layout()
                                fname_cm = f'confusion_matrix_{exit_label}.png'
                                plt.savefig(fname_cm)
                                plt.close()
                                plot_files.append(fname_cm)
                                print(f"[AGGREGATE] Saved plot {fname_cm}")
                                # ROC (only for binary or needs micro/macro averaging for multi-class)
                                # Simplified: only plot if binary
                                if len(classes) == 2:
                                    # Need scores for ROC, not just predictions. Assuming preds are sufficient for now.
                                    fpr, tpr, _ = roc_curve(y_true, y_pred) # This uses predicted labels, ideally needs scores
                                    roc_auc = auc(fpr, tpr)
                                    plt.figure()
                                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                    plt.xlim([0.0, 1.0])
                                    plt.ylim([0.0, 1.05])
                                    plt.xlabel('False Positive Rate')
                                    plt.ylabel('True Positive Rate')
                                    plt.title(f'Receiver Operating Characteristic: {exit_label}')
                                    plt.legend(loc="lower right")
                                    fname_roc = f'roc_curve_{exit_label}.png'
                                    plt.savefig(fname_roc)
                                    plt.close()
                                    plot_files.append(fname_roc)
                                    print(f"[AGGREGATE] Saved plot {fname_roc}")
            except Exception as e:
                print(f"[AGGREGATE] Confusion/ROC plot error: {e}")

            # --- Generate Markdown Report ---
            stats_md = ""
            try:
                stats_md = "\n## Summary Statistics (Mean / Stddev)\n\n| Exit | Metric | Mean | Stddev |\n|---|---|---|---|\n"
                # Ensure grouping includes Hardware Profile if present
                group_cols = ['Exit']
                if 'Hardware Profile' in df_all_exit.columns:
                    group_cols.append('Hardware Profile')

                for group_key, group_df in df_all_exit.groupby(group_cols):
                    exit_label_md = str(group_key)
                    for metric in ["Accuracy", "Entropy", "Energy (J)", "Inference Time (s)"]:
                        if metric in group_df.columns:
                            vals = group_df[metric].dropna()
                            if len(vals): # Corrected: was if not len(vals)
                                stats_md += f"| {exit_label_md} | {metric} | {np.mean(vals):.4f} | {np.std(vals):.4f} |\n"
                print("[AGGREGATE] Generated summary stats table.")
            except Exception as e:
                stats_md = "\n*Error generating stats table.*\n"
                print(f"[AGGREGATE] Stats table error: {e}")

            with open("experiment_report.md", "w") as f:
                f.write("# Experiment Summary Report\n\n")
                f.write("## Aggregated Per-Exit Metrics\n\n")
                for pf in plot_files:
                    if os.path.exists(pf):
                        f.write(f"![{pf}]({pf})\n\n")
                f.write("## CSVs\n\n")
                if os.path.exists('experiment_aggregate.csv'):
                    f.write("- [experiment_aggregate.csv](experiment_aggregate.csv)\n")
                if os.path.exists('experiment_aggregate_per_exit.csv'):
                    f.write("- [experiment_aggregate_per_exit.csv](experiment_aggregate_per_exit.csv)\n")
                f.write(stats_md)
            print("[AGGREGATE] Saved experiment_report.md")

            # --- Email report if env var set --- (Assuming necessary mail setup)
            if os.getenv("EMAIL_REPORT_TO"):
                try:
                    import smtplib
                    from email.mime.multipart import MIMEMultipart
                    from email.mime.text import MIMEText
                    from email.mime.base import MIMEBase
                    from email import encoders
                    email_to = os.environ["EMAIL_REPORT_TO"]
                    email_from = os.environ.get("EMAIL_REPORT_FROM", "noreply@localhost") # Default sender
                    msg = MIMEMultipart()
                    msg['From'] = email_from
                    msg['To'] = email_to
                    msg['Subject'] = "Experiment Report Aggregation"
                    with open("experiment_report.md", "r") as f_report:
                        body = f_report.read()
                    msg.attach(MIMEText(body, 'plain'))
                    files_to_attach = plot_files + ['experiment_aggregate.csv', 'experiment_aggregate_per_exit.csv', 'experiment_report.md']
                    for fname in files_to_attach:
                        if os.path.exists(fname):
                            part = MIMEBase('application', 'octet-stream')
                            with open(fname, 'rb') as file_attach:
                                part.set_payload(file_attach.read())
                            encoders.encode_base64(part)
                            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(fname)}')
                            msg.attach(part)
                    # Assuming local SMTP server or properly configured MTA
                    s = smtplib.SMTP('localhost')
                    s.send_message(msg)
                    s.quit()
                    print(f"[EMAIL] Report emailed to {email_to}")
                except Exception as e:
                    print(f"[EMAIL] Emailing report failed: {e}")

            # --- Archive all report artifacts --- (ZipFile needs import)
            try:
                import zipfile
                with zipfile.ZipFile('experiment_report.zip', 'w') as zipf:
                    files_to_archive = plot_files + ['experiment_aggregate.csv', 'experiment_aggregate_per_exit.csv', 'experiment_report.md']
                    for fname in files_to_archive:
                        if os.path.exists(fname):
                            zipf.write(fname, os.path.basename(fname))
                print("[AGGREGATE] Archived all report artifacts to experiment_report.zip")
            except Exception as e:
                print(f"[AGGREGATE] Archiving failed: {e}")

            # --- Optional: Weights & Biases logging ---
            if os.getenv("WANDB_API_KEY") and os.getenv("WANDB_PROJECT") and os.getenv("WANDB_ENTITY"):
                try:
                    import wandb
                    wandb.login(key=os.environ["WANDB_API_KEY"])
                    run_name = f"experiment-aggregate-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}" # Requires pandas import
                    wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], name=run_name, reinit=True)
                    # Log aggregate CSVs as tables
                    if os.path.exists('experiment_aggregate.csv'):
                        wandb.log({"experiment_aggregate": wandb.Table(dataframe=pd.read_csv('experiment_aggregate.csv'))})
                    if os.path.exists('experiment_aggregate_per_exit.csv'):
                        wandb.log({"experiment_aggregate_per_exit": wandb.Table(dataframe=pd.read_csv('experiment_aggregate_per_exit.csv'))})
                    # Log plots
                    for pf in plot_files:
                        if os.path.exists(pf):
                            wandb.log({os.path.basename(pf): wandb.Image(pf)})
                    # Log report markdown file
                    if os.path.exists("experiment_report.md"):
                        wandb.save("experiment_report.md") # Save markdown file as artifact
                    wandb.finish()
                    print("[WANDB] Logged aggregate results and plots to Weights & Biases.")
                except Exception as e:
                    print(f"[WANDB] Logging failed: {e}")
            else:
                print('[AGGREGATE] WANDB environment variables not set. Skipping W&B logging.')
        else:
            print('[AGGREGATE] No per-exit CSV dataframes to process after reading files.')
    else:
        print('[AGGREGATE] No per-exit files found to aggregate.')



def main():
    parser = argparse.ArgumentParser(description="Evaluate neural network models")
    parser.add_argument("--model", type=str, default="all",
                        help="Model to evaluate (standard_alexnet, dynamic_alexnet, all)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--agent_path", type=str, default=None,
                        help="Path to RL agent checkpoint (for dynamic models)")
    parser.add_argument("--agent_type", type=str, default="q_learning",
                        choices=["q_learning"],
                        help="Type of RL agent")
    parser.add_argument("--batch_size", type=int, default=config["test_batch_size"],
                        help="Batch size for evaluation")
    parser.add_argument("--confidence_thresholds", type=float, nargs='+', default=None,
                        help="Custom confidence thresholds for early exits (list of floats)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on (cpu or cuda)")
    parser.add_argument("--hardware_profile", type=str, default=None, help="Hardware profile for energy scaling (e.g. nvidia_a100, loihi, cpu)")
    # Add the verify_quant argument parsing if it was removed previously
    parser.add_argument('--verify_quant', action='store_true', help='Verify quantization after loading model')
    parser.add_argument('--time_steps', type=int, default=None, help='Number of time steps for SNN evaluation (overrides model/config defaults if set)')
    args = parser.parse_args()

    # The QDSNN was rebuilt on the full 5-conv/3-fc AlexNet backbone, but this
    # script's per-exit MAC/AC breakdown is hardwired to the OLD standalone
    # 2-conv/2-fc layout (e.g. it feeds a 192*8*8=12288 vector into fc1, which
    # now expects 256*4*4=4096). Fail fast -- before the CIFAR load -- and point
    # to the unified evaluator that measures the QSNN and QDSNN on one path.
    if args.model == "branchy_quantized_snn":
        raise SystemExit(
            "[REDIRECT] Evaluate the QSNN/QDSNN pair with evaluate_quantized_pair.py:\n"
            "  python evaluate_quantized_pair.py "
            "--qsnn_path results/quantized_snn_alexnet_qat_int8.pth "
            "--qdsnn_path results/branchy_quantized_snn_qat.pth "
            "--agent_path results/rl_agent_qat.pkl\n"
            "evaluate.py now only covers the FP32 ANN reference models."
        )

    # Set device from argument
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    if args.hardware_profile:
        print(f"[INFO] Using hardware profile: {args.hardware_profile}")

    # Optionally, set random seed for reproducibility
    set_seed(42)
    # Get data loaders
    _, _, test_loader, _ = get_data_loaders(distributed=False, augment=False)

    # Use available device (CUDA if possible) for evaluation - determined later based on model type

    if args.model == "all":
        # Evaluate all models
        models_to_evaluate = ["standard_alexnet", "dynamic_alexnet"] # Add other models as needed

        for model_name in models_to_evaluate:
            print(f"--- Evaluating {model_name} ---")
            # Determine the specific model path for each model
            specific_model_path = None
            agent_path_to_load = None

            if model_name == "standard_alexnet":
                specific_model_path = args.model_path or "results/standard_alexnet.pth"
            elif model_name == "dynamic_alexnet":
                specific_model_path = args.model_path or "results/branchy_alexnet.pth"
                agent_path_to_load = args.agent_path or "results/rl_agent.pkl"
            elif model_name == "branchy_quantized_snn": # Add this case if needed for "all"
                specific_model_path = args.model_path or "results/branchy_quantized_snn_qat.pth"
                agent_path_to_load = args.agent_path or "results/rl_agent_qat.pkl"
            # Add other cases as necessary

            if not specific_model_path or not os.path.exists(specific_model_path):
                print(f"Skipping {model_name}: Model path not found or specified ({specific_model_path})")
                continue

            # --- Load Model ---
            model = load_model(model_name, specific_model_path, device, time_steps_from_args=args.time_steps)

            # --- Calibrate Temperatures (if applicable) ---
            if "branchy_quantized_snn" in model_name:
                if hasattr(model, 'temperature1') and hasattr(model, 'temperature2'):
                    print(f"[INFO] Calibrating temperatures for {model_name}...")
                    try:
                        from data_loader import get_calibration_loader
                        calibration_loader = get_calibration_loader(num_batches=20)
                    except ImportError:
                        print("[WARN] get_calibration_loader not found. Using subset of test_loader.")
                        import itertools
                        calibration_loader = list(itertools.islice(test_loader, 20))
                    model = calibrate_temperatures(model, calibration_loader, device)
                    print(f"[INFO] {model_name} Temperatures calibrated. T1: {model.temperature1.item():.4f}, T2: {model.temperature2.item():.4f}")
                else:
                    print(f"[INFO] {model_name} does not support temperature scaling. Skipping calibration.")

            # --- Load Agent (if applicable) ---
            if agent_path_to_load:
                if os.path.exists(agent_path_to_load):
                    print(f"[INFO] Loading agent for {model_name} from: {agent_path_to_load}")
                    agent = load_agent(agent_path_to_load, model, device, args.agent_type, n_confidence_bins=args.confidence_thresholds)
                    model.rl_agent = agent
                    if args.confidence_thresholds is not None: # Apply CLI override if provided
                        if hasattr(agent, 'set_confidence_thresholds'):
                            agent.set_confidence_thresholds(args.confidence_thresholds)
                        elif hasattr(agent, 'confidence_thresholds'):
                            agent.confidence_thresholds = args.confidence_thresholds
                        if hasattr(model, 'confidence_thresholds') and model.confidence_thresholds is not agent.confidence_thresholds: # Corrected: model.confidence_thresholds *is not* agent.confidence_thresholds
                            model.confidence_thresholds = args.confidence_thresholds
                        print(f"[INFO] CLI Overriding RL thresholds for {model_name}: {args.confidence_thresholds}")
                    print(f"[DEBUG] {model_name} will use thresholds: {getattr(model.rl_agent, 'confidence_thresholds', 'N/A')}")
                else:
                    print(f"[WARN] Agent path {agent_path_to_load} not found for {model_name}.")

            # --- Determine Eval Device and Evaluate ---
            eval_dev = device
            if getattr(model, "is_quantized", False) or (specific_model_path and specific_model_path.endswith('_int8.pth')):
                eval_dev = torch.device('cpu')
                print(f"[INFO] Forcing INT8 evaluation on CPU for {model_name}.")

            evaluate_model(model, model_name, test_loader, eval_dev, hardware_profile=args.hardware_profile, time_steps_from_args=args.time_steps)
            print(f"--- Finished evaluating {model_name} ---\n")

    else: # Single model evaluation path (RESTRUCTURED)
        # Determine the specific model path
        specific_model_path = None
        agent_path_to_load = None
        if args.model == "standard_alexnet":
            specific_model_path = args.model_path or "results/standard_alexnet.pth"
        elif args.model == "dynamic_alexnet":
            specific_model_path = args.model_path or "results/branchy_alexnet.pth"
            agent_path_to_load = args.agent_path or "results/rl_agent.pkl"
        else: # Handle unknown model or direct path case
            if args.model_path and os.path.exists(args.model_path):
                specific_model_path = args.model_path
                # Try to infer agent path if dynamic? - For now, requires explicit agent_path for unknown model types
                if args.agent_path:
                    agent_path_to_load = args.agent_path
            else:
                raise ValueError(f"Unknown model name '{args.model}' and no valid --model_path provided.")

        if not specific_model_path or not os.path.exists(specific_model_path):
            raise FileNotFoundError(f"Model path not found: {specific_model_path}")

        # --- 1. Load Model ---
        print(f"--- Evaluating Single Model: {args.model} ---")
        # Pass args.time_steps to load_model
        model = load_model(args.model, specific_model_path, device, time_steps_from_args=args.time_steps)

        # --- Optional: Early Quantization Check ---
        is_int8_model = specific_model_path and specific_model_path.endswith('_int8.pth')
        if args.verify_quant and args.model == "branchy_quantized_snn" and is_int8_model:
            print("[INFO] Verifying quantization of INT8 model after loading...")
            try:
                verify_quantization(model)
                print("[VERIFY] Post-load quantization verification passed.")
            except Exception as e:
                print(f"[VERIFY] Post-load quantization verification FAILED: {e}")

        # --- 2. Calibrate Temperatures (if applicable) ---
        if "branchy_quantized_snn" in args.model:
            if hasattr(model, 'temperature1') and hasattr(model, 'temperature2'):
                print("[INFO] Calibrating temperatures for BranchyQuantizedSNN...")
                # --- DIAGNOSTIC REMOVED: Re-enable calibration but ensure MAX_TEMP is reasonable ---
                # print("[DIAGNOSTIC] Forcing T1=1.0, T2=1.0 for BranchyQuantizedSNN by not running calibration logic.")
                # # Ensure temperatures are 1.0 if they were changed by a previous step or loaded model
                # with torch.no_grad():
                #     model.temperature1.fill_(1.0)
                #     model.temperature2.fill_(1.0)
                # print(f"[DIAGNOSTIC] Temperatures manually set. T1: {model.temperature1.item():.4f}, T2: {model.temperature2.item():.4f}")
                try:
                    from data_loader import get_calibration_loader
                    calibration_loader = get_calibration_loader(num_batches=20)
                except ImportError:
                    print("[WARN] get_calibration_loader not found. Using subset of test_loader for calibration.")
                    import itertools
                    calibration_loader = list(itertools.islice(test_loader, 20))
                # Pass args to calibrate_temperatures if you want to control MAX_TEMP from CLI eventually
                model = calibrate_temperatures(model, calibration_loader, device)
                print(f"[INFO] Temperatures calibrated. T1: {model.temperature1.item():.4f}, T2: {model.temperature2.item():.4f}")

                # --- REMOVED: Manually reset temperatures to 1.0 for diagnostics ---
                # if hasattr(model, 'temperature1') and hasattr(model, 'temperature2'):
                #     with torch.no_grad():
                #         model.temperature1.fill_(1.0)
                #         model.temperature2.fill_(1.0)
                #     print(f"[DIAGNOSTIC] Manually RESET temperatures to T1: {model.temperature1.item():.4f}, T2: {model.temperature2.item():.4f} to observe unscaled confidences.")
                # --- END REMOVED ---
            else:
                print("[INFO] Model does not support temperature scaling. Skipping calibration.")

        # --- 3. Load Agent (if applicable) ---
        if agent_path_to_load:
            if os.path.exists(agent_path_to_load):
                print(f"[INFO] Loading agent from: {agent_path_to_load}")
                # agent can be QLearningAgent instance or None if loaded_data is not QLearningAgent
                agent = load_agent(agent_path_to_load, model, device, args.agent_type, n_confidence_bins=args.confidence_thresholds)
                
                if agent is not None: # Agent loaded successfully and is a QLearningAgent
                    model.rl_agent = agent # Attach valid agent to model
                    if args.confidence_thresholds is not None:
                        thresholds_set_on_agent = False
                        if hasattr(agent, 'set_confidence_thresholds'):
                            agent.set_confidence_thresholds(args.confidence_thresholds)
                            print(f"[INFO] CLI Overriding RL agent confidence thresholds via set_confidence_thresholds(): {args.confidence_thresholds}")
                            thresholds_set_on_agent = True
                        elif hasattr(agent, 'confidence_thresholds'):
                            agent.confidence_thresholds = args.confidence_thresholds
                            print(f"[INFO] CLI Overriding RL agent confidence_thresholds attribute: {args.confidence_thresholds}")
                            thresholds_set_on_agent = True
                        else:
                            print(f"[WARN] Agent object loaded, but cannot set CLI confidence_thresholds on it directly (no suitable attribute/method found).")

                        # Also, always try to set model.confidence_thresholds directly from CLI if provided,
                        # as this is the most direct control for models that use this attribute.
                        if hasattr(model, 'confidence_thresholds'):
                            model.confidence_thresholds = args.confidence_thresholds
                            print(f"[INFO] Directly set model.confidence_thresholds from CLI: {model.confidence_thresholds}")
                        
                    # If agent is valid but CLI thresholds were not provided, ensure model uses agent's current thresholds
                    elif hasattr(model, 'confidence_thresholds') and hasattr(agent, 'confidence_thresholds') and agent.confidence_thresholds is not None:
                         model.confidence_thresholds = agent.confidence_thresholds
                         print(f"[INFO] No CLI thresholds. Synced model.confidence_thresholds with agent's loaded thresholds: {model.confidence_thresholds}")

                else: # agent is None, meaning load_agent failed to return a valid QLearningAgent
                    print(f"[WARN] RL Agent from {agent_path_to_load} was not a valid QLearningAgent object or failed to load. Model will not use an RL agent.")
                    model.rl_agent = None # Ensure model.rl_agent is None
                    # If agent loading failed, but CLI thresholds are given, apply them directly to the model
                    if hasattr(model, 'confidence_thresholds') and args.confidence_thresholds is not None:
                        model.confidence_thresholds = args.confidence_thresholds
                        print(f"[INFO] Agent loading failed. CLI thresholds {args.confidence_thresholds} applied directly to model.confidence_thresholds.")
            
            else: # Agent path does not exist
                print(f"[WARN] Agent path {agent_path_to_load} not found. Evaluation proceeds without agent.")
                model.rl_agent = None # Ensure model.rl_agent is None
                if hasattr(model, 'confidence_thresholds') and args.confidence_thresholds is not None:
                    model.confidence_thresholds = args.confidence_thresholds
                    print(f"[INFO] Agent path not found. CLI thresholds {args.confidence_thresholds} applied directly to model.confidence_thresholds.")
        
        else: # No agent_path_to_load was specified at all
            print("[INFO] No agent path specified. Model will run without an RL agent.")
            model.rl_agent = None # Ensure model.rl_agent is None
            if hasattr(model, 'confidence_thresholds') and args.confidence_thresholds is not None:
                model.confidence_thresholds = args.confidence_thresholds
                print(f"[INFO] No agent path given. CLI thresholds {args.confidence_thresholds} applied directly to model.confidence_thresholds.")

        # --- Specific handling for branchy_quantized_snn if no valid agent and CLI thresholds are given ---
        if "branchy_quantized_snn" in args.model and model.rl_agent is None and args.confidence_thresholds is not None:
            if hasattr(model, 'confidence_thresholds'):
                model.confidence_thresholds = args.confidence_thresholds
                print(f"[INFO] For {args.model}, no valid RL agent. CLI thresholds {args.confidence_thresholds} applied directly to model.confidence_thresholds.")
            else:
                print(f"[WARN] For {args.model}, no valid RL agent, and model has no 'confidence_thresholds' attribute to set from CLI.")

        # Final check of what thresholds the model's agent (if any) AND the model itself will use
        final_agent_thresholds_debug = "N/A (No RL Agent or agent has no threshold attribute)"
        if hasattr(model, 'rl_agent') and model.rl_agent is not None:
            if hasattr(model.rl_agent, 'get_confidence_thresholds'):
                final_agent_thresholds_debug = model.rl_agent.get_confidence_thresholds()
            elif hasattr(model.rl_agent, 'confidence_thresholds'): # Fallback direct attribute check
                final_agent_thresholds_debug = model.rl_agent.confidence_thresholds
        print(f"[DEBUG] model.rl_agent is {type(model.rl_agent)} and will use thresholds: {final_agent_thresholds_debug}")
        
        if hasattr(model, 'confidence_thresholds'):
            print(f"[DEBUG] model.confidence_thresholds are currently set to: {model.confidence_thresholds}")
            # --- REMOVED TEMPORARY OVERRIDE FOR DIAGNOSIS ---
            # if "branchy_quantized_snn" in args.model:
            #     override_thresholds = [0.7, 0.75] # Example reasonable thresholds
            #     model.confidence_thresholds = override_thresholds
            #     print(f"[TEMP OVERRIDE] For {args.model}, manually setting model.confidence_thresholds to: {model.confidence_thresholds}")
            # --- END REMOVED TEMPORARY OVERRIDE ---
        else:
            print(f"[DEBUG] Model does not have a 'confidence_thresholds' attribute.")

        # --- 4. Determine Eval Device and Evaluate ONCE ---
        eval_dev = device # Start with the user-requested device
        if getattr(model, "is_quantized", False) or is_int8_model:
            eval_dev = torch.device('cpu')
            print("[INFO] Forcing INT8 evaluation on CPU for the final evaluation run.")

        # --- MODIFIED: Loop for forced exit paths for branchy_quantized_snn ---
        if "branchy_quantized_snn" in args.model:
            # Create threshold suffix for filename if CLI thresholds are provided
            threshold_suffix = ""
            if args.confidence_thresholds is not None:
                threshold_str = "_".join([f"{t:.1f}" for t in args.confidence_thresholds])
                threshold_suffix = f"_th{threshold_str}"
            
            # --- ADDED: DYNAMIC EVALUATION RUN FIRST ---
            print(f"--- Starting DYNAMIC Evaluation for {args.model} (RL Agent Controlled) ---")
            
            # --- ADDED: Explicitly set T=1.0 for dynamic RL run after calibration/agent loading ---
            print(f"[DIAGNOSTIC] For DYNAMIC RL run, ensuring T1=1.0, T2=1.0 (post-calibration).")
            if hasattr(model, 'temperature1') and isinstance(model.temperature1, torch.nn.Parameter):
                with torch.no_grad(): # Ensure no gradient tracking for this modification
                    model.temperature1.fill_(1.0)
            else: # If not a parameter (e.g. direct tensor), reassign
                 model.temperature1 = torch.tensor(1.0, device=model.temperature1.device if hasattr(model.temperature1, 'device') else device)
            
            if hasattr(model, 'temperature2') and isinstance(model.temperature2, torch.nn.Parameter):
                with torch.no_grad():
                    model.temperature2.fill_(1.0)
            else:
                model.temperature2 = torch.tensor(1.0, device=model.temperature2.device if hasattr(model.temperature2, 'device') else device)
            # --- END ADDED T=1.0 FOR DYNAMIC RUN ---

            dynamic_model_name_for_log = f"{args.model}{threshold_suffix}_dynamic_rl_eval"
            evaluate_model(model, dynamic_model_name_for_log, test_loader, eval_dev,
                           hardware_profile=args.hardware_profile,
                           force_exit_path=None, # force_exit_path=None for dynamic RL behavior
                           time_steps_from_args=args.time_steps) 
            print(f"--- Finished DYNAMIC Evaluation for {args.model} (Logged as: {dynamic_model_name_for_log}) ---")
            # --- END ADDED DYNAMIC EVALUATION RUN ---

            # The TEMP OVERRIDE for thresholds and DIAGNOSTIC T reset are already applied to 'model' by this point.
            # They will be active, but force_exit_path will dictate the decision path.
            print(f"--- Starting Final Evaluation for {args.model} with Forced Exit Paths ---")
            
            # --- ADDED: Explicitly set T=1.0 for forced exit runs after calibration/agent loading ---
            print(f"[DIAGNOSTIC] For FORCED EXIT runs, ensuring T1=1.0, T2=1.0 (post-calibration).")
            if hasattr(model, 'temperature1') and isinstance(model.temperature1, torch.nn.Parameter):
                with torch.no_grad():
                    model.temperature1.fill_(1.0)
            else:
                 model.temperature1 = torch.tensor(1.0, device=model.temperature1.device if hasattr(model.temperature1, 'device') else device)

            if hasattr(model, 'temperature2') and isinstance(model.temperature2, torch.nn.Parameter):
                with torch.no_grad():
                    model.temperature2.fill_(1.0)
            else:
                model.temperature2 = torch.tensor(1.0, device=model.temperature2.device if hasattr(model.temperature2, 'device') else device)
            # --- END ADDED T=1.0 FOR FORCED RUNS ---

            for forced_path_idx in range(3): # 0 for Exit 1, 1 for Exit 2, 2 for Final Exit
                model_name_for_log = f"{args.model}{threshold_suffix}_forced_exit{forced_path_idx}"
                print(f"  --- Evaluating with force_exit_path = {forced_path_idx} (Log as: {model_name_for_log}) ---")
                
                # Reset exit counts for each forced path evaluation if they are part of the model
                # or ensure evaluate_model initializes them cleanly.
                # For now, assuming evaluate_model handles exit_counts initialization per call.
                
                evaluate_model(model, model_name_for_log, test_loader, eval_dev, 
                               hardware_profile=args.hardware_profile,
                               force_exit_path=forced_path_idx,
                               time_steps_from_args=args.time_steps)
                print(f"  --- Finished evaluating with force_exit_path = {forced_path_idx} ---")
            print(f"--- Finished All Forced Exit Path Evaluations for {args.model} ---")
        else:
            # Original single evaluation call for other models
            print(f"--- Starting Final Evaluation for {args.model} ---")
            evaluate_model(model, args.model, test_loader, eval_dev, hardware_profile=args.hardware_profile, time_steps_from_args=args.time_steps)
            print(f"--- Finished Final Evaluation for {args.model} ---")

        # --- Optional: Post-final-evaluation Verification ---
        # This might be redundant if checked earlier, unless evaluate_model modifies the model state unexpectedly.
        # if args.verify_quant and args.model == "branchy_quantized_snn" and is_int8_model:
        # print("[INFO] Verifying quantization post-final-evaluation...")
        # verify_quantization(model)

# Need BranchyQuantizedSNN for type checking maybe, handled in load_model
# Need utils functions if not imported globally

if __name__ == "__main__":
    aggregate_experiment_results()
    main()