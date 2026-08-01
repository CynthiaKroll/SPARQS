import argparse
import torch
import time
import os
import csv
import numpy as np
from tqdm import tqdm
import threading
import subprocess
import random
from collections import OrderedDict
import pandas as pd
import json

from models.lenet import LeNet
from models.branchy_models import BranchyLeNet
from models.lenet_snn import LeNetSNN
from models.quantized_snn_lenet import QuantizedSNNLeNet
from models.branchy_quantized_snn import BranchyQuantizedSNN
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

def get_device():
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    
    return device

def evaluate_model(model, model_name, test_loader, device, hardware_profile=None, force_exit_path=None, time_steps_from_args=None):
    
    from utils import (calculate_fair_theoretical_energy_from_dict, count_mac_operations,
                       count_memory_access, E_MAC_32BIT_J, E_MAC_8BIT_J, E_AC_8BIT_J, E_MEM_J_PER_BYTE)

    
    is_quantized_eval = False 
    precision_eval = 'FP32' 

    
    # The op/memory counters must be driven by REAL test images, one sample at
    # a time. Feeding torch.randn noise makes an early-exit model maximally
    # unconfident, so it never exits and the counters report the worst-case
    # full path -- that is why dynamic_lenet used to score MORE MACs (493K)
    # than plain lenet (416K). One sample at a time matters for the same
    # reason a batch hides it: a branchy model only skips a stage once EVERY
    # sample still active in the batch has exited.
    COUNTER_SAMPLES = 32
    try:
        sample_batch_for_shape, _ = next(iter(test_loader))

        if sample_batch_for_shape.dim() >= 4:
             _, C_shape, H_shape, W_shape = sample_batch_for_shape.shape[:4]
        else:
             C_shape, H_shape, W_shape = config["image_size"]  # MNIST: (1, 28, 28)
        counter_samples = sample_batch_for_shape[:COUNTER_SAMPLES].to(device)
    except Exception as e:
        print(f"[Eval] WARNING: could not draw real samples for counting ({e}); "
              f"falling back to noise, early-exit counts will be worst-case.")
        C_shape, H_shape, W_shape = config["image_size"]  # MNIST: (1, 28, 28)
        counter_samples = torch.randn(1, C_shape, H_shape, W_shape).to(device)

    n_counter_samples = counter_samples.size(0)
    dummy_sample_for_mem = counter_samples[:1]

    def _mean_over_samples(counter_fn):
        """Average a per-sample counter over real images, one sample at a time."""
        return sum(counter_fn(counter_samples[i:i + 1])
                   for i in range(n_counter_samples)) / n_counter_samples

    
    exit_counts = {1: 0, 2: 0, 3: 0}
    total_correct = 0
    total_samples = 0
    
    
    avg_acc = 0.0
    avg_energy = 0.0
    avg_time = 0.0
    avg_power = 0.0
    
    
    print(f"[Eval] Counting operations for {model_name}...")

    # Count ops on the INFERENCE path. For a branchy model, training_mode=True
    # runs every exit branch, so MACs would be over-counted; set it False first.
    if hasattr(model, 'training_mode'):
        model.training_mode = False

    mac_to_log_final = _mean_over_samples(
        lambda s: count_mac_operations(model, s))
    print(f"[Eval] MAC operations counted: {mac_to_log_final} "
          f"(mean over {n_counter_samples} real samples)")


    if not model_name.startswith("snn_") and model_name != "quantized_snn_lenet" and model_name != "branchy_quantized_snn":
        ann_mem_bytes_val = _mean_over_samples(
            lambda s: count_memory_access(model, s, is_snn=False))
        ann_mem_accesses_val = ann_mem_bytes_val # Store it
        print(f"[Eval] ANN Memory access counted: {ann_mem_bytes_val} bytes")
    else:

        ann_mem_bytes_val = 0
        ann_mem_accesses_val = 0
        print(f"[Eval] Skipping ANN memory counting for SNN model {model_name}")
    
    
    metrics_dict = {
        'macs': mac_to_log_final,
        'ann_mem_accesses': ann_mem_bytes_val,
    }
    fair_theoretical_energy_j, source_info = calculate_fair_theoretical_energy_from_dict(metrics_dict, model_name)
    print(f"[Eval] Fair Theoretical Energy calculated: {fair_theoretical_energy_j:.5f} J (Source: {source_info})")

    
    ac_ops_to_log_final = np.nan
    snn_mem_bytes_val = 0
    snn_mem_accesses_val = 0
    spike_count_to_log_final = 0
    syops_to_log_final = 0
    
    
    if (model_name.startswith("snn_") or model_name == "quantized_snn_lenet" or model_name == "branchy_quantized_snn") and time_steps_from_args:
        print(f"[Eval] SNN detected. Counting AC operations with time_steps={time_steps_from_args}...")
        try:
            # Poisson-encoded noise fires at a different rate than real digits,
            # so the spike/AC counts must come from real samples as well.
            ac_sum = spk_sum = 0.0
            for i in range(n_counter_samples):
                ac_i, spk_i = count_ac_operations(model, counter_samples[i:i + 1],
                                                  time_steps_from_args, debug=(i == 0))
                ac_sum += ac_i
                spk_sum += spk_i
            ac_ops_to_log_final = ac_sum / n_counter_samples
            spike_count_to_log_final = spk_sum / n_counter_samples
            print(f"[Eval] AC operations counted: {ac_ops_to_log_final}, "
                  f"Spike count: {spike_count_to_log_final} "
                  f"(mean over {n_counter_samples} real samples)")


            snn_mem_bytes_val = _mean_over_samples(
                lambda s: count_memory_access(model, s, is_snn=True))
            snn_mem_accesses_val = snn_mem_bytes_val
            print(f"[Eval] SNN Memory access counted: {snn_mem_bytes_val} bytes")
        except Exception as e:
            print(f"[WARN] AC operation counting failed for {model_name}: {e}")
            ac_ops_to_log_final = np.nan
    
    model.eval()
    
    total_time_ms = 0  
    
    
    if USE_NVML:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        
        avg_energy = 0.0
        avg_power = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            
            if hasattr(model, 'training_mode'):
                model.training_mode = False
            
            
            if model_name.startswith("snn_") or model_name == "quantized_snn_lenet" or model_name == "branchy_quantized_snn":
                
                num_steps = time_steps_from_args if time_steps_from_args else config["num_steps"]

                spike_train = images.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)  # [T, B, C, H, W]
                model_output = model(spike_train)
            else:
                
                model_output = model(images)
            
            if isinstance(model_output, tuple) and len(model_output) == 2:

                final_outputs, exit_points = model_output
                outputs = final_outputs
                # Record where samples actually left the network; without this
                # an early-exit model reports no exit statistics at all, and
                # its op counts can't be sanity-checked against its routing.
                for k in (1, 2, 3):
                    exit_counts[k] += (exit_points == k).sum().item()
            elif isinstance(model_output, list):
                
                outputs = model_output[-1] 
            else:
                
                outputs = model_output
            
            
            if model_name.startswith("snn_") or model_name in ["quantized_snn_lenet", "branchy_quantized_snn"]:
                if outputs.dim() == 3:  # [time_steps, batch_size, num_classes]
                    outputs = outputs.sum(dim=0)  # Sum over time dimension
            
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size
    
    
    if USE_NVML:
        end_event.record()
        torch.cuda.synchronize()
        
        total_time_ms = start_event.elapsed_time(end_event)
        
        avg_time = (total_time_ms / 1000.0) / total_samples if total_samples > 0 else 0.0
        
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    else:
        
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        avg_time = 0.001  
    
    print(f"\n[Eval] Initial Results for {model_name}:")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  Inference Time: {avg_time:.6f} s")
    
    
    fair_theoretical_energy_j_val = np.nan
    projected_neuromorphic_energy_val = 0.0
    
    
    if not model_name.startswith("snn_") and pd.notna(fair_theoretical_energy_j):
        fair_theoretical_energy_j_val = fair_theoretical_energy_j
        print(f"  Using early calculated Fair Theoretical Energy for {model_name}: {fair_theoretical_energy_j_val:.4e} J")
    
    
    if (model_name.startswith("snn_") or model_name == "quantized_snn_lenet" or model_name == "branchy_quantized_snn") and pd.notna(ac_ops_to_log_final):
        snn_mem_bytes_val = _mean_over_samples(
            lambda s: count_memory_access(model, s, is_snn=True))
        snn_mem_accesses_val = snn_mem_bytes_val 
        ac_energy_j = ac_ops_to_log_final * E_AC_8BIT_J 
        mem_energy_j_snn = snn_mem_bytes_val * E_MEM_J_PER_BYTE
        fair_theoretical_energy_j_val = ac_energy_j + mem_energy_j_snn
        print(f"  Generic SNN {model_name} Fair Theoretical Energy (AC ops based): {fair_theoretical_energy_j_val:.4e} J (ACs: {ac_ops_to_log_final}, SNN Mem: {snn_mem_bytes_val})")
    
        
        from config import E_AC_NEUROMORPHIC_J, E_MEM_NEUROMORPHIC_J_PER_BYTE
        neuromorphic_ac_energy = ac_ops_to_log_final * E_AC_NEUROMORPHIC_J
        neuromorphic_mem_energy = snn_mem_bytes_val * E_MEM_NEUROMORPHIC_J_PER_BYTE
        projected_neuromorphic_energy_val = neuromorphic_ac_energy + neuromorphic_mem_energy
        print(f"  Generic SNN {model_name} Projected Neuromorphic Energy (AC ops based): {projected_neuromorphic_energy_val:.4e} J")

    elif (model_name.startswith("snn_") or model_name == "quantized_snn_lenet" or model_name == "branchy_quantized_snn"): # SNN but AC ops were not available
         snn_mem_bytes_val = _mean_over_samples(
            lambda s: count_memory_access(model, s, is_snn=True))
         snn_mem_accesses_val = snn_mem_bytes_val
         
         mac_proxy_energy_snn = mac_to_log_final * E_MAC_8BIT_J 
         mem_energy_j_snn = snn_mem_bytes_val * E_MEM_J_PER_BYTE
         fair_theoretical_energy_j_val = mac_proxy_energy_snn + mem_energy_j_snn
         print(f"[WARN] Generic SNN {model_name}: AC ops unavailable. Fair Theoretical Energy using MACs proxy: {fair_theoretical_energy_j_val:.4e} J")
         projected_neuromorphic_energy_val = 0.0 # Cannot reliably project without ACs

    
    if pd.notna(fair_theoretical_energy_j_val):
        
        avg_energy = fair_theoretical_energy_j_val
        
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
        ("macs", mac_to_log_final),
        ("ann_mem_accesses", ann_mem_accesses_val),
        ("ann_theoretical_energy", fair_theoretical_energy_j if pd.notna(fair_theoretical_energy_j) else 0.0),
        ("ac_ops", ac_ops_to_log_final if pd.notna(ac_ops_to_log_final) else 0),
        ("spike_count", spike_count_to_log_final),
        ("syops", syops_to_log_final),
        ("snn_mem_accesses", snn_mem_accesses_val),
        ("fair_theoretical_energy", fair_theoretical_energy_j_val if pd.notna(fair_theoretical_energy_j_val) else 0.0),
        ("projected_neuromorphic_energy", projected_neuromorphic_energy_val)
    ])

    # Record the time window for spiking models so the plot T-labels come from
    # data, not visualize.py's fallback. ANNs have no time dimension.
    is_snn = (model_name.startswith("snn_") or model_name == "quantized_snn_lenet"
              or model_name == "branchy_quantized_snn")
    if is_snn:
        results_data["num_steps"] = time_steps_from_args if time_steps_from_args else config["num_steps"]

    # Exit routing, for models that actually have exits. Column names match
    # those written by evaluate_quantized_pair.py so both feed visualize.py.
    if sum(exit_counts.values()) > 0 and total_samples > 0:
        results_data["exit1_percentage"] = 100.0 * exit_counts[1] / total_samples
        results_data["exit2_percentage"] = 100.0 * exit_counts[2] / total_samples
        results_data["final_exit_percentage"] = 100.0 * exit_counts[3] / total_samples
        print(f"  Exit distribution: exit1={results_data['exit1_percentage']:.2f}%, "
              f"exit2={results_data['exit2_percentage']:.2f}%, "
              f"final={results_data['final_exit_percentage']:.2f}%")
    
    
    os.makedirs("results/logs", exist_ok=True)
    csv_path = f"results/logs/{model_name}_eval.csv"
    
    df = pd.DataFrame([results_data])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return results_data

def load_model(model_name, model_path, device, time_steps_from_args=None, confidence_thresholds=None):
    
    
    if model_name == "branchy_quantized_snn" and os.path.exists(model_path):
        
        fp32_path = model_path.replace('.pth', '_fp32.pth')
        if os.path.exists(fp32_path):
            print(f"[INFO] Found FP32 version: {fp32_path}")
            model_path = fp32_path
        else:
            
            print(f"[INFO] Attempting to load quantized branchy SNN with dequantization")
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    
                    model = BranchyQuantizedSNN(num_classes=config["num_classes"])
                    model.training_mode = False
                    
                    
                    state_dict = {}
                    for name, param in checkpoint.items():
                        if hasattr(param, 'dequantize'):
                            state_dict[name] = param.dequantize()
                            print(f"[INFO] Dequantized parameter: {name}")
                        else:
                            state_dict[name] = param
                    
                    model.load_state_dict(state_dict, strict=False)
                    model.to(device)
                    model.eval()
                    
                    
                    if confidence_thresholds is not None:
                        model.confidence_thresholds = confidence_thresholds
                        print(f"[INFO] Set confidence thresholds to {confidence_thresholds}")
                    
                    print(f"[INFO] Successfully loaded dequantized branchy quantized SNN")
                    return model
                    
            except Exception as e:
                print(f"[WARN] Dequantization failed: {e}")
        
        
    
    if model_name == "lenet":
        model = LeNet(num_classes=config["num_classes"])
    elif model_name == "dynamic_lenet":
        model = BranchyLeNet(num_classes=config["num_classes"])
        
        model.training_mode = False
        print(f"[INFO] Set dynamic_lenet to inference mode")
    elif model_name == "snn_lenet":
        model = LeNetSNN(num_classes=config["num_classes"],
                         time_steps=time_steps_from_args if time_steps_from_args else config["num_steps"])
        print(f"[INFO] Loaded SNN LeNet model (T={model.time_steps})")
    elif model_name == "quantized_snn_lenet":
        
        model_time_steps = time_steps_from_args if time_steps_from_args else config["quantized_num_steps"]
        model = QuantizedSNNLeNet(num_classes=config["num_classes"], time_steps=model_time_steps)
        print(f"[INFO] Loaded Quantized SNN LeNet model with time_steps={model_time_steps}")
    elif model_name == "branchy_quantized_snn":
        model = BranchyQuantizedSNN(num_classes=config["num_classes"])
        model.training_mode = False  
        print(f"[INFO] Loaded Branchy Quantized SNN model in inference mode")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load model weights: {e}. Using random weights.")
    else:
        print(f"Warning: Model path {model_path} does not exist. Using random weights.")
    
    model.to(device)
    model.eval()
    
    
    if confidence_thresholds is not None and hasattr(model, 'confidence_thresholds'):
        model.confidence_thresholds = confidence_thresholds
        print(f"[INFO] Set confidence thresholds to {confidence_thresholds}")
    
    return model

def aggregate_experiment_results():
    
    logs_dir = "results/logs"
    if not os.path.exists(logs_dir):
        print(f"Logs directory {logs_dir} does not exist.")
        return
    
    csv_files = [f for f in os.listdir(logs_dir) if f.endswith('_eval.csv')]
    
    if not csv_files:
        print("No evaluation CSV files found.")
        return
    
    all_results = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(logs_dir, csv_file)
        print(f"[AGGREGATE DEBUG] Attempting to read: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                all_results.append(df)
                print(f"Successfully read {csv_file} with {len(df)} rows")
            else:
                print(f"Warning: {csv_file} is empty")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if all_results:
        
        combined_df = pd.concat(all_results, ignore_index=True, sort=False)
        
        
        agg_path = os.path.join(logs_dir, "aggregated_results.csv")
        combined_df.to_csv(agg_path, index=False)
        print(f"Aggregated results saved to {agg_path}")
        
        
        print("\n=== EXPERIMENT SUMMARY ===")
        print(combined_df[['model_name', 'accuracy', 'energy', 'inference_time', 'power']].to_string(index=False))
    else:
        print("No valid CSV files found to aggregate.")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LeNet models')
    # NOTE: the quantized pair (quantized_snn_lenet / branchy_quantized_snn) is
    # NOT evaluated here. This legacy per-model evaluator feeds a direct-repeat
    # spike train (the models are trained with Poisson encoding) and its
    # load_model cannot restore INT8 packed weights (strict=False silently loads
    # nothing). Use evaluate_quantized_pair.py for those -- it measures both on
    # one identical path.
    parser.add_argument('--model', type=str, required=True,
                       choices=['lenet', 'dynamic_lenet', 'snn_lenet'],
                       help='Model to evaluate (FP32 baselines; quantized pair -> evaluate_quantized_pair.py)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--time_steps', type=int, default=config["num_steps"],
                       help='Number of time steps for SNN models (default: config.NUM_STEPS)')
    parser.add_argument('--aggregate_only', action='store_true',
                       help='Only aggregate existing results without running evaluation')
    parser.add_argument('--hardware_profile', type=str, default='a100',
                       choices=['a100', 'v100', 'rtx3090', 'neuromorphic'],
                       help='Hardware profile for energy calculations')
    parser.add_argument('--confidence_thresholds', type=float, nargs=2, default=None,
                       help='Confidence thresholds for early exits (e.g., 0.6 0.7)')
    
    args = parser.parse_args()
    
    
    set_seed(42)
    
    if args.aggregate_only:
        aggregate_experiment_results()
        return
    
    device = get_device()
    
    
    _, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        test_batch_size=args.batch_size
    )
    
    
    model = load_model(args.model, args.model_path, device, args.time_steps, args.confidence_thresholds)
    
    
    results = evaluate_model(
        model, 
        args.model, 
        test_loader, 
        device,
        hardware_profile=args.hardware_profile,
        time_steps_from_args=args.time_steps
    )
    
    print(f"\nEvaluation completed for {args.model}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Energy: {results['energy']:.6e} J")
    print(f"Power: {results['power']:.4f} W")

if __name__ == "__main__":
    main()