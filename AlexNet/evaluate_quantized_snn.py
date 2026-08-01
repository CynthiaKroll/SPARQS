"""
Evaluate a quantized SNN AlexNet model.
- Loads a state_dict saved from QAT.
- Performs inference on the CPU.
- Logs evaluation results to a CSV file.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.quantized_snn_alexnet import QuantizedSNNAlexNet, prepare_quantized_snn_alexnet_for_qat
import pandas as pd
import os
import time
from datetime import datetime
import numpy as np # Add missing import
import argparse # Added: For command-line arguments
from utils import (
    count_ac_operations, 
    count_memory_access, 
    calculate_fair_theoretical_energy_from_dict, # Use centralized function
    E_MAC_32BIT_J, E_MAC_8BIT_J, E_AC_8BIT_J, E_MEM_J_PER_BYTE # Import constants if needed directly (though calc fn uses them)
)

# NVML setup for hardware energy measurement (same as evaluate.py)
USE_NVML = False
if torch.cuda.is_available():
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        USE_NVML = True
        print("NVML initialized for hardware energy measurement")
    except Exception as e:
        print(f"Warning: NVML init failed: {e}")
else:
    print("Warning: CUDA unavailable; energy measurement disabled.")

# --- Configuration ---
# batch_size = 128 # Moved to argparse
# Ensure time_steps is defined, as it's crucial for SNNs
# DEFAULT_TIME_STEPS = 4 # Moved to argparse
# time_steps = DEFAULT_TIME_STEPS # Will be set by argparse
model_path = 'quantized_snn_alexnet_qat.pth'
# For now, we'll hardcode the model name. You'll change this when evaluating different models.
current_model_name = "Quantized SNN AlexNet (QSNN)" 
csv_results_path = "results/logs/snn_quantized_alexnet_eval.csv"

device = torch.device('cpu')

# Placeholder for average synaptic density for SyOps calculation
# TODO: Research or estimate a more accurate average synaptic density for AlexNet layers for precise SyOps.
# Current value of 1.0 implies SyOps = AC_Ops.
AVG_SYNAPTIC_DENSITY = 1.0 

# Attempt to load energy constants from config.py for projected neuromorphic energy
E_MEM_NEUROMORPHIC_J_DEFAULT = 0.2e-12 # Default if not in config
try:
    from config import config as project_config
    E_SPIKE_NEUROMORPHIC_J = project_config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_spike", 0.1e-12) # J per spike
    E_AC_NEUROMORPHIC_J = project_config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_ac", E_SPIKE_NEUROMORPHIC_J) # J per AC
    E_MEM_NEUROMORPHIC_J = project_config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_mem_access", E_MEM_NEUROMORPHIC_J_DEFAULT) # J per Byte Access
except ImportError:
    print("Warning: config.py not found. Using default neuromorphic energy constants.")
    E_SPIKE_NEUROMORPHIC_J = 0.1e-12 # J
    E_AC_NEUROMORPHIC_J = 0.1e-12 # J
    E_MEM_NEUROMORPHIC_J = E_MEM_NEUROMORPHIC_J_DEFAULT # J

# --- Ensure results directory exists ---
os.makedirs(os.path.dirname(csv_results_path), exist_ok=True)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Evaluate Quantized SNN AlexNet")
parser.add_argument('--model_path', type=str, default='quantized_snn_alexnet_qat.pth', help='Path to the quantized SNN model state_dict')
parser.add_argument('--time_steps', type=int, default=32, help='Number of time steps for SNN evaluation (more steps typically improve accuracy, try 32-128)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
parser.add_argument('--encoding', type=str, default='direct', choices=['direct', 'poisson', 'rate', 'temporal_contrast'], 
                   help='Spike encoding method (poisson/rate often improve accuracy)')
parser.add_argument('--output_method', type=str, default='mean', choices=['mean', 'last', 'max', 'weighted_late'], 
                   help='Method to process SNN outputs across time steps')
parser.add_argument('--beta', type=float, default=0.95, help='Beta parameter for LIF neurons (0.9-0.99, higher values = longer memory)')
args = parser.parse_args()

# Update configuration from args
model_path = args.model_path
time_steps = args.time_steps
batch_size = args.batch_size
encoding_method = args.encoding
output_method = args.output_method
beta = args.beta
# Note: Higher time_steps (32-128) generally improve SNN accuracy but increase inference time
# csv_results_path will use the default for now, but could be an arg

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# --- Model Instantiation and Preparation ---
num_classes = 10
thresholds = None # Assuming no specific thresholds are loaded/set here
time_steps_param = time_steps # Use time_steps from args

print(f"Instantiating base model structure: {current_model_name} with T={time_steps_param}, Beta={beta}")
model = QuantizedSNNAlexNet(num_classes=num_classes, beta=beta, thresholds=thresholds, time_steps=time_steps_param)
model.to(device)

print("Preparing model for QAT to match saved model structure...")
model = prepare_quantized_snn_alexnet_for_qat(model)

print("Converting model to quantized inference structure...")
model.eval()
model = torch.ao.quantization.convert(model, inplace=False) 
print("Model structure prepared and converted.")


# --- Load the Saved State Dictionary ---
print(f"Loading model state_dict from: {model_path}")
try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model state_dict loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {model_path}.")
    exit()
except RuntimeError as e:
    print(f"ERROR: Could not load state_dict: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

model.eval()

# --- Spike Encoding Functions ---
def encode_direct(x, num_steps):
    """Basic encoding: repeat input across time steps"""
    return x.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)

def encode_poisson(x, num_steps):
    """Poisson encoding: convert pixel intensities to spike probabilities"""
    # Ensure x is in [0, 1] range
    x = torch.clamp(x, 0, 1)
    # Generate Poisson spikes
    spike_data = torch.rand(num_steps, *x.shape, device=x.device) < x.unsqueeze(0)
    return spike_data.float()

def encode_rate(x, num_steps):
    """Rate encoding: distribute spikes based on intensity"""
    x = torch.clamp(x, 0, 1)
    # Create time-distributed spikes
    time_indices = torch.arange(num_steps, device=x.device).float() / num_steps
    spike_probs = x.unsqueeze(0) * (1 - time_indices.view(-1, 1, 1, 1, 1))
    spike_data = torch.rand(num_steps, *x.shape, device=x.device) < spike_probs
    return spike_data.float()

def encode_temporal_contrast(x, num_steps):
    """Temporal contrast encoding: emphasize changes over time"""
    # Add small temporal variations to create more realistic spike patterns
    base_pattern = x.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
    temporal_noise = 0.1 * torch.randn_like(base_pattern)
    temporal_pattern = torch.sigmoid(base_pattern + temporal_noise)
    return temporal_pattern

# --- Evaluation Loop ---
print(f"--- Evaluating quantized model on {device} with {encoding_method} encoding and {output_method} output processing ---")

# Select encoding function
encoding_functions = {
    'direct': encode_direct,
    'poisson': encode_poisson, 
    'rate': encode_rate,
    'temporal_contrast': encode_temporal_contrast
}
encode_func = encoding_functions[encoding_method]

def process_outputs(outputs, method='mean'):
    """Process SNN outputs across time dimension"""
    if method == 'mean':
        return outputs.mean(dim=0)
    elif method == 'last':
        return outputs[-1]  # Use only the last time step
    elif method == 'max':
        return outputs.max(dim=0)[0]  # Max across time
    elif method == 'weighted_late':
        # Weight later time steps more heavily
        weights = torch.linspace(0.5, 1.0, outputs.size(0), device=outputs.device)
        weights = weights.view(-1, 1, 1)
        weighted_outputs = outputs * weights
        return weighted_outputs.sum(dim=0) / weights.sum()
    else:
        return outputs.mean(dim=0)

# Perform warmup passes for fair timing comparison (same as evaluate.py)
print(f"Performing warmup inference passes for {current_model_name}...")
dummy_input = torch.randn(8, 3, 32, 32, device=device)  # Batch of 8 CIFAR-10 sized images
with torch.no_grad():
    for _ in range(5):  # 5 warmup passes
        spike_train_warmup = encode_func(dummy_input, time_steps)
        _ = model(spike_train_warmup)

torch.cuda.synchronize() if torch.cuda.is_available() else None  # Ensure GPU operations complete
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    torch.mps.synchronize()  # Ensure MPS operations complete

# Initialize tracking variables
total_correct = 0
total_samples = 0
total_time = 0

# Use NVML total-energy API and CUDA events for energy/time measurement (same as evaluate.py)
if USE_NVML:
    # Sync and record start energy/time
    torch.cuda.synchronize()
    start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # Ensure GPU/MPS operations from previous iterations are complete
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
        
        # Start timing for this batch
        start_time = time.time()
        
        try:
            spike_train = encode_func(images, time_steps)
            outputs = model(spike_train)
            if outputs.dim() == 3 and outputs.size(0) == time_steps:
                outputs_processed = process_outputs(outputs, output_method)
            elif outputs.dim() == 2:
                outputs_processed = outputs
            else:
                print(f"Warning: Unexpected output dimensions: {outputs.shape}. Skipping batch.")
                continue
            _, predicted = torch.max(outputs_processed.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size
            
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(test_loader)}, Correct: {(predicted == labels).sum().item()}/{labels.size(0)}")
        except Exception as e:
            print(f"Error during inference on batch {i+1}: {e}")
            break
        
        # Ensure GPU/MPS operations are complete before stopping timer
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
            
        end_time = time.time()
        
        # Accumulate metrics
        hw_time = end_time - start_time
        total_time += hw_time

# Record end energy/time if using NVML events (same as evaluate.py)
if USE_NVML:
    end_event.record()
    torch.cuda.synchronize()
    end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
    # Elapsed GPU time in milliseconds
    total_time_ms = start_event.elapsed_time(end_event)
    # Energy in Joules (mJ to J)
    energy_j = (end_energy - start_energy) / 1000.0
    # Compute per-sample averages
    avg_energy = energy_j / total_samples if total_samples > 0 else 0.0
    avg_time = (total_time_ms / 1000.0) / total_samples if total_samples > 0 else 0.0
    avg_power = energy_j / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0
    # Calculate average accuracy
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"[NVML] Hardware measured - Energy: {avg_energy:.6e} J, Time: {avg_time:.6f} s, Power: {avg_power:.4f} W")
else:
    # Calculate average metrics when NVML not available
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    avg_energy = 0.0  # Will be overridden with theoretical energy later
    avg_time = total_time / total_samples if total_samples > 0 else 0.0
    avg_power = 0.0  # Will be calculated from theoretical energy later
    print(f"[NO NVML] Using timing measurements - Time: {avg_time:.6f} s per sample")

accuracy = avg_acc * 100.0  # Convert to percentage
avg_inference_time_total = avg_time
avg_inference_time_per_step = avg_inference_time_total / time_steps if time_steps > 0 else 0

print(f"Quantized SNN AlexNet Test Accuracy ({device}, T={time_steps}, Encoding={encoding_method}, Output={output_method}, Beta={beta}): {accuracy:.2f}% ({total_correct}/{total_samples})")
print(f"Energy: {avg_energy:.6e} J")
print(f"Average total inference time per sample: {avg_inference_time_total*1000:.3f} ms")
print(f"Average inference time per step: {avg_inference_time_per_step*1000:.3f} ms/step")
print(f"Power: {avg_power:.4f} W")
print("Evaluation finished.")

# --- Detailed Metric Calculation (after eval loop) ---
print("Calculating detailed metrics...")

dummy_input_for_ops = torch.randn(1, 3, 32, 32).to(device) # Single sample, 3 channels, 32x32

# Initialize metrics to ensure they exist
mac_ops_final = 0
ac_ops_final = 0
total_spike_count_final = 0
syops_final = 0 # SyOps might not be directly calculated by count_ac_operations
ann_mem_access_final = 0
snn_mem_access_final = 0
fair_theoretical_energy_j = 0.0
projected_neuromorphic_energy_j = 0.0 # Initialize
energy_model_type = "QuantizedSNN" # Default for this script

# 1. Operation Counts
try:
    # count_ac_operations from utils.py returns (ac_ops, spike_ops)
    returned_ac_ops, returned_spike_ops = count_ac_operations(model, dummy_input_for_ops, time_steps=time_steps)
    ac_ops_final = returned_ac_ops
    total_spike_count_final = returned_spike_ops
    # Calculate SyOps
    syops_final = ac_ops_final * AVG_SYNAPTIC_DENSITY
    print(f"Calculated AC Ops: {ac_ops_final}, Spike Count: {total_spike_count_final}, SyOps: {syops_final}")
except Exception as e:
    print(f"Warning: Could not calculate AC operations or spike counts. Error Type: {type(e).__name__}, Message: {e}")
    # ac_ops_final, total_spike_count_final remain 0

# MAC ops for Quantized SNN: typically considered 0 unless there's a non-spiking initial layer.
# The current QuantizedSNNAlexNet design has spiking neurons after every conv/linear layer that processes features.
# The very first conv might be FP32 before quantization in some generic SNNs, but here it's quantized.
# For simplicity and consistency with how SNNs are often evaluated, we'll assume MACs are negligible for the core SNN part.
# If there was a dedicated non-spiking pre-processing block whose MACs are significant, one would count them.
mac_ops_final = 0 # Primarily AC-based

# 2. Memory Accesses
# Ensure these are initialized before the (now bypassed) try-except block
ann_mem_access_final = 0.0 # SNNs don't have ANN memory access
snn_mem_access_final = 0.0 # Initialize
try:
    print(f"[Eval QSNN] Counting Memory Accesses (is_snn=True, time_steps={time_steps})...")
    # Call the centralized function from utils.py
    # Pass the dummy tensor and specify it's an SNN with its time_steps
    snn_mem_access_final = count_memory_access(model, dummy_input_for_ops, is_snn=True, time_steps_from_caller=time_steps)
    print(f"[Eval QSNN] SNN Memory Accesses counted: {snn_mem_access_final}")
except Exception as e_mem:
    print(f"[Eval QSNN Error] Failed to count Memory Accesses: {e_mem}")
    snn_mem_access_final = np.nan # Indicate failure

# 3. Fair Theoretical Energy Calculation (Using Centralized Function)
fair_theoretical_energy_j = np.nan
calc_source = "Not Calculated"
temp_res_dict_snn = {
    'macs': mac_ops_final, # Should be 0
    'ac_ops': ac_ops_final,
    'ann_mem_accesses': ann_mem_access_final, # Should be 0.0
    'snn_mem_accesses': snn_mem_access_final,
    'precision': 'INT8' # Explicitly state precision for QSNN
}
try:
    print(f"[Eval QSNN] Calculating Fair Theoretical Energy...")
    fair_theoretical_energy_j, calc_source = calculate_fair_theoretical_energy_from_dict(temp_res_dict_snn, "snn_quantized_alexnet")
    if pd.notna(fair_theoretical_energy_j):
        print(f"[Eval QSNN] Fair Theoretical Energy calculated: {fair_theoretical_energy_j:.4g} J (Source: {calc_source})")
    else:
        print(f"[Eval QSNN Warning] Fair Theoretical Energy calculation failed. Reason: {calc_source}")
except Exception as e_fair_eng:
    print(f"[Eval QSNN Error] Exception during Fair Theoretical Energy calculation: {e_fair_eng}")
    fair_theoretical_energy_j = np.nan
    calc_source = "Calculation Exception"

# 4. Projected Neuromorphic Energy (still calculated locally, depends on AC ops and SNN mem)
projected_neuromorphic_energy_j = np.nan
try:
    if pd.notna(ac_ops_final) and pd.notna(snn_mem_access_final):
        # Use constants loaded from config or defaults earlier
        neuromorphic_ac_energy = ac_ops_final * E_AC_NEUROMORPHIC_J
        neuromorphic_mem_energy = snn_mem_access_final * E_MEM_NEUROMORPHIC_J # Assumes E_MEM_NEUROMORPHIC_J is per-byte
        projected_neuromorphic_energy_j = neuromorphic_ac_energy + neuromorphic_mem_energy
        print(f"[Eval QSNN] Projected Neuromorphic Energy calculated: {projected_neuromorphic_energy_j:.4g} J")
    else:
        print("[Eval QSNN Warning] Cannot calculate projected neuromorphic energy due to missing AC ops or SNN memory access data.")
except Exception as e_neuro_eng:
    print(f"[Eval QSNN Error] Exception during Projected Neuromorphic Energy calculation: {e_neuro_eng}")
    projected_neuromorphic_energy_j = np.nan

# --- Calculate Theoretical Power if NVML was not available ---
if not USE_NVML and pd.notna(fair_theoretical_energy_j) and avg_time > 0:
    # Use theoretical energy to calculate power: P = E / t
    avg_power = fair_theoretical_energy_j / avg_time
    avg_energy = fair_theoretical_energy_j  # Use theoretical energy instead of 0.0
    print(f"[THEORETICAL POWER] Calculated from fair theoretical energy:")
    print(f"  Energy: {avg_energy:.6e} J")
    print(f"  Time: {avg_time:.6f} s") 
    print(f"  Power: {avg_power:.4f} W")
elif not USE_NVML:
    print(f"[WARN] Cannot calculate theoretical power - fair energy calculation failed or timing is zero")
    print(f"  fair_theoretical_energy_j: {fair_theoretical_energy_j}")
    print(f"  avg_time: {avg_time}")

print("Detailed metric calculation finished.")

# --- Storing Results ---
# ... (rest of the script for CSV saving)
# Ensure these calculated values are included in the results_data dictionary for CSV
# result_data needs: 'mac_operations', 'ac_operations', 'spike_count', 'syops', 
#                    'ann_memory_access', 'snn_memory_access', 'fair_theoretical_energy_j',
#                    'energy_model_type' 
# ...

# --- Storing Results ---
results_data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": current_model_name,
    "model_path": model_path,
    "batch_size": batch_size,
    "time_steps": time_steps,
    "accuracy_percent": accuracy,
    "total_correct": total_correct,
    "total_samples": total_samples,
    "energy": avg_energy,  # Hardware-measured energy (J) - same as evaluate.py
    "inference_time": avg_inference_time_total,  # Inference time per sample (s) - same as evaluate.py
    "power": avg_power,  # Power consumption (W) - same as evaluate.py
    "inference_time_per_step_ms": avg_inference_time_per_step * 1000 if time_steps > 0 else 0, # SNN-specific
    # Detailed metrics (same structure as evaluate.py)
    "macs": mac_ops_final,
    "ac_operations": ac_ops_final,
    "spike_count": total_spike_count_final,
    "syops": syops_final,
    "ann_mem_accesses": ann_mem_access_final,
    "snn_mem_accesses": snn_mem_access_final if pd.notna(snn_mem_access_final) else 0.0,
    "fair_theoretical_energy_j": fair_theoretical_energy_j if pd.notna(fair_theoretical_energy_j) else np.nan,
    "projected_neuromorphic_energy": projected_neuromorphic_energy_j if pd.notna(projected_neuromorphic_energy_j) else np.nan,
    "energy_calculation_source": calc_source,
    "energy_model_type": energy_model_type # e.g., "SNN", "ANN", "QuantizedSNN"
}

df_new = pd.DataFrame([results_data])
if os.path.exists(csv_results_path):
    df_existing = pd.read_csv(csv_results_path)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_combined = df_new

# Ensure all columns from results_data are present, adding missing ones with NaN
for col in results_data.keys():
    if col not in df_combined.columns:
        df_combined[col] = np.nan # Add column if it's missing from existing file
        # Ensure the new data is placed in the last row for the new column
        if df_combined.index.size > 0: # Check if DataFrame is not empty
             df_combined.iloc[-1, df_combined.columns.get_loc(col)] = results_data[col]

df_combined.to_csv(csv_results_path, index=False)
print(f"Appended results to {csv_results_path}")

# Added main guard for direct execution
if __name__ == '__main__':
    # This script is typically run directly, so the argument parsing and main logic execute.
    # No explicit main() function needed unless preferred for structure.
    pass
