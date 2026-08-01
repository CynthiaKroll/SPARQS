import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import random
import matplotlib as mpl
from matplotlib.lines import Line2D
from cycler import cycler
from collections import defaultdict
import seaborn as sns
import json

# Import the centralized calculation function and constants from utils
from utils import calculate_fair_theoretical_energy_from_dict
# The constants (E_MAC_32BIT_J, etc.) are now defined in utils.py and used by the imported function.

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Set a fixed random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
results_dir = "results/logs"
plots_dir = "results/plots"
hardware_profile = "CPU"  # Defaulting to CPU as primary context

# Define specific CSV files for certain models, overriding the default "<model_key>_eval.csv"
MODEL_FILE_OVERRIDES = {
    "branchy_quantized_snn_mlp_th0.6_0.8": "branchy_quantized_snn_mlp_th0.6_0.8_dynamic_rl_eval_eval.csv",
    "branchy_quantized_snn_mlp_th0.2_0.4": "branchy_quantized_snn_mlp_th0.2_0.4_dynamic_rl_eval_eval.csv"
    # Add other overrides here if needed
}

# Create a consistent color scheme for models
def get_model_colors(models):
    """Dynamically assign distinct colors for each model."""
    palette = sns.color_palette("colorblind", n_colors=len(models))
    return {m: palette[i] for i, m in enumerate(models)}

# Ensure plots directory exists and clear old plots
os.makedirs(plots_dir, exist_ok=True)
for fname in os.listdir(plots_dir):
    if fname.endswith('.png'):
        try:
            os.remove(os.path.join(plots_dir, fname))
        except Exception:
            pass

# Print diagnostic information
print(f"Looking for CSV files in: {os.path.abspath(results_dir)}")
print(f"Using hardware profile: {hardware_profile}")

# Only include these models for plotting - 5-layer MLP variants
INCLUDE_MODELS = [
    "standard_mlp",                          # Standard MLP
    "snn_mlp",                              # SNN MLP
    "branchy_mlp",                          # Branchy MLP
    "quantized_snn_mlp",                    # Quantized SNN MLP
    "branchy_quantized_snn_mlp",            # Branchy Quantized SNN MLP (default)
    "branchy_quantized_snn_mlp_th0.2_0.4",  # Branchy Quantized SNN MLP (0.2, 0.4 thresholds)
    "branchy_quantized_snn_mlp_th0.6_0.7",  # Branchy Quantized SNN MLP (0.6, 0.7 thresholds)
]

# Ensure correct display names for all models
MODEL_DISPLAY_NAMES = {
    "standard_mlp": "Standard MLP",
    "snn_mlp": "SNN MLP",
    "branchy_mlp": "Branchy MLP",
    "quantized_snn_mlp": "Quantized SNN MLP",
    "branchy_quantized_snn_mlp": "Branchy Quantized SNN MLP",
    "branchy_quantized_snn_mlp_th0.2_0.4": "Branchy Quantized SNN MLP (th=0.2,0.4)",
    "branchy_quantized_snn_mlp_th0.6_0.7": "Branchy Quantized SNN MLP (th=0.6,0.7)",
}

print(f"Including only models: {INCLUDE_MODELS}")
print(f"Model display names: {MODEL_DISPLAY_NAMES}")

# Load results from CSV files
all_results = {}
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    eval_files = [f for f in files if f.endswith('_eval.csv')]
    
    # Collect all potential CSV files for models we might want to plot
    candidate_files = {}
    for model_key in INCLUDE_MODELS:
        if model_key in MODEL_FILE_OVERRIDES:
            candidate_files[model_key] = MODEL_FILE_OVERRIDES[model_key]
        else:
            # Default pattern
            default_file_pattern = f"{model_key}_eval.csv"
            if default_file_pattern in files:
                candidate_files[model_key] = default_file_pattern

    print(f"Found {len(files)} files in logs dir. Processing specified models.")

    # Loop through the models we want to include
    for model_name in INCLUDE_MODELS:
        file_to_load = None
        if model_name in MODEL_FILE_OVERRIDES:
            file_to_load = MODEL_FILE_OVERRIDES[model_name]
            print(f"DEBUG: For model '{model_name}', using override file: '{file_to_load}'")
        else:
            # Fallback to original logic: find a file that starts with model_name and ends with _eval.csv
            potential_files = [f for f in files if f.startswith(model_name) and f.endswith('_eval.csv')]
            if potential_files:
                potential_files.sort(key=len)  # Prefer shorter names like "model_eval.csv"
                file_to_load = potential_files[0]
                
                simple_default_file = f"{model_name}_eval.csv"
                if simple_default_file in files:
                    file_to_load = simple_default_file
                elif potential_files:
                    file_to_load = potential_files[0]

        if not file_to_load or not os.path.exists(os.path.join(results_dir, file_to_load)):
            print(f"DEBUG: No suitable CSV file found or specified for model '{model_name}'. Skipping.")
            continue
        
        file_path = os.path.join(results_dir, file_to_load)
        print(f"DEBUG: Attempting to load {model_name} from {file_path}")

        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                result_dict = df.iloc[-1].to_dict()

                # Standardize essential fields to NaN if missing
                essential_fields = [
                    'accuracy', 'accuracy_percent', 'inference_time', 'inference_time_per_sample', 
                    'eval_duration_seconds', 'total_samples', 'ac_ops', 'ac_operations', 'syops', 
                    'ann_mem_accesses', 'snn_mem_accesses', 'ann_memory_access', 'snn_memory_access',
                    'macs', 'operation_count', 'energy', 'power', 'fair_theoretical_energy', 
                    'fair_theoretical_energy_j', 'projected_neuromorphic_energy', 'device', 'precision',
                    'time_steps', 'num_steps'
                ]
                for field in essential_fields:
                    if field not in result_dict:
                        result_dict[field] = np.nan
                
                # Data Cleaning and Prioritization
                
                # Accuracy (convert % to fraction 0-1)
                acc_raw = np.nan
                if 'accuracy_percent' in result_dict and pd.notna(result_dict['accuracy_percent']):
                    acc_raw = result_dict['accuracy_percent']
                elif 'accuracy' in result_dict and pd.notna(result_dict['accuracy']):
                    acc_raw = result_dict['accuracy']
                
                result_dict['accuracy'] = np.nan  # Default to NaN for the final processed value
                if pd.notna(acc_raw):
                    try:
                        acc_f = float(acc_raw)
                        # Case 1: Value is already a fraction (0 to 1)
                        if acc_f >= 0.0 and acc_f <= 1.0:
                            result_dict['accuracy'] = acc_f
                        # Case 2: Value is a percentage (1 to 100)
                        elif acc_f > 1.0 and acc_f <= 100.0:
                             result_dict['accuracy'] = acc_f / 100.0
                        # Case 3: Unusual values - flag as problematic
                        else:
                             print(f"  WARNING ({model_name}) Unusual accuracy value: {acc_f}. Setting to NaN.")
                             result_dict['accuracy'] = np.nan
                    except (ValueError, TypeError) as e:
                         print(f"  WARNING ({model_name}) Cannot convert accuracy '{acc_raw}' to float: {e}")
                         result_dict['accuracy'] = np.nan
                
                # Skip models with very low accuracy (< 30%) as they're likely broken
                if pd.notna(result_dict['accuracy']) and result_dict['accuracy'] < 0.30:
                    print(f"  SKIPPING ({model_name}) due to very low accuracy: {result_dict['accuracy']:.3f}")
                    continue
                
                # Inference time (prioritize specific fields)
                if 'inference_time_per_sample' in result_dict and pd.notna(result_dict['inference_time_per_sample']):
                    result_dict['inference_time'] = result_dict['inference_time_per_sample']
                elif 'inference_time' in result_dict and pd.notna(result_dict['inference_time']):
                    pass  # Use existing value
                else:
                    result_dict['inference_time'] = np.nan
                
                # Operations (prioritize specific fields for different model types)
                if 'ac_ops' in result_dict and pd.notna(result_dict['ac_ops']):
                    pass  # Use existing value
                elif 'ac_operations' in result_dict and pd.notna(result_dict['ac_operations']):
                    result_dict['ac_ops'] = result_dict['ac_operations']
                else:
                    result_dict['ac_ops'] = np.nan
                
                if 'macs' in result_dict and pd.notna(result_dict['macs']):
                    pass  # Use existing value
                elif 'operation_count' in result_dict and pd.notna(result_dict['operation_count']):
                    result_dict['macs'] = result_dict['operation_count']
                else:
                    result_dict['macs'] = np.nan
                
                # Memory accesses (prioritize specific fields)
                if 'ann_mem_accesses' in result_dict and pd.notna(result_dict['ann_mem_accesses']):
                    pass  # Use existing value
                elif 'ann_memory_access' in result_dict and pd.notna(result_dict['ann_memory_access']):
                    result_dict['ann_mem_accesses'] = result_dict['ann_memory_access']
                else:
                    result_dict['ann_mem_accesses'] = np.nan
                
                if 'snn_mem_accesses' in result_dict and pd.notna(result_dict['snn_mem_accesses']):
                    pass  # Use existing value
                elif 'snn_memory_access' in result_dict and pd.notna(result_dict['snn_memory_access']):
                    result_dict['snn_mem_accesses'] = result_dict['snn_memory_access']
                else:
                    result_dict['snn_mem_accesses'] = np.nan
                
                # Fair theoretical energy calculation (using centralized function)
                fair_energy_j, calc_source = calculate_fair_theoretical_energy_from_dict(result_dict, model_name)
                result_dict['fair_theoretical_energy_j'] = fair_energy_j
                result_dict['energy_calc_source'] = calc_source
                
                # Store final result
                all_results[model_name] = result_dict
                print(f"Successfully loaded {model_name} with accuracy {result_dict['accuracy']:.4f}")
            else:
                print(f"Empty CSV file for {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

print(f"\nLoaded {len(all_results)} models: {list(all_results.keys())}")

# Filter to only include models we want to plot and have data for
filtered_results = {k: v for k, v in all_results.items() if k in INCLUDE_MODELS}
models_to_plot = list(filtered_results.keys())

if not models_to_plot:
    print("No valid models found for plotting. Exiting.")
    exit()

print(f"Plotting {len(models_to_plot)} models: {models_to_plot}")

# Create color map and display names for filtered models
model_colors = get_model_colors(models_to_plot)
display_names = {k: MODEL_DISPLAY_NAMES.get(k, k) for k in models_to_plot}

print(f"Model colors: {model_colors}")
print(f"Display names: {display_names}")

def plot_accuracy_vs_power(results, color_map, display_names, filename="accuracy_vs_power.png"):
    """Plot accuracy vs power consumption."""
    print("\n--- Plotting Accuracy vs Power ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    accuracies = []
    powers = []
    model_names = []
    colors = []
    
    for model_name, data in results.items():
        accuracy = data.get('accuracy', np.nan)
        power = data.get('power', np.nan)
        
        print(f"  {model_name}: Accuracy={accuracy:.4f}, Power={power:.6e} W")
        
        if pd.notna(accuracy) and pd.notna(power) and power > 0:
            accuracies.append(accuracy)
            powers.append(power)
            model_names.append(model_name)
            colors.append(color_map[model_name])
    
    if not accuracies:
        print("No valid accuracy/power data found. Skipping accuracy vs power plot.")
        return
    
    # Create scatter plot
    scatter = ax.scatter(powers, accuracies, c=colors, s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for i, (power, accuracy, model_name) in enumerate(zip(powers, accuracies, model_names)):
        ax.annotate(display_names[model_name], 
                   (power, accuracy), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, ha='left')
    
    ax.set_xlabel('Power Consumption (W)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Accuracy vs Power Consumption - 5-Layer MLP Models', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs power plot to {filename}")

def plot_inference_time(results, color_map, display_names, filename="inference_time.png"):
    """Plot inference time comparison."""
    print("\n--- Plotting Inference Time ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = []
    inference_times = []
    colors = []
    
    for model_name, data in results.items():
        inf_time = data.get('inference_time', np.nan)
        
        print(f"  {model_name}: Inference time={inf_time:.6e} s")
        
        if pd.notna(inf_time) and inf_time > 0:
            model_names.append(display_names[model_name])
            inference_times.append(inf_time)
            colors.append(color_map[model_name])
    
    if not inference_times:
        print("No valid inference time data found. Skipping inference time plot.")
        return
    
    # Create bar plot
    bars = ax.bar(range(len(model_names)), inference_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Inference Time (s)', fontsize=14)
    ax.set_title('Inference Time Comparison - 5-Layer MLP Models', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, time in zip(bars, inference_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2e}s',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved inference time plot to {filename}")

def plot_energy_consumption(results, color_map, display_names, filename="energy_consumption.png"):
    """Plot energy consumption comparison."""
    print("\n--- Plotting Energy Consumption ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = []
    energies = []
    colors = []
    
    for model_name, data in results.items():
        # Prioritize fair theoretical energy, fallback to regular energy
        energy = data.get('fair_theoretical_energy_j', data.get('energy', np.nan))
        
        print(f"  {model_name}: Energy={energy:.6e} J")
        
        if pd.notna(energy) and energy > 0:
            model_names.append(display_names[model_name])
            energies.append(energy)
            colors.append(color_map[model_name])
    
    if not energies:
        print("No valid energy data found. Skipping energy consumption plot.")
        return
    
    # Create bar plot
    bars = ax.bar(range(len(model_names)), energies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Energy Consumption (J)', fontsize=14)
    ax.set_title('Energy Consumption Comparison - 5-Layer MLP Models', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.2e}J',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved energy consumption plot to {filename}")

def plot_accuracy_vs_time(results, color_map, display_names, filename="accuracy_vs_time.png"):
    """Plot accuracy vs inference time."""
    print("\n--- Plotting Accuracy vs Time ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    accuracies = []
    times = []
    model_names = []
    colors = []
    
    for model_name, data in results.items():
        accuracy = data.get('accuracy', np.nan)
        inf_time = data.get('inference_time', np.nan)
        
        print(f"  {model_name}: Accuracy={accuracy:.4f}, Time={inf_time:.6e} s")
        
        if pd.notna(accuracy) and pd.notna(inf_time) and inf_time > 0:
            accuracies.append(accuracy)
            times.append(inf_time)
            model_names.append(model_name)
            colors.append(color_map[model_name])
    
    if not accuracies:
        print("No valid accuracy/time data found. Skipping accuracy vs time plot.")
        return
    
    # Create scatter plot
    scatter = ax.scatter(times, accuracies, c=colors, s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for i, (time, accuracy, model_name) in enumerate(zip(times, accuracies, model_names)):
        ax.annotate(display_names[model_name], 
                   (time, accuracy), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, ha='left')
    
    ax.set_xlabel('Inference Time (s)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Accuracy vs Inference Time - 5-Layer MLP Models', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs time plot to {filename}")

def plot_operation_counts(results, color_map, display_names, filename="operation_counts.png"):
    """Plot operation counts comparison."""
    print("\n--- Plotting Operation Counts ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = []
    mac_ops = []
    ac_ops = []
    colors = []
    
    for model_name, data in results.items():
        macs = data.get('macs', np.nan)
        acs = data.get('ac_ops', np.nan)
        
        print(f"  {model_name}: MACs={macs}, AC ops={acs}")
        
        model_names.append(display_names[model_name])
        mac_ops.append(macs if pd.notna(macs) else 0)
        ac_ops.append(acs if pd.notna(acs) else 0)
        colors.append(color_map[model_name])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    # Create grouped bar plot
    bars1 = ax.bar(x - width/2, mac_ops, width, label='MAC Operations', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, ac_ops, width, label='AC Operations', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Operation Count', fontsize=14)
    ax.set_title('Operation Counts Comparison - 5-Layer MLP Models', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved operation counts plot to {filename}")

def plot_memory_accesses(results, color_map, display_names, filename="memory_accesses.png"):
    """Plot memory access comparison."""
    print("\n--- Plotting Memory Accesses ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = []
    ann_mem = []
    snn_mem = []
    colors = []
    
    for model_name, data in results.items():
        ann_mem_val = data.get('ann_mem_accesses', np.nan)
        snn_mem_val = data.get('snn_mem_accesses', np.nan)
        
        print(f"  {model_name}: ANN mem={ann_mem_val}, SNN mem={snn_mem_val}")
        
        model_names.append(display_names[model_name])
        ann_mem.append(ann_mem_val if pd.notna(ann_mem_val) else 0)
        snn_mem.append(snn_mem_val if pd.notna(snn_mem_val) else 0)
        colors.append(color_map[model_name])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    # Create grouped bar plot
    bars1 = ax.bar(x - width/2, ann_mem, width, label='ANN Memory Accesses', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, snn_mem, width, label='SNN Memory Accesses', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Memory Accesses (Bytes)', fontsize=14)
    ax.set_title('Memory Access Comparison - 5-Layer MLP Models', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved memory accesses plot to {filename}")

def plot_power_consumption(results, color_map, display_names, filename="power_consumption.png"):
    """Plot power consumption comparison."""
    print("\n--- Plotting Power Consumption ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = []
    powers = []
    colors = []
    
    for model_name, data in results.items():
        power = data.get('power', np.nan)
        
        print(f"  {model_name}: Power={power:.6e} W")
        
        if pd.notna(power) and power > 0:
            model_names.append(display_names[model_name])
            powers.append(power)
            colors.append(color_map[model_name])
    
    if not powers:
        print("No valid power data found. Skipping power consumption plot.")
        return
    
    # Create bar plot
    bars = ax.bar(range(len(model_names)), powers, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Power Consumption (W)', fontsize=14)
    ax.set_title('Power Consumption Comparison - 5-Layer MLP Models', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{power:.2e}W',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved power consumption plot to {filename}")

def plot_projected_neuromorphic_energy(results, color_map, display_names, filename="projected_neuromorphic_energy.png"):
    """Plot projected neuromorphic energy for SNN models."""
    print("\n--- Plotting Projected Neuromorphic Energy ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = []
    neuromorphic_energies = []
    colors = []
    
    for model_name, data in results.items():
        neuro_energy = data.get('projected_neuromorphic_energy', np.nan)
        
        print(f"  {model_name}: Neuromorphic energy={neuro_energy:.6e} J")
        
        if pd.notna(neuro_energy) and neuro_energy > 0:
            model_names.append(display_names[model_name])
            neuromorphic_energies.append(neuro_energy)
            colors.append(color_map[model_name])
    
    if not neuromorphic_energies:
        print("No valid neuromorphic energy data found. Skipping neuromorphic energy plot.")
        return
    
    # Create bar plot
    bars = ax.bar(range(len(model_names)), neuromorphic_energies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Projected Neuromorphic Energy (J)', fontsize=14)
    ax.set_title('Projected Neuromorphic Energy - SNN Models', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, energy in zip(bars, neuromorphic_energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.2e}J',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved projected neuromorphic energy plot to {filename}")

def plot_fair_theoretical_energy(results, color_map, display_names, filename="fair_theoretical_energy.png"):
    """Plot fair theoretical energy comparison."""
    print("\n--- Plotting Fair Theoretical Energy ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = []
    energies = []
    colors = []
    
    for model_name, data in results.items():
        energy = data.get('fair_theoretical_energy_j', np.nan)
        
        print(f"  {model_name}: Fair theoretical energy={energy:.6e} J")
        
        if pd.notna(energy) and energy > 0:
            model_names.append(display_names[model_name])
            energies.append(energy)
            colors.append(color_map[model_name])
    
    if not energies:
        print("No valid fair theoretical energy data found. Skipping fair theoretical energy plot.")
        return
    
    # Create bar plot
    bars = ax.bar(range(len(model_names)), energies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Fair Theoretical Energy (J)', fontsize=14)
    ax.set_title('Fair Theoretical Energy Comparison - 5-Layer MLP Models', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.2e}J',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved fair theoretical energy plot to {filename}")

# Generate all plots
print("\n=== Generating Visualization Plots ===")

try:
    plot_accuracy_vs_power(filtered_results, model_colors, display_names)
    plot_accuracy_vs_time(filtered_results, model_colors, display_names)
    plot_energy_consumption(filtered_results, model_colors, display_names)
    plot_inference_time(filtered_results, model_colors, display_names)
    plot_power_consumption(filtered_results, model_colors, display_names)
    plot_operation_counts(filtered_results, model_colors, display_names)
    plot_memory_accesses(filtered_results, model_colors, display_names)
    plot_projected_neuromorphic_energy(filtered_results, model_colors, display_names)
    plot_fair_theoretical_energy(filtered_results, model_colors, display_names)
    
    print(f"\n=== All plots saved to {plots_dir} ===")
except Exception as e:
    print(f"Error generating plots: {e}")
    import traceback
    traceback.print_exc()