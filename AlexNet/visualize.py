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
# Use consistent LaTeX-compatible fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts are LaTeX-compatible
plt.rcParams['text.usetex'] = False  # Keep False for compatibility, but use LaTeX-style fonts
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20
# Ensure all text is black for publication
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
# Set DPI for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Set a fixed random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
results_dir = "results/logs"
plots_dir = "results/plots"
hardware_profile = "CPU"  # CHANGED: Defaulting to CPU as primary context

# --- ADD THIS DICTIONARY ---
# Define specific CSV files for certain models, overriding the default "<model_key>_eval.csv"
MODEL_FILE_OVERRIDES = {
    "branchy_quantized_snn_th0.6_0.7": "branchy_quantized_snn_th0.6_0.7_dynamic_rl_eval_eval.csv",
    "branchy_quantized_snn_th0.2_0.4": "branchy_quantized_snn_th0.2_0.4_dynamic_rl_eval_eval.csv"
    # Add other overrides here if needed, e.g.:
    # "snn_quantized_alexnet": "snn_quantized_alexnet_custom_run_eval.csv"
}
# --- END OF ADDITION ---

# Create a consistent color scheme for models
def get_model_colors(models):
    """Dynamically assign distinct colors for each model."""
    palette = sns.color_palette("colorblind", n_colors=len(models))
    return {m: palette[i] for i, m in enumerate(models)}

# Define model display names for cleaner plotting
# Note: Model display names are now defined in MODEL_DISPLAY_NAMES dictionary below

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

# Only include these models for plotting - now with both QDSNN threshold variants
INCLUDE_MODELS = [
    "standard_alexnet",           # AlexNet
    "dynamic_alexnet",           # Dynamic AlexNet
    "snn_standard_alexnet",      # SNN AlexNet
    "snn_quantized_alexnet",     # Quantized SNN AlexNet (QSNN, static SNN, quantized)
    "branchy_quantized_snn_th0.6_0.7",  # QDSNN with conservative thresholds
    "branchy_quantized_snn_th0.2_0.4"   # QDSNN with aggressive thresholds
]

# Ensure correct display names for all models including both QDSNN variants
MODEL_DISPLAY_NAMES = {
    "standard_alexnet": "AlexNet",
    "dynamic_alexnet": "Dynamic AlexNet",
    "snn_standard_alexnet": "SNN AlexNet",
    "snn_quantized_alexnet": "QSNN",
    "branchy_quantized_snn_th0.6_0.7": "QDSNN (T=0.6/0.7)",
    "branchy_quantized_snn_th0.2_0.4": "QDSNN (T=0.2/0.4)",
}

print(f"Including only models: {INCLUDE_MODELS}")
print(f"Model display names: {MODEL_DISPLAY_NAMES}")

# Load results from CSV files
all_results = {}
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    eval_files = [f for f in files if f.endswith('_eval.csv')] # Keep this for discovery if needed for non-overridden models
    
    # Collect all potential CSV files for models we might want to plot
    # This includes override files and default pattern matches.
    candidate_files = {}
    for model_key in INCLUDE_MODELS:
        if model_key in MODEL_FILE_OVERRIDES:
            candidate_files[model_key] = MODEL_FILE_OVERRIDES[model_key]
        else:
            # Default pattern
            default_file_pattern = f"{model_key}_eval.csv"
            # Check if the default file exists, otherwise we might have files like _eval_eval.csv etc.
            # For simplicity, we'll stick to the original logic for non-overridden files for now,
            # assuming they follow the _eval.csv pattern or are found by the general loop.
            # A more robust way would be to explicitly list all files or have a priority system.
            if default_file_pattern in files: # Check if the simple pattern exists
                 candidate_files[model_key] = default_file_pattern
            # If not, the general loop below might pick up other variants for non-overridden models.

    print(f"Found {len(files)} files in logs dir. Processing specified models.") # Adjusted print

    # Loop through the models we want to include
    for model_name in INCLUDE_MODELS:
        file_to_load = None
        if model_name in MODEL_FILE_OVERRIDES:
            file_to_load = MODEL_FILE_OVERRIDES[model_name]
            print(f"DEBUG: For model '{model_name}', using override file: '{file_to_load}'")
        else:
            # Fallback to original logic: find a file that starts with model_name and ends with _eval.csv
            # This part of the logic might need refinement if multiple files match (e.g., xxx_eval.csv, xxx_eval_eval.csv)
            # For now, we'll try the most basic match first.
            potential_files = [f for f in files if f.startswith(model_name) and f.endswith('_eval.csv')]
            if potential_files:
                # If multiple files match (e.g., model_eval.csv, model_extra_eval.csv), prioritize the simplest one
                # or define a more specific rule. For now, take the first one found by a simple sort.
                potential_files.sort(key=len) # Prefer shorter names like "model_eval.csv"
                file_to_load = potential_files[0] 
                # Check if a more specific "eval_eval.csv" exists for this model if it's not overridden
                # This part is tricky; explicit overrides are better.
                # Example: if model_name is "snn_quantized_alexnet", and "snn_quantized_alexnet_eval_eval.csv" exists,
                # it might be preferred over "snn_quantized_alexnet_eval.csv" if the latter is older.
                # The current override system handles "branchy_quantized_snn" correctly.
                
                # A simpler fallback for non-overridden models:
                simple_default_file = f"{model_name}_eval.csv"
                if simple_default_file in files:
                    file_to_load = simple_default_file
                elif potential_files: # If the simple default isn't there, take the shortest matching one
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

                # --- Standardize essential fields to NaN if missing ---
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
                
                # --- Data Cleaning and Prioritization (incorporating previous logic) ---
                
                # Accuracy (convert % to fraction 0-1)
                # --- REVISED LOGIC: Prioritize accuracy_percent if valid, then accuracy ---
                acc_raw = np.nan
                if 'accuracy_percent' in result_dict and pd.notna(result_dict['accuracy_percent']):
                    acc_raw = result_dict['accuracy_percent']
                    print(f"  DEBUG ({model_name}) Acc Source: Using 'accuracy_percent': {acc_raw}") # DEBUG
                elif 'accuracy' in result_dict and pd.notna(result_dict['accuracy']):
                    acc_raw = result_dict['accuracy']
                    print(f"  DEBUG ({model_name}) Acc Source: Using 'accuracy': {acc_raw}") # DEBUG
                else:
                     print(f"  DEBUG ({model_name}) Acc Source: No valid 'accuracy' or 'accuracy_percent' found.") # DEBUG
                
                result_dict['accuracy'] = np.nan # Default to NaN for the final processed value
                print(f"  DEBUG ({model_name}) Acc Raw Value Selected: {acc_raw} (Type: {type(acc_raw)})") # DEBUG
                if pd.notna(acc_raw):
                    try:
                        acc_f = float(acc_raw)
                        print(f"  DEBUG ({model_name}) Acc Float Value: {acc_f}") # DEBUG
                        # Case 1: Value is already a fraction (0 to 1)
                        if acc_f >= 0.0 and acc_f <= 1.0:
                            result_dict['accuracy'] = acc_f
                            print(f"  DEBUG ({model_name}) Acc Case 1: Fraction {acc_f}") # DEBUG
                        # Case 2: Value is a percentage (1 to 100)
                        elif acc_f > 1.0 and acc_f <= 100.0:
                             result_dict['accuracy'] = acc_f / 100.0
                             print(f"  DEBUG ({model_name}) Acc Case 2: Percentage {acc_f}") # DEBUG
                        # Case 3: Value might be percentage * 100 (e.g., 6495.0)
                        elif acc_f > 100.0:
                            print(f"  DEBUG ({model_name}) Acc Case 3 Check: Value > 100 ({acc_f})") # DEBUG
                            potential_pcnt = acc_f / 100.0
                            is_likely_scaled_int = abs(acc_f - round(acc_f)) < 1e-9 
                            print(f"    -> Potential %: {potential_pcnt}, Is Int-like: {is_likely_scaled_int}") # DEBUG
                            if potential_pcnt >= 0 and potential_pcnt <= 100.0 and is_likely_scaled_int:
                                result_dict['accuracy'] = potential_pcnt / 100.0 # Convert to fraction
                                print(f"    -> SUCCESS: Interpreted {acc_raw} as {potential_pcnt}%, Stored Fraction: {result_dict['accuracy']}") # DEBUG
                            else:
                                print(f"    -> FAILED: Conditions not met for Case 3.") # DEBUG
                        else: # Handles cases like exactly 0.0 if not caught by >=0 above
                            if acc_f == 0.0:
                                result_dict['accuracy'] = 0.0
                                print(f"  DEBUG ({model_name}) Acc Case 0: Zero value.") # DEBUG
                            else:
                                print(f"  DEBUG ({model_name}) Acc Case ?: Unhandled value {acc_f}") # DEBUG

                    except ValueError as e_acc:
                        print(f"  DEBUG ({model_name}) Acc Error: Could not convert {acc_raw} to float: {e_acc}") # DEBUG
                        pass # remains NaN
                else:
                    print(f"  DEBUG ({model_name}) Acc Raw Value is NaN/None.") # DEBUG
                
                # DEBUG: Print key loaded values before processing
                print(f"  DEBUG ({model_name}): Initial raw values - Acc: {result_dict.get('accuracy', 'N/A')}, Acc%: {result_dict.get('accuracy_percent', 'N/A')}, Time: {result_dict.get('inference_time', 'N/A')}, FairEnergyJ: {result_dict.get('fair_theoretical_energy_j', 'N/A')}, Energy: {result_dict.get('energy', 'N/A')}, AC_Ops: {result_dict.get('ac_ops', 'N/A')}, MACs: {result_dict.get('macs', 'N/A')}")

                # Inference Time (prioritize specific columns, fallback to calculation)
                inf_time_raw = result_dict.get('inference_time', np.nan)
                if model_name == "snn_quantized_alexnet" and pd.notna(result_dict.get('inference_time_per_sample')):
                     try: result_dict['inference_time'] = float(result_dict['inference_time_per_sample'])
                     except: result_dict['inference_time'] = np.nan
                elif pd.isna(inf_time_raw) and pd.notna(result_dict.get('eval_duration_seconds')) and pd.notna(result_dict.get('total_samples')):
                    try:
                        total_time = float(result_dict['eval_duration_seconds'])
                        num_samples = float(result_dict['total_samples'])
                        result_dict['inference_time'] = total_time / num_samples if num_samples > 0 else np.nan
                    except:
                         result_dict['inference_time'] = np.nan
                elif pd.notna(inf_time_raw): # Ensure existing value is float
                     try: result_dict['inference_time'] = float(inf_time_raw)
                     except: result_dict['inference_time'] = np.nan
                # else: remains NaN
                
                # AC Ops (Fallback from ac_operations for specific SNNs)
                snn_ac_fallback_keys = ["snn_quantized_alexnet", "branchy_quantized_snn_th0.6_0.7", "branchy_quantized_snn_th0.2_0.4", "snn_standard_alexnet"]
                if model_name in snn_ac_fallback_keys and (pd.isna(result_dict.get('ac_ops')) or result_dict.get('ac_ops') == 0):
                    if pd.notna(result_dict.get('ac_operations')):
                        try: 
                            val = int(float(result_dict['ac_operations']))
                            if val > 0: result_dict['ac_ops'] = val
                        except: pass

                # Memory Accesses (Fallback for QSNN)
                if model_name == "snn_quantized_alexnet":
                    if pd.isna(result_dict.get('ann_mem_accesses')) and pd.notna(result_dict.get('ann_memory_access')):
                        try: result_dict['ann_mem_accesses'] = float(result_dict['ann_memory_access'])
                        except: pass
                    if pd.isna(result_dict.get('snn_mem_accesses')) and pd.notna(result_dict.get('snn_memory_access')):
                        try: result_dict['snn_mem_accesses'] = float(result_dict['snn_memory_access'])
                        except: pass
                
                # MACs / operation_count (ensure integer)
                op_c = result_dict.get('macs', result_dict.get('operation_count', np.nan))
                if pd.notna(op_c):
                     try: result_dict['macs'] = int(float(op_c)) # Store in macs
                     except: result_dict['macs'] = np.nan
                else: 
                     result_dict['macs'] = np.nan
                
                # --- Calculate Fair Theoretical Energy if missing --- (USING IMPORTED FUNCTION)
                fair_energy_j = result_dict.get('fair_theoretical_energy_j', result_dict.get('fair_theoretical_energy', np.nan))
                if pd.isna(fair_energy_j) or fair_energy_j <= 0: # Check if missing or invalid
                     print(f"[Data Load] Fair theoretical energy missing for {model_name}. Attempting calculation...")
                     # Call the imported function from utils.py
                     calculated_energy_j, calc_source = calculate_fair_theoretical_energy_from_dict(result_dict, model_name)
                     if pd.notna(calculated_energy_j) and calculated_energy_j > 0:
                         result_dict['fair_theoretical_energy_j'] = calculated_energy_j
                         print(f"[Data Load] Calculated fair_theoretical_energy_j for {model_name}: {calculated_energy_j:.4g} J (Source: {calc_source})")
                     else:
                         result_dict['fair_theoretical_energy_j'] = np.nan # Ensure it's NaN if calculation failed
                         print(f"[Data Load] Calculation failed for {model_name}. Reason: {calc_source}")
                else: # If value exists, ensure it's float
                     try:
                         result_dict['fair_theoretical_energy_j'] = float(fair_energy_j)
                     except:
                         result_dict['fair_theoretical_energy_j'] = np.nan

                # --- Load exit distribution ---
                dist_file = os.path.join(results_dir, f"{model_name}_exit_dist.json")
                if os.path.exists(dist_file):
                    try:
                        with open(dist_file, 'r') as f:
                            result_dict['exit_distribution'] = json.load(f)
                    except Exception as e:
                        print(f"Error loading exit distribution {dist_file}: {e}")
                
                # Store final processed dict
                all_results[model_name] = result_dict
                # print(f"Loaded results for {model_name}: {result_dict}") # Can enable for full dict view
                print(f"  DEBUG ({model_name}): Stored final values - Acc: {result_dict.get('accuracy', 'N/A')*100:.2f}%, Time: {result_dict.get('inference_time', 'N/A'):.4f}s, FairEnergyJ: {result_dict.get('fair_theoretical_energy_j', 'N/A'):.4g} J") # Summary Debug
            else:
                print(f"Warning: CSV file is empty: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
else:
    print(f"Error: Directory {results_dir} does not exist")

print(f"Visualizing models: {list(all_results.keys())}")

# Define plotting functions

def plot_accuracy_vs_power(results, color_map, display_names, filename="accuracy_vs_power.png"):
    """
    Plot accuracy vs THEORETICAL power with model names annotated, hardware markers, and SNN timesteps.
    Theoretical Power = Fair Theoretical Energy / Inference Time.
    """
    import math
    plt.figure(figsize=(14, 10))  # Larger figure for better label visibility
    ax = plt.gca()

    plot_data = []
    for model_key, res in results.items():
        accuracy = res.get('accuracy', np.nan)
        if pd.notna(accuracy) and accuracy > 1.0: # Ensure accuracy is fraction for calculation
            accuracy = accuracy / 100.0
        
        fair_energy_j = res.get('fair_theoretical_energy_j', np.nan)
        inference_time_s = res.get('inference_time', np.nan)
        theoretical_power_w = np.nan

        if pd.notna(fair_energy_j) and pd.notna(inference_time_s) and inference_time_s > 0:
            theoretical_power_w = fair_energy_j / inference_time_s
        
        # Determine device marker
        device_str = str(res.get('device', 'cpu')).lower()
        marker = 's' # default to CPU
        if 'cuda' in device_str or 'nvidia' in device_str:
            marker = 'o' # GPU
        elif 'neuromorphic' in device_str:
            marker = 'D' # Neuromorphic

        # SNN Timestep Labeling
        label = display_names.get(model_key, model_key)
        if model_key.startswith('snn_') or model_key in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']:
            steps = res.get('time_steps', res.get('num_steps', np.nan))
            if pd.notna(steps):
                try: label += f' (T={int(float(steps))})'
                except: pass

        if pd.notna(accuracy) and pd.notna(theoretical_power_w):
            plot_data.append({
                'label': label,
                'raw_label': display_names.get(model_key, model_key), # For annotation without T=
                'power': theoretical_power_w,
                'accuracy_pcnt': accuracy * 100.0,
                'color': color_map.get(model_key, 'blue'),
                'marker': marker,
                'model_key': model_key # for debugging or other purposes
            })
        else:
            print(f"[AccPwrPlot WARN] Skipping {model_key} due to missing data for theoretical power (Energy: {fair_energy_j}, Time: {inference_time_s}) or accuracy ({accuracy}).")

    if not plot_data:
        print(f"WARNING: Skipping {filename} because no valid data points found for theoretical power.")
        plt.close()
        return

    # Plot scatter points
    for data_point in plot_data:
        ax.scatter(data_point['power'], data_point['accuracy_pcnt'], s=150, # Increased size
                   color=data_point['color'],
                   edgecolor='black', linewidth=1.2, alpha=0.85,
                   marker=data_point['marker'],
                   label=data_point['label']) # Full label for legend if needed, though annotations are primary

    # Annotate points (using the raw label for annotation to avoid clutter if T= is in legend)
    for data_point in plot_data:
        ax.annotate(data_point['raw_label'],
                    xy=(data_point['power'], data_point['accuracy_pcnt']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=14,
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.4", fc='white', ec="black", lw=1.0, alpha=0.9))


    # Formatting
    ax.set_xlabel('Power (Watts)', fontsize=18, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    ax.set_title('Accuracy vs. Power Trade-off', fontsize=20, fontweight='bold', pad=25)
    ax.grid(True, which="both", linestyle='--', alpha=0.4, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make remaining spines thicker
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Legend for marker shapes (if needed, annotations might be enough)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='GPU', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='CPU', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        # Line2D([0], [0], marker='D', color='w', label='Neuromorphic', markerfacecolor='gray', markersize=10, markeredgecolor='black')
    ]
    # Create a consolidated legend if there are multiple marker types
    unique_markers_in_plot = {dp['marker'] for dp in plot_data}
    active_legend_elements = [le for le in legend_elements if le.get_marker() in unique_markers_in_plot]
    if active_legend_elements:
        legend = ax.legend(handles=active_legend_elements, loc='lower right', fontsize=14, 
                          title="Hardware Type", title_fontsize=14, frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.9, edgecolor='black')

    # Caption
    caption_text = (
        "Power calculated as (Energy / Inference Time). Lower power and higher accuracy are better. "
        "Model markers indicate primary hardware used for inference time measurement."
    )
    plt.figtext(0.5, 0.01, caption_text, ha="center", fontsize=11, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")

def plot_inference_time(results, color_map, display_names, filename="inference_time.png"):
    """Plot inference time comparison with horizontal bars, showing precision and SNN timesteps."""
    plt.figure(figsize=(14, 10))  # Larger figure for better label visibility

    any_cpu = False # Still useful for other potential warnings, but not for title/caption differentiation
    model_info_list = []

    for m_key in results.keys():
        result = results[m_key]
        label_parts = [display_names.get(m_key, m_key)]

        is_snn_model = 'snn' in m_key or m_key in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']
        current_steps = np.nan

        if is_snn_model:
            s_val = result.get('time_steps', result.get('num_steps', np.nan))
            if pd.notna(s_val):
                try:
                    current_steps = int(float(s_val))
                    if current_steps > 0:
                        label_parts.append(f'(T={current_steps})')
                except (ValueError, TypeError):
                    pass

        precision_label = '(FP32)'
        if 'quant' in m_key or 'int8' in m_key:
            precision_label = '(INT8)'
        elif pd.notna(result.get('precision')):
            precision_label = f"({str(result.get('precision')).upper()})"
        label_parts.append(precision_label)
            
        model_info_list.append({
            'key': m_key,
            'display_label': " ".join(label_parts),
            'time_ms': result.get('inference_time', 0) * 1000,
            'time_std_ms': result.get('inference_time_std', 0) * 1000,
            'is_snn': is_snn_model,
            'steps': current_steps,
        })

    model_info_list.sort(key=lambda x: x['time_ms'] if pd.notna(x['time_ms']) else float('inf'))

    ax = plt.gca()
    
    plot_labels = [info['display_label'] for info in model_info_list]
    plot_times = [info['time_ms'] if pd.notna(info['time_ms']) else 0 for info in model_info_list]
    plot_errors = [info['time_std_ms'] if pd.notna(info['time_std_ms']) else 0 for info in model_info_list]
    plot_colors = [color_map.get(info['key'], 'gray') for info in model_info_list]

    bars = ax.barh(plot_labels, plot_times, xerr=plot_errors, color=plot_colors, alpha=0.8, capsize=5)
    ax.set_xlabel('Inference Time (ms)', fontsize=18, fontweight='bold')
    # Generic title
    title_str = f'Inference Time Comparison'
    ax.set_title(title_str, fontsize=20, fontweight='bold', pad=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make remaining spines thicker
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    # Improve grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, linewidth=0.8)

    max_time_for_text_offset = np.nanmax(plot_times) if any(pd.notna(pt) for pt in plot_times) else 1.0

    for i, bar in enumerate(bars):
        info = model_info_list[i]
        total_time_ms = info['time_ms']
        
        annotation_text = ""
        if pd.isna(total_time_ms):
            annotation_text = "N/A"
        elif total_time_ms > 1000:
            annotation_text = f'{total_time_ms/1000:.3f} s'
        else:
            annotation_text = f'{total_time_ms:.3f} ms'

        if info['is_snn'] and pd.notna(info['steps']) and info['steps'] > 0 and pd.notna(total_time_ms):
            time_per_step_ms = total_time_ms / info['steps']
            annotation_text += f'\n({time_per_step_ms:.3f} ms/step)'
        
        text_x_position = bar.get_width() + (plot_errors[i] if pd.notna(plot_errors[i]) else 0) + max_time_for_text_offset * 0.01
        
        vertical_alignment = 'center'
        if '\n' in annotation_text:
            vertical_alignment = 'top'

        ax.text(text_x_position, bar.get_y() + bar.get_height()/2,
                annotation_text, 
                va=vertical_alignment, 
                ha='left',
                fontsize=12,
                fontweight='bold',
                color='black')

    # Generic caption
    base_caption = f"Inference times measured on {hardware_profile.replace('_', ' ').title()} hardware. "
    snn_caption = "SNN (T=steps) total time shown; per-step time in parentheses. "
    final_caption = base_caption + snn_caption
    plt.figtext(0.5, 0.01, final_caption.strip(), ha="center", fontsize=11, style='italic', wrap=True)

    if not any(pd.notna(t) for t in plot_times):
        print(f"WARNING: Skipping {filename} because input data is empty or all zero/NaN.")
        plt.close()
        return

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")

def plot_accuracy_vs_time(results, color_map, display_names, filename="accuracy_vs_time.png"):
    """
    Plot accuracy vs inference time with model names annotated and improvements:
    - Marker shape for CPU/GPU/neuromorphic
    - Pareto frontier
    """
    import math
    plt.figure(figsize=(14, 10))  # Larger figure for better label visibility
    ax = plt.gca()

    JOULE_TO_MILLIJOULE = 1000
    plot_points_data = []

    for model_key, res in results.items():
        time_ms = res.get('inference_time', np.nan) * 1000
        time_std_ms = res.get('inference_time_std', 0) * 1000 # Default to 0 if not present
        
        accuracy_val = res.get('accuracy', np.nan)
        # Ensure accuracy is in percentage for plotting
        if pd.notna(accuracy_val):
            if accuracy_val <= 1.0 and accuracy_val > 0: # if it's a fraction 0 to 1
                accuracy_pcnt = accuracy_val * 100.0
            else: # if it's already a percentage or other format, use as is (or cap)
                accuracy_pcnt = min(accuracy_val, 100.0) 
        else:
            accuracy_pcnt = np.nan
            
        accuracy_std_pcnt = res.get('accuracy_std', 0)
        # If accuracy_std is a fraction, convert to pcnt. Assume it matches 'accuracy' format.
        if accuracy_std_pcnt <= 1.0 and accuracy_std_pcnt > 0 and res.get('accuracy', 0) <=1.0:
            accuracy_std_pcnt *= 100.0
        
        energy_mj = np.nan
        raw_energy = res.get('fair_theoretical_energy_j', np.nan) # NEW - ensure consistency
        if pd.notna(raw_energy):
            energy_mj = raw_energy * JOULE_TO_MILLIJOULE

        device_str = str(res.get('device', 'cpu')).lower()
        marker_shape = 's' # Default CPU
        if 'cuda' in device_str or 'nvidia' in device_str: marker_shape = 'o' # GPU
        elif 'neuromorphic' in device_str: marker_shape = 'D' # Neuromorphic

        # SNN Timestep Labeling for annotation/legend
        current_display_name = display_names.get(model_key, model_key)
        label_for_plot = current_display_name # Base name for annotation
        if model_key.startswith('snn_') or model_key in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']:
            steps = res.get('time_steps', res.get('num_steps', np.nan))
            if pd.notna(steps):
                try: current_display_name += f' (T={int(float(steps))})' # Full name for legend
                except: pass
        
        if pd.notna(time_ms) and pd.notna(accuracy_pcnt):
            plot_points_data.append({
                'model_key': model_key,
                'time_ms': time_ms,
                'time_std_ms': time_std_ms if time_std_ms > 0 else None, # Use None for errorbar if std is 0
                'accuracy_pcnt': accuracy_pcnt,
                'accuracy_std_pcnt': accuracy_std_pcnt if accuracy_std_pcnt > 0 else None,
                'energy_mj': energy_mj,
                'marker': marker_shape,
                'color': color_map.get(model_key, 'blue'),
                'display_name_full': current_display_name, # With T= for legend
                'display_name_annot': label_for_plot # Simpler name for annotation
            })

    if not plot_points_data:
        print(f"WARNING: Skipping {filename} due to no valid data points.")
        plt.close()
        return

    # Plot points with error bars if std_dev data is available
    for p_data in plot_points_data:
        ax.errorbar(p_data['time_ms'], p_data['accuracy_pcnt'], 
                    xerr=p_data['time_std_ms'], yerr=p_data['accuracy_std_pcnt'],
                    fmt=p_data['marker'], color=p_data['color'], ecolor='gray',
                    elinewidth=2, capsize=5, markersize=12, markeredgecolor='black', 
                    markeredgewidth=2, linewidth=2, alpha=0.9, label=p_data['display_name_full'])

    # Pareto frontier (based on mean values)
    # Extract mean points for Pareto calculation, handling potential NaNs if any point was skipped
    mean_points_for_pareto = sorted(
        [(p['time_ms'], p['accuracy_pcnt']) for p in plot_points_data if pd.notna(p['time_ms']) and pd.notna(p['accuracy_pcnt'])],
        key=lambda x: x[0]
    )
    if mean_points_for_pareto:
        pareto_points = []
        max_accuracy_pareto = -float('inf')
        for point in mean_points_for_pareto:
            if point[1] > max_accuracy_pareto:
                pareto_points.append(point)
                max_accuracy_pareto = point[1]
        if pareto_points:
            pareto_x, pareto_y = zip(*pareto_points)
            ax.plot(pareto_x, pareto_y, 'k--', alpha=0.8, linewidth=3, label="Pareto Frontier (mean values)")

    # Annotate points (using simpler display name)
    for p_data in plot_points_data:
        ax.annotate(p_data['display_name_annot'],
                    xy=(p_data['time_ms'], p_data['accuracy_pcnt']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.4", fc='white', ec="black", lw=1.5, alpha=0.9))

    # Formatting
    ax.set_xlabel('Inference Time (ms)', fontsize=18, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    ax.set_title('Accuracy vs. Inference Time Trade-off', fontsize=20, fontweight='bold', pad=25)
    ax.grid(True, which="both", linestyle='--', alpha=0.4, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make remaining spines thicker
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Legend for marker shapes and Pareto Frontier
    handles, labels = ax.get_legend_handles_labels()
    # Add hardware type legend elements if not duplicative
    legend_elements_hw = [
        Line2D([0], [0], marker='o', color='w', label='GPU', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='CPU', markerfacecolor='gray', markersize=10, markeredgecolor='black')
    ]
    # Filter handles to prevent duplicate model names if errorbar added them
    # A bit complex due to errorbar potentially adding multiple handles per label.
    # Simplification: Rely on point annotations and a separate hardware legend.
    # ax.legend(handles=handles, loc='best') # Default legend for models and Pareto
    
    # Create a new legend for hardware types explicitly
    # Collect unique markers used in this plot
    unique_markers_in_plot_acc = {dp['marker'] for dp in plot_points_data}
    active_legend_elements_acc = [le for le in legend_elements_hw if le.get_marker() in unique_markers_in_plot_acc]

    if active_legend_elements_acc:
         # Add Pareto to this legend if present
        if any("Pareto Frontier" in lbl for lbl in labels):
             active_legend_elements_acc.append(Line2D([0], [0], linestyle='--', color='k', label='Pareto Frontier (mean)'))
        legend = ax.legend(handles=active_legend_elements_acc, loc='upper right', fontsize=14, 
                          title="Hardware / Frontier", title_fontsize=14, frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.9, edgecolor='black')
    elif any("Pareto Frontier" in lbl for lbl in labels): # Only Pareto
        legend = ax.legend(handles=[Line2D([0], [0], linestyle='--', color='k', label='Pareto Frontier (mean)')], 
                          loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True, 
                          framealpha=0.9, edgecolor='black')


    # Caption
    caption_text = (
        "SNNs may be less accurate but can be far more energy efficient on neuromorphic hardware. "
        "Markers indicate hardware. Pareto frontier on mean values."
    )
    plt.figtext(0.5, 0.005, caption_text, ha="center", fontsize=11, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")


def plot_energy_consumption(results, color_map, display_names, filename="energy_consumption.png"):
    """Plot energy consumption (Joules) for each model."""
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    models = list(results.keys())
    energies = []
    labels = [display_names.get(m, m) for m in models]
    colors_for_plot = [] # Store colors in the order of plotting

    JOULE_TO_MILLIJOULE = 1000

    for m_name in models:
        res = results[m_name]
        energy_val = res.get('energy', np.nan) # Default to NaN

        # For SNN types, if 'energy' is NaN or 0, try to use a theoretical energy
        is_snn_type = m_name.startswith('snn_') or m_name in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']
        if is_snn_type and (pd.isna(energy_val) or energy_val == 0):
            if pd.notna(res.get('fair_theoretical_energy_j')):
                energy_val = res['fair_theoretical_energy_j']
                # print(f"[Energy Cons Plot] Using 'fair_theoretical_energy_j' for {m_name}: {energy_val}")
            elif pd.notna(res.get('snn_theoretical_energy')):
                energy_val = res['snn_theoretical_energy']
                # print(f"[Energy Cons Plot] Using 'snn_theoretical_energy' for {m_name}: {energy_val}")
            # else: print(f"[Energy Cons Plot] No suitable energy for SNN {m_name}")
        
        energies.append(energy_val * JOULE_TO_MILLIJOULE if pd.notna(energy_val) else np.nan)
        colors_for_plot.append(color_map.get(m_name, 'gray'))

    # Filter out NaN values for plotting and sorting, keeping labels and colors aligned
    valid_indices = [i for i, e in enumerate(energies) if pd.notna(e)]
    if not valid_indices:
        print(f"WARNING: Skipping {filename} because all energy values are NaN or missing after fallbacks.")
        plt.close()
        return

    sorted_labels = [labels[i] for i in valid_indices]
    sorted_energies = [energies[i] for i in valid_indices]
    sorted_colors = [colors_for_plot[i] for i in valid_indices]

    # Sort by energy values for better visualization
    sort_order = np.argsort(sorted_energies)
    final_labels = [sorted_labels[i] for i in sort_order]
    final_energies = [sorted_energies[i] for i in sort_order]
    final_colors = [sorted_colors[i] for i in sort_order]

    bars = ax.barh(final_labels, final_energies, color=final_colors, alpha=0.8)
    ax.set_xlabel('Energy (mJ)', fontsize=18, fontweight='bold')
    ax.set_title('Energy Consumption Comparison', fontsize=20, fontweight='bold', pad=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, linewidth=0.8)
    
    # Add value annotations on bars
    for i, (bar, energy) in enumerate(zip(bars, final_energies)):
        ax.text(energy + max(final_energies) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{energy:.2f}', va='center', ha='left', fontsize=12, fontweight='bold')
    # Add caption with hardware and simulation note
    plt.figtext(0.5, 0.01,
                f"Hardware: {hardware_profile.replace('_', ' ').title()} | Note: SNN metrics from software simulation; neuromorphic hardware may differ.",
                ha="center", fontsize=10, style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")


def plot_power_consumption(results, color_map, display_names, filename="power_consumption.png"):
    """Plot power consumption (Watts) for each model, using theoretical for SNNs."""
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    models = list(results.keys())
    power_data = [] # List to store dicts of {label, power, color, source}

    ENERGY_THRESHOLD_BYTES = 1.0 # If theoretical energy > this, assume it's bytes needing conversion

    for m_name in models:
        res = results[m_name]
        power_val = np.nan
        power_source = "N/A" # To track if power is measured or theoretical
        
        time_val = res.get('inference_time', np.nan)
        is_snn_type = m_name.startswith('snn_') or m_name in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']

        if is_snn_type:
            # For SNNs, prioritize power from fair theoretical energy
            theoretical_power_snn = np.nan
            if pd.notna(res.get('fair_theoretical_energy_j')) and pd.notna(time_val) and time_val > 0:
                theoretical_power_snn = res['fair_theoretical_energy_j'] / time_val
                power_source = "Theoretical (Fair Energy / Time)"
            elif pd.notna(res.get('snn_theoretical_energy')) and pd.notna(time_val) and time_val > 0:
                energy_j = res['snn_theoretical_energy']
                if energy_j > ENERGY_THRESHOLD_BYTES: # Assumed bytes, convert
                    mem_e = energy_j * E_MEM_J_PER_BYTE
                    ac_ops_val = res.get('ac_ops', res.get('ac_operations', 0))
                    ac_e = 0
                    if pd.notna(ac_ops_val):
                        try: ac_e = int(float(ac_ops_val)) * E_AC_8BIT_J
                        except: pass # ac_e remains 0
                    energy_j = mem_e + ac_e
                
                if pd.notna(energy_j):
                    theoretical_power_snn = energy_j / time_val
                    power_source = "Theoretical (SNN Energy / Time)"
            
            if pd.notna(theoretical_power_snn):
                power_val = theoretical_power_snn
            else:
                # Fallback to reported power if theoretical calculation failed for SNN
                power_val = res.get('power', np.nan)
                if pd.notna(power_val):
                    power_source = "Reported (Measured)"
                # print(f"[Power Plot DEBUG] {m_name} (SNN): No theoretical power, using reported 'power': {power_val} W")
        else: # For non-SNN models, use reported power
            power_val = res.get('power', np.nan)
            if pd.notna(power_val):
                power_source = "Reported (Measured)"
        
        power_data.append({
            'label': display_names.get(m_name, m_name),
            'power': power_val,
            'color': color_map.get(m_name, 'gray'),
            'source': power_source
        })

    # Filter out NaN power values before sorting and plotting
    valid_power_data = [d for d in power_data if pd.notna(d['power'])]
    if not valid_power_data:
        print(f"WARNING: Skipping {filename} because all power values are NaN or missing after fallbacks.")
        plt.close()
        return

    # Sort by power values for better visualization
    valid_power_data.sort(key=lambda x: x['power'])

    final_labels = [d['label'] for d in valid_power_data]
    final_powers = [d['power'] for d in valid_power_data]
    final_colors = [d['color'] for d in valid_power_data]
    final_sources = [d['source'] for d in valid_power_data]

    bars = ax.barh(final_labels, final_powers, color=final_colors, alpha=0.8)
    ax.set_xlabel('Power (Watts)', fontsize=18, fontweight='bold')
    ax.set_title('Power Consumption Comparison', fontsize=20, fontweight='bold', pad=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, linewidth=0.8)

    # Annotate bars with values
    for i, bar_obj in enumerate(bars):
        width = final_powers[i]
        source_info = final_sources[i]
        label_val = f'{width:.2f} W'
        if pd.isna(width):
            label_val = "N/A"
        
        # Position N/A text or actual value text
        text_x = ax.get_xlim()[0] + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if pd.isna(width) else width + 0.01 * width
        ha_align = 'left'
        
        ax.text(text_x, bar_obj.get_y() + bar_obj.get_height()/2, 
                label_val, va='center', ha=ha_align, fontsize=12, fontweight='bold', 
                color='black' if pd.notna(width) else 'dimgrey')

    # Updated caption
    caption_text = (
        f"Hardware: {hardware_profile.replace('_', ' ').title()}. "
        "SNN power is derived theoretically (Fair Energy/Time or SNN Energy/Time). "
        "ANN/other model power is from reported/measured values. Lower is better."
    )
    plt.figtext(0.5, 0.01, caption_text, ha="center", fontsize=9, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for caption
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")


# Additional publication-quality plots

def plot_projected_neuromorphic_energy(results, color_map, display_names, filename="projected_neuromorphic_energy.png"):
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    models = list(results.keys())
    # Collect data, storing 0 or NaN explicitly
    projected_data = []
    for m in models:
        val = results[m].get('projected_neuromorphic_energy', np.nan)
        # Convert to mJ, handling potential NaN
        if pd.notna(val):
            projected_data.append(val * 1000) 
        else:
            projected_data.append(np.nan)
    
    x = np.arange(len(models))
    
    # Separate non-zero/valid data for plotting bars
    plot_indices = [i for i, val in enumerate(projected_data) if pd.notna(val) and val > 0]
    plot_x = x[plot_indices]
    plot_values = [projected_data[i] for i in plot_indices]
    plot_colors = [color_map.get(models[i], 'gray') for i in plot_indices]

    if not plot_values: # If no positive values exist
         # Handle the case where all values are 0 or NaN
         # Plot on a linear scale or show a message
         ax.set_yticks([])
         ax.text(0.5, 0.5, 'No positive projected energy data available.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
         print(f"[Proj Energy Plot] Skipping log scale for {filename} as no positive data found.")
    else:
        # Convert to horizontal bars
        plot_labels = [display_names.get(models[i], models[i]) for i in plot_indices]
        bars = ax.barh(plot_labels, plot_values, color=plot_colors, alpha=0.8)
        ax.set_xscale('log')
        # Annotate only the plotted bars (w > 0)
        for i, bar in enumerate(bars):
            w = plot_values[i]
            # Annotation for positive values
            ax.annotate(f'{w:.2f}', xy=(w, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center', 
                        fontsize=12, fontweight='bold')

    ax.set_xlabel('Projected Neuromorphic Energy (mJ)', fontsize=18, fontweight='bold')
    ax.set_title('Projected Neuromorphic Energy', fontsize=20, fontweight='bold', pad=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, linewidth=0.8)

    # Annotate N/A for zero or NaN values (for models not plotted)
    all_model_labels = [display_names.get(m, m) for m in models]
    plotted_model_indices = set(plot_indices)
    
    for i, val in enumerate(projected_data):
        if not (pd.notna(val) and val > 0) and i not in plotted_model_indices:
            # Add N/A bars for missing models
            model_label = all_model_labels[i]
            ax.barh([model_label], [0.1], color='lightgrey', alpha=0.3)
            ax.text(0.1, len(all_model_labels) - i - 1, 'N/A', 
                    ha='left', va='center', fontsize=12, fontweight='bold', color='grey')
            print(f"[Proj Energy Plot] Zero or NaN projected energy for {models[i]}, annotating N/A.")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")

def plot_syops(results, color_map, display_names, filename="syops.png"):
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    models = list(results.keys())
    # Collect data, storing NaN explicitly
    syops_data = []
    for m in models:
        val = results[m].get('syops', np.nan)
        if pd.notna(val):
             try: syops_data.append(int(float(val)))
             except: syops_data.append(np.nan) # Treat conversion errors as NaN
        else:
             syops_data.append(np.nan)

    x = np.arange(len(models))

    # Separate non-zero/valid data for plotting bars
    plot_indices = [i for i, val in enumerate(syops_data) if pd.notna(val) and val > 0]
    plot_x = x[plot_indices]
    plot_values = [syops_data[i] for i in plot_indices]
    # Use a consistent color for SyOps unless we want per-model colors
    plot_colors = [color_map.get(models[i], 'tab:blue') for i in plot_indices] # Keep per-model color if desired

    if not plot_values:
         ax.set_yticks([])
         ax.text(0.5, 0.5, 'No positive SyOps data available.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
         print(f"[SyOps Plot] Skipping log scale for {filename} as no positive data found.")
    else:
        # Convert to horizontal bars  
        plot_labels = [display_names.get(models[i], models[i]) for i in plot_indices]
        bars = ax.barh(plot_labels, plot_values, color=plot_colors, alpha=0.8)
        ax.set_xscale('log')
        # Annotate only the plotted bars (w > 0)
        for i, bar in enumerate(bars):
            w = plot_values[i]
            ax.annotate(f'{int(w):,}', xy=(w, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center', 
                        fontsize=12, fontweight='bold')

    ax.set_xlabel('SyOps', fontsize=18, fontweight='bold')
    ax.set_title('SyOps (Spike Operations)', fontsize=20, fontweight='bold', pad=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, linewidth=0.8)

    # Annotate N/A for zero or NaN values (for models not plotted)
    all_model_labels = [display_names.get(m, m) for m in models]
    plotted_model_indices = set(plot_indices)

    for i, val in enumerate(syops_data):
        if not (pd.notna(val) and val > 0) and i not in plotted_model_indices:
            # Add N/A bars for missing models
            model_label = all_model_labels[i]
            ax.barh([model_label], [1], color='lightgrey', alpha=0.3)
            ax.text(1, len(all_model_labels) - i - 1, 'N/A', 
                    ha='left', va='center', fontsize=12, fontweight='bold', color='grey')
            print(f"[SyOps Plot] Zero or NaN SyOps for {models[i]}, annotating N/A.")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")

def plot_memory_accesses(results, color_map, display_names, filename="memory_accesses.png"):
    """
    Plot memory accesses for each model with improved annotations and error handling.
    """
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    models = list(results.keys())
    ann_mem = [results[m].get('ann_mem_accesses', 0) for m in models]
    snn_mem = [results[m].get('snn_mem_accesses', 0) for m in models]
    model_labels = [display_names.get(m, m) for m in models]
    
    # Create horizontal grouped bars
    y_pos = np.arange(len(models))
    height = 0.35
    bars_ann = ax.barh(y_pos - height/2, ann_mem, height, label='ANN/Quantized', color='tab:orange', alpha=0.8)
    bars_snn = ax.barh(y_pos + height/2, snn_mem, height, label='SNN', color='tab:cyan', alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Memory Accesses', fontsize=18, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_labels, fontsize=14)
    ax.set_title('Memory Accesses', fontsize=20, fontweight='bold', pad=25)
    legend = ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True, framealpha=0.9, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, linewidth=0.8)
    # Improved annotation: scientific notation, N/A for zero, robust error handling (horizontal bars)
    for bar in list(bars_ann) + list(bars_snn):
        w = bar.get_width()
        if w > 0:
            ax.annotate(f'{w:.2e}', xy=(w, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center', 
                        fontsize=12, fontweight='bold')
        else:
            ax.annotate('N/A', xy=(1e-6, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center', 
                        fontsize=12, fontweight='bold')
    plt.figtext(0.5, 0.01, "Memory access = number of reads/writes to main memory during inference. SNNs may have lower memory access due to event-driven operation.", ha="center", fontsize=10, style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")


def plot_operation_counts(all_results, model_colors, display_names, output_dir="results/plots"):
    """Plots MACs vs ACs for different models, including SNN timesteps and clarifying captions."""
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(14, 10)) # Larger figure for better visibility

    # Define custom order from high to low operation counts
    model_order = [
        'standard_alexnet',           # AlexNet (highest operations)
        'dynamic_alexnet',           # Dynamic AlexNet 
        'snn_standard_alexnet',      # SNN AlexNet
        'snn_quantized_alexnet',     # Quantized SNN AlexNet
        'branchy_quantized_snn_th0.6_0.7',  # QDSNN conservative
        'branchy_quantized_snn_th0.2_0.4'   # QDSNN aggressive (lowest operations)
    ]
    legend_handles = [] # Store handles for custom legend

    ann_color = 'mediumseagreen'
    snn_color = 'mediumpurple'
    
    x_labels = [] # Store final labels for x-axis
    x_ticks = [] # Store positions for x-axis ticks

    for i, model_name in enumerate(model_order):
        if model_name in all_results:
            data = all_results[model_name]
            base_label = display_names.get(model_name, model_name)
            final_label = base_label # Start with the base name
            
            # Determine ops and color: Prioritize AC for SNN types, MAC for others
            ops_value = np.nan
            color = 'grey' # Default color if no ops found
            op_type = 'N/A'
            is_snn_type = model_name.startswith('snn_') or model_name == 'branchy_quantized_snn'

            # --- Determine Operation Type and Value ---
            if model_name in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']: # Specific handling for QDSNN variants
                 ac_val = data.get('ac_ops', data.get('ac_operations', np.nan))
                 if pd.notna(ac_val) and ac_val > 0:
                     ops_value = ac_val
                     color = snn_color
                     op_type = 'AC'
                 else: # Fallback to MACs for QDSNN if ACs are zero/NaN (unlikely but safe)
                     mac_val = data.get('macs', data.get('operation_count', np.nan))
                     if pd.notna(mac_val) and mac_val > 0:
                         ops_value = mac_val
                         color = ann_color
                         op_type = 'MAC'
            elif is_snn_type: # Standard SNNs
                ac_val = data.get('ac_ops', data.get('ac_operations', np.nan))
                if pd.notna(ac_val) and ac_val > 0:
                    ops_value = ac_val
                    color = snn_color
                    op_type = 'AC'
            else: # ANN models
                mac_val = data.get('macs', data.get('operation_count', np.nan))
                if pd.notna(mac_val) and mac_val > 0:
                    ops_value = mac_val
                    color = ann_color
                    op_type = 'MAC'

            # --- Add Timestep (T) label for SNNs ---
            if is_snn_type:
                steps = np.nan
                if model_name == 'snn_standard_alexnet': # Use prior knowledge
                     steps = 32
                elif model_name == 'snn_quantized_alexnet': # Use prior knowledge
                     steps = 4
                else: # Fetch for others like QDSNN variants
                     steps_raw = data.get('time_steps', data.get('num_steps', np.nan))
                     if pd.notna(steps_raw):
                         try: steps = int(float(steps_raw))
                         except: pass # remains NaN

                if pd.notna(steps):
                    final_label += f' (T={steps})'
                else:
                    final_label += ' (T=N/A)' # Indicate if T is unknown
            
            if not pd.isna(ops_value) and ops_value > 0:
                # Store for plotting later as horizontal bar
                x_labels.append(final_label) # Add the final label with T value
                x_ticks.append(i) # Store the tick position
                
            else: # Handle cases where ops are NaN or zero
                x_labels.append(final_label + "\n(No Ops Data)")
                x_ticks.append(i)


    # Now create horizontal bars for all models
    ops_values = []
    ops_colors = []
    final_labels = []
    
    for i, model_name in enumerate(model_order):
        if model_name in all_results:
            data = all_results[model_name]
            base_label = display_names.get(model_name, model_name)
            final_label = base_label
            
            # Determine ops and color: Prioritize AC for SNN types, MAC for others
            ops_value = np.nan
            color = 'grey'
            is_snn_type = model_name.startswith('snn_') or model_name == 'branchy_quantized_snn'

            if model_name in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']:
                 ac_val = data.get('ac_ops', data.get('ac_operations', np.nan))
                 if pd.notna(ac_val) and ac_val > 0:
                     ops_value = ac_val
                     color = snn_color
                 else:
                     mac_val = data.get('macs', data.get('operation_count', np.nan))
                     if pd.notna(mac_val) and mac_val > 0:
                         ops_value = mac_val
                         color = ann_color
            elif is_snn_type:
                ac_val = data.get('ac_ops', data.get('ac_operations', np.nan))
                if pd.notna(ac_val) and ac_val > 0:
                    ops_value = ac_val
                    color = snn_color
            else:
                mac_val = data.get('macs', data.get('operation_count', np.nan))
                if pd.notna(mac_val) and mac_val > 0:
                    ops_value = mac_val
                    color = ann_color

            # Add Timestep (T) label for SNNs
            if is_snn_type:
                steps = np.nan
                if model_name == 'snn_standard_alexnet':
                     steps = 32
                elif model_name == 'snn_quantized_alexnet':
                     steps = 4
                else:
                     steps_raw = data.get('time_steps', data.get('num_steps', np.nan))
                     if pd.notna(steps_raw):
                         try: steps = int(float(steps_raw))
                         except: pass

                if pd.notna(steps):
                    final_label += f' (T={steps})'
                else:
                    final_label += ' (T=N/A)'
            
            ops_values.append(ops_value if pd.notna(ops_value) and ops_value > 0 else 1)  # Use 1 for N/A values
            ops_colors.append(color)
            final_labels.append(final_label)

    # Create horizontal bars
    y_positions = np.arange(len(final_labels))
    bars = ax.barh(y_positions, ops_values, color=ops_colors, alpha=0.85)
    
    # Annotate bars
    for i, (bar, ops_val) in enumerate(zip(bars, ops_values)):
        if ops_val > 1:  # Real values
            text_color = 'black'
            if ops_val >= 1e9:
                text = f"{ops_val/1e9:.2f}G"
            elif ops_val >= 1e6:
                text = f"{ops_val/1e6:.2f}M"
            elif ops_val >= 1e3:
                text = f"{ops_val/1e3:.1f}K"
            else:
                text = f"{ops_val:.0f}"
            
            ax.text(ops_val * 1.05, bar.get_y() + bar.get_height()/2, text, 
                    ha='left', va='center', fontsize=12, color=text_color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="black", lw=1.0, alpha=0.9))
        else:  # N/A values
            ax.text(1.1, bar.get_y() + bar.get_height()/2, 'N/A', 
                    ha='left', va='center', fontsize=12, fontweight='bold', color='grey')

    # --- Create Custom Legend ---
    mac_patch = plt.Rectangle((0, 0), 1, 1, color=ann_color, label='MAC Operations (ANNs)')
    ac_patch = plt.Rectangle((0, 0), 1, 1, color=snn_color, label='AC Operations (SNNs)')
    legend = ax.legend(handles=[mac_patch, ac_patch], loc='upper right', fontsize=14, 
                      frameon=True, fancybox=True, shadow=True, framealpha=0.9, edgecolor='black')

    # Customize y-axis with model labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(final_labels, fontsize=14)

    # Set title and labels
    ax.set_title('Operation Counts per Inference', fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('Number of Operations (Log Scale)', fontsize=18, fontweight='bold')
    
    # Make x-axis logarithmic for better visualization
    ax.set_xscale('log')
    
    # Add a vertical grid for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # --- Add Explanatory Caption ---
    caption_text = (
        "Operation counts per inference. MACs (Multiply-Accumulates) reported for ANNs (green); "
        "ACs (Accumulates/Synaptic Ops) reported for SNNs (purple). SNN labels include timesteps (T). "
        "Note: MACs and ACs have different computational/energy costs (MAC >> AC). "
        "This plot shows raw counts; see energy/power plots for efficiency comparisons."
    )
    plt.figtext(0.5, 0.01, caption_text, ha="center", fontsize=9, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect for caption
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'operation_counts.png'), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(output_dir, 'operation_counts.png')}")
    plt.close()

def plot_energy_measured_vs_theoretical(results, color_map, display_names, filename="energy_measured_vs_theoretical.png"):
    """
    Plot Fair Theoretical Energy for each model. 
    The previously included 'Measured' energy (likely NVML board energy) has been removed 
    as it's not directly comparable to model-intrinsic theoretical energy.
    Adjusts SNN AlexNet (T=32) to T=4 for this plot if detected.
    """
    import math
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    models = list(results.keys())
    JOULE_TO_MILLIJOULE = 1000
    ENERGY_THRESHOLD_FOR_BYTES = 1.0 

    plot_entries = [] # Store dicts of {label, energy_mj, color}

    print("\nDEBUG: Fair Theoretical Energy Calculation for Plot (energy_measured_vs_theoretical):")
    for m_key in models:
        res = results[m_key]
        theo_energy_j = np.nan
        source_debug = "N/A"
        current_display_name = display_names.get(m_key, m_key)

        # DEBUG PRINT: Show initial energy value before potential adjustment
        if m_key == "snn_standard_alexnet":
            initial_energy_j_debug = res.get('fair_theoretical_energy_j', res.get('snn_theoretical_energy', np.nan)) # Check primary sources
            print(f"  DEBUG ({m_key} Energy Scaling): Initial Energy Value (J) before check: {initial_energy_j_debug}")

        if pd.notna(res.get('fair_theoretical_energy_j')):
            theo_energy_j = res['fair_theoretical_energy_j']
            source_debug = "direct 'fair_theoretical_energy_j'"
        else:
            # Fallback logic (existing)
            is_snn = m_key.startswith('snn_') or m_key in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']
            if is_snn:
                if pd.notna(res.get('snn_theoretical_energy')):
                    val = res['snn_theoretical_energy']
                    if val > ENERGY_THRESHOLD_FOR_BYTES: 
                        mem_energy_j = val * E_MEM_J_PER_BYTE
                        ac_ops = res.get('ac_ops', res.get('ac_operations',0))
                        ac_energy_j = 0
                        if pd.notna(ac_ops):
                            try: ac_energy_j = int(float(ac_ops)) * E_AC_8BIT_J
                            except: pass 
                        theo_energy_j = mem_energy_j + ac_energy_j
                        source_debug = "fallback SNN (mem bytes + AC ops)"
                    else: 
                        theo_energy_j = val
                        source_debug = "fallback SNN (direct Joules)"
            else: # ANN models fallback
                if pd.notna(res.get('ann_theoretical_energy')):
                    val = res['ann_theoretical_energy']
                    if val > ENERGY_THRESHOLD_FOR_BYTES:
                        mem_energy_j = val * E_MEM_J_PER_BYTE
                        macs = res.get('macs', res.get('operation_count', 0))
                        mac_energy_j = 0
                        if pd.notna(macs):
                            try: mac_energy_j = int(float(macs)) * E_MAC_32BIT_J
                            except: pass
                        theo_energy_j = mem_energy_j + mac_energy_j
                        source_debug = "fallback ANN (mem bytes + MAC ops)"
                    else: 
                        theo_energy_j = val
                        source_debug = "fallback ANN (direct Joules)"
            if pd.isna(theo_energy_j):
                 print(f"  [MvsT Plot WARN] {m_key}: Could not determine theoretical energy even with fallbacks.")

        # --- Special adjustment for SNN AlexNet T=32 to T=4 for this plot ---
        original_timesteps_for_snn_alexnet = np.nan
        if m_key == "snn_standard_alexnet":
            # DEBUG PRINT: Show the energy value *just before* the scaling logic activates
            print(f"  DEBUG ({m_key} Energy Scaling): Energy Value (J) entering scaling logic: {theo_energy_j}")
            # Check original timesteps. Assume 'time_steps' or 'num_steps' holds this.
            original_steps_val = res.get('time_steps', res.get('num_steps', np.nan))
            if pd.notna(original_steps_val):
                try: original_timesteps_for_snn_alexnet = int(float(original_steps_val))
                except: pass

            if original_timesteps_for_snn_alexnet == 32 and pd.notna(theo_energy_j):
                scaling_factor = 4.0 / 32.0
                adjusted_energy_j = theo_energy_j * scaling_factor
                print(f"  ADJUSTING {m_key} from T=32 to T=4 for this plot. Original Energy (J): {theo_energy_j:.4g}, Adjusted Energy (J): {adjusted_energy_j:.4g}")
                theo_energy_j = adjusted_energy_j
                current_display_name += " (T=4, Adjusted)" # Update display name
            elif pd.notna(original_timesteps_for_snn_alexnet): # If T is not 32, append its actual T value
                 current_display_name += f" (T={original_timesteps_for_snn_alexnet})"
        elif m_key.startswith('snn_') or m_key in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']: # For other SNNs, add their timesteps
            steps_val = res.get('time_steps', res.get('num_steps', np.nan))
            if pd.notna(steps_val):
                try: current_display_name += f" (T={int(float(steps_val))})"
                except: pass
        # --- End of SNN AlexNet adjustment ---

        print(f"  Model: {m_key:<30} | Source: {source_debug:<30} | Final Energy (J) for plot: {theo_energy_j}")
        
        if pd.notna(theo_energy_j):
            plot_entries.append({
                'label': current_display_name,
                'energy_mj': theo_energy_j * JOULE_TO_MILLIJOULE,
                'color': color_map.get(m_key, 'gray')
            })

    if not plot_entries:
        print(f"WARNING: Skipping {filename} because no models had valid theoretical energy data.")
        plt.close()
        return

    # Sort by theoretical energy for better visualization
    plot_entries.sort(key=lambda x: x['energy_mj'])
    
    sorted_labels = [entry['label'] for entry in plot_entries]
    sorted_energies_mj = [entry['energy_mj'] for entry in plot_entries]
    sorted_colors = [entry['color'] for entry in plot_entries]

    # Horizontal bars
    bars = ax.barh(sorted_labels, sorted_energies_mj, color=sorted_colors, alpha=0.8)
    
    ax.set_xlabel('Energy Consumption (mJ)', fontsize=18, fontweight='bold')
    ax.set_xscale('log')
    ax.set_title('Energy Consumption', fontsize=20, fontweight='bold', pad=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, linewidth=0.8)

    for bar_obj in bars:
        w = bar_obj.get_width()
        if w > 0:
            ax.annotate(f'{w:.3g}', 
                        xy=(w, bar_obj.get_y() + bar_obj.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', 
                        ha='left', va='center', fontsize=12, fontweight='bold')

    caption_text = (
        "Energy Consumption accounts for operations (MACs/ACs) and memory accesses, normalized for SNN timesteps (where applicable) and bit precision. "
        "SNN AlexNet (T=32) energy is adjusted to T=4 for this specific plot. Lower is better."
    )
    plt.figtext(0.5, 0.005, caption_text, ha="center", fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjusted rect for potentially longer caption
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")

def plot_fair_theoretical_energy(results, color_map, display_names, filename="fair_theoretical_energy.png"):
    """Plot Fair Theoretical Energy, prioritizing 'fair_theoretical_energy_j' and falling back to others."""
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    models = list(results.keys())
    JOULE_TO_MILLIJOULE = 1000
    ENERGY_THRESHOLD_FOR_BYTES = 1.0 # If theoretical energy > this, assume it's bytes and needs conversion
    fair_energies_mj = []
    labels = []
    colors = []

    for model_key in models:
        res = results[model_key]
        energy_val_j = np.nan
        source_debug = "NaN"
        current_display_name_fte = display_names.get(model_key, model_key) # Use a distinct variable for label

        is_snn_type = model_key.startswith('snn_') or model_key in ['branchy_quantized_snn_th0.6_0.7', 'branchy_quantized_snn_th0.2_0.4']

        if pd.notna(res.get('fair_theoretical_energy_j')):
            energy_val_j = res['fair_theoretical_energy_j']
            source_debug = "'fair_theoretical_energy_j'"
        elif is_snn_type and pd.notna(res.get('snn_theoretical_energy')):
            val = res['snn_theoretical_energy']
            if val > ENERGY_THRESHOLD_FOR_BYTES: # Likely bytes
                mem_energy_j = val * E_MEM_J_PER_BYTE
                ac_ops = res.get('ac_ops', 0)
                # TODO: Differentiate SNN AC bit precision if data available, default to 8-bit
                ac_energy_j = ac_ops * E_AC_8BIT_J 
                energy_val_j = mem_energy_j + ac_energy_j
                source_debug = f"Converted 'snn_theoretical_energy' (mem_bytes {val:.0f}) + AC_energy"
            else: # Already Joules
                energy_val_j = val
                source_debug = "'snn_theoretical_energy' (as Joules)"
        elif not is_snn_type and pd.notna(res.get('ann_theoretical_energy')):
            val = res['ann_theoretical_energy']
            if val > ENERGY_THRESHOLD_FOR_BYTES: # Likely bytes
                mem_energy_j = val * E_MEM_J_PER_BYTE
                macs = res.get('macs', 0)
                # TODO: Differentiate ANN MAC bit precision if data available, default to 32-bit
                mac_energy_j = macs * E_MAC_32BIT_J
                energy_val_j = mem_energy_j + mac_energy_j
                source_debug = f"Converted 'ann_theoretical_energy' (mem_bytes {val:.0f}) + MAC_energy"
            else: # Already Joules
                energy_val_j = val
                source_debug = "'ann_theoretical_energy' (as Joules)"
        
        print(f"[Fair Energy Plot DEBUG] {model_key}: Using {source_debug}, value (J) before T-adjustment: {energy_val_j}")

        # --- Adjust SNN AlexNet energy to represent T=4 for this plot ---
        if model_key == "snn_standard_alexnet":
            native_t = 32 # Assuming the loaded value corresponds to T=32
            target_t = 4
            if pd.notna(energy_val_j):
                scaling_factor = float(target_t) / native_t
                adjusted_energy_j = energy_val_j * scaling_factor
                print(f"  [Fair Energy Plot] OVERRIDING {model_key} energy for T={target_t}. Original Energy (J): {energy_val_j:.4g}, New Energy (J): {adjusted_energy_j:.4g}")
                energy_val_j = adjusted_energy_j # Replace with scaled value
                current_display_name_fte = f"{display_names.get(model_key, model_key)} (Energy @ T={target_t})" # Specific label
            else:
                 current_display_name_fte += " (T=Unknown Energy)" # Label if energy was NaN
        elif is_snn_type: # For other SNNs (like QSNN, QDSNN), add their actual timesteps
            steps_val = res.get('time_steps', res.get('num_steps', np.nan))
            if pd.notna(steps_val):
                try: 
                    s_int = int(float(steps_val))
                    current_display_name_fte += f" (T={s_int})"
                except: pass # Keep original name if conversion fails
            # else: keep original name if T is unknown
        # --- End of SNN AlexNet adjustment ---

        if pd.notna(energy_val_j):
            fair_energies_mj.append(energy_val_j * JOULE_TO_MILLIJOULE)
        else:
            fair_energies_mj.append(np.nan)
        colors.append(color_map.get(model_key, 'grey'))
        labels.append(current_display_name_fte) # Use the potentially modified label

    if not any(pd.notna(e) and e > 0 for e in fair_energies_mj): # Check if all values are zero, NaN or less
        print(f"WARNING: Skipping {filename} because all Fair Theoretical Energy values are zero, NaN or missing.")
        plt.close() # Close the figure to prevent empty plot display/save
        return

    # Sort by energy values for better visualization
    sorted_indices = np.argsort(fair_energies_mj)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_energies = [fair_energies_mj[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # Keep as horizontal bars
    bars = ax.barh(sorted_labels, sorted_energies, color=sorted_colors, alpha=0.8)
    ax.set_xlabel('Energy Consumption (mJ)', fontsize=18, fontweight='bold')
    ax.set_title('Energy Consumption Comparison', fontsize=20, fontweight='bold', pad=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, linewidth=0.8)

    # Annotate bars with values (horizontal bars)
    for bar_obj, energy in zip(bars, sorted_energies):
        label_val = f'{energy:.3g}' # Use general format for better readability
        if energy == 0:
            label_val = "N/A"
            ax.text(0.01, bar_obj.get_y() + bar_obj.get_height()/2, label_val, 
                   ha='left', va='center', fontsize=12, fontweight='bold')
        else:
            ax.text(energy + energy*0.01, bar_obj.get_y() + bar_obj.get_height()/2, label_val, 
                   ha='left', va='center', fontsize=12, fontweight='bold')
    
    # Add a caption about the energy calculation
    plt.figtext(0.5, 0.01,
                "Energy Consumption prioritizes 'fair_theoretical_energy_j' column which accounts for SNN timesteps, bit precision, etc. Falls back to other theoretical energy values if primary is missing.",
                ha="center", fontsize=9, style='italic')

    plt.tight_layout()
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_early_exit_distribution(data_file="results/logs/early_exit_distribution.csv", filename="early_exit_distribution.png"):
    """Generate early exit distribution plot from CSV data"""
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Data file {data_file} not found. Please run early exit evaluation first using evaluate_early_exit_distribution.py")
        return
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    
    # Prepare data for plotting
    exit_names = ["Exit 1", "Exit 2", "Final Exit"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Get model data
    models = []
    distributions = []
    
    for _, row in df.iterrows():
        model_name = row['model_name']
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        models.append(display_name)
        
        # Extract exit percentages
        exit_dist = []
        for i in range(3):  # 3 exits maximum
            exit_dist.append(row.get(f'exit_{i}_percentage', 0.0))
        distributions.append(exit_dist)
    
    # Create stacked horizontal bars
    y_pos = np.arange(len(models))
    bar_width = 0.8
    
    for i, (model, distribution) in enumerate(zip(models, distributions)):
        left = 0
        for exit_idx, (percentage, color) in enumerate(zip(distribution, colors)):
            if percentage > 0:
                bar = ax.barh(i, percentage, left=left, height=bar_width, 
                            color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add percentage labels if significant
                if percentage > 5:  # Only label if > 5%
                    ax.text(left + percentage/2, i, f'{percentage:.1f}%', 
                           ha='center', va='center', fontweight='bold', 
                           fontsize=12, color='white' if percentage > 15 else 'black')
                
                left += percentage
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Percentage of Samples (%)', fontweight='bold')
    ax.set_title('Early Exit Distribution Across Models', fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black') 
                      for color in colors]
    ax.legend(legend_elements, exit_names, loc='lower right', frameon=True, 
             fancybox=True, shadow=True)
    
    # Style the axes
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, alpha=0.4, linewidth=0.8, axis='x')
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Early exit distribution plot saved to {save_path}")

def plot_class_distribution_per_exit(data_file="results/logs/early_exit_distribution.csv", filename="class_distribution_per_exit.png"):
    """Plot class distribution for each exit point from CSV data"""
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Data file {data_file} not found. Please run early exit evaluation first using evaluate_early_exit_distribution.py")
        return
    
    # CIFAR-10 class names
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Find models that have early exit data
    models_with_exits = []
    for _, row in df.iterrows():
        model_name = row['model_name']
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        # Check if this model has class distribution data
        has_class_data = False
        for exit_idx in range(3):
            for class_name in CIFAR10_CLASSES:
                col_name = f'exit_{exit_idx}_class_{class_name}_percentage'
                if col_name in df.columns and pd.notna(row.get(col_name, np.nan)):
                    has_class_data = True
                    break
            if has_class_data:
                break
        
        if has_class_data:
            models_with_exits.append((model_name, display_name, row))
    
    if not models_with_exits:
        print("No class distribution data found in CSV file.")
        return
    
    # Find unique exits across all models
    all_exits = set()
    for model_name, display_name, row in models_with_exits:
        for exit_idx in range(3):
            exit_percentage = row.get(f'exit_{exit_idx}_percentage', 0.0)
            if exit_percentage > 0:
                all_exits.add(exit_idx)
    
    max_exits = len(all_exits)
    sorted_exits = sorted(all_exits)
    
    if max_exits == 0:
        print("No exit data found.")
        return
    
    # Create subplot grid
    fig, axes = plt.subplots(len(models_with_exits), max_exits, 
                            figsize=(6*max_exits, 6*len(models_with_exits)), dpi=300)
    
    # Handle single subplot case
    if len(models_with_exits) == 1 and max_exits == 1:
        axes = np.array([[axes]])
    elif len(models_with_exits) == 1:
        axes = axes.reshape(1, -1)
    elif max_exits == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each model
    for model_idx, (model_name, display_name, row) in enumerate(models_with_exits):
        # Plot class distribution for exits that exist for this model
        for col_idx, exit_idx in enumerate(sorted_exits):
            if col_idx >= max_exits:
                break
                
            ax = axes[model_idx, col_idx]
            
            # Check if this exit has data for this model
            exit_percentage = row.get(f'exit_{exit_idx}_percentage', 0.0)
            
            if exit_percentage > 0:
                class_percentages = []
                for class_name in CIFAR10_CLASSES:
                    col_name = f'exit_{exit_idx}_class_{class_name}_percentage'
                    percentage = row.get(col_name, 0.0)
                    class_percentages.append(percentage if pd.notna(percentage) else 0.0)
                
                # Create bar plot
                bars = ax.bar(CIFAR10_CLASSES, class_percentages, color='#4ECDC4', alpha=0.8, 
                             edgecolor='black', linewidth=0.5)
                
                # Add percentage labels on bars
                for bar, percentage in zip(bars, class_percentages):
                    if percentage > 0.5:  # Only show labels for bars with significant height
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                               f'{percentage:.1f}%', ha='center', va='bottom', 
                               fontsize=10, fontweight='bold')
                
                # Customize the subplot
                ax.set_ylabel('Percentage of Samples (%)', fontweight='bold')
                ax.set_xlabel('Class Label', fontweight='bold')
                
                # Set title based on exit type
                if exit_idx == max(sorted_exits):
                    exit_name = f"Final Exit ({exit_idx+1})"
                else:
                    exit_name = f"Early Exit ({exit_idx+1})"
                
                ax.set_title(f"{display_name} - {exit_name}", fontweight='bold')
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
                
                # Style the axes
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                ax.grid(True, alpha=0.4, linewidth=0.8, axis='y')
                
                # Set consistent y-axis limits
                ax.set_ylim(0, max(max(class_percentages) * 1.1, 10))
            else:
                # Hide subplot if this exit doesn't exist for this model
                ax.set_visible(False)
    
    # Add overall title
    fig.suptitle('CIFAR-10 - Class Distributions Across Exits', fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Class distribution plot saved to {save_path}")

def plot_early_exit_distribution_from_json(json_file, model_display_name=None, filename="early_exit_distribution.png"):
    """Plot early exit distribution from the new JSON format produced by evaluate.py."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    with open(json_file, 'r') as f:
        data = json.load(f)
    total_samples = data.pop('total_samples', None)
    if total_samples is None or total_samples == 0:
        print(f"No total_samples in {json_file}, cannot plot.")
        return
    # Sort exits by index
    exit_indices = sorted(int(k) for k in data.keys() if k.isdigit())
    percentages = [100.0 * data[str(idx)] / total_samples for idx in exit_indices]
    exit_labels = [f"Exit {idx+1}" for idx in exit_indices]
    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(exit_labels, percentages, color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(exit_labels)], alpha=0.8, edgecolor='black')
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%", va='center', fontsize=12)
    ax.set_xlabel('Percentage of Samples (%)', fontweight='bold')
    if model_display_name:
        ax.set_title(f'Exit Distribution: {model_display_name}', fontweight='bold')
    else:
        ax.set_title('Exit Distribution', fontweight='bold')
    ax.set_xlim(0, 100)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Early exit distribution plot saved to {save_path}")

# Generate plots if results are available
if all_results:
    model_colors = get_model_colors(list(all_results.keys()))
    display_names = MODEL_DISPLAY_NAMES

    # --- Inference Time Plots ---
    # Revert to single call for inference time plot
    print("\n--- Generating Combined Inference Time Plot ---")
    plot_inference_time(all_results, model_colors, display_names, filename="inference_time.png") # Ensure original filename is used

    # gpu_results_inf = {k: v for k, v in all_results.items() if 'cuda' in str(v.get('device', 'cpu')).lower() or 'nvidia' in str(v.get('device', 'cpu')).lower()}
    # cpu_results_inf = {k: v for k, v in all_results.items() if not ('cuda' in str(v.get('device', 'cpu')).lower() or 'nvidia' in str(v.get('device', 'cpu')).lower())}

    # if gpu_results_inf:
    #     print("\n--- Generating GPU Inference Time Plot ---")
    #     plot_inference_time(gpu_results_inf, model_colors, display_names, filename="inference_time_gpu.png", device_context="GPU")
    # else:
    #     print("\nNo GPU results found for separate inference time plot.")

    # if cpu_results_inf:
    #     print("\n--- Generating CPU Inference Time Plot ---")
    #     plot_inference_time(cpu_results_inf, model_colors, display_names, filename="inference_time_cpu.png", device_context="CPU")
    # else:
    #     print("\nNo CPU results found for separate inference time plot.")
    
    # Optionally, plot a combined one if both CPU and GPU results exist and it's desired
    # if gpu_results_inf and cpu_results_inf:
    # print("\n--- Generating Combined Inference Time Plot ---")
    # plot_inference_time(all_results, model_colors, display_names, filename="inference_time_combined.png", device_context="Combined")
    # else:
    # plot_inference_time(all_results, model_colors, display_names, filename="inference_time.png") # Default combined if not explicitly separated


    # --- Other Plots (Original Calls) ---
    plot_accuracy_vs_time(all_results, model_colors, display_names)
    plot_accuracy_vs_power(all_results, model_colors, display_names)
    plot_energy_consumption(all_results, model_colors, display_names)
    plot_power_consumption(all_results, model_colors, display_names)
    
    plot_operation_counts(all_results, model_colors, display_names)
    plot_energy_measured_vs_theoretical(all_results, model_colors, display_names)
    
    plot_projected_neuromorphic_energy(all_results, model_colors, display_names)
    plot_syops(all_results, model_colors, display_names)
    plot_memory_accesses(all_results, model_colors, display_names)
    plot_fair_theoretical_energy(all_results, model_colors, display_names)
    plot_early_exit_distribution()  # Plot early exit distribution if data exists
    plot_class_distribution_per_exit()  # Plot class-wise exit distribution if data exists
    
    print(f"\nAll plots saved to {plots_dir}")
else:
    print("No results found. Please run evaluation first.")

if __name__ == "__main__":
    # ... existing plotting calls ...
    # Example usage for a single model:
    # Find a QDSNN or branchy model exit dist file
    logs_dir = "results/logs"
    for fname in os.listdir(logs_dir):
        if fname.endswith("_exit_dist.json"):
            model_name = fname.replace("_exit_dist.json", "")
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            plot_early_exit_distribution_from_json(os.path.join(logs_dir, fname), model_display_name=display_name, filename=f"{model_name}_exit_distribution.png")