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

from config import config
from utils import calculate_fair_theoretical_energy_from_dict, E_MAC_32BIT_J, E_MAC_8BIT_J, E_AC_8BIT_J, E_MEM_J_PER_BYTE

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

np.random.seed(42)
random.seed(42)

results_dir = "results/logs"
plots_dir = "results/plots"
hardware_profile = "CPU"

MODEL_FILE_OVERRIDES = {
}

def get_model_colors(models):
    palette = sns.color_palette("colorblind", n_colors=len(models))
    return {m: palette[i] for i, m in enumerate(models)}


def is_snn_model(model_key):
    """True for every spiking variant, quantized or not.

    'quantized_snn_lenet' matches neither the 'snn_' prefix nor the branchy
    name, so spelling this predicate out inline silently dropped the QSNN from
    the SNN-only plots.
    """
    return (model_key.startswith('snn_')
            or model_key == 'branchy_quantized_snn'
            or model_key == 'quantized_snn_lenet')


def get_fair_energy_j(res):
    """Fair theoretical energy in Joules, or NaN if genuinely unavailable.

    evaluate.py writes the column as 'fair_theoretical_energy' while the
    MLP/AlexNet evaluators write 'fair_theoretical_energy_j'; accept both.  A
    stored 0.0 means "not measured for this model class" (SNNs never populate
    'ann_theoretical_energy'), so zeros must not terminate the fallback chain.
    """
    for key in ('fair_theoretical_energy_j', 'fair_theoretical_energy',
                'ann_theoretical_energy', 'dynamic_theoretical_energy', 'energy'):
        value = res.get(key, np.nan)
        if pd.notna(value) and value > 0:
            return value
    return np.nan

os.makedirs(plots_dir, exist_ok=True)
for fname in os.listdir(plots_dir):
    if fname.endswith('.png'):
        try:
            os.remove(os.path.join(plots_dir, fname))
        except Exception:
            pass

print(f"Looking for CSV files in: {os.path.abspath(results_dir)}")
print(f"Using hardware profile: {hardware_profile}")

INCLUDE_MODELS = [
    "lenet",
    "dynamic_lenet",
    "snn_lenet",
    "quantized_snn_lenet",
    "branchy_quantized_snn",
]

MODEL_DISPLAY_NAMES = {
    "lenet": "LeNet",
    "dynamic_lenet": "Dynamic LeNet",
    "snn_lenet": "SNN LeNet", 
    "quantized_snn_lenet": "Quantized SNN LeNet (QSNN)",
    "branchy_quantized_snn": "Quantized Dynamic LeNet (QDSNN)",
}

print(f"Including only models: {INCLUDE_MODELS}")
print(f"Model display names: {MODEL_DISPLAY_NAMES}")

all_results = {}
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    eval_files = [f for f in files if f.endswith('_eval.csv')]
    
    candidate_files = {}
    for model_key in INCLUDE_MODELS:
        if model_key in MODEL_FILE_OVERRIDES:
            candidate_files[model_key] = MODEL_FILE_OVERRIDES[model_key]
        else:
            default_file_pattern = f"{model_key}_eval.csv"
            if default_file_pattern in files:
                 candidate_files[model_key] = default_file_pattern

    print(f"Found {len(files)} files in logs dir. Processing specified models.")
    
    for model_name in INCLUDE_MODELS:
        file_to_load = None
        if model_name in MODEL_FILE_OVERRIDES:
            file_to_load = MODEL_FILE_OVERRIDES[model_name]
            print(f"DEBUG: For model '{model_name}', using override file: '{file_to_load}'")
        else:
            potential_files = [f for f in files if f.startswith(model_name) and f.endswith('_eval.csv')]
            if potential_files:
                potential_files.sort(key=len)
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
                
                
                acc_raw = np.nan
                if 'accuracy_percent' in result_dict and pd.notna(result_dict['accuracy_percent']):
                    acc_raw = result_dict['accuracy_percent']
                    print(f"  DEBUG ({model_name}) Acc Source: Using 'accuracy_percent': {acc_raw}")
                elif 'accuracy' in result_dict and pd.notna(result_dict['accuracy']):
                    acc_raw = result_dict['accuracy']
                    print(f"  DEBUG ({model_name}) Acc Source: Using 'accuracy': {acc_raw}")
                else:
                     print(f"  DEBUG ({model_name}) Acc Source: No valid 'accuracy' or 'accuracy_percent' found.")
                
                result_dict['accuracy'] = np.nan
                print(f"  DEBUG ({model_name}) Acc Raw Value Selected: {acc_raw} (Type: {type(acc_raw)})")
                if pd.notna(acc_raw):
                    try:
                        acc_f = float(acc_raw)
                        print(f"  DEBUG ({model_name}) Acc Float Value: {acc_f}")
                        if acc_f >= 0.0 and acc_f <= 1.0:
                            result_dict['accuracy'] = acc_f
                            print(f"  DEBUG ({model_name}) Acc Case 1: Fraction {acc_f}")
                        elif acc_f > 1.0 and acc_f <= 100.0:
                             result_dict['accuracy'] = acc_f / 100.0
                             print(f"  DEBUG ({model_name}) Acc Case 2: Percentage converted to {result_dict['accuracy']}")
                        else:
                            print(f"  DEBUG ({model_name}) Acc Invalid range: {acc_f}")
                    except (ValueError, TypeError) as e:
                        print(f"  DEBUG ({model_name}) Acc Conversion Error: {e}")
                
                if 'inference_time_per_sample' in result_dict and pd.notna(result_dict['inference_time_per_sample']):
                    result_dict['inference_time'] = result_dict['inference_time_per_sample']
                elif 'inference_time' in result_dict and pd.notna(result_dict['inference_time']):
                    pass
                else:
                    result_dict['inference_time'] = np.nan
                
                if 'ac_operations' in result_dict and pd.notna(result_dict['ac_operations']):
                    result_dict['ac_ops'] = result_dict['ac_operations']
                elif pd.isna(result_dict.get('ac_ops', np.nan)):
                    result_dict['ac_ops'] = 0
                
                if 'operation_count' in result_dict and pd.notna(result_dict['operation_count']):
                    result_dict['macs'] = result_dict['operation_count']
                elif pd.isna(result_dict.get('macs', np.nan)):
                    result_dict['macs'] = 0
                
                if 'ann_memory_access' in result_dict and pd.notna(result_dict['ann_memory_access']):
                    result_dict['ann_mem_accesses'] = result_dict['ann_memory_access']
                elif pd.isna(result_dict.get('ann_mem_accesses', np.nan)):
                    result_dict['ann_mem_accesses'] = 0
                
                if 'snn_memory_access' in result_dict and pd.notna(result_dict['snn_memory_access']):
                    result_dict['snn_mem_accesses'] = result_dict['snn_memory_access']
                elif pd.isna(result_dict.get('snn_mem_accesses', np.nan)):
                    result_dict['snn_mem_accesses'] = 0
                
                if 'fair_theoretical_energy_j' in result_dict and pd.notna(result_dict['fair_theoretical_energy_j']):
                    result_dict['energy'] = result_dict['fair_theoretical_energy_j']
                elif 'fair_theoretical_energy' in result_dict and pd.notna(result_dict['fair_theoretical_energy']):
                    result_dict['energy'] = result_dict['fair_theoretical_energy']
                elif pd.isna(result_dict.get('energy', np.nan)):
                    result_dict['energy'] = 0
                
                if 'num_steps' in result_dict and pd.notna(result_dict['num_steps']):
                    result_dict['time_steps'] = result_dict['num_steps']
                elif pd.isna(result_dict.get('time_steps', np.nan)):
                    # Fall back to the configured T (not a hardcoded 32) so the
                    # plot labels track config.NUM_STEPS when a CSV predates the
                    # time_steps column.
                    result_dict['time_steps'] = config["num_steps"]
                
                all_results[model_name] = result_dict
                print(f"  Loaded {model_name}: Acc={result_dict.get('accuracy', 'N/A'):.4f}")
            else:
                print(f"  DEBUG: {file_to_load} is empty.")
        except Exception as e:
            print(f"  ERROR loading {file_to_load}: {e}")
    
    else:
        print(f"Results directory {results_dir} does not exist!")
        pass

print(f"\nLoaded {len(all_results)} model results")
for name, data in all_results.items():
    display_name = MODEL_DISPLAY_NAMES.get(name, name)
    acc = data.get('accuracy', np.nan)
    energy = data.get('energy', np.nan)
    print(f"  {display_name}: Accuracy={acc:.4f}, Energy={energy:.6e} J")

model_colors = get_model_colors(list(all_results.keys()))
display_names = {k: MODEL_DISPLAY_NAMES.get(k, k) for k in all_results.keys()}

def plot_accuracy_vs_power(results, color_map, display_names, filename="accuracy_vs_power.png"):
    import math
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    plot_data = []
    for model_key, res in results.items():
        accuracy = res.get('accuracy', np.nan)
        if pd.notna(accuracy) and accuracy > 1.0:
            accuracy = accuracy / 100.0
        
        fair_energy_j = get_fair_energy_j(res)

        inference_time_s = res.get('inference_time', np.nan)
        theoretical_power_w = np.nan

        if pd.notna(fair_energy_j) and pd.notna(inference_time_s) and inference_time_s > 0:
            theoretical_power_w = fair_energy_j / inference_time_s
        
        device_str = str(res.get('device', 'cpu')).lower()
        marker = 's'
        if 'cuda' in device_str or 'nvidia' in device_str:
            marker = 'o'
        elif 'neuromorphic' in device_str:
            marker = 'D'

        label = display_names.get(model_key, model_key)
        if is_snn_model(model_key):
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

    for data_point in plot_data:
        ax.scatter(data_point['power'], data_point['accuracy_pcnt'], s=150, # Increased size
                   color=data_point['color'],
                   edgecolor='black', linewidth=1.2, alpha=0.85,
                   marker=data_point['marker'],
                   label=data_point['label']) # Full label for legend if needed, though annotations are primary

    for data_point in plot_data:
        ax.annotate(data_point['raw_label'],
                    xy=(data_point['power'], data_point['accuracy_pcnt']),
                    xytext=(7, 7),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc=data_point['color'], ec="black", lw=0.5, alpha=0.3))


    ax.set_xlabel('Theoretical Power (Watts, Fair Energy / Time)', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Accuracy vs. Theoretical Power Trade-off', fontsize=16, pad=20)
    ax.grid(True, which="both", linestyle='--', alpha=0.3) # Grid for major and minor log ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='GPU', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='CPU', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
    ]
    unique_markers_in_plot = {dp['marker'] for dp in plot_data}
    active_legend_elements = [le for le in legend_elements if le.get_marker() in unique_markers_in_plot]
    if active_legend_elements:
        ax.legend(handles=active_legend_elements, loc='lower right', fontsize=11, title="Hardware Type")

    caption_text = (
        "Theoretical Power calculated as (Fair Theoretical Energy / Inference Time). Lower power and higher accuracy are better. "
        "Model markers indicate primary hardware used for inference time measurement."
    )
    plt.figtext(0.5, 0.01, caption_text, ha="center", fontsize=9, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")

def plot_inference_time(results, color_map, display_names, filename="inference_time.png"):
    plt.figure(figsize=(10, 8))

    any_cpu = False
    model_info_list = []

    for m_key in results.keys():
        result = results[m_key]
        label_parts = [display_names.get(m_key, m_key)]

        is_snn_model = 'snn' in m_key or m_key == 'branchy_quantized_snn'
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

    bars = ax.barh(plot_labels, plot_times, xerr=plot_errors, color=plot_colors, alpha=0.8, capsize=3)
    ax.set_xlabel('Inference Time (ms, lower is better)')
    title_str = f'Inference Time Comparison on {hardware_profile.replace("_", " ").title()}'
    ax.set_title(title_str)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
                fontsize=9)

    base_caption = f"Inference times measured on {hardware_profile.replace('_', ' ').title()} hardware. "
    snn_caption = "SNN (T=steps) total time shown; per-step time in parentheses. "
    final_caption = base_caption + snn_caption
    plt.figtext(0.5, 0.01, final_caption.strip(), ha="center", fontsize=9, style='italic', wrap=True)

    if not any(pd.notna(t) for t in plot_times):
        print(f"WARNING: No valid timing data found for {filename}.")
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved plot: {os.path.join(plots_dir, filename)}")
    
    plt.close()

def plot_accuracy_vs_time(results, color_map, display_names, filename="accuracy_vs_time.png"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_data = []
    y_data = []
    colors = []
    labels = []
    
    for model_name, data in results.items():
        accuracy = data.get('accuracy', np.nan)
        inference_time = data.get('inference_time', np.nan)
        
        if pd.notna(accuracy) and pd.notna(inference_time) and inference_time > 0:
            x_data.append(inference_time * 1000)
            y_data.append(accuracy * 100)
            colors.append(color_map[model_name])
            labels.append(display_names.get(model_name, model_name))
    
    if x_data:
        scatter = ax.scatter(x_data, y_data, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1)
        
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]), xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Inference Time (ms)', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy vs Inference Time - LeNet Models', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
    else:
        print(f"No valid data for {filename}")

def plot_energy_consumption(results, color_map, display_names, filename="energy_consumption.png"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = []
    energies = []
    colors = []
    
    for model_name, data in results.items():
        energy = data.get('energy', np.nan)
        if pd.notna(energy) and energy > 0:
            models.append(display_names.get(model_name, model_name))
            energies.append(energy * 1000)
            colors.append(color_map[model_name])
    
    if models:
        bars = ax.bar(models, energies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{energy:.2f}mJ', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Energy Consumption (mJ)', fontweight='bold')
        ax.set_title('Energy Consumption Comparison - LeNet Models', fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
    else:
        print(f"No valid data for {filename}")

def plot_power_consumption(results, color_map, display_names, filename="power_consumption.png"):

    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = []
    powers = []
    colors = []
    
    for model_name, data in results.items():
        power = data.get('power', np.nan)
        if pd.notna(power) and power > 0:
            models.append(display_names.get(model_name, model_name))
            powers.append(power)
            colors.append(color_map[model_name])
    
    if models:
        bars = ax.bar(models, powers, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
        for bar, power in zip(bars, powers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{power:.2f}W', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Power Consumption (W)', fontweight='bold')
        ax.set_title('Power Consumption Comparison - LeNet Models', fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
    else:
        print(f"No valid data for {filename}")

def plot_projected_neuromorphic_energy(results, color_map, display_names, filename="projected_neuromorphic_energy.png"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = []
    energies = []
    colors = []
    
    for model_name, data in results.items():
        energy = data.get('projected_neuromorphic_energy', np.nan)
        if pd.notna(energy) and energy > 0:
            models.append(display_names.get(model_name, model_name))
            energies.append(energy * 1000)
            colors.append(color_map[model_name])
    
    if models:
        bars = ax.bar(models, energies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{energy:.2f}mJ', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Projected Neuromorphic Energy (mJ)', fontweight='bold')
        ax.set_title('Projected Neuromorphic Energy - LeNet Models', fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
    else:
        print(f"No valid data for {filename}")

def plot_memory_accesses(results, color_map, display_names, filename="memory_accesses.png"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = []
    ann_mem = []
    snn_mem = []
    colors = []
    
    for model_name, data in results.items():
        ann_mem_val = data.get('ann_mem_accesses', 0)
        snn_mem_val = data.get('snn_mem_accesses', 0)
        
        if ann_mem_val > 0 or snn_mem_val > 0:
            models.append(display_names.get(model_name, model_name))
            ann_mem.append(ann_mem_val / 1e6)
            snn_mem.append(snn_mem_val / 1e6)
            colors.append(color_map[model_name])
    
    if models:
        x_pos = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, ann_mem, width, label='ANN Memory', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, snn_mem, width, label='SNN Memory', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Memory Accesses (MB)', fontweight='bold')
        ax.set_title('Memory Access Comparison - LeNet Models', fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
    else:
        print(f"No valid data for {filename}")

def plot_operation_counts(all_results, model_colors, display_names, output_dir="results/plots"):
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(12, 7))

    model_order = list(MODEL_DISPLAY_NAMES.keys())
    legend_handles = []

    ann_color = 'mediumseagreen'
    snn_color = 'mediumpurple'
    
    x_labels = []
    x_ticks = []

    for i, model_name in enumerate(model_order):
        if model_name in all_results:
            data = all_results[model_name]
            base_label = display_names.get(model_name, model_name)
            final_label = base_label
            
            ops_value = np.nan
            color = 'grey'
            op_type = 'N/A'
            is_snn_type = is_snn_model(model_name)

            ac_val = data.get('ac_ops', data.get('ac_operations', np.nan))
            mac_val = data.get('macs', data.get('operation_count', np.nan))
            
            if is_snn_type:
                if pd.notna(ac_val) and ac_val > 0:
                    ops_value = ac_val
                    color = snn_color
                    op_type = 'AC'
                elif pd.notna(mac_val) and mac_val > 0:
                    ops_value = mac_val
                    color = ann_color
                    op_type = 'MAC (SNN fallback)'
                    base_label += ' [MAC fallback]'
            else:
                if pd.notna(mac_val) and mac_val > 0:
                    ops_value = mac_val
                    color = ann_color
                    op_type = 'MAC'

            if is_snn_type:
                steps = np.nan
                if model_name == 'snn_lenet':
                     steps = 32
                elif model_name == 'quantized_snn_lenet':
                     steps = 32
                else:
                     steps_raw = data.get('time_steps', data.get('num_steps', np.nan))
                     if pd.notna(steps_raw):
                         try: steps = int(float(steps_raw))
                         except: pass

                if pd.notna(steps):
                    final_label += f' (T={steps})'
                else:
                    final_label += ' (T=N/A)'
            
            if not pd.isna(ops_value) and ops_value > 0:
                bar = ax.bar(i, ops_value, color=color, alpha=0.85)
                x_labels.append(final_label)
                x_ticks.append(i)

                text_color = 'black'
                if ops_value >= 1e9:
                    text = f"{ops_value/1e9:.2f}G"
                elif ops_value >= 1e6:
                    text = f"{ops_value/1e6:.2f}M"
                elif ops_value >= 1e3:
                    text = f"{ops_value/1e3:.1f}K"
                else:
                    text = f"{ops_value:.0f}"
                
                ax.text(i, ops_value, text, 
                        ha='center', va='bottom', rotation=0, 
                        fontsize=10, color=text_color,
                        bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="black", lw=0.5, alpha=0.1))

            else:
                ax.bar(i, 1, color='lightgrey', alpha=0.5)
                ax.text(i, 1.1, 'N/A', ha='center', va='bottom', fontsize=8, color='grey')
                x_labels.append(final_label + "\n(No Ops Data)")
                x_ticks.append(i)


    mac_patch = plt.Rectangle((0, 0), 1, 1, color=ann_color, label='MAC Operations')
    ac_patch = plt.Rectangle((0, 0), 1, 1, color=snn_color, label='AC Operations (SNNs)')
    ax.legend(handles=[mac_patch, ac_patch], loc='upper right', fontsize=11)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    ax.set_title('Operation Counts per Inference')
    ax.set_ylabel('Number of Operations (Log Scale)')
    
    ax.set_yscale('log')
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    caption_text = (
        "Operation counts per inference. Green bars show MAC operations (ANNs and SNN fallback when AC data unavailable); "
        "Purple bars show AC operations (SNNs). SNN labels include timesteps (T). "
        "Note: MACs and ACs have different computational/energy costs (MAC >> AC). "
        "SNN models showing '[MAC fallback]' lack AC operation data and display MAC counts instead."
    )
    plt.figtext(0.5, 0.01, caption_text, ha="center", fontsize=9, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'operation_counts.png'), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(output_dir, 'operation_counts.png')}")
    plt.close()

def plot_fair_theoretical_energy(results, color_map, display_names, filename="fair_theoretical_energy.png"):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    models = list(results.keys())
    JOULE_TO_MILLIJOULE = 1000
    ENERGY_THRESHOLD_FOR_BYTES = 1.0
    fair_energies_mj = []
    labels = []
    colors = []

    for model_key in models:
        res = results[model_key]
        energy_val_j = np.nan
        source_debug = "NaN"
        current_display_name_fte = display_names.get(model_key, model_key)

        is_snn_type = is_snn_model(model_key)

        primary_energy_j = get_fair_energy_j(res)
        if pd.notna(primary_energy_j):
            energy_val_j = primary_energy_j
            source_debug = "fair theoretical energy"
        elif is_snn_type and pd.notna(res.get('snn_theoretical_energy')):
            val = res['snn_theoretical_energy']
            if val > ENERGY_THRESHOLD_FOR_BYTES:
                mem_energy_j = val * E_MEM_J_PER_BYTE
                ac_ops = res.get('ac_ops', 0)
                ac_energy_j = ac_ops * E_AC_8BIT_J 
                energy_val_j = mem_energy_j + ac_energy_j
                source_debug = f"Converted 'snn_theoretical_energy' (mem_bytes {val:.0f}) + AC_energy"
            else:
                energy_val_j = val
                source_debug = "'snn_theoretical_energy' (as Joules)"
        elif not is_snn_type and pd.notna(res.get('ann_theoretical_energy')):
            val = res['ann_theoretical_energy']
            if val > ENERGY_THRESHOLD_FOR_BYTES:
                mem_energy_j = val * E_MEM_J_PER_BYTE
                macs = res.get('macs', 0)
                mac_energy_j = macs * E_MAC_32BIT_J
                energy_val_j = mem_energy_j + mac_energy_j
                source_debug = f"Converted 'ann_theoretical_energy' (mem_bytes {val:.0f}) + MAC_energy"
            else:
                energy_val_j = val
                source_debug = "'ann_theoretical_energy' (as Joules)"
        
        print(f"[Fair Energy Plot DEBUG] {model_key}: Using {source_debug}, value (J) before T-adjustment: {energy_val_j}")

        # Report every model at the time window it was actually measured at.
        # There used to be a special case here that divided snn_lenet's energy
        # by 8 and relabelled it "@ T=4" while the quantized pair kept its
        # measured T=32 numbers -- that fabricated the comparison.
        if is_snn_type:
            steps_val = res.get('time_steps', res.get('num_steps', np.nan))
            if pd.notna(steps_val):
                try: 
                    s_int = int(float(steps_val))
                    current_display_name_fte += f" (T={s_int})"
                except: pass # Keep original name if conversion fails
            # else: keep original name if T is unknown
        # --- End of SNN LeNet adjustment ---

        if pd.notna(energy_val_j):
            fair_energies_mj.append(energy_val_j * JOULE_TO_MILLIJOULE)
        else:
            fair_energies_mj.append(np.nan)
        colors.append(color_map.get(model_key, 'grey'))
        labels.append(current_display_name_fte)

    if not any(pd.notna(e) and e > 0 for e in fair_energies_mj):
        print(f"WARNING: Skipping {filename} because all Fair Theoretical Energy values are zero, NaN or missing.")
        plt.close()
        return

    sorted_indices = np.argsort(fair_energies_mj)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_energies = [fair_energies_mj[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    bars = ax.barh(sorted_labels, sorted_energies, color=sorted_colors, alpha=0.8)
    ax.set_xlabel('Fair Theoretical Energy (mJ, lower is better)')
    ax.set_title('Fair Theoretical Energy Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar_obj, width in zip(bars, sorted_energies):
        label_val = f'{width:.3g}'
        if width == 0:
            label_val = "N/A"
            ax.text(0.01, bar_obj.get_y() + bar_obj.get_height()/2, label_val, va='center', ha='left', fontsize=10)
        else:
            ax.text(width + width*0.01, bar_obj.get_y() + bar_obj.get_height()/2, label_val, va='center', fontsize=10)
    
    plt.figtext(0.5, 0.01,
                "Fair Theoretical Energy prioritizes 'fair_theoretical_energy_j' column which accounts for SNN timesteps, bit precision, etc. Falls back to other theoretical energy values if primary is missing.",
                ha="center", fontsize=9, style='italic')

    plt.tight_layout()
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_accuracy_vs_time(results, color_map, display_names, filename="accuracy_vs_time.png"):
    import matplotlib.patches as patches
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    scatter_data = []
    for model_key, res in results.items():
        accuracy = res.get('accuracy', np.nan)
        if pd.notna(accuracy) and accuracy > 1.0:
            accuracy = accuracy / 100.0

        inference_time_s = res.get('inference_time', np.nan)
        inference_time_ms = inference_time_s * 1000 if pd.notna(inference_time_s) else np.nan

        device_str = str(res.get('device', 'cpu')).lower()
        marker = 's'
        if 'cuda' in device_str or 'nvidia' in device_str:
            marker = 'o'
        elif 'neuromorphic' in device_str:
            marker = 'D'

        label = display_names.get(model_key, model_key)
        if is_snn_model(model_key):
            steps = res.get('time_steps', res.get('num_steps', np.nan))
            if pd.notna(steps):
                try: label += f' (T={int(float(steps))})'
                except: pass

        if pd.notna(accuracy) and pd.notna(inference_time_ms):
            scatter_data.append({
                'label': label,
                'raw_label': display_names.get(model_key, model_key),
                'time_ms': inference_time_ms,
                'accuracy_pcnt': accuracy * 100.0,
                'color': color_map.get(model_key, 'blue'),
                'marker': marker,
                'model_key': model_key
            })
        else:
            print(f"[AccTimeScatter WARN] Skipping {model_key} due to missing accuracy ({accuracy}) or inference time ({inference_time_ms}).")

    if not scatter_data:
        print(f"WARNING: Skipping {filename} because no valid data points found.")
        plt.close()
        return

    for data_point in scatter_data:
        ax.scatter(data_point['time_ms'], data_point['accuracy_pcnt'], s=150, 
                   color=data_point['color'],
                   edgecolor='black', linewidth=1.2, alpha=0.85,
                   marker=data_point['marker'])

    for data_point in scatter_data:
        ax.annotate(data_point['raw_label'],
                    xy=(data_point['time_ms'], data_point['accuracy_pcnt']),
                    xytext=(7, 7),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc=data_point['color'], ec="black", lw=0.5, alpha=0.3))

    ax.set_xlabel('Inference Time (ms, lower is better)', fontsize=14)
    ax.set_ylabel('Accuracy (%, higher is better)', fontsize=14)
    ax.set_title('Accuracy vs. Inference Time Trade-off', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='GPU', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='CPU', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
    ]
    unique_markers = {dp['marker'] for dp in scatter_data}
    active_legend_elements = [le for le in legend_elements if le.get_marker() in unique_markers]
    if active_legend_elements:
        ax.legend(handles=active_legend_elements, loc='lower right', fontsize=11, title="Hardware Type")

    caption_text = (
        "Inference times measured per hardware type. Higher accuracy and lower inference time are better. "
        "SNN models show total inference time across all time steps."
    )
    plt.figtext(0.5, 0.01, caption_text, ha="center", fontsize=9, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")
    plt.close()

def plot_energy_consumption(results, color_map, display_names, filename="energy_consumption.png"):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    plot_entries = []
    for model_key, res in results.items():
        energy_j = get_fair_energy_j(res)

        if pd.notna(energy_j) and energy_j > 0:
            energy_mj = energy_j * 1000
            label = display_names.get(model_key, model_key)
            
            if is_snn_model(model_key):
                steps = res.get('time_steps', res.get('num_steps', np.nan))
                if pd.notna(steps):
                    try: label += f' (T={int(float(steps))})'
                    except: pass
            
            plot_entries.append({
                'model_key': model_key,
                'label': label,
                'energy_mj': energy_mj,
                'color': color_map.get(model_key, 'grey')
            })

    if not plot_entries:
        print(f"WARNING: Skipping {filename} because no valid energy data found.")
        plt.close()
        return

    plot_entries.sort(key=lambda x: x['energy_mj'])
    
    sorted_labels = [entry['label'] for entry in plot_entries]
    sorted_energies_mj = [entry['energy_mj'] for entry in plot_entries]
    sorted_colors = [entry['color'] for entry in plot_entries]

    x_positions = np.arange(len(sorted_labels))
    bars = ax.bar(x_positions, sorted_energies_mj, width=0.6, color=sorted_colors, alpha=0.75)
    
    ax.set_ylabel('Theoretical Energy (mJ, log scale, lower is better)')
    ax.set_yscale('log')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_labels, rotation=30, ha='right', fontsize=10)
    ax.set_title(f'Theoretical Energy Comparison on {hardware_profile.replace("_", " ").title()}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar_obj in bars:
        h = bar_obj.get_height()
        if h > 0:
            ax.annotate(f'{h:.3g}', 
                        xy=(bar_obj.get_x() + bar_obj.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points', 
                        ha='center', va='bottom', fontsize=9, rotation=0)

    caption_text = (
        "Theoretical Energy accounts for operations (MACs/ACs) and memory accesses, normalized for SNN timesteps (where applicable) and bit precision. "
        "Lower is better."
    )
    plt.figtext(0.5, 0.005, caption_text, ha="center", fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")
    plt.close()

def plot_power_consumption(results, color_map, display_names, filename="power_consumption.png"):

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    plot_entries = []
    for model_key, res in results.items():
        fair_energy_j = get_fair_energy_j(res)

        inference_time_s = res.get('inference_time', np.nan)

        if pd.notna(fair_energy_j) and pd.notna(inference_time_s) and inference_time_s > 0:
            theoretical_power_w = fair_energy_j / inference_time_s
            label = display_names.get(model_key, model_key)
            
            if is_snn_model(model_key):
                steps = res.get('time_steps', res.get('num_steps', np.nan))
                if pd.notna(steps):
                    try: label += f' (T={int(float(steps))})'
                    except: pass
            
            plot_entries.append({
                'model_key': model_key,
                'label': label,
                'power_w': theoretical_power_w,
                'color': color_map.get(model_key, 'grey')
            })

    if not plot_entries:
        print(f"WARNING: Skipping {filename} because no valid power data found.")
        plt.close()
        return

    plot_entries.sort(key=lambda x: x['power_w'])
    
    sorted_labels = [entry['label'] for entry in plot_entries]
    sorted_powers_w = [entry['power_w'] for entry in plot_entries]
    sorted_colors = [entry['color'] for entry in plot_entries]

    x_positions = np.arange(len(sorted_labels))
    bars = ax.bar(x_positions, sorted_powers_w, width=0.6, color=sorted_colors, alpha=0.75)
    
    ax.set_ylabel('Theoretical Power (W, log scale, lower is better)')
    ax.set_yscale('log')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_labels, rotation=30, ha='right', fontsize=10)
    ax.set_title(f'Theoretical Power Comparison on {hardware_profile.replace("_", " ").title()}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar_obj in bars:
        h = bar_obj.get_height()
        if h > 0:
            ax.annotate(f'{h:.3g}', 
                        xy=(bar_obj.get_x() + bar_obj.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points', 
                        ha='center', va='bottom', fontsize=9, rotation=0)

    caption_text = (
        "Theoretical Power calculated as (Fair Theoretical Energy / Inference Time). "
        "Lower is better."
    )
    plt.figtext(0.5, 0.005, caption_text, ha="center", fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")
    plt.close()

def plot_projected_neuromorphic_energy(results, color_map, display_names, filename="projected_neuromorphic_energy.png"):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    plot_entries = []
    for model_key, res in results.items():
        if not is_snn_model(model_key):
            continue
            
        neuromorphic_energy_j = res.get('projected_neuromorphic_energy', np.nan)
        
        if pd.notna(neuromorphic_energy_j) and neuromorphic_energy_j > 0:
            energy_mj = neuromorphic_energy_j * 1000
            label = display_names.get(model_key, model_key)
            
            steps = res.get('time_steps', res.get('num_steps', np.nan))
            if pd.notna(steps):
                try: label += f' (T={int(float(steps))})'
                except: pass
            
            plot_entries.append({
                'model_key': model_key,
                'label': label,
                'energy_mj': energy_mj,
                'color': color_map.get(model_key, 'grey')
            })

    if not plot_entries:
        print(f"WARNING: Skipping {filename} because no valid neuromorphic energy data found.")
        plt.close()
        return

    plot_entries.sort(key=lambda x: x['energy_mj'])
    
    sorted_labels = [entry['label'] for entry in plot_entries]
    sorted_energies_mj = [entry['energy_mj'] for entry in plot_entries]
    sorted_colors = [entry['color'] for entry in plot_entries]

    x_positions = np.arange(len(sorted_labels))
    bars = ax.bar(x_positions, sorted_energies_mj, width=0.6, color=sorted_colors, alpha=0.75)
    
    ax.set_ylabel('Projected Neuromorphic Energy (mJ, log scale, lower is better)')
    ax.set_yscale('log')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_labels, rotation=30, ha='right', fontsize=10)
    ax.set_title('Projected Neuromorphic Energy Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar_obj in bars:
        h = bar_obj.get_height()
        if h > 0:
            ax.annotate(f'{h:.3g}', 
                        xy=(bar_obj.get_x() + bar_obj.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points', 
                        ha='center', va='bottom', fontsize=9, rotation=0)

    caption_text = (
        "Projected Neuromorphic Energy estimates energy consumption on neuromorphic hardware using "
        "lower energy per spike and memory access. Lower is better."
    )
    plt.figtext(0.5, 0.005, caption_text, ha="center", fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {os.path.join(plots_dir, filename)}")
    plt.close()

if all_results:
    model_colors = get_model_colors(list(all_results.keys()))
    display_names = MODEL_DISPLAY_NAMES

    print("\n--- Generating Combined Inference Time Plot ---")
    plot_inference_time(all_results, model_colors, display_names, filename="inference_time.png")


    plot_accuracy_vs_time(all_results, model_colors, display_names)
    plot_accuracy_vs_power(all_results, model_colors, display_names)
    plot_energy_consumption(all_results, model_colors, display_names)
    plot_power_consumption(all_results, model_colors, display_names)
    
    plot_operation_counts(all_results, model_colors, display_names)
    
    plot_projected_neuromorphic_energy(all_results, model_colors, display_names)
    plot_memory_accesses(all_results, model_colors, display_names)
    plot_fair_theoretical_energy(all_results, model_colors, display_names)
    
    print(f"\nAll plots saved to {plots_dir}")
else:
    print("No results found. Please run evaluation first.")