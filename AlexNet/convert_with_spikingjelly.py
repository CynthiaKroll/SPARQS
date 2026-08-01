# Prepend project root to allow top-level imports
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import time
import pandas as pd
from config import config
from utils import calculate_energy, calculate_power
import argparse
from data_loader import get_data_loaders
from torchvision import datasets, transforms
from models.alexnet import AlexNet
from spikingjelly.activation_based.ann2snn import Converter
from spikingjelly.activation_based.ann2snn.modules import VoltageScaler
import pynvml
import threading

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Baseline idle power
pwr_idle = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)/1000.0

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def main(args):
    device = get_device()
    print(f'Using device: {device}')
    # Precise measurement using NVML total-energy & CUDA events
    torch.cuda.synchronize()
    start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)  # in mJ
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Load trained ANN
    ann = AlexNet(num_classes=10).to(device)
    ann.load_state_dict(torch.load(args.ann_checkpoint, map_location=device))
    ann.eval()

    # Convert ANN to SNN via SpikingJelly
    print('Converting ANN to SNN with SpikingJelly Converter...')
    # Calibration loader: raw CIFAR-10 in [0,1] without normalization
    # Calibration uses same normalization as ANN training
    conv_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    conv_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=conv_transform)
    conv_loader = torch.utils.data.DataLoader(conv_dataset, batch_size=args.calib_batch_size, shuffle=False, num_workers=4)
    converter = Converter(dataloader=conv_loader, device=device, mode=args.mode, momentum=args.momentum, fuse_flag=True)
    snn_model = converter(ann).to(device)
    snn_model.eval()

    # Debug: print threshold scales
    print("Threshold scales for each VoltageScaler:")
    for name, module in snn_model.named_modules():
        if isinstance(module, VoltageScaler):
            print(f"{name}: {module.scale.item():.4f}")

    # Evaluate SNN
    # Load test loader from data_loader
    _, _, test_loader, _ = get_data_loaders(augment=False)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

    total, correct = 0, 0
    total_inference_time = 0.0

    # --- Start energy and time measurement JUST BEFORE SNN EVALUATION ---
    torch.cuda.synchronize()
    start_energy_eval = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)  # in mJ
    start_event_eval = torch.cuda.Event(enable_timing=True)
    end_event_eval   = torch.cuda.Event(enable_timing=True)
    start_event_eval.record()
    # --- End of new placement for start measurement ---

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            inputs = images
            # Reset SNN internal state (e.g., IFNode) if applicable
            for m in snn_model.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            # --- Measure inference time for this batch ---
            batch_start = time.time()
            mem_list = []
            for t in range(args.time_steps):
                out_step = snn_model(inputs)
                mem_list.append(out_step)
            batch_end = time.time()
            total_inference_time += (batch_end - batch_start)
            out = torch.stack(mem_list, dim=0)  # [T, B, C]
            # Sum membrane potentials over time for classification
            scores = out.sum(dim=0)
            _, preds = scores.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    # Record end event & compute energy/time
    end_event_eval.record() # Use the eval-specific end event
    torch.cuda.synchronize()
    end_energy_eval = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)  # in mJ
    # Use eval-specific start/end events for time
    total_time = start_event_eval.elapsed_time(end_event_eval) / 1000.0  # seconds (hardware measured)
    # Use eval-specific start/end energy
    energy     = (end_energy_eval - start_energy_eval) / 1000.0          # J (hardware measured)

    # Accuracy as fraction for consistency with other scripts
    acc = correct / total if total > 0 else 0
    print(f'\n[RESULTS] SpikingJelly SNN Accuracy: {acc*100:.2f}%')
    # Use both hardware-measured and software-timed inference
    inference_time = total_inference_time / total if total > 0 else 0   # sec per sample (Python timing)
    print(f'[RESULTS] Total measured inference time (hardware): {total_time:.4f} s')
    print(f'[RESULTS] Total measured inference time (Python): {total_inference_time:.4f} s')
    print(f'[RESULTS] Per-sample inference time: {inference_time:.6f} s')
    print(f'[RESULTS] Total measured energy: {energy:.6f} J')

    # --- Operation counting for fair comparison ---
    import traceback
    from utils import count_ac_operations, count_memory_access, calculate_theoretical_energy
    # Import neuromorphic constants from config
    try:
        from config import config as project_config
        E_SPIKE_NEUROMORPHIC_J = project_config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_spike", 0.1e-12)
        E_AC_NEUROMORPHIC_J = project_config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_ac", E_SPIKE_NEUROMORPHIC_J)
        E_MEM_NEUROMORPHIC_J_PER_BYTE = project_config.get("hardware_constants", {}).get("neuromorphic", {}).get("energy_per_mem_access", 0.2e-12)
    except ImportError:
        print("Warning: config.py not found or neuromorphic constants missing. Using default neuromorphic energy constants for projection.")
        E_SPIKE_NEUROMORPHIC_J = 0.1e-12
        E_AC_NEUROMORPHIC_J = 0.1e-12
        E_MEM_NEUROMORPHIC_J_PER_BYTE = 0.2e-12

    batch_images, _ = next(iter(test_loader))
    batch_images = batch_images[:1].to(device)  # Ensure batch size 1
    ac_ops = syops = spike_count = snn_mem_accesses = snn_theo_energy = projected_neuromorphic_energy = 0
    try:
        # For SpikingJelly, count ops for a single time step and scale up
        ac_ops_single_step, spike_count_single_step = count_ac_operations(snn_model, batch_images, time_steps=1)
        ac_ops = ac_ops_single_step * args.time_steps
        spike_count = spike_count_single_step * args.time_steps
        syops = ac_ops # Assuming SyOps = AC_Ops for this context
        
        snn_mem_accesses = count_memory_access(snn_model, batch_images, is_snn=True, time_steps_from_caller=1)
        snn_mem_accesses *= args.time_steps # Scale memory access for all time steps

        # Calculate standard SNN theoretical energy (using default utils constants)
        _, snn_theo_energy = calculate_theoretical_energy(0, ac_ops, spike_count, 0, snn_mem_accesses)
        
        # Calculate projected neuromorphic energy using constants from config
        projected_energy_ac_neuromorphic = ac_ops * E_AC_NEUROMORPHIC_J
        projected_energy_spikes_neuromorphic = spike_count * E_SPIKE_NEUROMORPHIC_J
        projected_energy_mem_neuromorphic = snn_mem_accesses * E_MEM_NEUROMORPHIC_J_PER_BYTE
        
        projected_neuromorphic_energy = projected_energy_ac_neuromorphic + \
                                          projected_energy_spikes_neuromorphic + \
                                          projected_energy_mem_neuromorphic

    except Exception as e:
        print(f"[WARNING] Op counting or energy projection failed for SpikingJelly SNN: {e}")

    # Calculate projected inference time for SNN on neuromorphic hardware
    if isinstance(spike_count, (int, float)):
        projected_inference_time = spike_count * 1e-6  # 1us per spike
    else:
        projected_inference_time = 0

    print(f'[DEBUG] AC ops: {ac_ops}')
    print(f'[DEBUG] Spike count: {spike_count}')
    print(f'[DEBUG] SyOps: {syops}')
    print(f'[DEBUG] SNN memory accesses: {snn_mem_accesses}')
    print(f'[DEBUG] SNN theoretical energy: {snn_theo_energy}')
    print(f'[DEBUG] Projected neuromorphic energy: {projected_neuromorphic_energy}')
    print(f'[DEBUG] Projected inference time (neuromorphic): {projected_inference_time}')

    power = energy / total_time if total_time > 0 else 0  # W
    print(f'[RESULTS] AC ops: {ac_ops}')
    print(f'[RESULTS] Spike count: {spike_count}')
    print(f'[RESULTS] SNN memory accesses: {snn_mem_accesses}')
    print(f'[RESULTS] SNN theoretical energy: {snn_theo_energy}')
    print(f'[RESULTS] Projected neuromorphic energy: {projected_neuromorphic_energy}')
    print(f'[RESULTS] Projected inference time (neuromorphic): {projected_inference_time}')
    print(f'[RESULTS] Power: {power:.6f} W')

    # Save SpikingJelly evaluation to CSV (all relevant metrics, consistent with evaluate.py)
    os.makedirs('results/logs', exist_ok=True)
    df = pd.DataFrame({
        'model_name': ['snn_standard_alexnet'],
        'weights': [args.ann_checkpoint],
        'thresholds': [args.mode],
        'num_steps': [args.time_steps],
        'accuracy': [acc],
        'inference_time': [inference_time],
        'energy': [energy],
        'power': [power],
        'ac_ops': [ac_ops],
        'syops': [syops],
        'spike_count': [spike_count],
        'snn_mem_accesses': [snn_mem_accesses],
        'snn_theoretical_energy': [snn_theo_energy],
        'projected_neuromorphic_energy': [projected_neuromorphic_energy],
        'projected_inference_time': [projected_inference_time]
    })
    csv_path = os.path.join('results', 'logs', 'snn_standard_alexnet_eval.csv')
    print(f'[RESULTS] Saving DataFrame with columns: {df.columns.tolist()}')
    print(df.head())
    df.to_csv(csv_path, index=False)
    print(f"[RESULTS] SpikingJelly evaluation saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert AlexNet ANN to SNN using SpikingJelly')
    parser.add_argument('--ann_checkpoint', type=str, required=True, help='Path to ANN .pth checkpoint')
    parser.add_argument('--time_steps', type=int, default=100, help='Simulation time steps (T)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--calib_batch_size', type=int, default=128, help='Calibration batch size for SNN converter')
    parser.add_argument('--mode', type=str, default='99.9%', help="Conversion mode: 'max', '99.9%%', or float (0<x<=1)")
    parser.add_argument('--momentum', type=float, default=1.0, help='Momentum for voltage calibration (1.0 for pure max)')
    args = parser.parse_args()
    main(args)
