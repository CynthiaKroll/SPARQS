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
from torch.ao.nn.quantized.modules.conv import Conv2d as QuantizedConv2d
from torch.ao.nn.quantized.modules.linear import Linear as QuantizedLinear
import snntorch
import pandas as pd


# --- Theoretical energy constants ---
# Imported from config.py (single source of truth). They used to be redefined
# here AND inside calculate_theoretical_energy with values that CONTRADICTED
# config.py (E_MEM 4e-12 here vs an implied 160e-12 in the theoretical fn), so
# the energy numbers depended on which function produced them.


def compute_accuracy(outputs, labels):
    """Compute accuracy from model outputs and labels."""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def calculate_power(energy, inference_time):
    """Calculate power consumption (P = E / t)."""
    if inference_time is None or inference_time <= 0:
        return 0.0
    return energy / inference_time


def count_mac_operations(model, input_data):
    """Count MAC operations during a forward pass.

    Handles standard and quantized Conv/Linear layers. Only layers actually
    executed contribute (so a QDSNN that skips deeper layers counts fewer MACs).
    """
    if isinstance(input_data, tuple):
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_data, device=device)
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
        inp_shape = input_val[0].size()
        output_shape = output_val.size()
        if len(inp_shape) == 4:
            in_channels = inp_shape[1]
        elif len(inp_shape) == 3:
            in_channels = inp_shape[0]
        else:
            return
        out_channels = output_shape[-3]
        out_h, out_w = output_shape[-2], output_shape[-1]
        kernel_h, kernel_w = module.kernel_size
        macs += (in_channels * kernel_h * kernel_w) * (out_channels * out_h * out_w)

    def linear_hook(module, input_val, output_val):
        nonlocal macs
        macs += module.in_features * module.out_features

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
    version charged every model -- including the SNNs -- per MAC, defeating the
    whole point of the SNN energy argument."""
    if hardware_profile is None:
        hardware_profile = config.get("hardware_profile", "a100")
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
        hardware_profile = config.get("hardware_profile", "a100")
    if hardware_profile in config.get("hardware_constants", {}):
        gpu_flops = config["hardware_constants"][hardware_profile]["gpu_flops"]
    else:
        gpu_flops = 10e12

    if isinstance(input_data, tuple):
        device = next(model.parameters()).device
        input_tensor = torch.randn(1, *input_data, device=device)
    else:
        input_tensor = input_data
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

    macs = count_mac_operations(model, input_data)
    theoretical_time = (macs * 2) / gpu_flops if gpu_flops > 0 else 0

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    actual_time = (time.time() - start_time) / repeats
    print(f"Inference time - Theoretical: {theoretical_time:.6f} s, Actual: {actual_time:.6f} s")
    return actual_time


def count_ac_operations(model, input_data, time_steps, debug=False):
    """Count accumulate (AC) operations for an SNN by counting emitted spikes.

    Two bugs used to make this unreliable:
      1. The hook read `out[0]` correctly, but detection keyed off both
         `snntorch.LIF` and `snntorch.Leaky`; more importantly it built a 5-D
         tensor by repeating a sample and, on any exception, a bare `except`
         returned (0, 0) so AC ops silently came back as zero.
    Both SNN models now Poisson-encode a raw 4-D image batch internally and loop
    over their own time window, so the hooks fire once per timestep.
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
        raise TypeError("input_data must be tuple (C,H,W) or torch.Tensor")

    if sample_input.dim() == 5:      # already a spike train -> take one frame
        sample_input = sample_input[0]
    elif sample_input.dim() == 3:    # single sample -> add batch dim
        sample_input = sample_input.unsqueeze(0)
    elif sample_input.dim() != 4:
        raise ValueError(f"count_ac_operations expects 4D (B,C,H,W), got {sample_input.shape}")

    observed_spikes = []

    def spike_counting_hook(module, inp, out):
        # snntorch Leaky returns (spk, mem); count spikes (out[0]) not membrane.
        spk = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(spk):
            observed_spikes.append((spk > 0).sum().item())

    if not any(isinstance(m, snntorch.Leaky) for m in model.modules()):
        return 0, 0
    hooks = [m.register_forward_hook(spike_counting_hook)
             for m in model.modules() if isinstance(m, snntorch.Leaky)]

    model.eval()
    try:
        with torch.no_grad():
            _ = model(sample_input)
        total_spike_ops = sum(observed_spikes)
    finally:
        for h in hooks:
            h.remove()

    if debug:
        print(f"[count_ac_operations] spikes={total_spike_ops} over {len(observed_spikes)} LIF invocations")
    return total_spike_ops, total_spike_ops  # one spike drives one accumulate


def count_syops(model, input_data, time_steps):
    """Synaptic operations == AC operations for these spiking models."""
    return count_ac_operations(model, input_data, time_steps)[0]


def count_memory_access(model, sample_input_or_shape, is_snn=False, time_steps=None, batch_size=1):
    """Count bytes moved (weights + biases + input/output activations) for one inference.

    The old version ran an SNN by calling the model once per timestep from the
    OUTSIDE while the model also looped over time INTERNALLY (T*T passes) and
    hardcoded num_steps=32 regardless of the model's actual time window. Now the
    hooks fire once per internal timestep and we simply sum them -- no external
    loop, no hardcoded T, no division.
    """
    if isinstance(sample_input_or_shape, torch.Tensor):
        sample_input = sample_input_or_shape
        if sample_input.dim() == 5:      # [T,B,C,H,W] -> one frame (model re-encodes)
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
        # Quantized layers expose weight()/bias() as callables, not attributes.
        w = module.weight() if callable(getattr(module, 'weight', None)) else getattr(module, 'weight', None)
        b = module.bias() if callable(getattr(module, 'bias', None)) else getattr(module, 'bias', None)
        weight_bytes = w.nelement() * w.element_size() if torch.is_tensor(w) else 0
        bias_bytes = b.nelement() * b.element_size() if torch.is_tensor(b) else 0
        in_t = inp[0]
        input_bytes = in_t.nelement() * in_t.element_size() if torch.is_tensor(in_t) else 0
        out_t = output[0] if isinstance(output, tuple) else output
        output_bytes = out_t.nelement() * out_t.element_size() if torch.is_tensor(out_t) else 0
        total_memory_bytes += weight_bytes + bias_bytes + input_bytes + output_bytes

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, QuantizedConv2d, QuantizedLinear)):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    try:
        with torch.no_grad():
            _ = model(sample_input)  # both SNN models encode + loop internally
    finally:
        for h in hooks:
            h.remove()
    return int(total_memory_bytes)


def calculate_theoretical_energy(mac_count, ac_count, spike_count, ann_mem_bytes, snn_mem_bytes):
    """Theoretical ANN vs SNN energy, using the constants from config.py."""
    ann_total_energy = mac_count * E_MAC_32BIT_J + ann_mem_bytes * E_MEM_J_PER_BYTE
    snn_total_energy = ac_count * E_AC_8BIT_J + snn_mem_bytes * E_MEM_J_PER_BYTE
    print(f"ANN Energy = {ann_total_energy:.2e} J | SNN Energy = {snn_total_energy:.2e} J")
    return ann_total_energy, snn_total_energy


def calculate_fair_theoretical_energy_from_dict(res, model_key):
    """Fair theoretical energy: SNNs billed per-AC + memory, ANNs per-MAC + memory.

    Both quantized spiking models must be recognised as SNNs, or the QSNN gets
    billed per-MAC while the QDSNN gets billed per-AC -- an apples-to-oranges gap.
    """
    energy_j = np.nan
    source = "Not Calculated"
    is_snn_type = model_key.startswith('snn_') or 'snn' in model_key
    is_quantized = 'quant' in model_key

    try:
        if is_snn_type:
            ac_ops = res.get('ac_ops', res.get('ac_operations', np.nan))
            mem_accesses = res.get('snn_mem_accesses', res.get('snn_memory_access', np.nan))
            if pd.notna(ac_ops) and ac_ops > 0:
                ops_cost = float(ac_ops) * E_AC_8BIT_J
                source = "AC Ops"
            else:
                return np.nan, "Missing SNN Ops"
            mem_cost = float(mem_accesses) * E_MEM_J_PER_BYTE if pd.notna(mem_accesses) and mem_accesses > 0 else 0
            energy_j = ops_cost + mem_cost
            source += " + SNN Mem" if mem_cost > 0 else " (Mem Missing)"
        else:
            mac_ops = res.get('macs', res.get('operation_count', np.nan))
            mem_accesses = res.get('ann_mem_accesses', res.get('ann_memory_access', np.nan))
            precision_cost = E_MAC_8BIT_J if is_quantized else E_MAC_32BIT_J
            if pd.notna(mac_ops) and mac_ops > 0:
                ops_cost = float(mac_ops) * precision_cost
                source = "MAC Ops (%s)" % ("INT8" if is_quantized else "FP32")
            else:
                return np.nan, "Missing ANN Ops"
            mem_cost = float(mem_accesses) * E_MEM_J_PER_BYTE if pd.notna(mem_accesses) and mem_accesses > 0 else 0
            energy_j = ops_cost + mem_cost
            source += " + ANN Mem" if mem_cost > 0 else " (Mem Missing)"

        if pd.notna(energy_j) and energy_j >= 0:
            return energy_j, source
        return np.nan, "Calculation Failed"
    except Exception as e:
        print(f"[Calc FairEnergy ERROR] {model_key}: {e}")
        return np.nan, "Calculation Error"


def encode_direct(x, num_steps):
    """Kept for backward compatibility; prefer models.quantized_snn_lenet.encode_poisson."""
    return x.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
