"""
Unified evaluation of the QSNN baseline vs the QDSNN (early-exit) model.

The whole point of this script is that BOTH models are measured by the SAME code
path: same dataset, same time window T, same Poisson encoder, same spike/AC
counter, same memory counter, same energy constants. Previously the QSNN was
evaluated by evaluate_quantized_snn.py and the QDSNN by evaluate.py, at
different T, with different encoders -- so the reported power/delay gap was an
artifact of the measurement, not of early exiting.

Metrics per model:
  - accuracy (%)
  - AC operations and spike count per sample
  - memory bytes touched per sample
  - theoretical energy (J) = AC * E_AC + bytes * E_MEM
  - avg timesteps to exit and avg exit depth (QSNN: always T and always 3)
  - measured wall-clock latency per sample
  - power (J/s) = energy / latency

Usage:
  python evaluate_quantized_pair.py \
      --qsnn_path results/quantized_snn_alexnet_qat_int8.pth \
      --qdsnn_path results/branchy_quantized_snn_qat.pth \
      --agent_path results/rl_agent_qat.pkl
"""
import argparse
import os
import time

import pandas as pd
import torch

from config import config, E_AC_8BIT_J, E_MEM_J_PER_BYTE
from data_loader import get_data_loaders
from models.branchy_quantized_snn import BranchyQuantizedSNN
from models.quantized_snn_alexnet import (
    QuantizedSNNAlexNet,
    prepare_quantized_snn_alexnet_for_qat,
    encode_poisson,
    get_quant_engine,
)
from models.rl_environment import QLearningAgent
from utils import count_ac_operations, count_memory_access


def load_qsnn(path, time_steps, device):
    """Load the QSNN. Handles both fake-quant (QAT) and converted INT8 checkpoints."""
    model = QuantizedSNNAlexNet(num_classes=config["num_classes"],
                                beta=config["beta"],
                                time_steps=time_steps)
    if path and os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        is_int8 = path.endswith('_int8.pth')
        if is_int8:
            # Rebuild the INT8 skeleton so the quantized weights actually fit.
            torch.backends.quantized.engine = get_quant_engine()
            model = prepare_quantized_snn_alexnet_for_qat(model)
            model.eval().cpu()
            model = torch.ao.quantization.convert(model, inplace=False)
        else:
            model = prepare_quantized_snn_alexnet_for_qat(model)
        # Strict load: a skeleton mismatch must FAIL LOUDLY. strict=False here
        # can silently load nothing and evaluate random weights.
        model.load_state_dict(state)
        print(f"[QSNN] loaded {path} (int8={is_int8})")
    else:
        print(f"[QSNN] WARNING: {path} not found -- evaluating an UNTRAINED model")
    return model.eval().to(device)


def load_snn(path, time_steps, device):
    """Load the FP32 SNN baseline: the SAME backbone as the QSNN, trained WITHOUT
    QAT (train_quantized_snn_alexnet.py --no_qat). Its QuantStub/DeQuantStub were
    never prepared, so the checkpoint is a plain FP32 state_dict -- no prepare_qat
    here or the keys won't match. Same Poisson encoding + T + counters as the QSNN,
    so it drops straight into run_eval as a non-branchy model."""
    model = QuantizedSNNAlexNet(num_classes=config["num_classes"],
                                beta=config["beta"],
                                time_steps=time_steps)
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"[SNN] loaded FP32 checkpoint {path}")
    else:
        print(f"[SNN] WARNING: {path} not found -- evaluating an UNTRAINED model")
    return model.eval().to(device)


def load_qdsnn(path, agent_path, thresholds, time_steps, device):
    """Load the QDSNN (fake-quant / QAT checkpoint) and its RL exit thresholds.

    train_branchy_quantized_snn now saves the FAKE-QUANT model (the one whose
    Val Acc was measured), not a torch.quantization.convert()-ed INT8 copy. The
    destructive INT8 conversion of this hybrid SNN collapsed a ~65% model to
    ~10%, and it was saved unchecked because the good pre-convert val acc gated
    the save. We rebuild the identical QAT structure (fuse + prepare_qat with
    the SAME qconfig training used) so the fused-ConvBn + observer keys line up,
    load, and evaluate the fake-quant model directly -- no convert. 8-bit energy
    comes from the op-counters, not real INT8 kernels, so precision is unchanged
    by running the fake-quant graph in FP32 numerics."""
    model = BranchyQuantizedSNN(num_classes=config["num_classes"],
                                time_window=time_steps,
                                beta=config["beta"])
    if path and os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        # Reproduce training's QAT preparation EXACTLY (train mode is required so
        # fuse keeps BatchNorm live via intrinsic ConvBn, matching the saved keys).
        model.train()
        model.fuse_model()
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        )
        torch.quantization.prepare_qat(model, inplace=True)
        # Strict load: a skeleton mismatch must FAIL LOUDLY, not silently
        # evaluate random weights.
        model.load_state_dict(state)
        # Neutralize any over-sharpened exit temperatures baked in by the
        # (removed) calibration -- the exit decision now uses RAW accumulated
        # confidence. No-op for fresh checkpoints (already 1.0).
        from models.branchy_quantized_snn import reset_exit_temperatures
        reset_exit_temperatures(model)
        print(f"[QDSNN] loaded fake-quant checkpoint {path}")
    else:
        print(f"[QDSNN] WARNING: {path} not found -- evaluating an UNTRAINED model")

    # Exit policy: explicit thresholds win, else the RL agent's learned ones.
    if thresholds:
        model.prepare_for_inference(confidence_thresholds=thresholds)
    elif agent_path and os.path.exists(agent_path):
        agent = QLearningAgent(n_exits=model.n_exits)
        agent.load(agent_path)
        model.prepare_for_inference(rl_agent=agent)
    print(f"[QDSNN] confidence thresholds: {model.confidence_thresholds}")
    return model.eval().to(device)


def run_eval(model, name, test_loader, device, time_steps, is_branchy, max_batches=None):
    """Accuracy + exit statistics + wall-clock latency, over the test set."""
    correct = 0
    total = 0
    exit_counts = {1: 0, 2: 0, 3: 0}
    timestep_sum = 0.0
    depth_sum = 0.0
    elapsed = 0.0

    with torch.no_grad():
        for bi, (images, labels) in enumerate(test_loader):
            if max_batches is not None and bi >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if is_branchy:
                exit_outputs, final_output, exit_pts = model(images)
                # Each sample is classified by the head it actually exited at.
                logits = final_output.clone()
                for i, out in enumerate(exit_outputs):
                    mask = (exit_pts.to(device) == (i + 1))
                    if mask.any():
                        logits[mask] = out[mask]
            else:
                spike_train = encode_poisson(images, time_steps)
                logits = model(spike_train).mean(dim=0)  # MEAN over time (matches training)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if is_branchy:
                for k in (1, 2, 3):
                    exit_counts[k] += (exit_pts == k).sum().item()
                timestep_sum += model.last_exit_timesteps.float().sum().item()
                depth_sum += model.last_exit_points.float().sum().item()
            else:
                # The QSNN always runs the full network for the full time window.
                exit_counts[3] += labels.size(0)
                timestep_sum += time_steps * labels.size(0)
                depth_sum += 3 * labels.size(0)

    return {
        "model": name,
        "accuracy_pct": 100.0 * correct / total,
        "avg_timesteps_to_exit": timestep_sum / total,
        "avg_exit_depth": depth_sum / total,
        "exit1_pct": 100.0 * exit_counts[1] / total,
        "exit2_pct": 100.0 * exit_counts[2] / total,
        "final_exit_pct": 100.0 * exit_counts[3] / total,
        "latency_s_per_sample": elapsed / total,
    }


def measure_cost(model, device, time_steps, images=None, n_samples=32):
    """Per-sample AC ops, spikes, memory bytes and theoretical energy.

    The counters MUST be driven one sample at a time. The QDSNN only skips a
    deeper layer once *every still-active sample in the batch* has exited, so a
    batched measurement would hide most of the per-sample compute saving -- the
    layer keeps running as long as any one sample in the batch is still active.
    We therefore average over `n_samples` single-image forward passes.

    Uses real test images (if given) so spike counts reflect actual data
    sparsity rather than uniform noise."""
    if images is None:
        images = torch.rand(n_samples, *config["input_size"], device=device)
    else:
        images = images[:n_samples].to(device)

    ac_sum = spk_sum = mem_sum = 0.0
    n = images.size(0)
    for i in range(n):
        one = images[i:i+1]
        ac, spk = count_ac_operations(model, one, time_steps)
        mem = count_memory_access(model, one, is_snn=True, time_steps=time_steps)
        ac_sum += ac
        spk_sum += spk
        mem_sum += mem

    ac_ops = ac_sum / n
    spikes = spk_sum / n
    mem_bytes = mem_sum / n
    energy_j = ac_ops * E_AC_8BIT_J + mem_bytes * E_MEM_J_PER_BYTE
    return {
        "ac_ops": ac_ops,
        "spike_count": spikes,
        "mem_bytes": mem_bytes,
        "energy_j": energy_j,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate QSNN vs QDSNN identically")
    parser.add_argument("--qsnn_path", type=str,
                        default="results/quantized_snn_alexnet_qat.pth")
    parser.add_argument("--qdsnn_path", type=str,
                        default="results/branchy_quantized_snn_qat.pth")
    parser.add_argument("--agent_path", type=str, default="results/rl_agent_qat.pkl")
    parser.add_argument("--confidence_thresholds", type=float, nargs=2, default=None,
                        help="Override the RL-learned exit thresholds")
    parser.add_argument("--time_steps", type=int, default=config["num_steps"])
    parser.add_argument("--cost_samples", type=int, default=32,
                        help="Number of single-image passes averaged for the "
                             "per-sample AC/memory cost measurement")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit test batches (INT8 runs on CPU, so the full "
                             "set at T=32 is slow; use this for a quick check)")
    parser.add_argument("--output_csv", type=str, default="results/qsnn_vs_qdsnn.csv")
    parser.add_argument("--snn_path", type=str, default=None,
                        help="Optional FP32 SNN baseline (trained with --no_qat). "
                             "If given, added as a third row measured on the same path.")
    args = parser.parse_args()

    # We evaluate the FAKE-QUANT (QAT) models, which run in FP32 numerics on any
    # device, so use the GPU when present -- the old real-INT8-kernel path was
    # pinned to CPU and made this eval slow. 8-bit energy still comes from the
    # op-counters (measure_cost), not from real INT8 kernels, so the device does
    # not change any reported energy/accuracy number.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.quantized.engine = get_quant_engine()
    print(f"Device: {device} | engine: {torch.backends.quantized.engine} "
          f"| T={args.time_steps}")

    _, _, test_loader, _ = get_data_loaders(distributed=False, augment=False)

    # A batch of real test images for the per-sample cost measurement.
    cost_images, _ = next(iter(test_loader))
    cost_images = cost_images[:args.cost_samples].to(device)

    qsnn = load_qsnn(args.qsnn_path, args.time_steps, device)
    qdsnn = load_qdsnn(args.qdsnn_path, args.agent_path,
                       args.confidence_thresholds, args.time_steps, device)

    # QSNN and QDSNN stay rows[0]/[1] so the comparison summary below is unchanged;
    # the optional FP32 SNN baseline is appended as an extra row.
    model_specs = [("QSNN (quantized_snn_alexnet)", qsnn, False),
                   ("QDSNN (branchy_quantized_snn)", qdsnn, True)]
    if args.snn_path:
        snn = load_snn(args.snn_path, args.time_steps, device)
        model_specs.append(("SNN (fp32_snn_alexnet)", snn, False))

    rows = []
    for name, model, is_branchy in model_specs:
        print(f"\n=== Evaluating {name} ===")
        row = run_eval(model, name, test_loader, device, args.time_steps, is_branchy,
                       max_batches=args.max_batches)
        row.update(measure_cost(model, device, args.time_steps,
                                images=cost_images, n_samples=args.cost_samples))
        row["power_w"] = (row["energy_j"] / row["latency_s_per_sample"]
                          if row["latency_s_per_sample"] > 0 else float('nan'))
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["model", "accuracy_pct", "ac_ops", "spike_count", "mem_bytes",
            "energy_j", "power_w", "latency_s_per_sample",
            "avg_timesteps_to_exit", "avg_exit_depth",
            "exit1_pct", "exit2_pct", "final_exit_pct"]
    df = df[cols]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4g}")
    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    if len(rows) >= 2:
        q, d = rows[0], rows[1]   # QSNN, QDSNN (SNN, if present, is rows[2])
        print(f"\nQDSNN vs QSNN:")
        print(f"  accuracy : {d['accuracy_pct']:.2f}% vs {q['accuracy_pct']:.2f}% "
              f"({d['accuracy_pct'] - q['accuracy_pct']:+.2f} pp)")
        if q["energy_j"] > 0:
            print(f"  energy   : {d['energy_j']/q['energy_j']:.3f}x "
                  f"({100 * (1 - d['energy_j']/q['energy_j']):+.1f}% saved)")
        if q["power_w"] > 0:
            print(f"  power    : {d['power_w']/q['power_w']:.3f}x")
        if q["latency_s_per_sample"] > 0:
            print(f"  delay    : {d['latency_s_per_sample']/q['latency_s_per_sample']:.3f}x "
                  f"(avg timesteps {d['avg_timesteps_to_exit']:.2f} vs "
                  f"{q['avg_timesteps_to_exit']:.2f})")

    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
