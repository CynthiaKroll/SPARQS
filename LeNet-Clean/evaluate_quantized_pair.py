"""
Unified evaluation of the QSNN baseline vs the QDSNN (early-exit) model.

The whole point of this script is that BOTH models are measured by the SAME code
path: same dataset (MNIST), same time window T, same Poisson encoder, same
spike/AC counter, same memory counter, same energy constants. Previously the
QSNN and QDSNN were evaluated by evaluate.py at different (hardcoded) time steps,
with a direct-repeat encoder that fed analog images rather than spikes -- so any
reported power/delay gap was an artifact of the measurement, not of early exit.

Metrics per model:
  - accuracy (%)
  - AC operations and spike count per sample
  - memory bytes touched per sample
  - theoretical energy (J) = AC * E_AC + bytes * E_MEM
  - avg timesteps to exit and avg exit depth (QSNN: always T and always 3)
  - measured wall-clock latency per sample (timed at batch size 1, to match the
    regime the AC/memory counters are driven in)
  - power (J/s) = energy / latency

Usage:
  python evaluate_quantized_pair.py \
      --qsnn_path results/quantized_snn_lenet_int8.pth \
      --qdsnn_path results/branchy_quantized_snn_int8.pth \
      --agent_path results/rl_agent_qdsnn.pkl
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.ao.quantization as tq

from config import config, E_AC_8BIT_J, E_MEM_J_PER_BYTE
from data_loader import get_data_loaders
from models.branchy_quantized_snn import BranchyQuantizedSNN
from models.quantized_snn_lenet import (
    QuantizedSNNLeNet,
    prepare_quantized_snn_lenet_for_qat,
    encode_poisson,
    get_quant_engine,
)
from models.rl_environment import QLearningAgent
from utils import count_ac_operations, count_memory_access


def _looks_int8(state):
    return any('_packed_params' in k or k.endswith('.scale') for k in state.keys())


def load_qsnn(path, time_steps, device):
    """Load the QSNN. Handles both FP32/fake-quant and converted INT8 checkpoints."""
    model = QuantizedSNNLeNet(num_classes=config["num_classes"],
                              beta=config["snn"]["beta"], time_steps=time_steps)
    if path and os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        if _looks_int8(state):
            torch.backends.quantized.engine = get_quant_engine()
            model = prepare_quantized_snn_lenet_for_qat(model)
            model.eval().cpu()
            model = tq.convert(model, inplace=False)
        # Strict load: a skeleton mismatch must FAIL LOUDLY. strict=False here
        # can silently load nothing and evaluate random weights.
        model.load_state_dict(state)
        print(f"[QSNN] loaded {path} (int8={_looks_int8(state)})")
    else:
        print(f"[QSNN] WARNING: {path} not found -- evaluating an UNTRAINED model")
    return model.eval().to(device)


def load_qdsnn(path, agent_path, thresholds, time_steps, device):
    """Load the QDSNN and its RL-learned exit thresholds."""
    model = BranchyQuantizedSNN(num_classes=config["num_classes"],
                                time_window=time_steps, beta=config["snn"]["beta"])
    if path and os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        if _looks_int8(state):
            torch.backends.quantized.engine = get_quant_engine()
            model.fuse_model()
            model.qconfig = tq.QConfig(
                activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
                weight=tq.MinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
            )
            import snntorch
            for _, m in model.named_modules():
                if isinstance(m, snntorch.Leaky):
                    m.qconfig = None
            tq.prepare_qat(model, inplace=True)
            model.eval().cpu()
            model = tq.convert(model, inplace=False)
        # Strict load: a skeleton mismatch must FAIL LOUDLY, not silently
        # evaluate random weights.
        model.load_state_dict(state)
        print(f"[QDSNN] loaded {path} (int8={_looks_int8(state)})")
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
    correct = total = 0
    exit_counts = {1: 0, 2: 0, 3: 0}
    timestep_sum = depth_sum = elapsed = 0.0

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
                logits = final_output.clone()
                for i, out in enumerate(exit_outputs):
                    mask = (exit_pts.to(device) == (i + 1))
                    if mask.any():
                        logits[mask] = out[mask]
            else:
                spike_train = encode_poisson(images, time_steps)
                logits = model(spike_train).sum(dim=0)

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
        # Batched timing, kept for reference only -- it cannot show an
        # early-exit benefit (see measure_latency). Not used for power.
        "batched_latency_s_per_sample": elapsed / total,
    }


def measure_cost(model, device, time_steps, images, n_samples=32):
    """Per-sample AC ops, spikes, memory bytes and theoretical energy.

    The counters MUST be driven one sample at a time: the QDSNN only skips a
    deeper layer once EVERY still-active sample in the batch has exited, so a
    batched measurement would hide most of the per-sample compute saving."""
    images = images[:n_samples].to(device)
    ac_sum = spk_sum = mem_sum = 0.0
    n = images.size(0)
    for i in range(n):
        one = images[i:i + 1]
        ac, spk = count_ac_operations(model, one, time_steps)
        mem = count_memory_access(model, one, is_snn=True, time_steps=time_steps)
        ac_sum += ac
        spk_sum += spk
        mem_sum += mem
    ac_ops, spikes, mem_bytes = ac_sum / n, spk_sum / n, mem_sum / n
    energy_j = ac_ops * E_AC_8BIT_J + mem_bytes * E_MEM_J_PER_BYTE
    return {"ac_ops": ac_ops, "spike_count": spikes, "mem_bytes": mem_bytes, "energy_j": energy_j}


def measure_latency(model, device, time_steps, images, is_branchy,
                    n_samples=32, warmup=5):
    """Wall-clock latency per sample, timed ONE sample at a time.

    This has to match the regime measure_cost counts in, for the same reason:
    the QDSNN only skips a deeper layer once EVERY still-active sample in the
    batch has exited, so a batched timing loop reports roughly the QSNN's
    latency however aggressively the exit policy fires.  Timing a batch while
    counting per-sample makes the energy and latency columns two different
    experiments, and their ratio (power) meaningless.
    """
    images = images[:n_samples].to(device)
    n = images.size(0)

    def _forward(one):
        if is_branchy:
            model(one)
        else:
            model(encode_poisson(one, time_steps))

    with torch.no_grad():
        for i in range(min(warmup, n)):  # let the lazy INT8 kernels settle
            _forward(images[i:i + 1])

        elapsed = 0.0
        for i in range(n):
            one = images[i:i + 1]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _forward(one)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0

    return {"latency_s_per_sample": elapsed / n}


def main():
    parser = argparse.ArgumentParser(description="Evaluate QSNN vs QDSNN identically")
    parser.add_argument("--qsnn_path", type=str, default="results/quantized_snn_lenet_int8.pth")
    parser.add_argument("--qdsnn_path", type=str, default="results/branchy_quantized_snn_int8.pth")
    parser.add_argument("--agent_path", type=str, default="results/rl_agent_qdsnn.pkl")
    parser.add_argument("--confidence_thresholds", type=float, nargs=2, default=None,
                        help="Override the RL-learned exit thresholds")
    # This script only ever evaluates the QAT pair, so its T is the quantized
    # window (4), not the FP32-SNN default (32). Must match how they were trained.
    parser.add_argument("--time_steps", type=int, default=config["quantized_num_steps"])
    parser.add_argument("--cost_samples", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit test batches (INT8 runs on CPU; use for a quick check)")
    parser.add_argument("--output_csv", type=str, default="results/qsnn_vs_qdsnn.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # The Poisson encoder is stochastic. Without a fixed seed the same weights
    # score differently run to run (observed +-0.06 pp), which is the same
    # order as the QDSNN-vs-QSNN accuracy gap being measured -- so an unseeded
    # comparison cannot distinguish a real difference from encoder noise.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cpu')  # INT8 quantized kernels only run on CPU
    torch.backends.quantized.engine = get_quant_engine()
    print(f"Device: {device} | engine: {torch.backends.quantized.engine} | T={args.time_steps}")

    _, test_loader = get_data_loaders(batch_size=config["batch_size"], augment=False)
    cost_images, _ = next(iter(test_loader))

    qsnn = load_qsnn(args.qsnn_path, args.time_steps, device)
    qdsnn = load_qdsnn(args.qdsnn_path, args.agent_path,
                       args.confidence_thresholds, args.time_steps, device)

    rows = []
    for name, model, is_branchy in (("QSNN (quantized_snn_lenet)", qsnn, False),
                                    ("QDSNN (branchy_quantized_snn)", qdsnn, True)):
        print(f"\n=== Evaluating {name} ===")
        row = run_eval(model, name, test_loader, device, args.time_steps, is_branchy,
                       max_batches=args.max_batches)
        row.update(measure_cost(model, device, args.time_steps, cost_images, args.cost_samples))
        row.update(measure_latency(model, device, args.time_steps, cost_images,
                                   is_branchy, args.cost_samples))
        # Both terms are per-sample, so their ratio is a meaningful power.
        row["power_w"] = (row["energy_j"] / row["latency_s_per_sample"]
                          if row["latency_s_per_sample"] > 0 else float('nan'))
        rows.append(row)

    cols = ["model", "accuracy_pct", "ac_ops", "spike_count", "mem_bytes",
            "energy_j", "power_w", "latency_s_per_sample",
            "batched_latency_s_per_sample",
            "avg_timesteps_to_exit", "avg_exit_depth",
            "exit1_pct", "exit2_pct", "final_exit_pct"]
    df = pd.DataFrame(rows)[cols]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4g}")
    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    if len(rows) == 2:
        q, d = rows[0], rows[1]
        print(f"\nQDSNN vs QSNN:")
        print(f"  accuracy : {d['accuracy_pct']:.2f}% vs {q['accuracy_pct']:.2f}% "
              f"({d['accuracy_pct'] - q['accuracy_pct']:+.2f} pp)")
        if q["energy_j"] > 0:
            print(f"  energy   : {d['energy_j']/q['energy_j']:.3f}x "
                  f"({100 * (1 - d['energy_j']/q['energy_j']):+.1f}% saved)")
        if q["latency_s_per_sample"] > 0:
            print(f"  delay    : {d['latency_s_per_sample']/q['latency_s_per_sample']:.3f}x "
                  f"(avg timesteps {d['avg_timesteps_to_exit']:.2f} vs {q['avg_timesteps_to_exit']:.2f})")

    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
