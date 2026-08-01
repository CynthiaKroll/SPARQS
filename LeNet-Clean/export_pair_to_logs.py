#!/usr/bin/env python3
"""Bridge results/qsnn_vs_qdsnn.csv into the per-model CSVs visualize.py reads.

evaluate_quantized_pair.py measures QSNN and QDSNN together (same encoder, same
INT8 kernels, same time window) and writes one combined CSV with its own column
names.  visualize.py, however, expects one results/logs/<model_key>_eval.csv per
model using the schema evaluate.py emits.  Without this translation step the
plots silently drop the quantized pair -- i.e. the two models the SPARQ claim
actually rests on.

Run it after evaluate_quantized_pair.py and before visualize.py.
"""

import argparse
import os

import pandas as pd

from config import config, E_AC_NEUROMORPHIC_J, E_MEM_NEUROMORPHIC_J_PER_BYTE

# The pair evaluator labels rows "QSNN (quantized_snn_lenet)" / "QDSNN
# (branchy_quantized_snn)"; the text in parentheses is the visualize.py key.
MODEL_KEYS = {
    "QSNN (quantized_snn_lenet)": "quantized_snn_lenet",
    "QDSNN (branchy_quantized_snn)": "branchy_quantized_snn",
}


def convert(row):
    """Map one pair-evaluator row onto the evaluate.py CSV schema."""
    ac_ops = float(row["ac_ops"])
    mem_bytes = float(row["mem_bytes"])
    energy_j = float(row["energy_j"])

    return {
        "model_name": MODEL_KEYS[row["model"]],
        "device": "cpu",  # INT8 quantized kernels only run on CPU
        "accuracy": float(row["accuracy_pct"]) / 100.0,
        "energy": energy_j,
        "inference_time": float(row["latency_s_per_sample"]),
        "power": float(row["power_w"]),
        # Pure INT8 SNNs: all compute is accumulate, so there are no MACs and no
        # ANN-side memory traffic to report.
        "macs": 0,
        "ann_mem_accesses": 0,
        "ann_theoretical_energy": 0.0,
        "ac_ops": ac_ops,
        "spike_count": float(row["spike_count"]),
        "syops": ac_ops,
        "snn_mem_accesses": mem_bytes,
        "fair_theoretical_energy": energy_j,
        "projected_neuromorphic_energy": (ac_ops * E_AC_NEUROMORPHIC_J
                                          + mem_bytes * E_MEM_NEUROMORPHIC_J_PER_BYTE),
        # Time window, so the plot T-labels match what the pair was measured at.
        "num_steps": config["quantized_num_steps"],
        # Early-exit telemetry, carried through for the QDSNN.
        "avg_timesteps_to_exit": row.get("avg_timesteps_to_exit"),
        "avg_exit_depth": row.get("avg_exit_depth"),
        "exit1_percentage": row.get("exit1_pct"),
        "exit2_percentage": row.get("exit2_pct"),
        "final_exit_percentage": row.get("final_exit_pct"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=str, default="results/qsnn_vs_qdsnn.csv")
    parser.add_argument("--logs_dir", type=str, default="results/logs")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise SystemExit(f"{args.input_csv} not found -- run evaluate_quantized_pair.py first.")

    df = pd.read_csv(args.input_csv)
    os.makedirs(args.logs_dir, exist_ok=True)

    for _, row in df.iterrows():
        if row["model"] not in MODEL_KEYS:
            print(f"[WARN] Unrecognised model label {row['model']!r}, skipping.")
            continue
        converted = convert(row)
        out_path = os.path.join(args.logs_dir, f"{converted['model_name']}_eval.csv")
        pd.DataFrame([converted]).to_csv(out_path, index=False)
        print(f"Wrote {out_path}  (acc {converted['accuracy']*100:.2f}%, "
              f"{converted['energy']:.4e} J, {converted['power']:.4g} W)")


if __name__ == "__main__":
    main()
