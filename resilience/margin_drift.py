"""
Mechanism test: SIGNED Top-1/Top-2 margin drift under weight-memory faults.

The metric's `D_M` component reports margin *collapse*, clipped to [0,100], so a
margin that WIDENS under corruption is indistinguishable from one that does not
move. That clipping hides the very effect this test is after: prior joint
assessment on static CNNs reports the margin WIDENING as fault-induced
prediction changes accumulate, whereas the SPARQ spiking models show confidence
falling (RESILIENCE_NOTES.md F4).

Measuring the signed drift across all three variants separates the two candidate
explanations:

  * if the FP32 ANN widens while BOTH spiking variants narrow, the driver is
    SPIKING (a rate code integrated over T ticks degrades under corruption,
    where a single-shot logit polarises);
  * if only the QDSNN narrows, the driver is DYNAMIC ROUTING, not spiking.

Margins are measured on the unchanged-prediction subset, as in the metric, so
the comparison is on the same class and is not contaminated by relabelling.

Run from the architecture's directory:
  PYTHONPATH=. ../venv/Scripts/python.exe ../resilience/margin_drift.py --arch lenet
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fault_inject import (inject_int8, inject_fp32, inject_int8_simulated,  # noqa: E402
                          snapshot_int8, restore_int8,
                          snapshot_fp32, restore_fp32)
from registry import load, prepare_input                                  # noqa: E402
from metrics import margin                                                # noqa: E402
from campaign import encode, forward                                      # noqa: E402


def run(spec, model_input):
    probs, exit_pts, _, _ = forward(spec, model_input)
    return probs.numpy(), (exit_pts.numpy() if exit_pts is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mlp", "lenet", "alexnet"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ber", type=float, nargs="*", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from config import config
    from data_loader import get_data_loaders
    try:
        loaders = get_data_loaders(batch_size=config["batch_size"], augment=False)
    except TypeError:               # AlexNet: no batch_size, returns 4 values
        loaders = get_data_loaders(augment=False)
    test_loader = loaders[2] if len(loaders) >= 4 else loaders[1]
    xs, ys, got = [], [], 0
    for x, y in test_loader:
        xs.append(x); ys.append(y); got += y.size(0)
        if got >= args.n:
            break
    images_raw = torch.cat(xs)[: args.n]
    labels = torch.cat(ys)[: args.n]

    print(f"{'variant':<14}{'BER':>9}{'MRAD%':>8}{'margin_0':>10}"
          f"{'margin_f':>10}{'drift%':>9}  direction")
    print("-" * 72)

    for variant in ("ann", "qsnn", "qdsnn"):
        spec = load(args.arch, variant)
        images = prepare_input(spec, images_raw)
        T = spec.T or 1
        train, _ = encode(spec, images, T)

        g_probs, _ = run(spec, train)
        g_pred = g_probs.argmax(1)
        g_marg = margin(g_probs)

        if getattr(spec, "sim_int8", False):
            snap, inject, restore = (snapshot_fp32(spec.model),
                                     inject_int8_simulated, restore_fp32)
        elif spec.is_int8:
            snap, inject, restore = snapshot_int8(spec.model), inject_int8, restore_int8
        else:
            snap, inject, restore = snapshot_fp32(spec.model), inject_fp32, restore_fp32

        # Each variant needs a BER range that actually damages IT: the FP32 ANN
        # saturates five orders of magnitude below the INT8 models, so a shared
        # grid would compare a destroyed network against an untouched one.
        if args.ber:
            grid = args.ber
        elif not spec.is_int8:
            grid = [1e-8, 3e-8, 1e-7] if spec.n_params > 1e6 else [1e-5, 3e-5, 1e-4]
        else:
            grid = ([1e-3, 3e-3, 1e-2] if spec.n_bits > 1e7 else [3e-4, 1e-3, 3e-3])

        for ber in grid:
            d_list, m0_list, mf_list, mrad_list = [], [], [], []
            for t in range(args.trials):
                rng = np.random.default_rng(args.seed + 1000 * t + int(ber * 1e12))
                inject(spec.model, ber, rng, snap)
                f_probs, _ = run(spec, train)
                restore(snap)

                f_pred = f_probs.argmax(1)
                f_marg = margin(f_probs)
                keep = ((f_pred == g_pred) & np.isfinite(g_marg) & np.isfinite(f_marg))
                if keep.sum() < 5:
                    continue
                m0, mf = g_marg[keep].mean(), f_marg[keep].mean()
                m0_list.append(m0); mf_list.append(mf)
                d_list.append(100.0 * (mf - m0) / m0 if m0 > 0 else np.nan)
                mrad_list.append(100.0 * (f_pred != g_pred).mean())

            if not d_list:
                print(f"{variant:<14}{ber:>9.0e}   (too few surviving predictions)")
                continue
            drift = float(np.nanmean(d_list))
            print(f"{variant:<14}{ber:>9.0e}{np.mean(mrad_list):>8.2f}"
                  f"{np.mean(m0_list):>10.4f}{np.mean(mf_list):>10.4f}{drift:>9.2f}"
                  f"  {'WIDENS  <-- polarises' if drift > 0 else 'narrows'}")
        print()


if __name__ == "__main__":
    main()
