"""
SPARQ resilience campaign driver.

Runs, for one architecture and one variant, a golden pass plus a sweep of
threat conditions (weight-memory bit flips and/or rho-budget input attacks),
and writes the six vulnerability components plus the modelled cost per
condition to CSV.

Run from the architecture's own directory:
  cd LeNet-Clean
  PYTHONPATH=. ../venv/Scripts/python.exe ../resilience/campaign.py \
      --arch lenet --variant qdsnn --n 512 --trials 3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fault_inject import (inject_int8, inject_fp32, inject_int8_simulated,   # noqa: E402
                          snapshot_int8, restore_int8,
                          snapshot_fp32, restore_fp32)
from registry import load, prepare_input                                     # noqa: E402
from cost import (SpikeCounter, load_topology, derive_topology,              # noqa: E402
                  run_cost, exposure)
from metrics import components, aggregate, W_FAULT, W_ATTACK                 # noqa: E402
from attack import worst_of_k                                                # noqa: E402


# --------------------------------------------------------------------------
def encode(spec, images, T):
    """Poisson-encode ONCE and reuse for every condition.

    The encoder is stochastic and its run-to-run noise is the same order as the
    effects being measured, so freezing one spike train removes it as a
    confounder entirely rather than merely seeding around it.
    """
    if not spec.is_spiking:
        return images, 0.0
    if spec.arch == "mlp":
        train = spec.model._encode_poisson(images.view(images.size(0), -1), T)
    elif spec.arch == "alexnet":
        from models.quantized_snn_alexnet import encode_poisson
        train = encode_poisson(images, T)
    else:
        from models.quantized_snn_lenet import encode_poisson
        train = encode_poisson(images, T)
    return train, float((train > 0).sum().item())


@torch.no_grad()
def forward(spec, model_input):
    """Unified inference. Returns (probs, exit_pts|None, exit_ts|None, lif_spikes)."""
    m = spec.model
    with SpikeCounter(m) as sc:
        if spec.is_dynamic:
            exit_outputs, final_output, exit_pts = m(model_input)
            logits = final_output.clone()
            for i, out in enumerate(exit_outputs):
                sel = (exit_pts == (i + 1))
                if sel.any():
                    logits[sel] = out[sel]
            exit_ts = m.last_exit_timesteps.clone()
        elif spec.is_spiking:
            logits = m(model_input).sum(dim=0)
            exit_pts = exit_ts = None
        else:
            logits = m(model_input)
            exit_pts = exit_ts = None
    return torch.softmax(logits, dim=1), exit_pts, exit_ts, list(sc.counts)


def observe(spec, topo, model_input, labels, input_spikes, T):
    """One condition -> everything the metric needs."""
    probs, exit_pts, exit_ts, lif = forward(spec, model_input)
    n = labels.size(0)
    if spec.is_spiking:
        c = run_cost(topo, input_spikes, lif, n, exit_pts, exit_ts, T)
    else:
        # A static network's cost is input-independent, so its inflation terms
        # are structurally zero. Fixed placeholders keep the ratio at exactly 1.
        c = dict(acs_per_sample=0.0, t_run=1.0, n_neurons_live=0.0,
                 energy_j=1.0, latency_s=1.0, power_w=float("nan"))
    return dict(pred=probs.argmax(1).numpy(), probs=probs.numpy(),
                label=labels.numpy(),
                exit=(exit_pts.numpy() if exit_pts is not None else None),
                exit_ts=(exit_ts.numpy() if exit_ts is not None else None),
                **c)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mlp", "lenet", "alexnet"])
    ap.add_argument("--variant", required=True, choices=["ann", "qsnn", "qdsnn"])
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresholds", type=float, nargs=2, default=None)
    ap.add_argument("--ber", type=float, nargs="*", default=None)
    ap.add_argument("--rho", type=float, nargs="*", default=None)
    ap.add_argument("--attack-k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = load(args.arch, args.variant, T=args.T)
    if args.thresholds and spec.is_dynamic:
        spec.model.prepare_for_inference(confidence_thresholds=list(args.thresholds))
        spec.thresholds = list(spec.model.confidence_thresholds)
    T = spec.T or 1
    print(f"[campaign] {spec}  thresholds={spec.thresholds}")

    from config import config
    from data_loader import get_data_loaders
    # The three modules' loaders disagree on both signature and arity:
    # MLP/LeNet take batch_size and return (train, test); AlexNet takes none and
    # returns (train, val, test, dataset). Select the test loader by position.
    try:
        loaders = get_data_loaders(batch_size=config["batch_size"], augment=False)
    except TypeError:
        loaders = get_data_loaders(augment=False)
    test_loader = loaders[2] if len(loaders) >= 4 else loaders[1]
    xs, ys, got = [], [], 0
    for x, y in test_loader:
        xs.append(x); ys.append(y); got += y.size(0)
        if got >= args.n:
            break
    images = prepare_input(spec, torch.cat(xs)[: args.n])
    labels = torch.cat(ys)[: args.n]

    # BER defaults track the bit budget: the MLP holds 524x more bits than LeNet
    # and absorbs proportionally more upsets before it degrades.
    ber_grid = args.ber if args.ber is not None else (
        [1e-4, 3e-4, 1e-3, 3e-3, 1e-2] if spec.n_bits < 1e7
        else [1e-3, 3e-3, 1e-2, 3e-2, 6e-2])
    rho_grid = args.rho if args.rho is not None else [0.005, 0.01, 0.02, 0.05]

    # Topology: use the Loihi export where one exists, otherwise profile the model.
    topo = load_topology(args.arch, args.variant if spec.is_spiking else "qdsnn")
    if topo is None:
        from registry import CATALOG
        exit_after = (CATALOG[args.arch][args.variant].get("exit_after")
                      or getattr(spec.model, "exit_after", None))
        topo = derive_topology(spec.model, images[:1], exit_after,
                               T=(T if spec.is_spiking else 1))
        print(f"[campaign] topology derived from the model: {topo.n_layers} layers, "
              f"{topo.n_neurons:,} spiking neurons")

    spike_train, in_spk = encode(spec, images, T)
    golden = observe(spec, topo, spike_train, labels, in_spk, T)
    g_acc = 100.0 * float((golden["pred"] == labels.numpy()).mean())
    X, phi_frac = exposure(topo, golden["exit"], spec.n_params,
                           8 if spec.is_int8 else 32)
    print(f"[golden] acc {g_acc:.2f}%  energy {golden['energy_j']:.3e} J  "
          f"t_run {golden['t_run']:.2f}  exposure {X:,.0f} bits (phi={phi_frac:.3f})")

    # Three fault paths: converted-INT8 modules, FP32 weights, and models that
    # deploy as INT8 but still hold FP32 tensors (AlexNet's fake-quant pair).
    if getattr(spec, "sim_int8", False):
        snap, inject, restore = (snapshot_fp32(spec.model),
                                 inject_int8_simulated, restore_fp32)
    elif spec.is_int8:
        snap, inject, restore = snapshot_int8(spec.model), inject_int8, restore_int8
    else:
        snap, inject, restore = snapshot_fp32(spec.model), inject_fp32, restore_fp32

    rows = [dict(arch=args.arch, variant=args.variant, thresholds=str(spec.thresholds),
                 T=T, threat="clean", intensity=0.0, flips=0.0, acc=g_acc,
                 D_A=0.0, D_M=0.0, I_E=0.0, I_L=0.0, U=0.0, H=0.0, score=0.0,
                 energy_j=golden["energy_j"], t_run=golden["t_run"],
                 acs=golden["acs_per_sample"], neurons=golden["n_neurons_live"],
                 exposure_bits=X, phi=phi_frac, n=args.n)]

    def record(threat, lam, obs_list, flips=np.nan):
        comps = [components(golden, o) for o in obs_list]
        agg = {k: float(np.nanmean([c[k] for c in comps])) for k in
               ("D_A", "D_M", "I_E", "I_L", "U", "H")}
        acc = float(np.mean([100.0 * (o["pred"] == labels.numpy()).mean() for o in obs_list]))
        w = W_FAULT if threat == "fault" else W_ATTACK
        rows.append(dict(arch=args.arch, variant=args.variant,
                         thresholds=str(spec.thresholds), T=T,
                         threat=threat, intensity=lam, flips=flips, acc=acc,
                         **agg, score=aggregate(agg, w),
                         energy_j=float(np.mean([o["energy_j"] for o in obs_list])),
                         t_run=float(np.mean([o["t_run"] for o in obs_list])),
                         # Raw op counts are recorded so energy can be re-priced
                         # under a different constant set without re-running the
                         # campaign (see sensitivity.py).
                         acs=float(np.mean([o["acs_per_sample"] for o in obs_list])),
                         neurons=float(np.mean([o["n_neurons_live"] for o in obs_list])),
                         exposure_bits=X, phi=phi_frac, n=args.n))
        print(f"  {threat:6s} {lam:<8.0e} acc {acc:6.2f}  D_A {agg['D_A']:6.2f}  "
              f"D_M {agg['D_M']:6.2f}  I_E {agg['I_E']:5.2f}  I_L {agg['I_L']:5.2f}  "
              f"U {agg['U']:5.2f}  H {agg['H']:5.2f}  -> {rows[-1]['score']:6.2f}")

    # ---- reliability: weight-memory bit flips ----
    print("\n[fault] weight-memory bit flips (per-bit BER)")
    for ber in ber_grid:
        obs, nfl = [], []
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + 1000 * t + int(ber * 1e9))
            nfl.append(inject(spec.model, ber, rng, snap))
            obs.append(observe(spec, topo, spike_train, labels, in_spk, T))
            restore(snap)
        record("fault", ber, obs, float(np.mean(nfl)))

    # ---- security: rho-budget input attack ----
    print("\n[attack] rho-budget worst-of-k input perturbation")

    def evaluate(imgs):
        tr, _ = encode(spec, imgs, T)
        return forward(spec, tr)[0]

    for rho in rho_grid:
        obs = []
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + 7000 * t + int(rho * 1e6))
            adv = worst_of_k(images, labels, rho, evaluate, rng, k=args.attack_k)
            tr, spk = encode(spec, adv, T)
            obs.append(observe(spec, topo, tr, labels, spk, T))
        record("attack", rho, obs)

    out = args.out or f"../resilience/results/{args.arch}_{args.variant}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
