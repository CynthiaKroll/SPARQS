"""
PHASE 0 -- the pivotal experiment, for any architecture in the registry.

Question: under weight-memory bit flips, does the QDSNN's exit confidence go
DOWN (dynamism self-corrects -- threatened inputs look uncertain and are routed
deeper) or UP (faults produce confidently-wrong predictions that fire exits
prematurely, so dynamism actively hurts)?

JASPER observed the dangerous direction on ResNet-50: the Top-1/Top-2 margin
WIDENED as fault-induced prediction changes accumulated. This tests whether that
carries over to a spiking dynamic network, where the confidence in question is
an accumulated-over-time exit statistic rather than a static logit.

Run from the architecture's own directory:
  PYTHONPATH=. ../venv/Scripts/python.exe ../resilience/phase0.py --arch lenet
  PYTHONPATH=. ../venv/Scripts/python.exe ../resilience/phase0.py --arch mlp
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fault_inject import inject_int8, snapshot_int8, restore_int8   # noqa: E402
from registry import load, prepare_input                             # noqa: E402


@torch.no_grad()
def run(model, images):
    """Per-sample (exit point, confidence at the taken exit, prediction)."""
    exit_outputs, final_output, exit_pts = model(images)
    e1a, e2a = exit_outputs
    probs = {1: torch.softmax(e1a / model.temperature1, dim=1),
             2: torch.softmax(e2a / model.temperature2, dim=1),
             3: torch.softmax(final_output, dim=1)}

    B = images.size(0)
    conf = torch.zeros(B)
    pred = torch.zeros(B, dtype=torch.long)
    for pt, pr in probs.items():
        m = (exit_pts == pt)
        if m.any():
            p, a = pr[m].max(dim=1)
            conf[m] = p.float()
            pred[m] = a
    return exit_pts.clone(), conf, pred


def get_samples(n):
    from config import config
    from data_loader import get_data_loaders
    _, test_loader = get_data_loaders(batch_size=config["batch_size"], augment=False)
    xs, ys, got = [], [], 0
    for x, y in test_loader:
        xs.append(x); ys.append(y); got += y.size(0)
        if got >= n:
            break
    return torch.cat(xs)[:n], torch.cat(ys)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mlp", "lenet"])
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresholds", type=float, nargs=2, default=None,
                    help="Override the RL-learned exit thresholds (they are "
                         "inference-time parameters, so no retraining is needed)")
    ap.add_argument("--ber", type=float, nargs="*",
                    default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = load(args.arch, "qdsnn", T=args.T)
    model = spec.model
    if args.thresholds:
        model.prepare_for_inference(confidence_thresholds=list(args.thresholds))
        spec.thresholds = list(model.confidence_thresholds)
    print(f"[phase0] {spec}")
    print(f"[phase0] thresholds = {spec.thresholds}")

    images, labels = get_samples(args.n)
    images = prepare_input(spec, images)
    snap = snapshot_int8(model)

    torch.manual_seed(args.seed)   # fix the Poisson encoder across all conditions
    g_exit, g_conf, g_pred = run(model, images)
    g_ok = (g_pred == labels)
    g_bad = ~g_ok
    g_depth = g_exit.float().mean().item()
    print(f"\n[golden] acc {100 * g_ok.float().mean():.2f}%  "
          f"exit dist {[int((g_exit == k).sum()) for k in (1, 2, 3)]}  "
          f"mean conf {g_conf.mean():.4f}")

    hdr = (f"{'BER':>8} {'flips':>7} {'acc%':>7} {'MRAD%':>7} {'conf':>7} "
           f"{'conf|ok':>8} {'conf|bad':>9} {'e1%':>6} {'e2%':>6} {'e3%':>6} {'depth':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    print(f"{0.0:>8.0e} {0:>7} {100 * g_ok.float().mean():>7.2f} {0.0:>7.2f} "
          f"{g_conf.mean():>7.4f} {g_conf[g_ok].mean():>8.4f} "
          f"{(g_conf[g_bad].mean() if g_bad.any() else float('nan')):>9.4f} "
          f"{100 * (g_exit == 1).float().mean():>6.1f} "
          f"{100 * (g_exit == 2).float().mean():>6.1f} "
          f"{100 * (g_exit == 3).float().mean():>6.1f} {g_depth:>6.2f}")

    rows = []
    for ber in args.ber:
        acc, mrad, cf, cf_ok, cf_bad, ed, dep, nfl = ([] for _ in range(8))
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + 1000 * t + int(ber * 1e9))
            n = inject_int8(model, ber, rng, snap)
            torch.manual_seed(args.seed)   # same encoder noise as the golden run
            f_exit, f_conf, f_pred = run(model, images)
            restore_int8(snap)

            ok = (f_pred == labels); bad = ~ok
            acc.append(100.0 * ok.float().mean().item())
            mrad.append(100.0 * (f_pred != g_pred).float().mean().item())
            cf.append(f_conf.mean().item())
            cf_ok.append(f_conf[ok].mean().item() if ok.any() else np.nan)
            cf_bad.append(f_conf[bad].mean().item() if bad.any() else np.nan)
            ed.append([100.0 * (f_exit == k).float().mean().item() for k in (1, 2, 3)])
            dep.append(f_exit.float().mean().item())
            nfl.append(n)

        e = np.mean(ed, axis=0)
        print(f"{ber:>8.0e} {int(np.mean(nfl)):>7} {np.mean(acc):>7.2f} {np.mean(mrad):>7.2f} "
              f"{np.mean(cf):>7.4f} {np.nanmean(cf_ok):>8.4f} {np.nanmean(cf_bad):>9.4f} "
              f"{e[0]:>6.1f} {e[1]:>6.1f} {e[2]:>6.1f} {np.mean(dep):>6.2f}")
        rows.append(dict(ber=ber, mrad=np.mean(mrad), conf=np.mean(cf),
                         conf_bad=np.nanmean(cf_bad), depth=np.mean(dep)))

    hi = [r for r in rows if r["mrad"] > 5.0]
    print("\n" + "=" * 78)
    if not hi:
        print("VERDICT: no BER in the sweep produced >5% MRAD -- extend the grid.")
    else:
        d_conf = np.mean([r["conf"] for r in hi]) - g_conf.mean().item()
        d_bad = np.nanmean([r["conf_bad"] for r in hi]) - (
            g_conf[g_bad].mean().item() if g_bad.any() else np.nan)
        d_depth = np.mean([r["depth"] for r in hi]) - g_depth
        print(f"{spec.arch.upper()} QDSNN, rows with MRAD > 5%:")
        print(f"  mean confidence shift     : {d_conf:+.4f}")
        print(f"  confidence-on-WRONG shift : {d_bad:+.4f}")
        print(f"  mean exit-depth shift     : {d_depth:+.3f}  "
              f"(+ = routed deeper, - = premature exits)")
        print("  -> " + ("confidence FALLS: dynamism SELF-CORRECTS."
                         if d_conf < 0 else
                         "confidence RISES: confidently-wrong premature exits."))
    print("=" * 78)


if __name__ == "__main__":
    main()
