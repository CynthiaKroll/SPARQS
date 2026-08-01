"""
PHASE 0 -- the pivotal experiment.

Question: under weight-memory bit flips, does the QDSNN's exit confidence go
DOWN (dynamism self-corrects: threatened inputs look uncertain and get routed
deeper) or UP (faults produce confidently-wrong predictions that fire early
exits prematurely, so dynamism actively hurts)?

JASPER observed the dangerous direction on ResNet-50: the Top-1/Top-2 margin
WIDENED as fault-induced prediction changes accumulated. This script tests
whether that carries over to a spiking dynamic network, where the confidence in
question is an accumulated-over-time exit statistic rather than a static logit.

The answer decides the narrative of the whole paper, so it runs before the
harness is built out.

Run from LeNet-Clean:
  PYTHONPATH=. ../venv/Scripts/python.exe ../resilience/phase0_lenet.py
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.ao.quantization as tq
import snntorch as snn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fault_inject import inject_int8, snapshot_int8, restore_int8, bit_budget


def build_qdsnn(ckpt, T, reset):
    from config import config
    from models.branchy_quantized_snn import BranchyQuantizedSNN
    from models.quantized_snn_lenet import get_quant_engine

    model = BranchyQuantizedSNN(num_classes=config["num_classes"],
                                time_window=T, beta=config["snn"]["beta"])
    torch.backends.quantized.engine = get_quant_engine()
    model.fuse_model()
    model.qconfig = tq.QConfig(
        activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=tq.MinMaxObserver.with_args(dtype=torch.qint8,
                                           qscheme=torch.per_tensor_symmetric))
    for _, m in model.named_modules():
        if isinstance(m, snn.Leaky):
            m.qconfig = None
    tq.prepare_qat(model, inplace=True)
    model.eval().cpu()
    model = tq.convert(model, inplace=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))  # strict
    model.eval().cpu()

    # The delay-aware QDSNN was retrained under HARD reset to match Loihi.
    if reset == "zero":
        for _, m in model.named_modules():
            if isinstance(m, snn.Leaky):
                m.reset_mechanism = "zero"
    return model


@torch.no_grad()
def run(model, images, labels, T):
    """Returns per-sample (exit_pt, confidence at the taken exit, prediction)."""
    exit_outputs, final_output, exit_pts = model(images)
    e1a, e2a = exit_outputs
    c1 = torch.softmax(e1a / model.temperature1, dim=1)
    c2 = torch.softmax(e2a / model.temperature2, dim=1)
    cf = torch.softmax(final_output, dim=1)

    B = images.size(0)
    conf = torch.zeros(B)
    pred = torch.zeros(B, dtype=torch.long)
    for pt, probs in ((1, c1), (2, c2), (3, cf)):
        m = (exit_pts == pt)
        if m.any():
            p, a = probs[m].max(dim=1)
            conf[m] = p.float()
            pred[m] = a
    return exit_pts.clone(), conf, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/branchy_quantized_snn_delay_hr_int8.pth")
    ap.add_argument("--agent", default="results/branchy_quantized_snn_delay_hr_int8_agent.pkl")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--reset", choices=["zero", "subtract"], default="zero")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ber", type=float, nargs="*",
                    default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from config import config
    from data_loader import get_data_loaders
    from models.rl_environment import QLearningAgent

    model = build_qdsnn(args.ckpt, args.T, args.reset)
    if os.path.exists(args.agent):
        agent = QLearningAgent(n_exits=model.n_exits)
        agent.load(args.agent)
        model.prepare_for_inference(rl_agent=agent)
    print(f"[phase0] thresholds = {model.confidence_thresholds}  T={args.T}  reset={args.reset}")

    n_params, n_bits = bit_budget(model, is_int8=True)
    print(f"[phase0] INT8 weight memory: {n_params:,} weights = {n_bits:,} bits")

    _, test_loader = get_data_loaders(batch_size=config["batch_size"], augment=False)
    imgs, labs = [], []
    got = 0
    for x, y in test_loader:
        imgs.append(x); labs.append(y); got += y.size(0)
        if got >= args.n:
            break
    images = torch.cat(imgs)[: args.n]
    labels = torch.cat(labs)[: args.n]

    snap = snapshot_int8(model)

    # ---- golden ----
    torch.manual_seed(args.seed)  # fix the Poisson encoder across conditions
    g_exit, g_conf, g_pred = run(model, images, labels, args.T)
    g_correct = (g_pred == labels)
    print(f"\n[golden] acc {100.0 * g_correct.float().mean():.2f}%  "
          f"exit dist {[int((g_exit == k).sum()) for k in (1, 2, 3)]}  "
          f"mean conf {g_conf.mean():.4f}")

    hdr = (f"{'BER':>8} {'flips':>7} {'acc%':>7} {'MRAD%':>7} {'conf':>7} "
           f"{'conf|ok':>8} {'conf|bad':>9} {'e1%':>6} {'e2%':>6} {'e3%':>6} {'depth':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))

    # golden row
    gb = ~g_correct
    print(f"{0.0:>8.0e} {0:>7} {100 * g_correct.float().mean():>7.2f} {0.0:>7.2f} "
          f"{g_conf.mean():>7.4f} {g_conf[g_correct].mean():>8.4f} "
          f"{(g_conf[gb].mean() if gb.any() else float('nan')):>9.4f} "
          f"{100 * (g_exit == 1).float().mean():>6.1f} {100 * (g_exit == 2).float().mean():>6.1f} "
          f"{100 * (g_exit == 3).float().mean():>6.1f} {g_exit.float().mean():>6.2f}")

    rows = []
    for ber in args.ber:
        acc, mrad, cf_all, cf_ok, cf_bad, ed, dep, nfl = ([] for _ in range(8))
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + 1000 * t + int(ber * 1e9))
            n = inject_int8(model, ber, rng, snap)
            torch.manual_seed(args.seed)  # same encoder noise as the golden run
            f_exit, f_conf, f_pred = run(model, images, labels, args.T)
            restore_int8(snap)

            ok = (f_pred == labels)
            bad = ~ok
            acc.append(100.0 * ok.float().mean().item())
            mrad.append(100.0 * (f_pred != g_pred).float().mean().item())
            cf_all.append(f_conf.mean().item())
            cf_ok.append(f_conf[ok].mean().item() if ok.any() else np.nan)
            cf_bad.append(f_conf[bad].mean().item() if bad.any() else np.nan)
            ed.append([100.0 * (f_exit == k).float().mean().item() for k in (1, 2, 3)])
            dep.append(f_exit.float().mean().item())
            nfl.append(n)

        e = np.mean(ed, axis=0)
        print(f"{ber:>8.0e} {int(np.mean(nfl)):>7} {np.mean(acc):>7.2f} {np.mean(mrad):>7.2f} "
              f"{np.mean(cf_all):>7.4f} {np.nanmean(cf_ok):>8.4f} {np.nanmean(cf_bad):>9.4f} "
              f"{e[0]:>6.1f} {e[1]:>6.1f} {e[2]:>6.1f} {np.mean(dep):>6.2f}")
        rows.append(dict(ber=ber, flips=float(np.mean(nfl)), acc=np.mean(acc),
                         mrad=np.mean(mrad), conf=np.mean(cf_all),
                         conf_ok=np.nanmean(cf_ok), conf_bad=np.nanmean(cf_bad),
                         e1=e[0], e2=e[1], e3=e[2], depth=np.mean(dep)))

    # ---- verdict ----
    base = g_conf.mean().item()
    hi = [r for r in rows if r["mrad"] > 5.0]
    print("\n" + "=" * 78)
    if not hi:
        print("VERDICT: no BER in the sweep produced >5% MRAD -- extend the grid.")
    else:
        d_conf = np.mean([r["conf"] - base for r in hi])
        d_bad = np.nanmean([r["conf_bad"] for r in hi]) - (
            g_conf[gb].mean().item() if gb.any() else np.nan)
        d_depth = np.mean([r["depth"] for r in hi]) - g_exit.float().mean().item()
        print(f"Over the rows with MRAD > 5%:")
        print(f"  mean confidence shift          : {d_conf:+.4f}")
        print(f"  confidence-on-WRONG shift      : {d_bad:+.4f}")
        print(f"  mean exit-depth shift          : {d_depth:+.3f}  "
              f"(+ = routed deeper, - = premature exits)")
        if d_conf < 0:
            print("  -> confidence FALLS under faults: dynamism SELF-CORRECTS.")
        else:
            print("  -> confidence RISES under faults: confidently-wrong premature exits.")
    print("=" * 78)


if __name__ == "__main__":
    main()
