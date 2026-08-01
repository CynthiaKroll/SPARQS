"""
Diagnose WHY the QDSNN never early-exits.

The threshold sweep (0.5 .. 0.8) produced exit1%=exit2%=0 at every setting, so
the exit policy is not the problem -- the exit heads simply never reach the
required confidence. This script measures, on the test set:

  1. Each exit head's STANDALONE accuracy (how good is it, ignoring confidence?)
  2. The DISTRIBUTION of its max-softmax confidence (how confident does it get?)
  3. For a range of thresholds, what fraction WOULD exit and at what accuracy.

That separates the two possible causes:
  - heads are inaccurate      -> need a stronger ExitClassifier / more training
  - heads are accurate but under-confident -> need temperature/threshold recalibration

Exits are disabled during measurement (thresholds forced to 1.1, which no
softmax max can exceed) so the accumulated exit logits are collected for EVERY
sample over the full time window, exactly as the exit decision would see them.

Usage:
  python diagnose_exits.py --qdsnn_path results_willi/branchy_quantized_snn_qat.pth
  python diagnose_exits.py --qdsnn_path <path> --max_batches 20   # quick pass
"""
import argparse

import torch

from config import config
from data_loader import get_data_loaders
from evaluate_quantized_pair import load_qdsnn


def summarize(name, logits, temperature, labels):
    """Accuracy + confidence distribution for one exit head."""
    conf = torch.softmax(logits / temperature, dim=1).max(dim=1)[0]
    pred = logits.argmax(dim=1)
    acc = 100.0 * (pred == labels).float().mean().item()
    qs = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    pcts = torch.quantile(conf.float(), qs)
    print(f"\n--- {name} ---")
    print(f"  standalone accuracy : {acc:.2f}%")
    print(f"  confidence  min/med/max : "
          f"{pcts[0]:.4f} / {pcts[2]:.4f} / {pcts[6]:.4f}")
    print(f"  confidence  p25/p75/p90/p99 : "
          f"{pcts[1]:.4f} / {pcts[3]:.4f} / {pcts[4]:.4f} / {pcts[5]:.4f}")
    return conf, pred, acc


def main():
    ap = argparse.ArgumentParser(description="Diagnose QDSNN early-exit behaviour")
    ap.add_argument("--qdsnn_path", type=str,
                    default="results/branchy_quantized_snn_qat.pth")
    ap.add_argument("--time_steps", type=int, default=config["num_steps"])
    ap.add_argument("--max_batches", type=int, default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | T={args.time_steps}")

    _, _, test_loader, _ = get_data_loaders(distributed=False, augment=False)

    # thresholds=1.1 -> no softmax max can exceed it -> nothing exits, so every
    # sample's exit logits accumulate over the full time window.
    model = load_qdsnn(args.qdsnn_path, agent_path=None, thresholds=[1.1, 1.1],
                       time_steps=args.time_steps, device=device)

    e1_all, e2_all, fa_all, y_all = [], [], [], []
    with torch.no_grad():
        for bi, (images, labels) in enumerate(test_loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            exit_outputs, final_output, exit_pts = model(images)
            e1_all.append(exit_outputs[0].cpu())
            e2_all.append(exit_outputs[1].cpu())
            fa_all.append(final_output.cpu())
            y_all.append(labels.cpu())
            if exit_pts is not None and (exit_pts != 3).any():
                print("[WARN] a sample exited early despite threshold 1.1 -- "
                      "exit gating is not behaving as assumed")

    e1 = torch.cat(e1_all); e2 = torch.cat(e2_all)
    fa = torch.cat(fa_all); y = torch.cat(y_all)
    t1 = model.temperature1.detach().cpu()
    t2 = model.temperature2.detach().cpu()
    print(f"\nSamples: {y.numel()} | learned temperatures: "
          f"T1={t1.item():.4f}  T2={t2.item():.4f}")
    print(f"Accumulated-logit magnitude (mean |logit|): "
          f"exit1={e1.abs().mean():.4f}  exit2={e2.abs().mean():.4f}  "
          f"final={fa.abs().mean():.4f}")

    conf1, pred1, _ = summarize("EXIT 1 (after conv block 2)", e1, t1, y)
    conf2, pred2, _ = summarize("EXIT 2 (after conv block 4)", e2, t2, y)
    summarize("FINAL EXIT (fc3)", fa, torch.ones(1), y)

    # What would actually happen at each candidate threshold?
    print("\n--- Would-be exit rates by threshold "
          "(exit1 checked first, exit2 on the remainder) ---")
    print(f"{'thr':>6} {'exit1%':>8} {'exit2%':>8} {'final%':>8} {'overall acc%':>13}")
    correct_final = (fa.argmax(1) == y)
    for thr in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        take1 = conf1 > thr
        take2 = (~take1) & (conf2 > thr)
        rest = ~(take1 | take2)
        correct = ((take1 & (pred1 == y)).sum()
                   + (take2 & (pred2 == y)).sum()
                   + (rest & correct_final).sum()).item()
        n = y.numel()
        print(f"{thr:>6.2f} {100.0 * take1.sum().item() / n:>8.2f} "
              f"{100.0 * take2.sum().item() / n:>8.2f} "
              f"{100.0 * rest.sum().item() / n:>8.2f} "
              f"{100.0 * correct / n:>13.2f}")

    print("\nRead this as: if the highest confidence any sample reaches is well "
          "below 0.5, no threshold in the swept range can ever fire an exit -- "
          "the heads need to be stronger or the confidence recalibrated.")


if __name__ == "__main__":
    main()
