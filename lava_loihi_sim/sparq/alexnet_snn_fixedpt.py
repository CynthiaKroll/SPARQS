"""
AlexNet FP32-SNN on Loihi-2: PTQ -> bit-accurate fixed-point sim (torch replica).

The FP32 SNN (fp32_snn_alexnet.pth, trained with --no_qat) has FP32 weights, which
Loihi cannot run (fixed-point hardware). We post-training-quantize (PTQ) it to int8
and run it through the SAME `Stage`/`run` fixed-point Loihi-2 model as the QSNN
(alexnet_loihi_fixedpt.py), so its Ops/energy are directly comparable to the QSNN /
QDSNN rows -- unlike the analytical 369.9 mJ (FP32-memory, off-chip cost model).

Expect it to land ~on the QSNN: same backbone + same Poisson encoding, PTQ vs QAT
weights (PTQ usually ~1-2 pp lower accuracy). It exists for completeness; on Loihi
the FP32-vs-int8 distinction disappears (everything is int8).

Usage (from AlexNet/):
  PYTHONPATH=. ../venv/Scripts/python.exe \
    ../lava_loihi_sim/sparq/alexnet_snn_fixedpt.py --n 200 [--delay]
"""
import argparse
import os
import sys

import torch
import torch.ao.quantization as tq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alexnet_loihi_fixedpt import extract_stages, Stage, run  # reuse the QSNN sim


def ptq_int8(fp32_ckpt, dev, T, calib_batches=30):
    """Load the FP32 SNN, PTQ it to int8 via the clean-checkpoint calibration recipe
    (floor near-zero BN gammas so the QAT ConvBn train-forward can't do 0/0;
    calibrate with fake-quant DISABLED so observers collect clean ranges), convert."""
    from config import config
    from data_loader import get_data_loaders
    from models.quantized_snn_alexnet import (
        QuantizedSNNAlexNet, prepare_quantized_snn_alexnet_for_qat, encode_poisson)

    m = QuantizedSNNAlexNet(num_classes=config["num_classes"], beta=config["beta"],
                            time_steps=T)
    m.load_state_dict(torch.load(fp32_ckpt, map_location="cpu"))   # plain FP32 weights
    m = prepare_quantized_snn_alexnet_for_qat(m)                   # fuse ConvBn + observers

    with torch.no_grad():                                          # floor ~zero BN gammas
        for name, mod in m.named_modules():
            g = getattr(mod, "weight", None)
            if torch.is_tensor(g) and torch.is_tensor(getattr(mod, "running_mean", None)):
                near0 = g.abs() < 1e-3
                if near0.any():
                    g[near0] = 1e-3

    tq.disable_fake_quant(m)                                       # clean FP32 forward
    train_loader, _, _, _ = get_data_loaders(distributed=False, augment=False)
    m.to(dev).train()
    with torch.no_grad():
        for i, (img, _) in enumerate(train_loader):
            if i >= calib_batches:
                break
            m(encode_poisson(img.to(dev), T))
    tq.enable_fake_quant(m)
    m.eval().cpu()
    return tq.convert(m, inplace=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--delay", action="store_true")
    ap.add_argument("--ckpt", default="results/fp32_snn_alexnet.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from config import config
    from models.quantized_snn_alexnet import get_quant_engine, encode_poisson
    T = config["num_steps"]
    torch.backends.quantized.engine = get_quant_engine()

    print(f"[alexnet-snn-fixedpt] PTQ of {args.ckpt} -> int8 ...")
    m = ptq_int8(args.ckpt, args.device, T)
    stages = [Stage(o, args.device) for o in extract_stages(m)]
    print(f"[alexnet-snn-fixedpt] {len(stages)} stages "
          f"({sum(s.has_lif for s in stages)} LIF + 1 readout), "
          f"delay={'ON' if args.delay else 'OFF'}, T={T}, dev={args.device}")

    import torchvision, torchvision.transforms as TT
    # Match TRAINING/off-chip normalization (data_loader.py), not the 0.247 variant.
    tfm = TT.Compose([TT.ToTensor(), TT.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))])
    ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))
    correct = 0; ac_run = 0.0; sp_run = None
    with torch.no_grad():
        for i in range(N):
            img, lab = ds[i]
            spk = encode_poisson(img.unsqueeze(0), T).to(args.device)
            logits, st = run(stages, spk, args.delay)
            correct += int(logits.argmax(1).item() == lab)
            ac_run += sum(st["ac"])
            sp_run = st["spikes"] if sp_run is None else [a + b for a, b in zip(sp_run, st["spikes"])]
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{N}  acc={correct/(i+1)*100:.2f}%  avg_ACs={ac_run/(i+1):.3e}")
    print(f"\n[alexnet-snn-fixedpt] N={N}  acc={correct/N*100:.2f}%  (ref FP32-SNN off-chip 67.21%)  "
          f"delay={'ON' if args.delay else 'OFF'}  t_run={st['t_run']}")
    print(f"[alexnet-snn-fixedpt] mean ACs/sample = {ac_run/N:.4e}  "
          f"(expect ~QSNN 5.8e8; feed to power_report.py for energy)")
    print(f"[alexnet-snn-fixedpt] mean spikes/stage = {[int(s/N) for s in sp_run]}")


if __name__ == "__main__":
    main()
