"""
DELAY-AWARE retrain of the LeNet QSNN (the "real fix" for LeNet on Loihi).

Runs in the MAIN env (py3.14 + torch, GPU). This is a SCAFFOLD: verify it trains
with --smoke, then launch the full run.

WHY: LeNet collapses on Loihi (best 71.5% vs 96% off-chip) because Loihi adds a
1-tick dendritic delay PER LAYER, and LeNet's small-fan-in conv neurons need many
ticks to integrate -- the delay truncates their effective window. The MLP (wide
fan-in) tolerates it; LeNet does not. The fix is to make the network LEARN under
that delay: during training, each stage consumes the PREVIOUS tick's output of
the stage below (a "pipelined" forward), exactly as the Lava sim runs it.

WHAT IT DOES:
  * Rebuilds QuantizedSNNLeNet, prepares QAT (same as train.py).
  * Splits the net into STAGES at every LIF (conv/linear+bn+lif = one stage; the
    final linear with no LIF is the readout stage).
  * Pipelined forward: stage k at tick t sees stage k-1's output from t-1, and
    stage 0 sees the input from t-1 (Dense buffers 1 tick). Membranes thread
    across ticks. Readout accumulates over all ticks.
  * Trains at T=16 (the sweep's LeNet operating point) with mixup + QAT.
  * Converts to INT8 and saves -> feed to export_lenet_qsnn.py as usual.

The trained checkpoint should recover LeNet's on-Loihi accuracy toward its ~96%
ceiling, because the delay is now in-distribution.

Usage (from LeNet-Clean):
  # verify it trains (a few dozen batches):
  PYTHONPATH=<LeNet-Clean> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/train_lenet_delay.py --smoke
  # full run:
  PYTHONPATH=<LeNet-Clean> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/train_lenet_delay.py --epochs 30 --T 16 \
      --out results/quantized_snn_lenet_delay_int8.pth
"""
import argparse
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
import snntorch as snn


def build_stages(model):
    """Split features_seq + classifier_seq into stages ending at each LIF.

    Returns (stages, readout_ops): stages is a list of op-lists each ending in a
    snn.Leaky; readout_ops is the trailing ops after the last LIF (final linear).
    Works AFTER prepare_qat -- fused conv-bn is a single module, dropped BN shows
    as Identity, both fine because we only split on snn.Leaky.
    """
    flat = list(model.features_seq) + list(model.classifier_seq)
    stages, cur = [], []
    for op in flat:
        cur.append(op)
        if isinstance(op, snn.Leaky):
            stages.append(cur); cur = []
    readout_ops = cur          # trailing ops with no LIF (the final linear)
    return stages, readout_ops


def run_stage(ops, x, mem, to_fp32):
    """Apply a stage's ops in order; thread the LIF membrane."""
    for op in ops:
        if isinstance(op, snn.Leaky):
            x = to_fp32(x)
            if mem is None or mem.shape != x.shape:
                mem = torch.zeros_like(x)
            x, mem = op(x, mem)
        else:
            x = op(x)
    return x, mem


def delay_forward(model, x, num_classes):
    """Pipelined forward with 1-tick inter-stage delay (Loihi-faithful).

    x: [T, B, C, H, W] spike train. Returns [T, B, num_classes] readout stack.
    """
    stages, readout_ops = model._delay_stages
    to_fp32 = model._to_fp32
    T = x.shape[0]; B = x.shape[1]
    dev = x.device
    n = len(stages)

    mems = [None] * n
    prev_out = [None] * n        # each stage's output at t-1
    prev_in = None               # input at t-1 (stage 0 is delayed too)
    zeros_logits = torch.zeros(B, num_classes, device=dev)
    out_rec = []

    for t in range(T):
        xt = model.quant(x[t])
        new_out = [None] * n
        for k, ops in enumerate(stages):
            src = prev_in if k == 0 else prev_out[k - 1]
            if src is None:
                continue                       # pipeline not filled to this stage yet
            new_out[k], mems[k] = run_stage(ops, src, mems[k], to_fp32)

        # readout: apply trailing ops to the last stage's output (delayed 1 tick)
        last = prev_out[n - 1]
        if last is not None:
            r = last
            for op in readout_ops:
                r = op(r)
            out_rec.append(model.dequant(r))
        else:
            out_rec.append(zeros_logits)

        prev_out = new_out
        prev_in = xt

    return torch.stack(out_rec, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=16)
    # match LeNet-Clean/config.py: epochs=20, early_stopping_patience=10
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=10,
                    help='early-stopping patience (0 disables); mirrors config.py')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--out', default='results/quantized_snn_lenet_delay_int8.pth')
    ap.add_argument('--smoke', action='store_true',
                    help='run ~30 train batches + a val pass and exit (verify only)')
    args = ap.parse_args()

    from config import config
    from data_loader import get_data_loaders
    from models.quantized_snn_lenet import (QuantizedSNNLeNet, get_quant_engine,
                                            convert_quantized_snn_lenet)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nc = config['num_classes']
    print(f"[delay-train] device={dev}  T={args.T}  epochs={args.epochs}  "
          f"{'SMOKE' if args.smoke else 'FULL'}")

    model = QuantizedSNNLeNet(num_classes=nc, beta=config['snn']['beta'], time_steps=args.T)
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
    model.to(dev)

    stages, readout = build_stages(model)
    model._delay_stages = (stages, readout)
    print(f"[delay-train] {len(stages)} spiking stages + readout "
          f"({sum(1 for _ in readout)} trailing ops) => pipeline depth {len(stages)+1}")

    from models.quantized_snn_lenet import encode_poisson
    train_loader, val_loader = get_data_loaders(batch_size=args.batch)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    def run_epoch(train, cap=None):
        model.train(train)
        tot = cor = 0; loss_sum = 0.0
        loader = train_loader if train else val_loader
        for bi, (img, lab) in enumerate(loader):
            if cap and bi >= cap:
                break
            img, lab = img.to(dev), lab.to(dev)
            spk = encode_poisson(img, args.T)
            with torch.set_grad_enabled(train):
                out = delay_forward(model, spk, nc).sum(0)
                loss = crit(out, lab)
            if train:
                opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(loss) and torch.isfinite(gn):
                    opt.step()
            loss_sum += loss.item() * lab.size(0)
            cor += (out.argmax(1) == lab).sum().item(); tot += lab.size(0)
        return loss_sum / max(tot, 1), 100.0 * cor / max(tot, 1)

    if args.smoke:
        tl, ta = run_epoch(True, cap=30)
        vl, va = run_epoch(False, cap=20)
        print(f"[smoke] after 30 train batches: train_loss={tl:.3f} train_acc={ta:.1f}%")
        print(f"[smoke] val (20 batches): loss={vl:.3f} acc={va:.1f}%")
        print("[smoke] OK -- delay-aware forward trains and backprops. "
              "Launch the full run to produce a checkpoint.")
        return

    best = 0.0
    patience_counter = 0
    for ep in range(args.epochs):
        tl, ta = run_epoch(True)
        vl, va = run_epoch(False)
        sched.step()
        print(f"epoch {ep+1}/{args.epochs}  train_loss={tl:.3f} train_acc={ta:.1f}%  "
              f"val_acc={va:.2f}%")
        if va > best:
            best = va
            patience_counter = 0
            int8 = convert_quantized_snn_lenet(copy.deepcopy(model).eval().cpu())
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
            torch.save(int8.state_dict(), args.out)
            print(f"   saved INT8 (val {va:.2f}%) -> {args.out}")
        else:
            # early stopping, same criterion as LeNet-Clean/train.py
            patience_counter += 1
            if args.patience and patience_counter >= args.patience:
                print(f"[delay-train] early stop at epoch {ep+1} "
                      f"(no val improvement for {args.patience} epochs)")
                break
    print(f"[delay-train] done. best val_acc={best:.2f}%  checkpoint={args.out}")


if __name__ == '__main__':
    main()
