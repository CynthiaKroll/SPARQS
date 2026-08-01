"""
Export the SPARQ QDSNN (`branchy_quantized_snn_mlp`) for the Lava/Loihi leg.

Runs in the MAIN env (py3.14 + torch).

WHY THIS ONE IS DIFFERENT
-------------------------
Loihi executes a STATIC graph for a fixed number of ticks. There is no
"stop early because this branch is confident enough" -- the softmax-and-compare
that drives SPARQ's early exit is not a neuromorphic primitive. So we use
PATH DECOMPOSITION (see [[loihi-implementation]]):

  1. PyTorch decides the exits (the real, trained behaviour) and we RECORD the
     per-sample (exit_point, exit_timestep) distribution.
  2. Loihi simulates the three static sub-networks that the exits select:
        path 1 : fc1 -> lif1                     (+ exit1 head)
        path 2 : fc1..fc3 -> lif1..lif3          (+ exit1, exit2 heads)
        path 3 : fc1..fc8 (full backbone)        (+ exit1, exit2 heads)
  3. Expected cost = sum_i P(exit=i) * E_loihi(path_i, ticks_i).

Reported honestly as "per-path hardware-simulated cost weighted by the measured
exit distribution", NOT as an end-to-end hardware run.

Cost accounting per _forward_inference (verified against the source):
  every active tick a sample pays  fc1 + exit1
  if it did not exit at 1, also    fc2 + fc3 + exit2
  if it did not exit at 2, also    fc4 + fc5 + fc6 + fc7 + fc8
Exit logits accumulate with decay d=0.9 (NOT the LIF beta).

Usage (from the 5-layer-MLP dir):
  PYTHONPATH=<5-layer-MLP> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/export_mlp_qdsnn.py --n 200 --T 4 \
      --out ../lava_loihi_sim/exports/mlp_qdsnn.npz
"""
import argparse
import os
import numpy as np
import torch
import snntorch as snn

EXIT_DECAY = 0.9          # `d` in _forward_inference


def quantize_int8_per_tensor(W):
    max_abs = float(W.abs().max().clamp(min=1e-12))
    scale = max_abs / 127.0
    w_int = torch.clamp(torch.round(W / scale), -127, 127)
    return w_int.numpy().astype(np.int8), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/branchy_quantized_snn_mlp.pth')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--out', default='../lava_loihi_sim/exports/mlp_qdsnn.npz')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from config import config
    from evaluate import load_model
    from utils import encode_poisson_rate

    model = load_model('branchy_quantized_snn_mlp', args.ckpt, 'cpu', time_steps=args.T)
    model.eval().cpu()
    thresholds = list(model.confidence_thresholds)
    print(f"[export] confidence_thresholds = {thresholds}")

    # --- backbone: 8 quantized Linears fc1..fc8 ---
    layers = []
    for i in range(1, 9):
        m = getattr(model, f'fc{i}')
        w = m.weight()
        if not getattr(w, 'is_quantized', False):
            raise RuntimeError(f"fc{i} is not quantized -- checkpoint/skeleton mismatch")
        b = m.bias()
        layers.append(dict(
            name=f'fc{i}',
            w_int=w.int_repr().numpy().astype(np.int8),
            scale=float(w.q_scale()),
            bias=(b.detach().numpy().astype(np.float32) if b is not None
                  else np.zeros(w.shape[0], np.float32)),
        ))

    # --- exit heads: FP32 (qconfig=None) -> quantize the same way ---
    exits = []
    for name, head in (('exit1', model.exit1), ('exit2', model.exit2)):
        lin = head.fc
        w_int, scale = quantize_int8_per_tensor(lin.weight.data.clone())
        exits.append(dict(
            name=name, w_int=w_int, scale=scale,
            bias=(lin.bias.data.clone().numpy().astype(np.float32)
                  if lin.bias is not None else np.zeros(w_int.shape[0], np.float32)),
        ))

    lifs = [m for m in model.modules() if isinstance(m, snn.Leaky)]
    beta = float(lifs[0].beta); thr = float(lifs[0].threshold)
    print(f"[export] backbone {len(layers)} int8 Linears, {len(exits)} exit heads, "
          f"{len(lifs)} LIF, beta={beta:.3f}, thr={thr:.3f}, T={args.T}")
    for L in layers + exits:
        print(f"   {L['name']:<7} {L['w_int'].shape}  scale={L['scale']:.4g}")

    # --- test data ---
    import torchvision
    from torchvision import transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))
    images = torch.stack([ds[i][0] for i in range(N)])
    labels = torch.tensor([ds[i][1] for i in range(N)])
    spikes = encode_poisson_rate(images, args.T)          # [T, N, 784]

    # --- PyTorch inference: the REAL exit decisions ---
    with torch.no_grad():
        exit_outputs, final_out, exit_pts = model(spikes)
    e1a, e2a = exit_outputs
    exit_pts = exit_pts.cpu()
    exit_ts = model.last_exit_timesteps.cpu()

    # compose per-sample prediction by exit point (paper-correct metric)
    composed = torch.where(
        (exit_pts == 1).unsqueeze(1), e1a,
        torch.where((exit_pts == 2).unsqueeze(1), e2a, final_out))
    pred = composed.argmax(1)
    acc = (pred == labels).float().mean().item()

    counts = {int(k): int((exit_pts == k).sum()) for k in (1, 2, 3)}
    print(f"[export] QDSNN acc (composed by exit) = {acc*100:.2f}%")
    print(f"[export] exit distribution = {counts}  "
          f"({', '.join(f'exit{k}:{v/N*100:.1f}%' for k, v in counts.items())})")
    print(f"[export] avg exit timestep = {exit_ts.float().mean():.2f}/{args.T}, "
          f"avg depth = {exit_pts.float().mean():.2f}/3")
    for k in (1, 2, 3):
        m = exit_pts == k
        if m.any():
            print(f"   exit{k}: n={int(m.sum())}  acc={(pred[m]==labels[m]).float().mean()*100:.1f}%  "
                  f"avg_ts={exit_ts[m].float().mean():.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = dict(
        n_layers=len(layers), n_exits=len(exits),
        layer_names=np.array([L['name'] for L in layers]),
        exit_names=np.array([e['name'] for e in exits]),
        beta=beta, threshold=thr, n_lif=7, T=args.T,
        readout='integrator', exit_decay=EXIT_DECAY,
        confidence_thresholds=np.array(thresholds, np.float32),
        # exit heads attach after these backbone layer indices (0-based):
        exit_after=np.array([0, 2], np.int64),   # exit1 after fc1(lif1), exit2 after fc3(lif3)
        spikes=spikes.permute(1, 0, 2).contiguous().numpy().astype(np.uint8),
        labels=labels.numpy().astype(np.int64),
        exit_points=exit_pts.numpy().astype(np.int64),
        exit_timesteps=exit_ts.numpy().astype(np.int64),
        golden_pred=pred.numpy().astype(np.int64),
        golden_acc=np.float32(acc), int8_acc=np.float32(acc),
        golden_fire=np.zeros(7, np.float32),
    )
    for k, L in enumerate(layers):
        blob[f'w{k}'] = L['w_int']; blob[f'scale{k}'] = np.float32(L['scale']); blob[f'bias{k}'] = L['bias']
    for j, E in enumerate(exits):
        blob[f'ew{j}'] = E['w_int']; blob[f'escale{j}'] = np.float32(E['scale']); blob[f'ebias{j}'] = E['bias']
    np.savez_compressed(args.out, **blob)
    print(f"[export] wrote {args.out}  (spikes {blob['spikes'].shape}, N={N})")


if __name__ == '__main__':
    main()
