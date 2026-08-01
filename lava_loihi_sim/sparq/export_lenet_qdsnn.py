"""
Export the SPARQ LeNet QDSNN (`branchy_quantized_snn_int8`) for the Lava leg.
MAIN env (py3.14 + torch). Conv/AvgPool unrolled to Dense -- see lenet_common.py.

Backbone (identical decomposition to the LeNet QSNN):
  L0 conv1          784  -> 4704  + LIF1     <- exit1 attaches here
  L1 pool1+conv2    4704 -> 1600  + LIF2     <- exit2 attaches here
  L2 pool2+fc1      1600 -> 120   + LIF_fc1
  L3 fc2            120  -> 84    + LIF_fc2
  L4 fc3            84   -> 10    (integrator readout)

Exit heads (ExitClassifier = AdaptiveAvgPool2d((1,1)) -> Linear(C,10)):
  exit1 on the (6,28,28) LIF1 map, exit2 on the (16,10,10) LIF2 map.
  The model applies them to the POOLED map (p1/p2), but global-average-pooling a
  2x2-average-pooled map equals global-average-pooling the unpooled map whenever
  the dims divide evenly (28/2=14, 10/2=5) -- so folding the head directly onto
  the LIF output is EXACT and avoids materialising the pooled tensor.

Exit logits accumulate with decay d=0.9 (NOT the LIF beta).
Path decomposition + the measured exit distribution: see lava_build_qdsnn.py.

Usage (from LeNet-Clean):
  PYTHONPATH=<LeNet-Clean> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/export_lenet_qdsnn.py --n 200 --T 4 \
      --out ../lava_loihi_sim/exports/lenet_qdsnn.npz
"""
import argparse
import os
import sys
import numpy as np
import torch
import snntorch as snn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lenet_common import (conv2d_to_dense, avgpool_matrix,
                          global_avgpool_linear_to_dense, pack_layer)

EXIT_DECAY = 0.9


def dq(m):
    """Dequantized (W, b) of a converted INT8 Conv2d/Linear."""
    w = m.weight()
    W = w.int_repr().float() * w.q_scale()
    b = m.bias()
    return W, (b.detach().float() if b is not None else torch.zeros(W.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/branchy_quantized_snn_int8.pth')
    ap.add_argument('--agent', default='results/branchy_quantized_snn_int8_agent.pkl')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--out', default='../lava_loihi_sim/exports/lenet_qdsnn.npz')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reset', choices=['zero', 'subtract'], default='zero',
                    help="LIF reset for the golden -- MUST match how the checkpoint "
                         "was trained. 'zero'=HARD reset (Loihi + delay-aware QDSNN "
                         "retrain). 'subtract'=repo default (old checkpoints).")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    import torch.ao.quantization as tq
    from config import config
    from models.branchy_quantized_snn import BranchyQuantizedSNN
    from models.quantized_snn_lenet import encode_poisson, get_quant_engine

    # Rebuild the INT8 skeleton exactly as evaluate_quantized_pair.load_qdsnn does.
    model = BranchyQuantizedSNN(num_classes=config["num_classes"],
                                time_window=args.T, beta=config["snn"]["beta"])
    state = torch.load(args.ckpt, map_location='cpu')
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
    model.load_state_dict(state)          # strict
    model.eval().cpu()
    print(f"[export] rebuilt INT8 skeleton and strict-loaded {args.ckpt}")

    # Match the checkpoint's reset so the golden reflects the true (Loihi) model.
    if args.reset == 'zero':
        from snntorch import surrogate
        sg = surrogate.atan()
        for attr in ['lif1', 'lif2', 'lif_fc1', 'lif_fc2']:
            old = getattr(model, attr)
            setattr(model, attr, snn.Leaky(beta=float(old.beta), threshold=float(old.threshold),
                    spike_grad=sg, init_hidden=False, reset_mechanism='zero'))
        print("[export] LIFs set to HARD reset (Loihi-faithful golden)")

    # RL-learned exit thresholds
    if os.path.exists(args.agent):
        from models.rl_environment import QLearningAgent
        agent = QLearningAgent(n_exits=2)
        agent.load(args.agent)
        model.prepare_for_inference(rl_agent=agent)
    thresholds = list(model.confidence_thresholds)
    print(f"[export] confidence_thresholds = {thresholds}")

    # --- backbone dense layers (pool folded into the FOLLOWING layer) ---
    W1, b1 = dq(model.conv1)
    M0, bias0, shape1 = conv2d_to_dense(W1, b1, (1, 28, 28), 1, 2)      # ->(6,28,28)

    P1, pooled1 = avgpool_matrix(shape1, 2, 2)                          # ->(6,14,14)
    W2, b2 = dq(model.conv2)
    Mc2, bias2, shape2 = conv2d_to_dense(W2, b2, pooled1, 1, 0)         # ->(16,10,10)
    M1 = Mc2 @ P1

    P2, _ = avgpool_matrix(shape2, 2, 2)                                # ->(16,5,5)
    W3, b3 = dq(model.fc1)
    M2 = W3.numpy().astype(np.float64) @ P2

    W4, b4 = dq(model.fc2)
    W5, b5 = dq(model.fc3)

    layers = [
        pack_layer('conv1', M0, bias0),
        pack_layer('pool1+conv2', M1, bias2),
        pack_layer('pool2+fc1', M2, b3.numpy().astype(np.float64)),
        pack_layer('fc2', W4.numpy().astype(np.float64), b4.numpy()),
        pack_layer('fc3', W5.numpy().astype(np.float64), b5.numpy()),
    ]

    # --- exit heads folded onto the UNPOOLED LIF maps (exact, see docstring) ---
    E0, eb0 = global_avgpool_linear_to_dense(model.exit1.fc.weight.data,
                                             model.exit1.fc.bias.data, shape1)
    E1, eb1 = global_avgpool_linear_to_dense(model.exit2.fc.weight.data,
                                             model.exit2.fc.bias.data, shape2)
    exits = [pack_layer('exit1', E0, eb0), pack_layer('exit2', E1, eb1)]

    lifs = [m for m in model.modules() if isinstance(m, snn.Leaky)]
    beta = float(lifs[0].beta); thr = float(lifs[0].threshold)
    print(f"[export] backbone {len(layers)} dense layers, {len(exits)} exit heads, "
          f"{len(lifs)} LIF, beta={beta:.3f}, thr={thr:.3f}, T={args.T}")
    for L in layers + exits:
        print(f"   {L['name']:<12} {L['w_int'].shape} scale={L['scale']:.4g} "
              f"fanout={L['fanout']:.1f}")

    # --- data + REAL exit decisions from PyTorch ---
    import torchvision
    from torchvision import transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))
    images = torch.stack([ds[i][0] for i in range(N)])
    labels = torch.tensor([ds[i][1] for i in range(N)])
    spk = encode_poisson(images, args.T)
    spikes_flat = spk.reshape(args.T, N, -1)

    with torch.no_grad():
        exit_outputs, final_out, exit_pts = model(spk)
    e1a, e2a = exit_outputs
    exit_pts = exit_pts.cpu()
    exit_ts = model.last_exit_timesteps.cpu()
    composed = torch.where((exit_pts == 1).unsqueeze(1), e1a,
                           torch.where((exit_pts == 2).unsqueeze(1), e2a, final_out))
    pred = composed.argmax(1)
    acc = (pred == labels).float().mean().item()
    counts = {int(k): int((exit_pts == k).sum()) for k in (1, 2, 3)}
    print(f"[export] QDSNN acc (composed by exit) = {acc*100:.2f}%")
    print(f"[export] exit distribution = {counts} "
          f"({', '.join(f'exit{k}:{v/N*100:.1f}%' for k, v in counts.items())})")
    print(f"[export] avg exit timestep = {exit_ts.float().mean():.2f}/{args.T}, "
          f"avg depth = {exit_pts.float().mean():.2f}/3")
    for k in (1, 2, 3):
        m = exit_pts == k
        if m.any():
            print(f"   exit{k}: n={int(m.sum())} acc={(pred[m]==labels[m]).float().mean()*100:.1f}% "
                  f"avg_ts={exit_ts[m].float().mean():.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = dict(
        n_layers=len(layers), n_exits=len(exits),
        layer_names=np.array([L['name'] for L in layers]),
        exit_names=np.array([e['name'] for e in exits]),
        beta=beta, threshold=thr, n_lif=4, T=args.T,
        readout='integrator', exit_decay=EXIT_DECAY,
        confidence_thresholds=np.array(thresholds, np.float32),
        exit_after=np.array([0, 1], np.int64),   # exit1 after L0, exit2 after L1
        ac_fanout=np.array([L['fanout'] for L in layers], np.float64),
        exit_fanout=np.array([e['fanout'] for e in exits], np.float64),
        spikes=spikes_flat.permute(1, 0, 2).contiguous().numpy().astype(np.uint8),
        labels=labels.numpy().astype(np.int64),
        exit_points=exit_pts.numpy().astype(np.int64),
        exit_timesteps=exit_ts.numpy().astype(np.int64),
        golden_pred=pred.numpy().astype(np.int64),
        golden_acc=np.float32(acc), int8_acc=np.float32(acc),
        golden_fire=np.zeros(4, np.float32),
    )
    for k, L in enumerate(layers):
        blob[f'w{k}'] = L['w_int']; blob[f'scale{k}'] = np.float32(L['scale'])
        blob[f'bias{k}'] = L['bias']
    for j, E in enumerate(exits):
        blob[f'ew{j}'] = E['w_int']; blob[f'escale{j}'] = np.float32(E['scale'])
        blob[f'ebias{j}'] = E['bias']
    np.savez_compressed(args.out, **blob)
    print(f"[export] wrote {args.out} (spikes {blob['spikes'].shape}, N={N})")


if __name__ == '__main__':
    main()
