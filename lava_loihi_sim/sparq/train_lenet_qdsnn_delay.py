"""
DELAY-AWARE retrain of the LeNet QDSNN (the "real fix" for the dynamic net on Loihi).

Sibling of train_lenet_delay.py (which does the QSNN). Same idea: model Loihi's
1-tick dendritic delay PER LAYER during training so the net learns under it -- but
here for the BRANCHY (early-exit) model, so it also threads the two exit heads and
the decaying accumulators through the pipeline, and (optionally) trains the RL
exit policy on the delay-consistent confidences.

WHY: the QSNN delay-retrain recovered LeNet on Loihi (84% -> 92% at T=16). The
QDSNN was NEVER retrained under the delay -- we only fixed the Lava readout math on
the old delay-blind checkpoint (22% -> 52% at T=4), which corrects the readout but
cannot overcome pipeline starvation. This script gives the QDSNN the same treatment
as the QSNN, so it recovers BOTH the early-exit AC/latency savings AND on-chip
accuracy.

FAITHFULNESS: this does NOT reimplement the network. It calls the model's own
_spiking_block / quant stubs / pool / exit heads / fc3, just re-SCHEDULED so that
stage k at tick t consumes stage k-1's output from t-1 (a pipelined forward),
exactly as the Lava path-decomposition sim runs each exit path. The exit heads hang
off their backbone stage with one extra tick of delay (they are real layers on
Loihi). Accumulators decay at 0.9 (matching the model: e/fa = e/fa*0.9 + head).

Backbone stages (each conv/fc + bn + LIF): conv1 -> conv2 -> fc1 -> fc2, readout fc3.
Exit1 head reads pool1 (after conv1); Exit2 head reads pool2 (after conv2).

Runs in the MAIN env (torch + GPU). Verify with --smoke, then launch the full run.

Usage (from LeNet-Clean):
  # verify it trains:
  PYTHONPATH=<LeNet-Clean> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/train_lenet_qdsnn_delay.py --smoke
  # full run:
  PYTHONPATH=<LeNet-Clean> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/train_lenet_qdsnn_delay.py --epochs 20 --T 16 \
      --out results/branchy_quantized_snn_delay_int8.pth
Then export_lenet_qdsnn.py --T 16 on the new checkpoint and re-run the Loihi sim.
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


# ------------------------------------------------------------------ helpers
# (copied from LeNet-Clean/train.py so this scaffold stays self-contained)
def _detach_lif_state(model):
    for mod in model.modules():
        if isinstance(mod, snn.Leaky):
            for attr in ('mem', 'spk'):
                v = getattr(mod, attr, None)
                if torch.is_tensor(v):
                    setattr(mod, attr, v.detach())
    return model


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, outputs, y_a, y_b, lam, weights):
    loss_a = sum(w * criterion(o, y_a) for w, o in zip(weights, outputs))
    loss_b = sum(w * criterion(o, y_b) for w, o in zip(weights, outputs))
    return lam * loss_a + (1 - lam) * loss_b


def _rl_update_from_batch(rl_agent, exit_outputs, final_output, labels, alpha):
    """Paper Sec. III-B sequential Q-learning; one episode per sample, one
    end_episode per batch. Identical to LeNet-Clean/train.py._rl_update_from_batch."""
    n_ex = len(exit_outputs)
    n_stages = n_ex + 1
    with torch.no_grad():
        confs = [torch.softmax(e, dim=1).max(1)[0] for e in exit_outputs]
        preds = [e.argmax(1) for e in exit_outputs]
        final_pred = final_output.argmax(1)
    for b in range(labels.size(0)):
        y = labels[b]
        for i in range(n_ex):
            state = rl_agent.get_state(i, confs[i][b].item())
            action = rl_agent.select_action(state)
            if action == 1:
                savings = (n_stages - (i + 1)) / n_stages
                r = (1.0 + alpha * savings) if preds[i][b] == y else -1.0
                rl_agent.update(state, 1, float(r), done=True)
                break
            if i == n_ex - 1:
                r = 1.0 if final_pred[b] == y else -1.0
                rl_agent.update(state, 0, float(r), done=True)
            else:
                next_state = rl_agent.get_state(i + 1, confs[i + 1][b].item())
                rl_agent.update(state, 0, 0.0, next_state=next_state, done=False)
    rl_agent.end_episode()


# ---------------------------------------------------- delay-aware QDSNN forward
def qdsnn_delay_forward(model, spike_train):
    """Pipelined (1-tick/layer delay) forward for the QDSNN.

    Returns (e1a, e2a, fa): the accumulated exit-1, exit-2 and final logits, exactly
    the three tensors the model's _forward_training returns -- but produced under the
    Loihi per-layer delay. Reuses the model's own ops so the trained net is
    bit-identical to the real model apart from the delay schedule.
    """
    T, B = spike_train.shape[0], spike_train.shape[1]
    dev = spike_train.device
    C = model.num_classes
    d = 0.9

    mems = [None] * 4
    e1a = torch.zeros(B, C, device=dev)
    e2a = torch.zeros(B, C, device=dev)
    fa = torch.zeros(B, C, device=dev)

    prev = [None] * 4       # backbone stage outputs at t-1: [p1, p2, spk_fc1, spk_fc2]
    prev_in = None          # input spike at t-1 (stage 0 is delayed too)

    for t in range(T):
        xt = spike_train[t]
        new = [None] * 4

        # --- backbone: stage k consumes stage k-1's output from t-1 ---
        if prev_in is not None:                                    # stage 0: conv1
            spk1, mems[0] = model._spiking_block(
                model.quant(prev_in), model.conv1, model.bn1, model.lif1, mems[0])
            new[0] = model.pool1(spk1)
        if prev[0] is not None:                                    # stage 1: conv2
            x = model.q1(prev[0])
            spk2, mems[1] = model._spiking_block(
                x, model.conv2, model.bn2, model.lif2, mems[1])
            new[1] = model.pool2(spk2)
        if prev[1] is not None:                                    # stage 2: fc1
            x = model.q2(model.flatten(prev[1]))
            spk_fc1, mems[2] = model._spiking_block(
                x, model.fc1, model.bn_fc1, model.lif_fc1, mems[2])
            new[2] = spk_fc1
        if prev[2] is not None:                                    # stage 3: fc2
            x = model.q3(prev[2])
            spk_fc2, mems[3] = model._spiking_block(
                x, model.fc2, model.bn_fc2, model.lif_fc2, mems[3])
            new[3] = spk_fc2
        if prev[3] is not None:                                    # readout: fc3
            f = model._maybe_dequant(model.fc3(model.q4(prev[3])))
            fa = fa * d + f

        # --- exit heads: hang off their backbone stage with +1 tick (real layers) ---
        if prev[0] is not None:
            e1a = e1a * d + model.exit1(prev[0])
        if prev[1] is not None:
            e2a = e2a * d + model.exit2(prev[1])

        prev = new
        prev_in = xt

    return e1a, e2a, fa


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=16)
    # match LeNet-Clean/config.py: epochs=20, early_stopping_patience=10
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=10,
                    help='early-stopping patience (0 disables)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--out', default='results/branchy_quantized_snn_delay_int8.pth')
    ap.add_argument('--reset', choices=['zero', 'subtract'], default='zero',
                    help="LIF reset. 'zero'=HARD reset (what Loihi does) -- REQUIRED "
                         "for the QDSNN: its rate-sensitive global-pool exit heads "
                         "collapse on Lava under the reset mismatch (verified: exit-2 "
                         "74%% hard vs 96%% subtract). 'subtract' = repo default (off-chip).")
    ap.add_argument('--no_rl', action='store_true',
                    help='disable RL exit-policy training (weights only)')
    ap.add_argument('--rl_bins', type=int, default=10)
    ap.add_argument('--smoke', action='store_true',
                    help='run ~30 train batches + a val pass and exit (verify only)')
    args = ap.parse_args()

    from config import config
    from data_loader import get_validation_loader
    from models.branchy_quantized_snn import BranchyQuantizedSNN
    from models.quantized_snn_lenet import encode_poisson, get_quant_engine
    from models.rl_environment import QLearningAgent

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nc = config['num_classes']
    use_rl = not args.no_rl
    print(f"[qdsnn-delay] device={dev}  T={args.T}  epochs={args.epochs}  "
          f"RL={'on' if use_rl else 'off'}  {'SMOKE' if args.smoke else 'FULL'}")

    model = BranchyQuantizedSNN(num_classes=nc, time_window=args.T,
                                beta=config['snn']['beta'])
    # Rebuild the LIFs with HARD reset to match Loihi (see --reset help). Must be
    # done at construction -- snntorch's reset_mechanism setter does NOT change the
    # actual reset computation post-hoc (verified: it keeps subtracting).
    if args.reset == 'zero':
        from snntorch import surrogate
        sg = surrogate.atan()
        for attr in ['lif1', 'lif2', 'lif_fc1', 'lif_fc2']:
            old = getattr(model, attr)
            setattr(model, attr, snn.Leaky(beta=float(old.beta), threshold=float(old.threshold),
                    spike_grad=sg, init_hidden=False, reset_mechanism='zero'))
        print("[qdsnn-delay] LIFs rebuilt with HARD reset (Loihi-faithful)")
    torch.backends.quantized.engine = get_quant_engine()
    model.fuse_model()                                    # TRAIN mode -> BN kept live
    model.qconfig = tq.QConfig(
        activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=tq.MinMaxObserver.with_args(dtype=torch.qint8,
                                           qscheme=torch.per_tensor_symmetric))
    for _, m in model.named_modules():
        if isinstance(m, snn.Leaky):
            m.qconfig = None
    tq.prepare_qat(model, inplace=True)
    model.to(dev)

    agent = None
    if use_rl:
        rl = config['rl']
        agent = QLearningAgent(n_exits=model.n_exits, n_confidence_bins=args.rl_bins,
                               learning_rate=rl['learning_rate'],
                               discount_factor=rl['discount_factor'],
                               exploration_rate=rl['exploration_rate'],
                               exploration_decay=rl['exploration_decay'],
                               min_exploration_rate=rl['min_exploration_rate'])
        model.rl_agent = agent

    train_loader, val_loader = get_validation_loader(batch_size=args.batch)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    exit_w = getattr(model, 'exit_loss_weights', [0.3, 0.3, 1.0])
    rl_alpha = config['rl'].get('reward_scale', 0.3)
    mixup_alpha = config['mixup_alpha']

    def run_train_epoch(cap=None):
        model.train()
        tot = cor = 0; loss_sum = 0.0
        for bi, (img, lab) in enumerate(train_loader):
            if cap and bi >= cap:
                break
            img, lab = img.to(dev), lab.to(dev)
            mixed, y_a, y_b, lam = mixup_data(img, lab, mixup_alpha)
            spk = encode_poisson(mixed, args.T)
            e1a, e2a, fa = qdsnn_delay_forward(model, spk)
            outputs = [e1a, e2a, fa]
            loss = mixup_criterion(crit, outputs, y_a, y_b, lam, exit_w)

            opt.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(gn):
                opt.step()
                loss_sum += loss.item() * lab.size(0)
            else:
                opt.zero_grad(set_to_none=True)
                print("[WARN] skipped non-finite step")
            cor += (fa.argmax(1) == lab).sum().item(); tot += lab.size(0)

            # RL: learn the exit policy on delay-consistent confidences
            if agent is not None:
                if lam < 1.0 - 1e-6:
                    with torch.no_grad():
                        r1, r2, rf = qdsnn_delay_forward(model, encode_poisson(img, args.T))
                    _rl_update_from_batch(agent, [r1, r2], rf, lab, rl_alpha)
                else:
                    _rl_update_from_batch(agent, [e1a.detach(), e2a.detach()],
                                          fa.detach(), lab, rl_alpha)
        return loss_sum / max(tot, 1), 100.0 * cor / max(tot, 1)

    @torch.no_grad()
    def run_val(cap=None):
        """Delay-faithful val. Reports final-exit (full-path) acc = the primary
        best-checkpoint metric, plus each exit path's own acc for insight."""
        model.eval()
        tot = 0; cor_f = cor_1 = cor_2 = 0
        for bi, (img, lab) in enumerate(val_loader):
            if cap and bi >= cap:
                break
            img, lab = img.to(dev), lab.to(dev)
            e1a, e2a, fa = qdsnn_delay_forward(model, encode_poisson(img, args.T))
            cor_f += (fa.argmax(1) == lab).sum().item()
            cor_1 += (e1a.argmax(1) == lab).sum().item()
            cor_2 += (e2a.argmax(1) == lab).sum().item()
            tot += lab.size(0)
        n = max(tot, 1)
        return 100.0*cor_f/n, 100.0*cor_1/n, 100.0*cor_2/n

    if args.smoke:
        tl, ta = run_train_epoch(cap=30)
        vf, v1, v2 = run_val(cap=20)
        print(f"[smoke] 30 train batches: loss={tl:.3f} final_acc={ta:.1f}%")
        print(f"[smoke] val(20): final={vf:.1f}%  exit1={v1:.1f}%  exit2={v2:.1f}%")
        if agent is not None:
            print(f"[smoke] RL thresholds={agent.get_confidence_thresholds()} "
                  f"episodes={agent.episode}")
        print("[smoke] OK -- delay-aware QDSNN forward trains + backprops. Launch full run.")
        return

    best = 0.0
    patience = 0
    for ep in range(args.epochs):
        tl, ta = run_train_epoch()
        vf, v1, v2 = run_val()
        sched.step()
        if agent is not None:
            model.confidence_thresholds = agent.get_confidence_thresholds()
            print(f"  RL thresholds={model.confidence_thresholds} "
                  f"eps={agent.exploration_rate:.3f} episodes={agent.episode}")
        print(f"epoch {ep+1}/{args.epochs}  train_loss={tl:.3f} train_final={ta:.1f}%  "
              f"val_final={vf:.2f}%  val_exit1={v1:.1f}%  val_exit2={v2:.1f}%")

        # select on the FINAL (deepest, hardest-under-delay) path accuracy
        if vf > best:
            best = vf
            patience = 0
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
            engine = get_quant_engine(); torch.backends.quantized.engine = engine
            _detach_lif_state(model)
            m_int8 = tq.convert(copy.deepcopy(model).eval().cpu(), inplace=False)
            torch.save(m_int8.state_dict(), args.out)
            model.to(dev)
            print(f"   saved INT8 (val_final {vf:.2f}%) -> {args.out}")
            if agent is not None:
                ap_path = args.out.replace('.pth', '_agent.pkl')
                agent.save(ap_path)
                print(f"   saved RL agent -> {ap_path}")
        else:
            patience += 1
            if args.patience and patience >= args.patience:
                print(f"[qdsnn-delay] early stop at epoch {ep+1} "
                      f"(no val_final improvement for {args.patience})")
                break
    print(f"[qdsnn-delay] done. best val_final={best:.2f}%  checkpoint={args.out}")


if __name__ == '__main__':
    main()
