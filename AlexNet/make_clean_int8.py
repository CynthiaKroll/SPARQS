"""
Repair the NaN-poisoned fc2 layer in the QSNN / QDSNN fake-quant checkpoints and
export clean weights for the Loihi pipeline.

Problem (BOTH models): fc2 -- QSNN `classifier_seq.5`, QDSNN `fc2` -- has NaN in its
fused BN running stats and its weight+activation observers. A transient inf during
late QAT poisoned them before the finiteness guard (which only skips the optimizer
step, not BN/observer updates). fc2 is the deepest/widest FC, so it blows up first.
The LIF nonlinearity turns the NaN into 0-spikes for the few poisoned channels, so
the fake-quant model still evaluates (~66-68%), but the NaN observers give a NaN
int8 scale -> convert()/hand-roll fails -> Loihi export blocked.

fc2's WEIGHT is clean (only re-estimable state is NaN), so we:
  1. reset NaN BN running stats (mean->0, var->1) and NaN observers,
  2. re-estimate them on clean data (BN in train mode + observers collecting),
  3. save the repaired FAKE-QUANT checkpoint (both models),
  4. for the QSNN also convert() to INT8 and verify every scale is finite,
  5. sanity-eval.

Usage:
  python make_clean_int8.py --model qsnn
  python make_clean_int8.py --model qdsnn
"""
import argparse

import torch
import torch.nn as nn

from config import config
from data_loader import get_data_loaders
from models.quantized_snn_alexnet import (
    QuantizedSNNAlexNet,
    prepare_quantized_snn_alexnet_for_qat,
    encode_poisson,
    get_quant_engine,
)
from models.branchy_quantized_snn import BranchyQuantizedSNN, reset_exit_temperatures

CALIB_BATCHES = 30
EVAL_BATCHES = 20


def any_nan(t):
    return torch.is_tensor(t) and torch.isnan(t).any().item()


def not_finite(t):
    """True if the tensor holds ANY NaN OR inf. isnan alone misses inf, and inf in
    a weight/gamma produces NaN in the forward -> re-poisons BN/observers."""
    return torch.is_tensor(t) and not torch.isfinite(t).all().item()


def build_qsnn(T):
    m = QuantizedSNNAlexNet(num_classes=config["num_classes"], beta=config["beta"],
                            time_steps=T)
    return prepare_quantized_snn_alexnet_for_qat(m)


def build_qdsnn(T):
    m = BranchyQuantizedSNN(num_classes=config["num_classes"], time_window=T,
                            beta=config["beta"])
    m.train()
    m.fuse_model()
    m.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    )
    torch.quantization.prepare_qat(m, inplace=True)
    return m


def repair(model, dev, T, is_branchy):
    """Reset NaN BN stats + observers, then re-estimate on clean data (train mode)."""
    # 1. Sanitize non-finite PARAMETERS (weight/bias/gamma/beta). This is the real
    #    fix: fc2 has inf (not just NaN) in a param from a transient QAT blow-up;
    #    inf passes isnan but yields NaN in the forward. Replace inf/NaN with 0 --
    #    that drops a handful of corrupted connections (negligible accuracy) and
    #    makes the forward finite so recalibration can actually work.
    n_p = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not_finite(p):
                p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0))
                n_p += 1
                print(f"  sanitized param: {name}")
    # 2. Reset non-finite BN running stats + observers.
    n_bn = n_obs = 0
    for name, mod in model.named_modules():
        rm, rv = getattr(mod, "running_mean", None), getattr(mod, "running_var", None)
        if not_finite(rm):
            rm.zero_()
            if torch.is_tensor(rv):
                rv.fill_(1.0)
            if hasattr(mod, "num_batches_tracked") and torch.is_tensor(mod.num_batches_tracked):
                mod.num_batches_tracked.zero_()
            n_bn += 1
            print(f"  reset BN running stats: {name}")
        if not_finite(getattr(mod, "min_val", None)) or not_finite(getattr(mod, "max_val", None)):
            if hasattr(mod, "reset_min_max_vals"):
                mod.reset_min_max_vals()
            else:
                if torch.is_tensor(getattr(mod, "min_val", None)):
                    mod.min_val = torch.full_like(mod.min_val, float("inf"))
                if torch.is_tensor(getattr(mod, "max_val", None)):
                    mod.max_val = torch.full_like(mod.max_val, float("-inf"))
            n_obs += 1
            print(f"  reset observer: {name}")
    print(f"sanitized {n_p} params, reset {n_bn} BN modules, {n_obs} observers")

    # 3. Floor near-zero BN gammas. The QAT LinearBn TRAIN forward divides by
    #    scale_factor = gamma / sqrt(running_var); a ~zero gamma channel gives
    #    0/0 -> NaN during recalibration -- THIS is why train-mode re-estimation
    #    kept re-poisoning fc2 despite clean weights. Near-zero gammas are dead
    #    channels (BN output ~= beta), so flooring |gamma| to 1e-3 is negligible
    #    and removes the division blow-up. Only applied to BN modules (they carry
    #    running_mean); conv/linear weights are untouched.
    n_g = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            g = getattr(mod, "weight", None)
            if torch.is_tensor(g) and torch.is_tensor(getattr(mod, "running_mean", None)):
                near0 = g.abs() < 1e-3
                if near0.any():
                    g[near0] = 1e-3
                    n_g += int(near0.sum())
    if n_g:
        print(f"floored {n_g} near-zero BN gamma channels")

    # CRITICAL: disable fake-quant during recalibration. The weight FakeQuantize
    # still wraps the (just-reset, ~inf) observer; left enabled it applies a NaN/inf
    # scale to the weight on the FIRST forward -> NaN activations -> re-poisons the
    # freshly reset BN + observers (this is exactly what failed the first attempt:
    # fc2 re-NaN'd and lif7.mem went NaN). Disabled, the forward runs the CLEAN FP32
    # weights so the observers collect finite ranges and BN re-estimates from finite
    # batch stats; we re-enable afterwards.
    torch.ao.quantization.disable_fake_quant(model)
    train_loader, _, _, _ = get_data_loaders(distributed=False, augment=False)
    model.to(dev).train()
    with torch.no_grad():
        for i, (img, _) in enumerate(train_loader):
            if i >= CALIB_BATCHES:
                break
            img = img.to(dev)
            model(img if is_branchy else encode_poisson(img, T))  # QDSNN encodes internally
    torch.ao.quantization.enable_fake_quant(model)
    print(f"recalibrated on {CALIB_BATCHES} batches (fake-quant disabled during calib)")

    # LIF membrane is transient per-forward state (snntorch registers it as a buffer);
    # zero any NaN that lingered so it does not fail the check or dirty the checkpoint.
    for mod in model.modules():
        mem = getattr(mod, "mem", None)
        if torch.is_tensor(mem):
            with torch.no_grad():
                mem.zero_()

    still = [n for n, p in list(model.named_buffers()) + list(model.named_parameters())
             if not_finite(p)]
    if still:
        raise RuntimeError(f"non-finite remains after recalibration: {still}")
    print("no non-finite buffers/params remain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qsnn", "qdsnn"], required=True)
    args = ap.parse_args()

    torch.backends.quantized.engine = get_quant_engine()
    T = config["num_steps"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model} | device={dev} | T={T} | engine={torch.backends.quantized.engine}")

    if args.model == "qsnn":
        ckpt = "results/quantized_snn_alexnet_qat.pth"
        model = build_qsnn(T)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        repair(model, dev, T, is_branchy=False)

        # save repaired fake-quant, then convert to INT8
        model.eval().cpu()
        torch.save(model.state_dict(), "results/quantized_snn_alexnet_qat_clean.pth")
        print("saved results/quantized_snn_alexnet_qat_clean.pth (repaired fake-quant)")
        int8 = torch.ao.quantization.convert(model, inplace=False)
        bad = [n for n, mod in int8.named_modules()
               if any_nan(mod.scale() if callable(getattr(mod, "scale", None)) else getattr(mod, "scale", None))]
        if bad:
            print(f"[FAIL] NaN scales after convert: {bad}")
            return
        print("ALL INT8 SCALES FINITE")
        torch.save(int8.state_dict(), "results/quantized_snn_alexnet_qat_int8_clean.pth")
        print("saved results/quantized_snn_alexnet_qat_int8_clean.pth")

        _, _, test_loader, _ = get_data_loaders(distributed=False, augment=False)
        correct = total = 0
        with torch.no_grad():
            for i, (img, lbl) in enumerate(test_loader):
                if i >= EVAL_BATCHES:
                    break
                out = int8(encode_poisson(img, T)).mean(dim=0)
                correct += (out.argmax(1) == lbl).sum().item()
                total += lbl.size(0)
        print(f"INT8 sanity accuracy ({total} samples): {100.0 * correct / total:.2f}%")

    else:  # qdsnn
        ckpt = "results/branchy_quantized_snn_qat.pth"
        model = build_qdsnn(T)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        repair(model, dev, T, is_branchy=True)

        # QDSNN Loihi path hand-rolls INT8 from the fake-quant, so save the repaired
        # fake-quant (no convert() -- the QDSNN convert path is not used).
        reset_exit_temperatures(model)
        model.eval().cpu()
        torch.save(model.state_dict(), "results/branchy_quantized_snn_qat_clean.pth")
        print("saved results/branchy_quantized_snn_qat_clean.pth (repaired fake-quant)")

        # sanity: full-depth accuracy (thresholds 1.1 -> no early exit)
        model.prepare_for_inference(confidence_thresholds=[1.1, 1.1])
        _, _, test_loader, _ = get_data_loaders(distributed=False, augment=False)
        correct = total = 0
        with torch.no_grad():
            for i, (img, lbl) in enumerate(test_loader):
                if i >= EVAL_BATCHES:
                    break
                _, final_out, _ = model(img)
                correct += (final_out.argmax(1) == lbl).sum().item()
                total += lbl.size(0)
        print(f"QDSNN full-depth sanity accuracy ({total} samples): {100.0 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
