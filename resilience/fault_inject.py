"""
Weight-memory bit-flip injection for the SPARQ resilience study.

Fault model
-----------
Faults are counted PER STORED BIT, not per weight. BER is a property of the
memory technology (upsets per stored bit), not of the network, so a model that
stores more bits presents a proportionally larger cross-section:

    n_flips = round(BER * n_params * b)      b = 32 (FP32 ANN), 8 (INT8 QSNN/QDSNN)

The FP32 ANN therefore receives 4x the flips per parameter at the same BER.
That asymmetry is deliberate and is itself a result: quantization buys a 4x
reduction in fault cross-section for free.

For INT8 models the flip happens in the INTEGER representation (int_repr), i.e.
exactly the bits a real INT8 weight memory would store, then the quantized
tensor is rebuilt with the original scale/zero-point.
"""
import numpy as np
import torch


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def quantized_weight_modules(model):
    """[(name, module)] for converted INT8 Conv2d/Linear carrying int8 weights."""
    out = []
    for name, m in model.named_modules():
        w = getattr(m, "weight", None)
        if not callable(w):
            continue
        try:
            t = m.weight()
        except Exception:
            continue
        if hasattr(t, "int_repr") and t.is_quantized:
            out.append((name, m))
    return out


def fp32_weight_tensors(model, include=("weight",)):
    """[(name, param)] for FP32 weight matrices (Conv/Linear), biases excluded.

    Biases are excluded because they are a negligible fraction of the stored
    bits and are typically held in a different (often protected) memory.
    """
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad and p.dtype != torch.float32:
            continue
        if p.dtype != torch.float32 or p.dim() < 2:
            continue
        if any(name.endswith(s) for s in include):
            out.append((name, p))
    return out


def bit_budget(model, is_int8):
    """Total stored weight bits -- the fault cross-section of this model."""
    if is_int8:
        mods = quantized_weight_modules(model)
        n = sum(m.weight().numel() for _, m in mods)
        return n, n * 8
    params = fp32_weight_tensors(model)
    n = sum(p.numel() for _, p in params)
    return n, n * 32


# --------------------------------------------------------------------------
# injection
# --------------------------------------------------------------------------
def _draw_flips(sizes, n_bits_per_elem, n_flips, rng):
    """Distribute n_flips uniformly over the concatenated bit space.

    Returns {tensor_idx: (elem_idx array, bit_idx array)}.
    """
    offsets = np.cumsum([0] + [s * n_bits_per_elem for s in sizes])
    total = offsets[-1]
    if n_flips <= 0 or total == 0:
        return {}
    n_flips = min(n_flips, total)
    picks = rng.choice(total, size=n_flips, replace=False)
    out = {}
    which = np.searchsorted(offsets, picks, side="right") - 1
    for ti in np.unique(which):
        local = picks[which == ti] - offsets[ti]
        out[int(ti)] = (local // n_bits_per_elem, local % n_bits_per_elem)
    return out


def snapshot_int8(model):
    """Save int8 weights so injection can be undone exactly."""
    return [(m, m.weight().int_repr().clone(), float(m.weight().q_scale()),
             int(m.weight().q_zero_point()),
             None if m.bias() is None else m.bias().clone())
            for _, m in quantized_weight_modules(model)]


def restore_int8(snap):
    for m, w_int, scale, zp, bias in snap:
        qw = torch._make_per_tensor_quantized_tensor(w_int.clone(), scale, zp)
        m.set_weight_bias(qw, bias)


def inject_int8(model, ber, rng, snap=None):
    """Flip round(ber * n_bits) bits across the INT8 weight memory.

    Returns the number of bits actually flipped.
    """
    snap = snap or snapshot_int8(model)
    sizes = [w.numel() for _, w, _, _, _ in snap]
    total_bits = sum(sizes) * 8
    n_flips = int(round(ber * total_bits))
    flips = _draw_flips(sizes, 8, n_flips, rng)

    for ti, (elem, bit) in flips.items():
        m, w_int, scale, zp, bias = snap[ti]
        arr = w_int.numpy().reshape(-1).copy().view(np.uint8)
        # XOR the selected bit of the selected byte. Repeated (elem, bit) pairs
        # cannot occur -- _draw_flips samples without replacement -- so a plain
        # XOR is correct and no accumulation guard is needed.
        arr[elem] ^= (np.uint8(1) << bit.astype(np.uint8))
        new = torch.from_numpy(arr.view(np.int8).reshape(w_int.shape).copy())
        qw = torch._make_per_tensor_quantized_tensor(new, scale, zp)
        m.set_weight_bias(qw, bias)
    return n_flips


def quantize_weights_inplace(model):
    """Round every FP32 weight matrix onto the INT8 grid, in place.

    Must be called ONCE before snapshotting when `inject_int8_simulated` is used,
    so that the fault-free run and the faulted runs sit on the same grid. Without
    it the golden pass would use raw FP32 weights while only the tensors that
    happened to receive a flip were quantized, and the difference between the two
    conditions would not be attributable to the injected faults alone.
    """
    with torch.no_grad():
        for _, p in fp32_weight_tensors(model):
            scale = float(p.abs().max().clamp(min=1e-12)) / 127.0
            p.copy_(torch.clamp(torch.round(p / scale), -127, 127) * scale)
    return model


def inject_int8_simulated(model, ber, rng, snap=None):
    """INT8 weight-memory faults on a model whose weights are stored as FP32.

    Needed for checkpoints that were quantization-aware trained but never passed
    through `convert()` (AlexNet's QDSNN), where the DEPLOYED representation is
    INT8 but the tensors in memory are float. The weight is quantized per-tensor
    symmetric -- the scheme the QAT qconfig uses -- the flip is applied in the
    integer domain, and the result is dequantized back.

    At BER=0 this is exactly the fake-quant weight the model already computes
    with, so the fault-free behaviour is unchanged; the bit budget is `8` bits
    per weight, matching a real INT8 weight memory rather than the FP32 tensor
    that happens to hold it.
    """
    snap = snap or snapshot_fp32(model)
    sizes = [p.numel() for p, _ in snap]
    n_flips = int(round(ber * sum(sizes) * 8))
    flips = _draw_flips(sizes, 8, n_flips, rng)

    with torch.no_grad():
        for ti, (elem, bit) in flips.items():
            p, orig = snap[ti]
            scale = float(orig.abs().max().clamp(min=1e-12)) / 127.0
            w_int = torch.clamp(torch.round(orig / scale), -127, 127)
            arr = w_int.cpu().numpy().astype(np.int8).reshape(-1).copy().view(np.uint8)
            arr[elem] ^= (np.uint8(1) << bit.astype(np.uint8))
            new = torch.from_numpy(arr.view(np.int8).reshape(orig.shape).copy())
            p.copy_((new.float() * scale).to(p.device, p.dtype))
    return n_flips


def snapshot_fp32(model):
    return [(p, p.detach().clone()) for _, p in fp32_weight_tensors(model)]


def restore_fp32(snap):
    with torch.no_grad():
        for p, orig in snap:
            p.copy_(orig)


def inject_fp32(model, ber, rng, snap=None):
    """Flip round(ber * n_params * 32) bits across the FP32 weight memory.

    Bits are flipped in the IEEE-754 representation, so an exponent-bit flip can
    produce the large-magnitude corruption that dominates FP32 fault behaviour.
    """
    snap = snap or snapshot_fp32(model)
    sizes = [p.numel() for p, _ in snap]
    total_bits = sum(sizes) * 32
    n_flips = int(round(ber * total_bits))
    flips = _draw_flips(sizes, 32, n_flips, rng)

    with torch.no_grad():
        for ti, (elem, bit) in flips.items():
            p, _ = snap[ti]
            arr = p.detach().cpu().numpy().reshape(-1).copy().view(np.uint32)
            arr[elem] ^= (np.uint32(1) << bit.astype(np.uint32))
            new = torch.from_numpy(arr.view(np.float32).reshape(p.shape).copy())
            p.copy_(new.to(p.device))
    return n_flips
