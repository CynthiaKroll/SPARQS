"""
Shared LeNet -> Loihi conversion helpers (runs in the MAIN env, torch).

DESIGN DECISION: unroll every Conv2d (and every AvgPool) into an equivalent
DENSE matrix, then reuse the Dense pipeline already validated end-to-end on the
5-layer MLP. Rationale:

  * A convolution is a linear map, so the unrolling is EXACT -- no accuracy loss.
  * Lava's Conv process uses a channel-last (X, Y, C) layout with its own weight
    ordering that did not match any straightforward transpose of PyTorch's NCHW
    in testing. Dense is already bit-verified against golden references, so this
    removes a whole class of layout risk.
  * AvgPool has no neuromorphic primitive (it produces graded values 0/.25/.5/
    .75/1, but Loihi passes BINARY spikes between layers). Folding the pool into
    the FOLLOWING layer's weights is exact and keeps every inter-layer signal
    binary. This is also what a real Loihi mapping would do, since the chip has
    no pooling unit.
  * LeNet is small: conv1 -> 4704x784, conv2(+pool1) -> 1600x4704. ~11M weights
    total vs the MLP's 32M, which ran fine.

AC-COUNT CAVEAT (important): the unrolled matrix is mostly ZEROS. Counting ACs
as spikes x out_features (correct for a true dense layer) would massively
overcount. So each layer also carries `ac_fanout` = mean number of NONZERO
weights per input unit, which is the real per-spike synaptic-operation count and
equals what conv connectivity would cost on hardware.
"""
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# BatchNorm folding
# --------------------------------------------------------------------------

def fold_bn_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Absorb BatchNorm2d into Conv2d weights. Returns (W, b) as tensors."""
    W = conv.weight.data.clone()
    b = conv.bias.data.clone() if conv.bias is not None else torch.zeros(W.shape[0])
    s = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    return W * s.reshape(-1, 1, 1, 1), (b - bn.running_mean.data) * s + bn.bias.data


def fold_bn_linear(lin: nn.Linear, bn: nn.BatchNorm1d):
    """Absorb BatchNorm1d into Linear weights. Returns (W, b) as tensors."""
    W = lin.weight.data.clone()
    b = lin.bias.data.clone() if lin.bias is not None else torch.zeros(W.shape[0])
    s = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    return W * s.unsqueeze(1), (b - bn.running_mean.data) * s + bn.bias.data


# --------------------------------------------------------------------------
# Unrolling to dense matrices
# --------------------------------------------------------------------------

def conv2d_to_dense(W, b, in_shape, stride=1, padding=0):
    """Unroll Conv2d into a dense matrix.

    W        : [C_out, C_in, kH, kW] tensor
    in_shape : (C_in, H, W)
    Returns  : (M [out_features, in_features] np.float64, bias [out_features],
                out_shape (C_out, H_out, W_out))
    """
    W = W.detach().numpy().astype(np.float64)
    C_out, C_in, kH, kW = W.shape
    Ci, H, Wd = in_shape
    assert Ci == C_in, f"channel mismatch {Ci} vs {C_in}"
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (Wd + 2 * padding - kW) // stride + 1

    M = np.zeros((C_out * H_out * W_out, C_in * H * Wd), np.float64)
    for co in range(C_out):
        for oi in range(H_out):
            for oj in range(W_out):
                orow = (co * H_out + oi) * W_out + oj
                for ci in range(C_in):
                    for ki in range(kH):
                        y = oi * stride + ki - padding
                        if y < 0 or y >= H:
                            continue
                        for kj in range(kW):
                            x = oj * stride + kj - padding
                            if x < 0 or x >= Wd:
                                continue
                            M[orow, (ci * H + y) * Wd + x] += W[co, ci, ki, kj]
    bias = np.repeat(b.detach().numpy().astype(np.float64), H_out * W_out)
    return M, bias, (C_out, H_out, W_out)


def avgpool_matrix(in_shape, k=2, stride=2):
    """Dense matrix for AvgPool2d over (C,H,W). Returns (P, out_shape)."""
    C, H, W = in_shape
    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1
    P = np.zeros((C * H_out * W_out, C * H * W), np.float64)
    for c in range(C):
        for oi in range(H_out):
            for oj in range(W_out):
                orow = (c * H_out + oi) * W_out + oj
                for a in range(k):
                    for bb in range(k):
                        y, x = oi * stride + a, oj * stride + bb
                        P[orow, (c * H + y) * W + x] = 1.0 / (k * k)
    return P, (C, H_out, W_out)


def global_avgpool_linear_to_dense(W, b, in_shape):
    """AdaptiveAvgPool2d((1,1)) + Linear(C -> n) as a dense matrix over (C,H,W).

    Used by the LeNet QDSNN exit heads (ExitClassifier: global pool then Linear).
    """
    W = W.detach().numpy().astype(np.float64)          # [n_out, C]
    C, H, Wd = in_shape
    n_out = W.shape[0]
    M = np.zeros((n_out, C * H * Wd), np.float64)
    for o in range(n_out):
        for c in range(C):
            M[o, c * H * Wd:(c + 1) * H * Wd] = W[o, c] / (H * Wd)
    return M, b.detach().numpy().astype(np.float64)


# --------------------------------------------------------------------------
# Quantization + AC accounting
# --------------------------------------------------------------------------

def quantize_int8(M):
    """Symmetric per-tensor INT8 of a dense matrix. Returns (int8, scale)."""
    max_abs = float(np.abs(M).max())
    if max_abs < 1e-12:
        return np.zeros_like(M, np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.round(M / scale), -127, 127).astype(np.int8)
    return q, scale


def ac_fanout(M_int8):
    """Mean NONZERO weights per input unit = true per-spike AC cost.

    The unrolled conv matrix is mostly zeros; spikes x out_features would
    hugely overcount ACs. This is the honest per-spike synaptic-op count.
    """
    nnz = int(np.count_nonzero(M_int8))
    n_in = M_int8.shape[1]
    return nnz / max(n_in, 1)


def pack_layer(name, M, bias):
    """Quantize a dense layer and package it for the npz."""
    q, scale = quantize_int8(M)
    return dict(name=name, w_int=q, scale=scale,
                bias=np.asarray(bias, np.float32), fanout=ac_fanout(q))
