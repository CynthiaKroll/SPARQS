"""
BatchNorm Folding + Int8 Quantization for Neuromorphic Deployment

Paper: Ahmed et al. (arXiv:2403.10173v4), Section 5.1, Equations 7-8

BN folding absorbs BatchNorm statistics and PLIF tau into conv weights:
    scale = (q_scale * weight_BN) / (tau * sqrt(Var_BN + eps))
    shift = (bias_conv - mean_BN) * weight_BN / (tau * sqrt(Var_BN + eps)) + bias_BN / tau

After folding, weights are quantized to int8 per-output-channel (paper reports
mAP 0.348 float16 -> 0.343 int8, minimal degradation).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List


def extract_plif_tau(neuron_module: nn.Module) -> float:
    """
    Extract learned tau value from a SpikingJelly ParametricLIFNode.

    The ParametricLIFNode stores a learnable parameter `w` where:
        tau = 1 / sigmoid(w)

    Args:
        neuron_module: PLIFNeuronMultiStep or PLIFNeuron wrapper

    Returns:
        tau: Learned time constant (float)
    """
    # Navigate to the underlying ParametricLIFNode
    if hasattr(neuron_module, 'node'):
        node = neuron_module.node
    else:
        node = neuron_module

    # ParametricLIFNode stores `w` parameter, tau = 1/sigmoid(w)
    if hasattr(node, 'w'):
        w = node.w.data
        tau = 1.0 / torch.sigmoid(w)
        return tau.item()
    elif hasattr(node, 'tau'):
        return node.tau.item()
    else:
        raise ValueError(
            f"Cannot extract tau from {type(neuron_module)}. "
            f"Expected ParametricLIFNode with 'w' or 'tau' attribute."
        )


def fold_bn_into_conv(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fold BatchNorm and PLIF tau into convolution weights (Paper Eq. 7-8).

    For neuromorphic deployment, BN statistics and the PLIF decay factor
    are absorbed into the convolution weights and bias:

        scale = gamma / (tau * sqrt(running_var + eps))
        W_folded = W_conv * scale.reshape(-1, 1, 1, 1)
        b_folded = (b_conv - running_mean) * scale + beta / tau

    where gamma = bn.weight, beta = bn.bias, and b_conv defaults to 0
    if the conv has no bias (bias=False).

    Args:
        conv: nn.Conv2d layer (may have bias=False)
        bn: nn.BatchNorm2d layer with running statistics
        tau: Learned PLIF time constant for this layer

    Returns:
        folded_weight: [C_out, C_in, kH, kW] folded weights
        folded_bias: [C_out] folded bias
    """
    assert bn.running_mean is not None, "BN must have running statistics (eval mode)"
    assert bn.running_var is not None, "BN must have running statistics (eval mode)"

    W = conv.weight.data.clone()  # [C_out, C_in, kH, kW]
    gamma = bn.weight.data.clone()  # [C_out]
    beta = bn.bias.data.clone()  # [C_out]
    mean = bn.running_mean.data.clone()  # [C_out]
    var = bn.running_var.data.clone()  # [C_out]
    eps = bn.eps

    # Conv bias (0 if bias=False)
    if conv.bias is not None:
        b_conv = conv.bias.data.clone()
    else:
        b_conv = torch.zeros(W.shape[0], device=W.device)

    # Paper Eq. 7: scale factor per output channel
    std = torch.sqrt(var + eps)
    scale = gamma / (tau * std)  # [C_out]

    # Paper Eq. 7: folded weight
    folded_weight = W * scale.reshape(-1, 1, 1, 1)

    # Paper Eq. 8: folded bias (shift)
    folded_bias = (b_conv - mean) * scale + beta / tau

    return folded_weight, folded_bias


def quantize_weights_int8(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to int8 using per-output-channel symmetric quantization.

    Paper Section 5.1: "int8 per-output-channel" quantization.

    For each output channel, find the max absolute value and compute
    a scale factor to map float values to [-127, 127].

    Args:
        weight: [C_out, C_in, kH, kW] float weights

    Returns:
        weight_int8: [C_out, C_in, kH, kW] quantized weights (int8 stored as float for Lava)
        scale: [C_out] per-channel scale factors (float_val = int8_val * scale)
    """
    C_out = weight.shape[0]

    # Per-output-channel max absolute value
    w_flat = weight.reshape(C_out, -1)
    max_abs = w_flat.abs().max(dim=1).values  # [C_out]

    # Avoid division by zero
    max_abs = torch.clamp(max_abs, min=1e-8)

    # Scale: maps float range to [-127, 127]
    scale = max_abs / 127.0  # [C_out]

    # Quantize
    weight_q = torch.round(weight / scale.reshape(-1, 1, 1, 1))
    weight_q = torch.clamp(weight_q, -127, 127)

    return weight_q, scale


def fold_and_quantize_snn_backbone(
    snn_backbone: nn.Module,
) -> Dict[str, Dict]:
    """
    Fold BN + tau and quantize all SNN backbone layers.

    Iterates over the 4 SNN blocks (block1-block4), extracting conv, bn,
    and PLIF neuron from each, then performs folding and quantization.

    Args:
        snn_backbone: SNNBackbone module (must be in eval mode)

    Returns:
        layers: Dict mapping block name -> {
            'weight_int8': quantized weights,
            'weight_scale': per-channel scales,
            'bias': folded bias (float),
            'tau': learned tau value,
            'in_channels': int,
            'out_channels': int,
            'kernel_size': int,
            'stride': int,
            'padding': int,
        }
    """
    snn_backbone.eval()
    layers = {}

    for block_name in ['block1', 'block2', 'block3', 'block4']:
        block = getattr(snn_backbone, block_name)

        # Extract components
        conv = block.conv
        bn = block.bn
        neuron = block.neuron
        tau = extract_plif_tau(neuron)

        # Fold BN into conv
        folded_weight, folded_bias = fold_bn_into_conv(conv, bn, tau)

        # Quantize to int8
        weight_int8, weight_scale = quantize_weights_int8(folded_weight)

        layers[block_name] = {
            'weight_int8': weight_int8,
            'weight_scale': weight_scale,
            'bias': folded_bias,
            'tau': tau,
            'in_channels': conv.in_channels,
            'out_channels': conv.out_channels,
            'kernel_size': conv.kernel_size[0],
            'stride': conv.stride[0],
            'padding': conv.padding[0],
        }

    return layers


def fold_and_quantize_snn_blocks(
    snn_blocks: nn.ModuleList,
) -> List[Dict]:
    """
    Fold BN + tau and quantize SNN blocks from a NAS ConfigurableBackbone.

    Works with nn.ModuleList of SNNConvBlock (the NAS model uses
    backbone.snn_blocks instead of backbone.snn.block1/2/3/4).

    Args:
        snn_blocks: nn.ModuleList of SNNConvBlock modules (must be in eval mode)

    Returns:
        layers: List of dicts (one per block), each containing:
            'weight_int8': quantized weights,
            'weight_scale': per-channel scales,
            'bias': folded bias (float),
            'tau': learned tau value,
            'in_channels': int,
            'out_channels': int,
            'kernel_size': int,
            'stride': int,
            'padding': int,
    """
    layers = []

    for i, block in enumerate(snn_blocks):
        block.eval()

        # Extract components (same structure as SNNConvBlock: .conv, .bn, .neuron)
        conv = block.conv
        bn = block.bn
        neuron = block.neuron
        tau = extract_plif_tau(neuron)

        # Fold BN into conv
        folded_weight, folded_bias = fold_bn_into_conv(conv, bn, tau)

        # Quantize to int8
        weight_int8, weight_scale = quantize_weights_int8(folded_weight)

        layers.append({
            'weight_int8': weight_int8,
            'weight_scale': weight_scale,
            'bias': folded_bias,
            'tau': tau,
            'in_channels': conv.in_channels,
            'out_channels': conv.out_channels,
            'kernel_size': conv.kernel_size[0],
            'stride': conv.stride[0],
            'padding': conv.padding[0],
        })

    return layers
