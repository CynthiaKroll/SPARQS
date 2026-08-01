"""
Security threat: shared rho-budget input perturbation.

Budget definition (comparable across all three variants)
--------------------------------------------------------
`rho` is the fraction of INPUT PIXELS the attacker controls, driven to maximum
magnitude:

  * static ANN : the selected pixel is driven to its range extreme.
  * spiking    : controlling a pixel means controlling its SPIKE TRAIN across
                 all T ticks (all-on or all-off).

A pixel's rate code over T ticks IS the spiking network's representation of that
pixel, so `rho` means the same thing for every variant and the SNN does not get
T x more perturbation power for free. This keeps the static ANN inside the joint
metric, which would otherwise be undefined for it.

Why gradient-free
-----------------
The converted INT8 modules (`torch.ao.nn.quantized.*`) implement no backward
pass, so a white-box gradient attack cannot be run against the deployed model.
We therefore use a black-box WORST-OF-K attack: draw k candidate pixel masks,
keep the one that most reduces the true-class probability. This is a genuine
adversary (it optimises against the model) and is budget-comparable, but it is
weaker than a white-box attack -- reported ASR should be read as a LOWER BOUND
on what a gradient adversary would achieve.
"""
import numpy as np
import torch


def _pixel_mask(shape_hw, rho, rng, device):
    """Boolean mask selecting a rho-fraction of pixel positions, per sample."""
    b, n = shape_hw
    k = int(round(rho * n))
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    if k <= 0:
        return mask
    for i in range(b):
        idx = rng.choice(n, size=min(k, n), replace=False)
        mask[i, torch.as_tensor(np.asarray(idx), device=device)] = True
    return mask


def apply_pixel_attack(images, mask, polarity):
    """Drive the masked pixels to a range extreme.

    Inputs are min-max normalised downstream, so the extremes of the incoming
    tensor are the strongest single-pixel perturbations available.
    """
    flat = images.reshape(images.size(0), -1).clone()
    lo, hi = float(flat.min()), float(flat.max())
    flat[mask] = hi if polarity > 0 else lo
    return flat.reshape(images.shape)


def worst_of_k(images, labels, rho, evaluate, rng, k=8):
    """Black-box rho-budget attack.

    `evaluate(imgs) -> probs [N,C]` is any callable running the model under test.
    For each candidate mask/polarity the true-class probability is measured, and
    the per-sample worst candidate is kept. Selection is PER SAMPLE, so the
    returned batch is the strongest combination found within the budget.
    """
    b = images.size(0)
    n = int(np.prod(images.shape[1:]))
    best = images.clone()
    best_p = evaluate(images)[torch.arange(b), labels].clone()

    for _ in range(k):
        for polarity in (+1, -1):
            mask = _pixel_mask((b, n), rho, rng, images.device)
            cand = apply_pixel_attack(images, mask, polarity)
            p = evaluate(cand)[torch.arange(b), labels]
            better = p < best_p
            if better.any():
                best[better] = cand[better]
                best_p[better] = p[better]
    return best
