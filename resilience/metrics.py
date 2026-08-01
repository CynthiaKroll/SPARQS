"""
The SPARQ resilience metric.

Six vulnerability components, each in [0,100] with higher = worse:

    D_A  accuracy    MRAD -- % of predictions differing from the fault-free run
    D_M  confidence  Top-1/Top-2 margin collapse on the unchanged-prediction subset
    I_E  power       energy inflation
    I_L  delay       latency inflation
    U    routing     exit-index displacement rate            (dynamic models only)
    H    hazard      exited SHALLOWER than golden AND now wrong (dynamic only)

D_A and D_M follow JASPER's definitions so the two studies stay comparable.
I_E, I_L, U and H are identically zero for a static network -- they are the
dynamic content that no existing joint metric carries.

Aggregation uses JASPER's geometric form, but applied to the COMPLEMENTARY
(resilience-polarity) terms rather than to the vulnerabilities themselves:

    R = Σ w_i v_i   under fault injection                    lower is better
    S = Σ w_i v_i   under adversarial attack                 lower is better
    SRS = sqrt((100 - R) · (100 - S))                        higher is better
    Phi = A^wA · (SRS/SRS_ref)^wR / (E^wE · L^wL)            higher is better

Why not JASPER's JVS = sqrt(R · S) directly: a geometric mean of two
VULNERABILITIES is pulled toward the SMALLER of the two, so a design point that
is nearly immune to one threat scores as robust however badly it fails the
other. On our own data the FP32 MLP ANN (R=46.29, S=1.44) scored 91.83 under
that form, statistically indistinguishable from the QDSNN's 91.87 despite
degrading 3.2x more under faults. Taking the geometric mean of (100-R) and
(100-S) instead makes a collapse on either axis unmaskable: the same ANN scores
72.76 against the QDSNN's 90.37.

Setting the four dynamic weights to zero makes R and S coincide with JASPER's
R_tot / S_tot, so the COMPONENTS stay directly comparable with that work; only
the aggregation differs, and it differs deliberately.
"""
import numpy as np

COMPONENTS = ("D_A", "D_M", "I_E", "I_L", "U", "H")

# Weights differ by threat axis, and the asymmetry is empirical (see
# RESILIENCE_NOTES.md F1/F3): under FAULTS a compute increase is the defence
# operating -- the network buying accuracy back by routing deeper -- so
# inflation is weighted lightly and the premature-exit hazard heavily. Under
# ATTACK a compute increase is the adversary's objective, so inflation is
# weighted heavily.
W_FAULT = dict(D_A=0.40, D_M=0.20, I_E=0.05, I_L=0.05, U=0.10, H=0.20)
W_ATTACK = dict(D_A=0.35, D_M=0.15, I_E=0.20, I_L=0.15, U=0.05, H=0.10)

# JASPER-equivalent weighting, for the "our metric generalizes theirs" column.
W_JASPER = dict(D_A=0.50, D_M=0.50, I_E=0.0, I_L=0.0, U=0.0, H=0.0)


def _squash(ratio):
    """Map an inflation ratio E_lambda/E_0 into [0,100).

    100·I/(1+I) with I = max(0, ratio-1): 0 when nothing inflates, 50 at a
    doubling, asymptotically 100. Bounded so a single runaway condition cannot
    dominate the weighted sum.
    """
    infl = max(0.0, float(ratio) - 1.0)
    return 100.0 * infl / (1.0 + infl)


def margin(probs):
    """Top-1 minus Top-2 probability, per sample."""
    top2 = np.sort(np.asarray(probs), axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def components(golden, threat):
    """Compute the six vulnerability components from a golden and a threat run.

    Each run is a dict with keys:
        pred    (N,)   predicted class
        probs   (N,C)  output probability distribution
        exit    (N,)   exit index, or None for a static/single-path model
        energy_j, t_run   scalars from cost.run_cost
    """
    g_pred = np.asarray(golden["pred"])
    f_pred = np.asarray(threat["pred"])
    n = len(g_pred)

    # --- D_A : MRAD, JASPER's definition (prediction change vs fault-free) ---
    d_a = 100.0 * float((f_pred != g_pred).mean())

    # --- D_M : margin collapse, measured only where the prediction survived ---
    # FP32 exponent-bit flips can drive a weight to inf/NaN, which propagates to
    # a NaN softmax. That is physically correct -- and part of why FP32 is
    # fragile -- but it leaves the margin undefined, so those samples are
    # excluded here. They are still counted by D_A, since a NaN argmax differs
    # from the golden prediction.
    g_marg, f_marg = margin(golden["probs"]), margin(threat["probs"])
    keep = (f_pred == g_pred) & np.isfinite(g_marg) & np.isfinite(f_marg)
    if keep.sum() >= 5:
        g_m = g_marg[keep].mean()
        f_m = f_marg[keep].mean()
        d_m = float(np.clip(100.0 * (1.0 - (f_m / g_m if g_m > 0 else 1.0)), 0.0, 100.0))
    else:
        # Too few surviving predictions for a stable estimate. JASPER hits the
        # same problem at high perturbation budgets and reports it as unreliable
        # rather than as a value.
        d_m = float("nan")

    # --- I_E / I_L : inflation of the modelled cost ---
    i_e = _squash(threat["energy_j"] / golden["energy_j"]) if golden["energy_j"] > 0 else 0.0
    i_l = _squash(threat["t_run"] / golden["t_run"]) if golden["t_run"] > 0 else 0.0

    # --- U / H : dynamic-only routing terms ---
    if golden.get("exit") is None or threat.get("exit") is None:
        u = h = 0.0
    else:
        g_e = np.asarray(golden["exit"])
        f_e = np.asarray(threat["exit"])
        u = 100.0 * float((f_e != g_e).mean())
        # The dangerous event: the threat pushed the sample to a SHALLOWER exit
        # AND the answer is now wrong -- cheaper and wrong. An inflation-only
        # view scores this as perfectly benign.
        shallower = (f_e < g_e)
        wrong = (f_pred != np.asarray(golden["label"]))
        h = 100.0 * float((shallower & wrong).mean())

    return dict(D_A=d_a, D_M=d_m, I_E=i_e, I_L=i_l, U=u, H=h, n=n)


def aggregate(comp, weights):
    """Weighted vulnerability score in [0,100]. NaN components are dropped and
    the remaining weights renormalized, so a missing D_M cannot silently zero
    out that part of the score."""
    tot = num = 0.0
    for k in COMPONENTS:
        v, w = comp.get(k, 0.0), weights.get(k, 0.0)
        if v is None or (isinstance(v, float) and np.isnan(v)) or w == 0:
            continue
        num += w * v
        tot += w
    return num / tot if tot > 0 else float("nan")


def srs(r, s):
    """Spiking Resilience Score, in [0,100] with HIGHER = more resilient.

    Geometric mean of the two complementary (resilience-polarity) terms, so a
    collapse on either threat axis drives the product toward zero and cannot be
    offset by strength on the other: the score is governed by the more exposed
    of the two surfaces. See the module docstring for why this is not JASPER's
    sqrt(R * S).
    """
    if np.isnan(r) or np.isnan(s):
        return float("nan")
    a = max(0.0, 100.0 - max(r, 0.0))
    b = max(0.0, 100.0 - max(s, 0.0))
    return float(np.sqrt(a * b))


def deficit(srs_value):
    """100 - SRS: the joint resilience deficit, lower = better.

    Plot-side companion to srs(). Ratios of SRS between variants sit close to
    unity and are unreadable on a shared axis; ratios of the deficit span a
    usable range, which is what the bar figures use.
    """
    return float(100.0 - srs_value)


def phi(acc, srs_value, energy, latency, ref, w=(0.4, 0.3, 0.15, 0.15)):
    """Pareto figure of merit, normalized to a reference design point.

    Weighted geometric mean (Σw = 1), so it is scale-free and unit-free, and
    zeroing a weight removes that concern from the objective. Phi(ref) = 1.
    SRS is already in resilience polarity, so it enters as a direct ratio.
    """
    wA, wR, wE, wL = w
    a = acc / ref["acc"]
    r = max(1e-9, srs_value) / max(1e-9, ref["srs"])
    e = energy / ref["energy"]
    l = latency / ref["latency"]
    return float((a ** wA) * (r ** wR) / ((e ** wE) * (l ** wL)))
