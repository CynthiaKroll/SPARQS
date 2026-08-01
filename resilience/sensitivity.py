"""
Cost-model sensitivity: is the joint metric's ranking an artefact of the
ESTIMATED Loihi-2 constants?

`I_E` and `I_L` are the only components of the metric that are derived from a
cost model rather than measured directly, and that cost model uses
E_synop = 8 pJ / E_neuron = 30 pJ, which Intel has never published for Loihi 2
(main.tex Tab. loihi_constants). A reviewer is entitled to ask whether the
conclusions move if those numbers are wrong.

This re-prices every recorded condition under the PUBLISHED Loihi-1 constants
(23.6 pJ / 81 pJ, Davies et al. 2018) and recomputes R, S and SRS. No campaign is
re-run: the raw operation counts (`acs`, `neurons`, `t_run`) are recorded in the
campaign CSVs precisely so energy can be re-derived afterwards.

Note the two constants do NOT scale by a common factor (23.6/8 = 2.95 against
81/30 = 2.70), so this is a genuine perturbation of the synaptic/neuron balance
and not a trivial rescaling that would cancel in the ratio by construction.

Usage (from the repo root):
  venv/Scripts/python.exe resilience/sensitivity.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")

from metrics import _squash, aggregate, srs, W_FAULT, W_ATTACK   # noqa: E402

CONSTANTS = {
    "Loihi-2 (estimated, used in the paper)": (8.0e-12, 30.0e-12),
    "Loihi-1 (published, Davies 2018)":       (23.6e-12, 81.0e-12),
}


def energy(acs, neurons, t_run, e_syn, e_neu):
    return acs * e_syn + neurons * t_run * e_neu


def rescore(df, e_syn, e_neu):
    """Recompute R / S / SRS under one constant set."""
    out = []
    for (arch, variant), g in df.groupby(["arch", "variant"]):
        clean = g[g.threat == "clean"]
        if clean.empty:
            continue
        c = clean.iloc[0]
        e0 = energy(c.acs, c.neurons, c.t_run, e_syn, e_neu)
        if not np.isfinite(e0) or e0 <= 0:
            e0 = 1.0

        scores = {}
        for threat, w in (("fault", W_FAULT), ("attack", W_ATTACK)):
            rows = g[g.threat == threat].sort_values("intensity")
            vals = []
            for _, r in rows.iterrows():
                el = energy(r.acs, r.neurons, r.t_run, e_syn, e_neu)
                comp = dict(D_A=r.D_A, D_M=r.D_M,
                            I_E=_squash(el / e0), I_L=_squash(r.t_run / c.t_run),
                            U=r.U, H=r.H)
                vals.append(aggregate(comp, w))
            scores[threat] = vals

        n = min(len(scores["fault"]), len(scores["attack"]))
        for i in range(n):
            out.append(dict(arch=arch, variant=variant, level=f"L{i+1}",
                            R=scores["fault"][i], S=scores["attack"][i],
                            SRS=srs(scores["fault"][i], scores["attack"][i])))
    return pd.DataFrame(out)


def main():
    frames = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.csv"))):
        b = os.path.basename(f)
        if b in ("joint_metric.csv",) or b.endswith("_lowber.csv"):
            continue
        d = pd.read_csv(f)
        if "acs" not in d.columns:
            print(f"  skipping {b}: predates the op-count columns")
            continue
        frames.append(d)
    if not frames:
        raise SystemExit("no campaign CSVs carrying `acs` -- re-run the campaign")
    df = pd.concat(frames, ignore_index=True)

    tables = {name: rescore(df, *c) for name, c in CONSTANTS.items()}
    base_name = list(CONSTANTS)[0]
    base = tables[base_name]

    print(f"\n{'arch':<9}{'variant':<8}{'level':<7}"
          + "".join(f"{n.split()[0]:>14}" for n in CONSTANTS)
          + f"{'delta':>9}{'rank':>7}")
    print("-" * 78)

    agree = total = 0
    for lvl in sorted(base.level.unique()):
        for arch in sorted(base.arch.unique()):
            sub = {n: t[(t.level == lvl) & (t.arch == arch)] for n, t in tables.items()}
            order = {n: list(s.sort_values("SRS", ascending=False).variant)
                     for n, s in sub.items()}
            same = len(set(map(tuple, order.values()))) == 1
            agree += bool(same); total += 1
            for _, r in sub[base_name].sort_values("SRS", ascending=False).iterrows():
                vals = [float(t[(t.level == lvl) & (t.arch == arch) &
                                (t.variant == r.variant)].SRS.iloc[0]) for t in tables.values()]
                d = 100.0 * (vals[1] - vals[0]) / vals[0] if vals[0] > 0 else float("nan")
                print(f"{arch:<9}{r.variant:<8}{lvl:<7}"
                      + "".join(f"{v:>14.2f}" for v in vals)
                      + f"{d:>8.1f}%{'  same' if same else '  CHANGED':>7}")
    print("-" * 78)
    print(f"Variant ordering preserved in {agree}/{total} (architecture, level) cells "
          f"when the published Loihi-1 constants replace the estimated Loihi-2 ones.")


if __name__ == "__main__":
    main()
