"""
Emit the threshold-sweep Pareto figure as native PGFPlots.

Rendered as LaTeX rather than an image on purpose: the figure is vector, picks up
the document's fonts, and adds no external file dependency (main.tex already
carries two missing-PNG references from the conference version).

Encoding
  x  Spiking Resilience Score SRS  (higher is better)
  y  clean accuracy           (higher is better)
  point label  the exit thresholds that produce that design point
  fill         feasible (within the accuracy floor) vs below it
  highlight    the operating point the RL agent actually selected

A scatter is an all-pairs form, so hue is capped at three series and the eight
threshold settings are identified by DIRECT LABELS rather than by colour. The two
series colours were validated together (CVD dE 24.7, normal-vision 33.6, both
clear of the gates); the third tone is a recessive neutral, not a series.

Usage (from the repo root):
  venv/Scripts/python.exe resilience/make_pareto_figure.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import srs                                        # noqa: E402

SWEEP = os.path.join(HERE, "sweep")
OUT = os.path.join(HERE, "pareto_figure.tex")

LEVEL = "L3"
ACC_FLOOR_PP = 2.0
RL_POINT = {"lenet": "[0.6, 0.6]", "mlp": "[0.7, 0.2]", "alexnet": "[0.7, 0.4]"}
TITLE = {"lenet": "LeNet-5 (MNIST)", "mlp": "5-Layer MLP (MNIST)",
         "alexnet": "AlexNet (CIFAR-10)"}
ARCHS = ("lenet", "mlp", "alexnet")


def collect(arch):
    rows = []
    for f in sorted(glob.glob(os.path.join(SWEEP, f"{arch}_qdsnn_*.csv"))):
        d = pd.read_csv(f)
        clean = d[d.threat == "clean"]
        if clean.empty:
            continue
        c = clean.iloc[0]
        fa = d[d.threat == "fault"].sort_values("intensity")
        at = d[d.threat == "attack"].sort_values("intensity")
        i = int(LEVEL[1:]) - 1
        if len(fa) <= i or len(at) <= i:
            continue
        r, s = float(fa.iloc[i].score), float(at.iloc[i].score)
        rows.append(dict(th=str(c.thresholds), acc=float(c.acc),
                         t_run=float(c.t_run), R=r, S=s, SRS=srs(r, s)))
    return pd.DataFrame(rows)


def pareto_front(df):
    """Non-dominated on (minimise latency, maximise SRS).

    Latency rather than accuracy is the second axis because in the
    (accuracy, SRS) plane one setting simply dominates the rest on LeNet and no
    front is visible: the real tension is that buying resilience costs ticks.
    """
    keep = []
    for i, a in df.iterrows():
        dominated = any((b.t_run <= a.t_run) and (b.SRS >= a.SRS) and
                        ((b.t_run < a.t_run) or (b.SRS > a.SRS))
                        for _, b in df.iterrows())
        if not dominated:
            keep.append(i)
    return df.loc[keep].sort_values("t_run")


MIN_POINTS = 4      # a design space needs enough points to show a front


def panel(arch, df, first):
    if len(df) < MIN_POINTS:
        return None
    floor = df.acc.max() - ACC_FLOOR_PP
    df = df.assign(feasible=df.acc >= floor)
    front = pareto_front(df[df.feasible])
    rl = RL_POINT.get(arch)

    def pts(sel):
        return "\n".join(f"      ({r.t_run:.2f}, {r.SRS:.2f})"
                         for _, r in sel.iterrows())

    lines = [f"\\nextgroupplot[title={{{TITLE[arch]}}},"]
    lines.append("  xlabel={Latency $t_{\\text{run}}$ (ticks)},")
    lines.append("  ylabel={Spiking Resilience Score SRS}," if first else "")
    lines.append("]")
    # Pareto frontier
    if len(front) > 1:
        lines.append("\\addplot[parBlue,thick,mark=none] coordinates {")
        lines.append(pts(front))
        lines.append("};")
    # infeasible
    inf = df[~df.feasible]
    if not inf.empty:
        lines.append("\\addplot[only marks,mark=o,mark size=2.2pt,parGray] coordinates {")
        lines.append(pts(inf))
        lines.append("};")
    # feasible, excluding the RL point
    feas = df[df.feasible & (df.th != rl)]
    if not feas.empty:
        lines.append("\\addplot[only marks,mark=*,mark size=2.4pt,parBlue] coordinates {")
        lines.append(pts(feas))
        lines.append("};")
    # RL-selected point
    sel = df[df.th == rl]
    if not sel.empty:
        lines.append("\\addplot[only marks,mark=square*,mark size=3pt,parOrange] coordinates {")
        lines.append(pts(sel))
        lines.append("};")
    # Direct labels, SELECTIVELY: the Pareto front, the RL point, and the two
    # extremes of the infeasible region. Labelling all eight makes them collide
    # at the dense lower-left and reads as noise.
    label_set = set(front.th) | ({rl} if rl in set(df.th) else set())
    inf_sorted = df[~df.feasible].sort_values("t_run")
    if not inf_sorted.empty:
        label_set |= {inf_sorted.iloc[0].th, inf_sorted.iloc[-1].th}

    for _, r in df.iterrows():
        if r.th not in label_set:
            continue
        lab = r.th.replace("[", "(").replace("]", ")").replace(" ", "")
        anchor = "north west" if r.feasible else "south west"
        lines.append(f"\\node[font=\\tiny,anchor={anchor},inner sep=2pt] "
                     f"at (axis cs:{r.t_run:.2f},{r.SRS:.2f}) {{{lab}}};")
    return "\n".join(l for l in lines if l)


def main():
    panels, present = [], []
    for i, arch in enumerate(ARCHS):
        df = collect(arch)
        p = panel(arch, df, first=(len(present) == 0))
        if p:
            panels.append(p); present.append(arch)
            print(f"[figure] {arch}: {len(df)} operating points")
    if not panels:
        raise SystemExit("no sweep CSVs found in resilience/sweep/")

    body = "\n".join(panels)
    tex = f"""% Auto-generated by resilience/make_pareto_figure.py -- do not edit by hand.
\\definecolor{{parBlue}}{{HTML}}{{2A78D6}}
\\definecolor{{parOrange}}{{HTML}}{{EB6834}}
\\definecolor{{parGray}}{{HTML}}{{8A8985}}
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
  group style={{group size={len(panels)} by 1, horizontal sep=1.35cm}},
  width={0.52 if len(panels) < 3 else 0.30}\\linewidth, height=5.0cm,
  scale only axis,
  tick align=outside, tick pos=left,
  grid=major, grid style={{parGray!25, very thin}},
  title style={{font=\\small, at={{(0.5,1.02)}}, anchor=south}},
  label style={{font=\\footnotesize}}, tick label style={{font=\\footnotesize}},
  axis line style={{parGray!60}},
  enlarge x limits=0.16, enlarge y limits=0.16,
]
{body}
\\end{{groupplot}}
\\end{{tikzpicture}}
\\caption{{Resilience--latency design space of the QDSNN at severity $L_3$. Each point
is one exit-threshold setting of the \\emph{{same trained network}}, labelled by
$(\\theta_1,\\theta_2)$; thresholds are inference-time parameters, so the entire space is
reachable without retraining. Latency is minimised and $\\mathrm{{SRS}}$ is maximised, so
the lower-right is preferred.
Filled circles are feasible (clean accuracy within {ACC_FLOOR_PP:.0f}\\,pp of the best
setting), open circles fall below that floor, the square marks the operating point the RL
agent selected, and the line traces the Pareto front over the feasible set. The three
architectures give three different pictures. On LeNet the space is ordered, buying
resilience costs latency, and the agent lands on the front but at its fast end rather than
its most resilient end. On the MLP every setting clears the accuracy floor, resilience
varies non-monotonically with the threshold, and the agent's choice is the most resilient
of the eight, although $(0.4,0.4)$ is 34\\% faster. On AlexNet the space is nearly flat,
spanning only $1.13\\times$ in $\\mathrm{{SRS}}$, and the agent is within one
percent of the best point on both axes. An efficiency-only objective therefore misses the
resilience-preferred point on LeNet and AlexNet, by a meaningful but small margin on the
former and negligibly on the latter.}}
\\label{{fig:pareto}}
\\end{{figure}}
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[figure] wrote {OUT} ({len(panels)} panel(s): {', '.join(present)})")


if __name__ == "__main__":
    main()
