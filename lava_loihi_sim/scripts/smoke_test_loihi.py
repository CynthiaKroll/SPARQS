"""
Phase 0 smoke test for the SPARQ Loihi effort.

Answers ONE question: does Intel Lava's fixed-point Loihi-2 simulation
(Loihi2SimCfg select_tag="fixed_pt") actually run a Dense -> LIF network
end-to-end on this Windows/py3.10 box, and can we read spikes back out?

It builds a tiny hand-made network:

    RingBuffer(input spikes) --Dense(W)--> LIF --> (Monitor + RingBuffer sink)

with integer weights and an integer threshold, runs it for T steps under the
bit-accurate fixed-point config, and prints the spikes it recorded. No SPARQ
checkpoints involved yet -- this only proves the toolchain works.

Run:
    venv-lava\\Scripts\\python.exe lava_loihi_sim\\scripts\\smoke_test_loihi.py
"""
import numpy as np

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as SpikeSource
from lava.proc.io.sink import RingBuffer as SpikeSink
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps


def main():
    np.random.seed(0)

    T = 8          # timesteps
    n_in = 6       # input neurons
    n_out = 4      # output (LIF) neurons

    # Deterministic binary input spike train, shaped [n_in, T] as RingBuffer wants.
    inp = (np.random.rand(n_in, T) > 0.5).astype(np.int32)

    # Integer weights in a Loihi-friendly range (8-bit signed-ish).
    W = np.random.randint(-20, 60, size=(n_out, n_in)).astype(np.int32)

    print("=" * 64)
    print("Phase 0 Loihi fixed-point smoke test")
    print("=" * 64)
    print(f"T={T}  n_in={n_in}  n_out={n_out}")
    print("input spikes [n_in, T]:")
    print(inp)

    # --- Build the process network ---
    src = SpikeSource(data=inp)                       # feeds one column per step
    dense = Dense(weights=W)                           # integer weights
    lif = LIF(shape=(n_out,), du=4095, dv=0,           # du/dv are fixed-point decays
              vth=40, bias_mant=0)                     # integer threshold
    sink = SpikeSink(shape=(n_out,), buffer=T)         # record output spikes

    src.s_out.connect(dense.s_in)
    dense.a_out.connect(lif.a_in)
    lif.s_out.connect(sink.a_in)

    # --- Run under bit-accurate fixed-point Loihi 2 simulation ---
    run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
    try:
        lif.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
        out_spikes = sink.data.get()                   # [n_out, T]
        v = lif.v.get()                                # final membrane
    finally:
        lif.stop()

    print("\noutput spikes [n_out, T]:")
    print(out_spikes.astype(int))
    print("total output spikes:", int(out_spikes.sum()))
    print("final membrane v:", np.asarray(v).astype(int))
    print("\nSMOKE TEST PASSED: fixed-point Loihi2SimCfg ran and returned spikes.")


if __name__ == "__main__":
    main()