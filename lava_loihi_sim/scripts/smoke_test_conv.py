"""
De-risk LeNet: does Lava's fixed-point Conv actually run under Loihi2SimCfg,
and does it compute a_out = W (*) spikes exactly for binary input?

LeNet needs Conv; the MLP work only exercised Dense. PyConvModelFixed exists
(tag 'fixed_pt') but is less exercised than Dense, so verify before building
the whole LeNet pipeline on top of it.

Checks:
  1. Conv -> LIF runs to completion under Loihi2SimCfg(select_tag='fixed_pt').
  2. Conv's integer output matches a numpy reference convolution exactly
     (accounting for Conv's 1-tick dendritic delay, same as Dense).
  3. Reports Lava's expected weight layout so the exporters use the right order.
"""
import numpy as np

from lava.proc.conv.process import Conv
from lava.proc.lif.process import LIF
from lava.proc.io.source import RingBuffer as SpikeSource
from lava.proc.io.sink import RingBuffer as SpikeSink
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps


def main():
    np.random.seed(0)
    T = 6
    C_in, H, W = 2, 8, 8
    C_out, K, S, P = 3, 3, 1, 1

    # Lava conv weight layout is (C_out, kH, kW, C_in) -- NOT torch's
    # (C_out, C_in, kH, kW). Verify and report.
    w = np.random.randint(-8, 9, size=(C_out, K, K, C_in)).astype(np.int32)
    x = (np.random.rand(T, C_in, H, W) > 0.6).astype(np.int32)

    print("=" * 64)
    print("Lava fixed-point Conv smoke test")
    print("=" * 64)
    print(f"input [T,C,H,W]={x.shape}  weight(Lava layout)={w.shape}  stride={S} pad={P}")

    # Lava conv is CHANNEL-LAST: input_shape=(X,Y,C_in), out=(X_out,Y_out,C_out).
    # RingBuffer therefore wants (X,Y,C_in,T).
    data = np.transpose(x, (2, 3, 1, 0))            # [H,W,C_in,T]
    src = SpikeSource(data=data)
    conv = Conv(input_shape=(H, W, C_in), weight=w, stride=(S, S), padding=(P, P))
    H_out = (H + 2 * P - K) // S + 1
    W_out = (W + 2 * P - K) // S + 1
    lif = LIF(shape=(H_out, W_out, C_out), du=4095, dv=0, vth=2 ** 17,
              bias_mant=0, bias_exp=0)
    sink = SpikeSink(shape=(H_out, W_out, C_out), buffer=T)

    src.s_out.connect(conv.s_in)
    conv.a_out.connect(lif.a_in)
    lif.s_out.connect(sink.a_in)

    try:
        src.run(condition=RunSteps(num_steps=T),
                run_cfg=Loihi2SimCfg(select_tag="fixed_pt"))
        v = np.asarray(lif.v.get())
        out = sink.data.get()
        print(f"RAN OK. out shape={out.shape}, final v shape={v.shape}, "
              f"|v|max={np.abs(v).max()}")
    finally:
        src.stop()

    # --- numpy reference: conv of the accumulated input over all ticks ---
    # LIF has dv=0 and a threshold too high to fire, so v == 64 * sum_t conv(x_t)
    # (act_shift=6), minus whatever is still in the 1-tick pipeline.
    wt = np.transpose(w, (0, 3, 1, 2)).astype(np.int64)     # -> (C_out,C_in,kH,kW)
    xp = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)))
    ref = np.zeros((H_out, W_out, C_out), np.int64)   # channel-last, like Lava
    for t in range(T - 1):          # last tick still in the dendritic pipeline
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = xp[t, :, i*S:i*S+K, j*S:j*S+K]
                    ref[i, j, co] += int((patch * wt[co]).sum())
    ref64 = ref * 64

    match = np.array_equal(v.astype(np.int64), ref64)
    print(f"\nnumpy reference (x64, excluding last pipelined tick) matches Lava v: {match}")
    if not match:
        diff = v.astype(np.int64) - ref64
        print(f"  max|diff|={np.abs(diff).max()}  (nonzero at {int((diff!=0).sum())} sites)")
        print("  -> weight layout or delay convention differs; inspect before building LeNet")
    print("\nCONCLUSION: fixed-point Conv is", "USABLE." if match else "NEEDS INVESTIGATION.")


if __name__ == "__main__":
    main()
