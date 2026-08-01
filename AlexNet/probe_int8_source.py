"""Probe AlexNet QSNN §2.1: does convert() yield valid INT8 from the fake-quant
checkpoint? Rebuild QAT skeleton -> load fake-quant state -> convert -> check every
quantized Conv/Linear has a finite int_repr() + q_scale(). Run from AlexNet/."""
import torch
import torch.ao.quantization as tq
from config import config
from models.quantized_snn_alexnet import (QuantizedSNNAlexNet, get_quant_engine,
                                          prepare_quantized_snn_alexnet_for_qat)

CKPT = 'results/quantized_snn_alexnet_qat.pth'
torch.backends.quantized.engine = get_quant_engine()

model = QuantizedSNNAlexNet(num_classes=config['num_classes'],
                            beta=config['beta'], time_steps=config['num_steps'])
model = prepare_quantized_snn_alexnet_for_qat(model)
state = torch.load(CKPT, map_location='cpu')
model.load_state_dict(state)
model.eval().cpu()
print(f"[probe] loaded fake-quant {CKPT} OK")

conv = tq.convert(model, inplace=False)
print("[probe] convert() returned without exception")

import torch.ao.nn.quantized as nnq
n_ok = n_bad = 0
for name, m in conv.named_modules():
    if isinstance(m, (nnq.Conv2d, nnq.Linear)):
        try:
            w = m.weight()
            ir = w.int_repr()
            per_ch = w.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine)
            if per_ch:
                sc = w.q_per_channel_scales()
                scmin, scmax = float(sc.min()), float(sc.max())
                sc_finite = bool(torch.isfinite(sc).all()) and scmin > 0
                sc_str = f"per-ch[{scmin:.4g}..{scmax:.4g}] n={sc.numel()}"
            else:
                s = w.q_scale()
                sc_finite = torch.isfinite(torch.tensor(s)) and s > 0
                sc_str = f"per-tensor {s:.5g}"
            ir_ok = torch.isfinite(ir.float()).all() and ir.abs().max().item() <= 127
            status = "OK" if (sc_finite and ir_ok) else "BAD"
            n_ok += status == "OK"; n_bad += status == "BAD"
            print(f"  {status:3s} {name:28s} shape={tuple(ir.shape)} {sc_str} "
                  f"|w|max={ir.abs().max().item()}")
        except Exception as e:
            n_bad += 1
            print(f"  ERR {name:28s} {type(e).__name__}: {e}")

print(f"\n[probe] {n_ok} valid INT8 layers, {n_bad} bad")
print("[probe] VERDICT:", "convert() WORKS -> int_repr() path, PER-CHANNEL scales"
      if n_bad == 0 and n_ok > 0 else "convert() FAILS -> hand-roll BN-fold+quantize")
