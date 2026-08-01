"""Validate AlexNet QSNN INT8 (§3.1). The local qat checkpoint has NaN in fc2's
BN running stats (old NaN-runaway leftover); the fake-quant forward is finite (BN
uses batch stats) but convert() folds the NaN running stats -> poisoned int8.
Fix: recalibrate BN (refresh running stats with a few forward batches), then
convert. Check every layer's int8 scale is finite and accuracy ~ 66.87%.

Run from AlexNet/ :  PYTHONPATH=. ../venv/Scripts/python.exe validate_int8.py --n 1000 --calib 8
"""
import argparse
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.ao.nn.quantized as nnq
import snntorch
from config import config
from models.quantized_snn_alexnet import (QuantizedSNNAlexNet, get_quant_engine,
                                          prepare_quantized_snn_alexnet_for_qat)

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--calib', type=int, default=8, help='calibration batches for BN refresh')
ap.add_argument('--bs', type=int, default=64)
args = ap.parse_args()
T = config['num_steps']
torch.backends.quantized.engine = get_quant_engine()

m = QuantizedSNNAlexNet(num_classes=config['num_classes'], beta=config['beta'], time_steps=T)
m = prepare_quantized_snn_alexnet_for_qat(m)
m.load_state_dict(torch.load('results/quantized_snn_alexnet_qat.pth', map_location='cpu'))
m.cpu()

import torchvision, torchvision.transforms as TT
tfm = TT.Compose([TT.ToTensor(), TT.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2470, 0.2435, 0.2616))])
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm)
train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tfm)
tl = torch.utils.data.DataLoader(train, batch_size=args.bs, shuffle=True)

# --- seed fc2's NaN weight observer from the (finite) raw linear weight ---
fc2 = m.classifier_seq[5]
w2 = fc2.weight.detach()
wfq = fc2.weight_fake_quant
with torch.no_grad():
    if wfq.min_val.dim() > 0:          # per-channel
        wfq.min_val.copy_(w2.amin(dim=1)); wfq.max_val.copy_(w2.amax(dim=1))
    else:
        wfq.min_val.fill_(float(w2.min())); wfq.max_val.fill_(float(w2.max()))
print(f"[recalib] seeded fc2 weight observer: finite="
      f"{torch.isfinite(wfq.min_val).all().item() and torch.isfinite(wfq.max_val).all().item()}")

# --- recalibrate BN + activation observers together (train mode, observers ON) ---
m.eval()
n_bn = 0
for mod in m.modules():
    if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
        mod.train(); mod.reset_running_stats(); mod.momentum = None
        n_bn += 1
    if 'ConvBn' in type(mod).__name__ or 'LinearBn' in type(mod).__name__:
        mod.train()
# reset fc2's NaN activation observer so it recomputes a finite range
apo = fc2.activation_post_process
if hasattr(apo, 'reset_min_max_vals'):
    apo.reset_min_max_vals()
for mod in m.modules():   # observers ON so the NaN activation ranges refresh
    if hasattr(mod, 'enable_observer'):
        mod.enable_observer()
print(f"[recalib] refreshing {n_bn} BN + activation observers over {args.calib} batches")
with torch.no_grad():
    for bi, (img, _) in enumerate(tl):
        if bi >= args.calib:
            break
        m(img)
m.eval()

bad = [k for k, v in {**dict(m.named_parameters()), **dict(m.named_buffers())}.items()
       if torch.is_tensor(v) and not torch.isfinite(v).all()]
print(f"[recalib] NaN tensors after recalibration: {len(bad)}  {bad[:6]}")

# --- convert -> int8, check per-layer scales ---
conv = tq.convert(m, inplace=False).eval().cpu()
nbad = 0
for name, mod in conv.named_modules():
    if isinstance(mod, (nnq.Conv2d, nnq.Linear)):
        w = mod.weight()
        sc = (w.q_per_channel_scales() if w.qscheme() in
              (torch.per_channel_symmetric, torch.per_channel_affine) else
              torch.tensor([w.q_scale()]))
        ok = bool(torch.isfinite(sc).all()) and float(sc.min()) > 0
        nbad += not ok
        print(f"  {'OK' if ok else 'BAD':3s} {name:20s} scale finite={ok}")
print(f"[convert] {nbad} bad layers")

N = min(args.n, len(test))
correct = 0
with torch.no_grad():
    for i in range(N):
        img, lab = test[i]
        out = conv(img.unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        correct += int(logits.argmax(1).item() == lab)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{N}  acc={correct/(i+1)*100:.2f}%")
print(f"\n[validate] INT8 (BN-recalibrated) acc = {correct/N*100:.2f}%  (ref 66.87%)  N={N}")
