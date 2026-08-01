import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.quantized_snn_alexnet import (
    QuantizedSNNAlexNet,
    prepare_quantized_snn_alexnet_for_qat,
    convert_quantized_snn_alexnet,
    encode_poisson,
    get_quant_engine,
)
from config import config
from data_loader import get_data_loaders
from train import mixup_data  # paper Sec. III-C: mixup for all models


def evaluate(model, loader, device, time_steps):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            spike_train = encode_poisson(images, time_steps)
            out = model(spike_train)          # [T, B, num_classes]
            logits = out.mean(dim=0)          # MEAN over time (matches training)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the quantized SNN AlexNet (QSNN)")
    parser.add_argument("--epochs", type=int, default=config["qat_num_epochs"])
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time_steps", type=int, default=config["num_steps"])
    parser.add_argument("--output_model", type=str,
                        default="results/quantized_snn_alexnet_qat.pth")
    parser.add_argument("--no_qat", action="store_true",
                        help="Train the SAME backbone in FP32 (no quantization) to "
                             "produce the FP32 SNN baseline. The QuantStub/DeQuantStub "
                             "are identities until prepare_qat swaps them, so skipping "
                             "it gives a pure FP32 SNN with identical encoding/T/eval.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Static input shapes: let cuDNN autotune conv kernels (~5-15% faster).
        torch.backends.cudnn.benchmark = True
    mode = "FP32 SNN (no QAT)" if args.no_qat else "QSNN (QAT)"
    print(f"Training on device: {device} | T={args.time_steps} | mode={mode} | engine={get_quant_engine()}")

    # get_data_loaders reads config["batch_size"]; without this line the
    # --batch_size flag was parsed but silently ignored. At T=32 the QAT
    # backward graph spans all 32 timesteps, so batch 128 does NOT fit in 8 GB
    # of GPU memory -- use --batch_size 32 (or 16) on such cards.
    config["batch_size"] = args.batch_size
    print(f"Batch size: {args.batch_size}")

    train_loader, val_loader, _, _ = get_data_loaders(distributed=False, augment=True)

    model = QuantizedSNNAlexNet(num_classes=config["num_classes"],
                                beta=config["beta"],
                                thresholds=None,
                                time_steps=args.time_steps)
    if not args.no_qat:
        model = prepare_quantized_snn_alexnet_for_qat(model)
    else:
        # FP32 baseline: leave the QuantStub/DeQuantStub as identities, no observers,
        # no fake-quant. The network is plain FP32 conv/BN/LIF end to end.
        print("Skipping prepare_qat: training the FP32 SNN baseline.")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)

            # Mixup augmentation (paper Sec. III-C), applied to the images
            # BEFORE spike encoding. config["mixup_alpha"]=0 disables it.
            mixed, y_a, y_b, lam = mixup_data(images, labels, config["mixup_alpha"])
            spike_train = encode_poisson(mixed, args.time_steps)

            optimizer.zero_grad()
            out = model(spike_train)               # [T, B, num_classes]
            # MEAN over time, not sum. sum(dim=0) scales the logits by T (=32),
            # which drives an unbounded logit/membrane runaway -> inf -> the QAT
            # observers and BN running-stats get poisoned mid-forward and every
            # subsequent batch is NaN. argmax(sum)==argmax(mean), so predictions
            # and accuracy are identical; this only rescales the loss/gradient.
            logits = out.mean(dim=0)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss.backward()
            # Clip gradients; skip the step entirely on non-finite gradients
            # (one poisoned step NaNs every weight and kills the rest of the run).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(grad_norm):
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
            else:
                optimizer.zero_grad(set_to_none=True)
                print("[WARN] Skipped optimizer step: non-finite loss/gradients in this batch")
            # NOTE: with mixup active, train accuracy is measured on BLENDED
            # images and will look low; only Val Acc is meaningful.
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            bar.set_postfix({'loss': running_loss / total, 'acc': 100. * correct / total})

        train_acc = 100. * correct / total
        val_acc = evaluate(model, val_loader, device, args.time_steps)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {running_loss/total:.4f} "
              f"| Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_dir = os.path.dirname(args.output_model)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), args.output_model)
            print(f"Best QAT (fake-quant) model saved to {args.output_model}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")

    if args.no_qat:
        # FP32 baseline: there is nothing to convert to INT8.
        print(f"FP32 SNN baseline saved to {args.output_model}")
        return

    # --- Optional: also export a real INT8 model (BEST-EFFORT) ---
    # evaluate_quantized_pair.py now evaluates the FAKE-QUANT checkpoint
    # (args.output_model) directly -- 8-bit energy comes from op-counting, not
    # real INT8 kernels -- so this converted file is only a bonus artifact. The
    # torch.quantization.convert of this hybrid SNN has been unreliable (NaN
    # scales when training observers were poisoned), so a failure here must NOT
    # sink the whole job after a good fake-quant model has already been saved.
    try:
        print("Converting best checkpoint to INT8 (optional artifact)...")
        model.load_state_dict(torch.load(args.output_model, map_location='cpu'))
        model.cpu()
        model_int8 = convert_quantized_snn_alexnet(model)
        int8_path = args.output_model.replace('.pth', '_int8.pth')
        torch.save(model_int8.state_dict(), int8_path)
        print(f"Converted INT8 QSNN saved to {int8_path}")
    except Exception as e:
        print(f"[WARN] INT8 export skipped ({type(e).__name__}: {e}). "
              f"The fake-quant checkpoint {args.output_model} is unaffected and "
              f"is what evaluate_quantized_pair.py uses.")


if __name__ == "__main__":
    main()
