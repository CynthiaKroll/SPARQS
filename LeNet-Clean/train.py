import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import snntorch

from models.lenet import LeNet
from models.branchy_models import BranchyLeNet
from models.branchy_quantized_snn import BranchyQuantizedSNN
from models.lenet_snn import LeNetSNN
from models.quantized_snn_lenet import QuantizedSNNLeNet, get_quant_engine

from config import config
from data_loader import get_data_loaders, get_validation_loader
import torch.nn.functional as F  # for softmax
from models.rl_environment import QLearningAgent


def _detach_lif_state(model):
    """Detach snntorch LIF membrane/spike buffers, which become non-leaf tensors
    after a forward+backward pass and otherwise block deepcopy / INT8 conversion."""
    for mod in model.modules():
        if isinstance(mod, snntorch.Leaky):
            for attr in ('mem', 'spk'):
                v = getattr(mod, attr, None)
                if torch.is_tensor(v):
                    setattr(mod, attr, v.detach())
    return model


def _rl_update_from_batch(rl_agent, exit_outputs, final_output, labels):
    """Train the Q-learning exit policy on one batch, following the paper (Sec. III-B).

    Each sample is one episode: at each early exit the agent picks {continue, exit}
    with an epsilon-greedy policy, receives the Eq.1 reward
        R = +1 + alpha * savings   (correct)      /   -1   (incorrect)
    and the Q-table is updated with the Bellman rule (Eq.2). Continuing carries no
    immediate reward; its value is bootstrapped from the next exit's state.

    end_episode() is called once per batch, which gives ~422 * epochs episodes
    (~8.4k over 20 epochs on MNIST) -- matching the paper's stated 5,000-10,000
    episodes. (The previous version force-updated BOTH actions with immediate
    rewards -- a contextual bandit, not the paper's sequential Q-learning -- and
    never called select_action at all.)
    """
    alpha = config["rl"].get("reward_scale", 0.3)
    n_ex = len(exit_outputs)
    n_stages = n_ex + 1  # early exits plus the final exit

    with torch.no_grad():
        confs = [torch.softmax(e, dim=1).max(1)[0] for e in exit_outputs]
        preds = [e.argmax(1) for e in exit_outputs]
        final_pred = final_output.argmax(1)

    for b in range(labels.size(0)):
        y = labels[b]
        for i in range(n_ex):
            state = rl_agent.get_state(i, confs[i][b].item())
            action = rl_agent.select_action(state)  # epsilon-greedy
            if action == 1:  # exit now -> terminal, reward from this exit's prediction
                savings = (n_stages - (i + 1)) / n_stages
                r = (1.0 + alpha * savings) if preds[i][b] == y else -1.0
                rl_agent.update(state, 1, float(r), done=True)
                break
            # action 0 = continue
            if i == n_ex - 1:
                # continuing past the last early exit lands at the final exit
                # (terminal, no compute saved)
                r = 1.0 if final_pred[b] == y else -1.0
                rl_agent.update(state, 0, float(r), done=True)
            else:
                next_state = rl_agent.get_state(i + 1, confs[i + 1][b].item())
                rl_agent.update(state, 0, 0.0, next_state=next_state, done=False)

    rl_agent.end_episode()  # one episode per batch


def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs and paired targets with lambda.'''
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, outputs, y_a, y_b, lam, weights):
    '''MixUp loss for multiple exit outputs.'''
    loss_a = sum(w * criterion(o, y_a) for w, o in zip(weights, outputs))
    loss_b = sum(w * criterion(o, y_b) for w, o in zip(weights, outputs))
    return lam * loss_a + (1 - lam) * loss_b


def verify_quantization(model):
    """Verify that model has been properly quantized (INT8 Conv/Linear present)."""
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.quantized.Conv2d))
    n_lin = sum(1 for m in model.modules() if isinstance(m, nn.quantized.Linear))
    print(f"Verified quantized layers: {n_conv} Conv2d + {n_lin} Linear (INT8)")
    return n_conv + n_lin


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device


def get_model(model_name, num_classes, **kwargs):
    if model_name == 'standard_lenet':
        return LeNet(num_classes=num_classes)
    elif model_name == 'branchy_lenet':
        return BranchyLeNet(num_classes=num_classes)
    elif model_name == 'branchy_quantized_snn':
        return BranchyQuantizedSNN(num_classes=num_classes,
                                   time_window=kwargs.get('time_window', config["quantized_num_steps"]),
                                   beta=kwargs.get('beta', config["snn"]["beta"]))
    elif model_name == 'lenet_snn':
        return LeNetSNN(num_classes=num_classes, beta=kwargs.get('beta', config["snn"]["beta"]),
                        time_steps=kwargs.get('time_window', config["num_steps"]))
    elif model_name == 'quantized_snn_lenet':
        return QuantizedSNNLeNet(num_classes=num_classes,
                                 beta=kwargs.get('beta', config["snn"]["beta"]),
                                 time_steps=kwargs.get('time_window', config["quantized_num_steps"]))
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ------------------------------------------------------------------ QDSNN
def train_branchy_quantized_snn(model, train_loader, val_loader, device, lr, num_epochs=100,
                                save_path=None, qat=False, rl_agent=None, agent_output=None):
    """Train the QDSNN (LeNet backbone + 2 early exits), optionally with real
    INT8 QAT and an RL-learned early-exit policy.

    The old code path never reached this function -- main() routed the QDSNN to
    train_snn_lenet, so it never did QAT and never trained the RL agent."""
    print("[INFO] Training Branchy Quantized SNN (QDSNN) LeNet...")

    if qat:
        print("[INFO] QDSNN QAT: fusing Conv/Linear+BN and preparing fake-quant "
              "(exit heads stay FP32)")
        torch.backends.quantized.engine = get_quant_engine()
        model.fuse_model()
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        )
        for _, module in model.named_modules():
            if isinstance(module, snntorch.Leaky):
                module.qconfig = None
        torch.quantization.prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixup augmentation (paper Sec. III-C), applied to the images BEFORE
            # the model's internal spike encoding. config["mixup_alpha"]=0 disables it.
            mixed, y_a, y_b, lam = mixup_data(images, labels, config["mixup_alpha"])
            exit_outputs, final_output, _ = model(mixed)  # model Poisson-encodes internally
            outputs = exit_outputs + [final_output]

            exit_loss_weights = getattr(model, 'exit_loss_weights', [0.3, 0.3, 1.0])
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam, exit_loss_weights)

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
            # images against unblended labels and will look low; only Val Acc is
            # meaningful.
            _, predicted = torch.max(final_output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_bar.set_postfix({'loss': running_loss / total, 'acc': 100. * correct / total})

            # --- RL agent: learn the exit-vs-continue policy from this batch ---
            # Mixed inputs make "is this prediction correct?" ill-defined, so when
            # mixup actually mixed (lam < 1) take a clean forward for the RL signal.
            if rl_agent is not None:
                if lam < 1.0 - 1e-6:
                    with torch.no_grad():
                        eo_rl, fo_rl, _ = model(images)
                else:
                    eo_rl, fo_rl = exit_outputs, final_output
                _rl_update_from_batch(rl_agent, eo_rl, fo_rl, labels)

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.2f}%")

        if rl_agent is not None:
            model.confidence_thresholds = rl_agent.get_confidence_thresholds()
            print(f"RL thresholds: {model.confidence_thresholds} | epsilon={rl_agent.exploration_rate:.3f} "
                  f"| episodes={rl_agent.episode}")

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                exit_outputs, final_output, exit_pts = model(images)

                # In eval mode the model takes the early-exit path, so each
                # sample must be scored by the exit it actually took:
                # final_output is FROZEN (near-zero) for early-exited samples,
                # and scoring it for everyone pins val acc at chance whenever
                # exits engage, wrecking best-checkpoint selection.
                outputs = final_output
                if exit_pts is not None and exit_outputs:
                    outputs = final_output.clone()
                    pts = exit_pts.to(outputs.device)
                    for i, eo in enumerate(exit_outputs):
                        mask = pts == (i + 1)
                        if mask.any():
                            outputs[mask] = eo[mask]

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                if qat:
                    engine = get_quant_engine()
                    torch.backends.quantized.engine = engine
                    _detach_lif_state(model)
                    m_conv = copy.deepcopy(model).eval().to('cpu')
                    m_conv = torch.quantization.convert(m_conv, inplace=False)
                    torch.save(m_conv.state_dict(), save_path)
                    model.to(device)
                    print(f"Converted INT8 model saved to {save_path} (engine={engine})")
                else:
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                if rl_agent is not None:
                    agent_path = agent_output or save_path.replace('.pth', '_agent.pkl')
                    rl_agent.save(agent_path)
                    print(f"RL agent saved to {agent_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model


# ------------------------------------------------------------------ QSNN
def train_quantized_snn_lenet(model, train_loader, val_loader, device, lr, num_epochs=100,
                              save_path=None, qat=False):
    """Train the QSNN (LeNet backbone, INT8 Conv/Linear + FP32 LIF).

    The old code path routed the QSNN to train_snn_lenet and never did QAT, so
    the 'quantized' checkpoint was actually FP32."""
    print("[INFO] Training Quantized SNN (QSNN) LeNet...")

    if qat:
        print("[INFO] QSNN QAT: fusing Conv/Linear+BN and preparing fake-quant")
        torch.backends.quantized.engine = get_quant_engine()
        model.fuse_model()
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        )
        for _, module in model.named_modules():
            if isinstance(module, snntorch.Leaky):
                module.qconfig = None
        torch.quantization.prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixup augmentation (paper Sec. III-C), applied to the images
            # BEFORE spike encoding. config["mixup_alpha"]=0 disables it.
            mixed, y_a, y_b, lam = mixup_data(images, labels, config["mixup_alpha"])
            outputs = model(mixed).sum(dim=0)  # [T,B,C] -> [B,C]; model encodes internally
            loss = mixup_criterion(criterion, [outputs], y_a, y_b, lam, [1.0])
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
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_bar.set_postfix({'loss': running_loss / total, 'acc': 100. * correct / total})

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {100.*correct/total:.2f}%")

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).sum(dim=0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                if qat:
                    engine = get_quant_engine()
                    torch.backends.quantized.engine = engine
                    _detach_lif_state(model)
                    m_conv = copy.deepcopy(model).eval().to('cpu')
                    m_conv = torch.quantization.convert(m_conv, inplace=False)
                    torch.save(m_conv.state_dict(), save_path)
                    verify_quantization(m_conv)
                    model.to(device)
                    print(f"Converted INT8 model saved to {save_path} (engine={engine})")
                else:
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model


# ------------------------------------------------------------------ plain SNN (FP32)
def train_snn_lenet(model, train_loader, val_loader, device, lr, num_epochs=100, save_path=None):
    """Train the plain FP32 SNN LeNet (LeNetSNN). Kept as an FP32 baseline; it is
    not part of the quantized QSNN/QDSNN comparison."""
    print(f"[INFO] Training FP32 SNN LeNet | lr={lr:.6f}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    time_steps = getattr(model, 'time_steps', config['num_steps'])

    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            spike_data = inputs.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
            optimizer.zero_grad()
            outputs = model(spike_data)
            if outputs.dim() == 3:
                outputs = outputs.sum(dim=0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            train_bar.set_postfix({'loss': running_loss / (train_bar.n + 1), 'acc': 100. * correct / total})

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                spike_data = inputs.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
                outputs = model(spike_data)
                if outputs.dim() == 3:
                    outputs = outputs.sum(dim=0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {val_acc:.2f}%")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


# ------------------------------------------------------------------ FP32 ANN
def train_standard_lenet(model, train_loader, val_loader, device, lr, num_epochs=100, save_path=None):
    """Train standard LeNet or the FP32 branchy LeNet."""
    print(f"[INFO] Training FP32 ANN LeNet | lr={lr:.6f}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        if hasattr(model, 'training_mode'):
            model.training_mode = True
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            model_out = model(inputs)
            outputs = model_out[0] if isinstance(model_out, tuple) else model_out
            if isinstance(outputs, list):
                weights = getattr(model, 'exit_loss_weights', [1.0 / len(outputs)] * len(outputs))
                loss = sum(w * criterion(o, labels) for w, o in zip(weights, outputs))
                predicted = outputs[-1].argmax(1)
            else:
                loss = criterion(outputs, labels)
                predicted = outputs.argmax(1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_bar.set_postfix({'loss': running_loss / (train_bar.n + 1), 'acc': 100. * correct / total})

        model.eval()
        if hasattr(model, 'training_mode'):
            model.training_mode = False
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                model_out = model(inputs)
                outputs = model_out[0] if isinstance(model_out, tuple) else model_out
                if isinstance(outputs, list):
                    outputs = outputs[-1]
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {val_acc:.2f}%")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train LeNet models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['standard_lenet', 'branchy_lenet', 'branchy_quantized_snn',
                                 'lenet_snn', 'quantized_snn_lenet'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=config['epochs'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['learning_rate'])
    parser.add_argument('--output_model', type=str, default='results/model.pth')
    parser.add_argument('--qat', action='store_true',
                        help='Enable Quantization-Aware Training (QSNN / QDSNN)')
    parser.add_argument('--time_window', type=int, default=None,
                        help='Time window T for SNN models. If omitted, each model '
                             'uses its own default: FP32 SNN = config.STANDARD_SNN_STEPS '
                             '(32), QAT pair = config.QUANTIZED_SNN_STEPS (4).')
    parser.add_argument('--beta', type=float, default=config['snn']['beta'])
    parser.add_argument('--use_rl', action='store_true',
                        help='Enable RL early-exit training for branchy_quantized_snn')
    parser.add_argument('--agent_output', type=str, default=None,
                        help='Path to save the RL agent Q-table')
    parser.add_argument('--rl_confidence_bins', type=int, default=config['rl']['n_confidence_bins'])
    args = parser.parse_args()

    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['learning_rate'] = args.lr

    device = get_device()
    # Train/val split off the MNIST TRAINING set. Validating on test_loader (as
    # before) leaked the test set into best-checkpoint selection/early stopping.
    train_loader, val_loader = get_validation_loader(batch_size=args.batch_size)

    # Only forward time_window when explicitly set, so get_model can apply the
    # correct per-model default (kwargs.get would return None, not the default,
    # if we passed time_window=None through).
    gm_kwargs = {'beta': args.beta}
    if args.time_window is not None:
        gm_kwargs['time_window'] = args.time_window
    model = get_model(args.model, config['num_classes'], **gm_kwargs).to(device)
    print(f"Created {args.model} with {sum(p.numel() for p in model.parameters())} parameters")

    if args.model == 'branchy_quantized_snn':
        agent = None
        if args.use_rl:
            rl_cfg = config["rl"]
            agent = QLearningAgent(
                n_exits=model.n_exits,
                n_confidence_bins=args.rl_confidence_bins,
                learning_rate=rl_cfg["learning_rate"],
                discount_factor=rl_cfg["discount_factor"],
                exploration_rate=rl_cfg["exploration_rate"],
                exploration_decay=rl_cfg["exploration_decay"],
                min_exploration_rate=rl_cfg["min_exploration_rate"],
            )
            model.rl_agent = agent
            print(f"RL agent enabled (exits={model.n_exits}, bins={args.rl_confidence_bins})")
        train_branchy_quantized_snn(model, train_loader, val_loader, device, args.lr,
                                    args.epochs, args.output_model, qat=args.qat,
                                    rl_agent=agent, agent_output=args.agent_output)
    elif args.model == 'quantized_snn_lenet':
        train_quantized_snn_lenet(model, train_loader, val_loader, device, args.lr,
                                  args.epochs, args.output_model, qat=args.qat)
    elif args.model == 'lenet_snn':
        train_snn_lenet(model, train_loader, val_loader, device, args.lr,
                        args.epochs, args.output_model)
    else:
        train_standard_lenet(model, train_loader, val_loader, device, args.lr,
                             args.epochs, args.output_model)

    print("Training completed!")


if __name__ == '__main__':
    main()
