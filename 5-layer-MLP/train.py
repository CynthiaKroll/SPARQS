import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import os
import time
import random
import numpy as np
from tqdm import tqdm

from models.mlp import MLP
from models.mlp_snn import MLPSNN
from models.branchy_mlp import BranchyMLP
from models.quantized_snn_mlp import (QuantizedSNNMLP, prepare_quantized_snn_mlp_for_qat,
                                      convert_quantized_snn_mlp)
from models.branchy_quantized_snn_mlp import BranchyQuantizedSNNMLP
from models.rl_environment import QLearningAgent

from config import config
from data_loader import get_data_loaders, get_calibration_loader, get_validation_loader
from utils import encode_poisson_rate, DEFAULT_SPIKE_GAIN
import torch.nn.functional as F
import snntorch


def _detach_lif_state(model):
    """Detach snntorch LIF membrane/spike buffers, which become non-leaf tensors
    after a forward+backward pass and otherwise block deepcopy/INT8 conversion."""
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

    end_episode() is called once per batch, which gives ~421 * epochs episodes
    (~8.4k over 20 epochs) -- matching the paper's stated 5,000-10,000 episodes.
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
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, outputs, y_a, y_b, lam, weights):
    '''MixUp loss for multiple exit outputs.'''
    losses_a = [criterion(o, y_a) for o in outputs]
    losses_b = [criterion(o, y_b) for o in outputs]
    loss_a = sum(w * l for w, l in zip(weights, losses_a))
    loss_b = sum(w * l for w, l in zip(weights, losses_b))
    return lam * loss_a + (1 - lam) * loss_b

def verify_quantization(model):
    quantized_convs = 0
    quantized_linears = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Conv2d):
            quantized_convs += 1
        elif isinstance(module, torch.nn.quantized.Linear):
            quantized_linears += 1
    print(f"Verified {quantized_convs} quantized Conv2d and {quantized_linears} quantized Linear layers")

def get_device():
    """
    Check for available GPU devices and return the appropriate device.
    Priority: CUDA > MPS > CPU
    """
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
    """Get model instance based on model name."""
    if model_name == 'mlp':
        return MLP(num_classes=num_classes)
    elif model_name == 'snn_mlp':
        # Get thresholds from the logged output if available, otherwise use defaults
        thresholds = kwargs.get('thresholds', None)
        return MLPSNN(num_classes=num_classes, 
                      beta=kwargs.get('beta', config['snn']['beta']),
                      thresholds=thresholds,
                      time_steps=kwargs.get('time_steps', config['num_steps']))
    elif model_name == 'branchy_mlp':
        return BranchyMLP(num_classes=num_classes)
    elif model_name == 'quantized_snn_mlp':
        return QuantizedSNNMLP(num_classes=num_classes,
                               beta=kwargs.get('beta', config['snn']['beta']),
                               time_steps=kwargs.get('time_steps', config['num_steps']))
    elif model_name == 'branchy_quantized_snn_mlp':
        return BranchyQuantizedSNNMLP(num_classes=num_classes,
                                      time_window=kwargs.get('time_steps', config['num_steps']),
                                      beta=kwargs.get('beta', config['snn']['beta']))
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_branchy_quantized_snn(model, train_loader, val_loader, device, lr, num_epochs=100, save_path=None, qat=False, rl_agent=None):
    """Train Branchy Quantized SNN MLP."""
    print("[INFO] Training Branchy Quantized SNN MLP...")

    if qat:
        print("[INFO] QDSNN QAT: fusing Linear+BN and preparing fake-quant (exit heads stay FP32)")
        model.fuse_model()
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        )
        torch.quantization.prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # Flatten images for MLP input (key difference from LeNet)
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # Mixup augmentation (paper Sec. III-C). config["mixup_alpha"]=0 disables it.
            mixed, y_a, y_b, lam = mixup_data(images, labels, config["mixup_alpha"])
            exit_outputs, final_output, _ = model(mixed)
            outputs = exit_outputs + [final_output]

            # Weighted mixup loss across all exits
            exit_loss_weights = model.exit_loss_weights if hasattr(model, 'exit_loss_weights') else [0.3, 0.3, 1.0]
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam, exit_loss_weights)

            loss.backward()

            # Clip gradients to stabilize QAT training; skip the step entirely on
            # non-finite gradients (one poisoned step NaNs the whole run).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(grad_norm):
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            else:
                optimizer.zero_grad(set_to_none=True)
                print("[WARN] Skipped optimizer step: non-finite loss/gradients in this batch")
            _, predicted = torch.max(outputs[-1], 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

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

        train_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.4f}")
        if rl_agent is not None:
            model.confidence_thresholds = rl_agent.get_confidence_thresholds()
            print(f"RL thresholds: {model.confidence_thresholds} | epsilon={rl_agent.exploration_rate:.3f} "
                  f"| episodes={rl_agent.episode}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Flatten images for MLP input
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)

                exit_outputs, final_output, exit_pts = model(images)

                # In eval mode the model takes the early-exit path, so each
                # sample must be scored by the exit it actually took. final_output
                # is FROZEN (near-zero) for early-exited samples -- scoring it for
                # everyone pinned val acc at chance whenever exits engaged, which
                # broke best-checkpoint selection and triggered early stopping.
                outputs = final_output
                if exit_pts is not None and exit_outputs:
                    outputs = final_output.clone()
                    pts = exit_pts.to(outputs.device)
                    for i, eo in enumerate(exit_outputs):
                        mask = pts == (i + 1)
                        if mask.any():
                            outputs[mask] = eo[mask]

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss/val_total:.4f} | Val Acc: {100*val_acc:.2f}%")
        
        # Learning rate scheduler step
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if qat:
                    # Convert a CPU copy to a real INT8 model for saving; keep the
                    # fake-quant model on-device so training can continue.
                    import copy
                    engine = ('fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines
                              else torch.backends.quantized.supported_engines[0])
                    torch.backends.quantized.engine = engine
                    # snntorch LIF neurons retain a non-leaf membrane buffer after a
                    # forward pass; detach it so the model can be deep-copied/converted.
                    _detach_lif_state(model)
                    m_conv = copy.deepcopy(model).eval().to('cpu')
                    m_conv = torch.quantization.convert(m_conv, inplace=False)
                    torch.save(m_conv.state_dict(), save_path)
                    print(f"Converted INT8 model saved to {save_path} (engine={engine})")
                else:
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                if rl_agent is not None:
                    agent_path = save_path.replace('.pth', '_agent.pkl')
                    rl_agent.save(agent_path)
                    print(f"RL agent saved to {agent_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print(f"Training completed. Best validation accuracy: {100*best_val_acc:.2f}%")
    return model

def train_standard_mlp(model, train_loader, val_loader, device, lr, num_epochs=100, save_path=None):
    """Train standard MLP."""
    print("[INFO] Training Standard MLP...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    # Paper (Sec. III-C) specifies a cosine annealing schedule for all models.
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc="Training")
        
        for images, labels in train_bar:
            images = images.view(images.size(0), -1).to(device)  # Flatten for MLP
            labels = labels.to(device)

            optimizer.zero_grad()
            # Mixup augmentation (paper Sec. III-C). config["mixup_alpha"]=0 disables it.
            mixed, y_a, y_b, lam = mixup_data(images, labels, config["mixup_alpha"])
            outputs = model(mixed)
            loss = mixup_criterion(criterion, [outputs], y_a, y_b, lam, [1.0])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': running_loss / (train_bar.n + 1),
                'acc': 100. * correct / total
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config["early_stopping_patience"]:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        scheduler.step()
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model

def train_quantized_snn_mlp(model, train_loader, val_loader, device, lr, num_epochs=100, save_path=None, qat=False):
    """Train quantized SNN MLP model with proper quantization-aware training."""
    print(f"[INFO] Training Quantized SNN MLP... QAT: {qat}")
    print(f"Using learning rate: {lr:.6f}")

    # Use the base LR directly, matching the LeNet/AlexNet QSNN trainers (all
    # quantized models train at 0.001 across the three modules). The old 5x
    # multiplier (effective 0.005) was both inconsistent and implicated in the
    # NaN divergence during QAT.
    snn_lr = lr
    
    # Enable QAT if requested
    if qat:
        print("[INFO] Enabling Quantization Aware Training (QAT): fusing Linear+BN, "
              "excluding LIF layers, inserting fake-quant modules")
        # Fusion is REQUIRED here: without it the converted INT8 model keeps
        # standalone FP32 BatchNorm1d layers that crash on quantized inputs.
        model = prepare_quantized_snn_mlp_for_qat(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=snn_lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Get time steps from model config 
    num_steps = getattr(model, 'time_steps', 32)
    
    def encode_poisson(x, num_steps, gain=DEFAULT_SPIKE_GAIN):
        """Canonical Poisson rate encoding (shared with evaluate.py, the models, and
        the op/energy counters). See utils.encode_poisson_rate."""
        return encode_poisson_rate(x, num_steps, gain)

    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Training")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup augmentation (paper Sec. III-C), applied to the images BEFORE
            # spike encoding. config["mixup_alpha"]=0 disables it.
            mixed, y_a, y_b, lam = mixup_data(inputs, labels, config["mixup_alpha"])

            # Convert images to spike trains [time_steps, batch_size, features]
            spike_trains = encode_poisson(mixed, num_steps)

            optimizer.zero_grad()

            # Forward pass through SNN - returns [time_steps, batch_size, num_classes]
            mem_outputs = model(spike_trains)

            # Sum membrane potentials over time for final prediction
            outputs = mem_outputs.sum(dim=0)  # [batch_size, num_classes]

            loss = mixup_criterion(criterion, [outputs], y_a, y_b, lam, [1.0])
            loss.backward()

            # Gradient clipping for SNN stability. If this batch produced
            # non-finite gradients (overflow/NaN), SKIP the optimizer step:
            # clipping cannot save it (inf norm -> 0 scale -> NaN weights) and a
            # single poisoned step makes the rest of the run unrecoverable
            # (train acc pinned at ~10% with loss=nan).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(loss) and torch.isfinite(grad_norm):
                optimizer.step()
                running_loss += loss.item()
            else:
                optimizer.zero_grad(set_to_none=True)
                print("[WARN] Skipped optimizer step: non-finite loss/gradients in this batch")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': running_loss / (train_bar.n + 1),
                'acc': 100. * correct / total
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Convert images to spike trains
                spike_trains = encode_poisson(inputs, num_steps)
                
                # Forward pass through SNN
                mem_outputs = model(spike_trains)
                
                # Sum membrane potentials over time for final prediction
                outputs = mem_outputs.sum(dim=0)  # [batch_size, num_classes]
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduler step
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                
                # Convert to quantized model before saving if QAT was used
                if qat:
                    print("[INFO] Converting QAT model to quantized model before saving")
                    # Detach non-leaf LIF membrane buffers so the model can be deep-copied.
                    _detach_lif_state(model)
                    model_quantized = convert_quantized_snn_mlp(model)
                    torch.save(model_quantized.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model

# REMOVED DUPLICATE train_snn_mlp FUNCTION - Using the one defined later in file

def train_snn_mlp(model, train_loader, val_loader, device, lr, num_epochs=100, save_path=None):
    """Train SNN MLP model with proper spike encoding - matches LeNet/AlexNet SNN pattern exactly."""
    print(f"[INFO] Training SNN MLP...")
    print(f"Using learning rate: {lr:.6f}")
    
    # Adjust learning rate for SNN - they typically need higher learning rates
    snn_lr = lr * 5.0  # 5x multiplier like LeNet/AlexNet SNNs
    print(f"Adjusted SNN learning rate: {snn_lr:.6f}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=snn_lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Get time steps from model config 
    num_steps = getattr(model, 'time_steps', 32)
    
    def encode_poisson(x, num_steps, gain=DEFAULT_SPIKE_GAIN):
        """Canonical Poisson rate encoding (shared with evaluate.py, the models, and
        the op/energy counters). See utils.encode_poisson_rate."""
        return encode_poisson_rate(x, num_steps, gain)

    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Training")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup augmentation (paper Sec. III-C), applied to the images BEFORE
            # spike encoding. config["mixup_alpha"]=0 disables it.
            mixed, y_a, y_b, lam = mixup_data(inputs, labels, config["mixup_alpha"])

            # Convert images to spike trains [time_steps, batch_size, features]
            spike_trains = encode_poisson(mixed, num_steps)

            optimizer.zero_grad()

            # Forward pass through SNN - returns [time_steps, batch_size, num_classes]
            mem_outputs = model(spike_trains)

            # Sum membrane potentials over time for final prediction
            outputs = mem_outputs.sum(dim=0)  # [batch_size, num_classes]

            loss = mixup_criterion(criterion, [outputs], y_a, y_b, lam, [1.0])
            loss.backward()

            # Gradient clipping for SNN stability. If this batch produced
            # non-finite gradients (overflow/NaN), SKIP the optimizer step:
            # clipping cannot save it (inf norm -> 0 scale -> NaN weights) and a
            # single poisoned step makes the rest of the run unrecoverable
            # (train acc pinned at ~10% with loss=nan).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(loss) and torch.isfinite(grad_norm):
                optimizer.step()
                running_loss += loss.item()
            else:
                optimizer.zero_grad(set_to_none=True)
                print("[WARN] Skipped optimizer step: non-finite loss/gradients in this batch")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': running_loss / (train_bar.n + 1),
                'acc': 100. * correct / total
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Convert images to spike trains
                spike_trains = encode_poisson(inputs, num_steps)
                
                # Forward pass through SNN
                mem_outputs = model(spike_trains)
                
                # Sum membrane potentials over time for final prediction
                outputs = mem_outputs.sum(dim=0)  # [batch_size, num_classes]
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduler step
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc

def train_branchy_mlp(model, train_loader, val_loader, device, num_epochs=100, save_path=None):
    """Train Branchy MLP with early exits."""
    print("[INFO] Training Branchy MLP...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Check if model has custom confidence thresholds that aren't the default.
    # Compare against the model's actual default [0.9, 0.8]; otherwise the default
    # itself reads as "custom" and the RL agent is never created.
    has_custom_thresholds = (hasattr(model, 'confidence_thresholds') and
                           model.confidence_thresholds is not None and
                           model.confidence_thresholds != [0.9, 0.8])
    
    if has_custom_thresholds:
        # Model has custom thresholds - don't override with RL agent
        print(f"Using fixed confidence thresholds: {model.confidence_thresholds}")
        rl_agent = None
        model.rl_agent = None
    else:
        # Initialize RL agent for dynamic inference
        rl_agent = QLearningAgent(
            n_exits=2,
            n_confidence_bins=config["rl"]["n_confidence_bins"],
            learning_rate=config["rl"]["learning_rate"],
            exploration_rate=config["rl"]["epsilon"],
            discount_factor=config["rl"]["gamma"],  # paper Sec. III-B: gamma = 0.9
        )
        model.rl_agent = rl_agent
        print(f"Using RL agent for dynamic thresholds")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc="Training")
        
        for images, labels in train_bar:
            images = images.view(images.size(0), -1).to(device)  # Flatten for MLP
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # Mixup augmentation (paper Sec. III-C). config["mixup_alpha"]=0 disables it.
            mixed, y_a, y_b, lam = mixup_data(images, labels, config["mixup_alpha"])
            exit_outputs, final_output, exit_points = model(mixed)
            outputs = exit_outputs + [final_output]

            # Weighted mixup loss across all exits
            loss_weights = [0.3, 0.3, 1.0]
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam, loss_weights)

            loss.backward()
            optimizer.step()

            # RL agent training. Mixed inputs make "prediction correct?" ill-defined,
            # so take a clean forward for the RL reward signal when lam < 1.
            if rl_agent is not None:
                if lam < 1.0 - 1e-6:
                    with torch.no_grad():
                        eo_rl, fo_rl, _ = model(images)
                else:
                    eo_rl, fo_rl = exit_outputs, final_output
                _rl_update_from_batch(rl_agent, eo_rl, fo_rl, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs[-1], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': running_loss / (train_bar.n + 1),
                'acc': 100. * correct / total
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                
                exit_outputs, final_output, _ = model(images)
                loss = criterion(final_output, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(final_output, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        if rl_agent is not None:
            model.confidence_thresholds = rl_agent.get_confidence_thresholds()
            print(f"RL Agent Thresholds: {model.confidence_thresholds} "
                  f"| epsilon={rl_agent.exploration_rate:.3f} | episodes={rl_agent.episode}")
        else:
            print(f"Fixed Thresholds: {model.confidence_thresholds}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")
                # Save RL agent only if it exists
                if rl_agent is not None:
                    agent_path = save_path.replace('.pth', '_agent.pkl')
                    rl_agent.save(agent_path)
                    print(f"RL agent saved to {agent_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config["early_stopping_patience"]:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        scheduler.step()
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train MLP models on MNIST")
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "snn_mlp", "branchy_mlp", "quantized_snn_mlp", "branchy_quantized_snn_mlp"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=config["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"],
                        help="Batch size")
    parser.add_argument("--time_steps", type=int, default=config["num_steps"],
                        help="Time steps for SNN models")
    parser.add_argument("--qat", action="store_true",
                        help="Enable quantization-aware training")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save trained model")
    parser.add_argument("--thresholds", type=float, nargs=2, default=None,
                        help="Confidence thresholds for branchy models (e.g., --thresholds 0.6 0.7)")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Get device
    device = get_device()
    
    # Load data: train/val split off the MNIST training set. (The previous code
    # built loaders with get_data_loaders and then immediately overwrote them
    # with get_validation_loader(), silently discarding --batch_size and the
    # training augmentation.)
    train_loader, val_loader = get_validation_loader(batch_size=args.batch_size)
    
    # Create model
    model = get_model(args.model, num_classes=config["num_classes"], 
                      time_steps=args.time_steps)
    
    # Set custom thresholds for branchy models
    if args.thresholds and "branchy" in args.model:
        model.confidence_thresholds = args.thresholds
        print(f"Using custom confidence thresholds: {args.thresholds}")
    
    model.to(device)
    
    print(f"Training {args.model} for {args.epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set default save path
    if args.save_path is None:
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        model_name_map = {
            "mlp": "standard_mlp.pth",
            "snn_mlp": "snn_mlp.pth",
            "branchy_mlp": "branchy_mlp.pth",
            "quantized_snn_mlp": "quantized_snn_mlp.pth",
            "branchy_quantized_snn_mlp": "branchy_quantized_snn_mlp.pth"
        }
        args.save_path = os.path.join(save_dir, model_name_map[args.model])
    
    # Train model
    if args.model == "mlp":
        train_standard_mlp(model, train_loader, val_loader, device, args.lr, args.epochs, args.save_path)
    elif args.model == "snn_mlp":
        train_snn_mlp(model, train_loader, val_loader, device, args.lr, args.epochs, args.save_path)
    elif args.model == "branchy_mlp":
        train_branchy_mlp(model, train_loader, val_loader, device, args.epochs, args.save_path)
    elif args.model == "quantized_snn_mlp":
        train_quantized_snn_mlp(model, train_loader, val_loader, device, args.lr, args.epochs, args.save_path, args.qat)
    elif args.model == "branchy_quantized_snn_mlp":
        # Create the RL agent to learn exit thresholds, UNLESS fixed thresholds were
        # passed via --thresholds. (The old `hasattr(model,'confidence_thresholds')`
        # guard was always True, so the agent was never created.)
        rl_agent = None
        if args.thresholds is None:
            rl_agent = QLearningAgent(
                n_exits=model.n_exits,
                n_confidence_bins=config["rl"]["n_confidence_bins"],
                learning_rate=config["rl"]["learning_rate"],
                exploration_rate=config["rl"]["epsilon"],
                discount_factor=config["rl"]["gamma"],
            )
            model.rl_agent = rl_agent
            print("[INFO] QDSNN: RL agent will learn exit thresholds")
        else:
            model.confidence_thresholds = list(args.thresholds)
            print(f"[INFO] QDSNN: using fixed thresholds {model.confidence_thresholds} (RL disabled)")
        train_branchy_quantized_snn(model, train_loader, val_loader, device, args.lr, args.epochs, args.save_path, args.qat, rl_agent)
    
    print("Training completed!")

if __name__ == "__main__":
    main()