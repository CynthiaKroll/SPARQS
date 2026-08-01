import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
from tqdm import tqdm
import numpy as np
import snntorch

from models.alexnet import AlexNet
from models.branchy_models import BranchyAlexNet
from models.branchy_quantized_snn import BranchyQuantizedSNN
from models.alexnet_qcfs import AlexNetQCFS # Import the QCFS model
from models.quantized_snn_alexnet import get_quant_engine

from config import config
from data_loader import get_data_loaders, get_calibration_loader
import torch.nn.functional as F  # for softmax
from models.rl_environment import QLearningAgent


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

    end_episode() is called once per batch, which gives ~352 * epochs episodes
    (~10.5k over 30 epochs on CIFAR-10) -- matching the paper's stated
    5,000-10,000 episodes. (The previous version force-updated BOTH actions with
    immediate rewards -- a contextual bandit, not the paper's sequential
    Q-learning -- and never called select_action at all.)
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
    quantized_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.quantized.Conv2d):
            assert module.weight().dtype == torch.qint8, f"{name} not quantized"
            quantized_layers += 1
    print(f"Verified {quantized_layers} quantized Conv2d layers")

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Input shapes are static (fixed batch/T/resolution), so let cuDNN
        # autotune conv kernels once and reuse them (~5-15% faster convs).
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device

def get_model(model_name, num_classes, **kwargs):
    if model_name == 'standard_alexnet':
        from models.alexnet import AlexNet
        return AlexNet(num_classes=num_classes)
    elif model_name == 'branchy_alexnet':
        return BranchyAlexNet(num_classes=num_classes)
    elif model_name == 'branchy_quantized_snn':
        return BranchyQuantizedSNN(num_classes=num_classes,
                                   time_window=kwargs.get('time_window', config["num_steps"]),
                                   beta=kwargs.get('beta', config["beta"]))
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_branchy_quantized_snn(model, train_loader, val_loader, device, lr, num_epochs=100,
                                save_path=None, qat=False, rl_agent=None, agent_output=None):
    """Train the QDSNN (AlexNet backbone + 2 early exits), optionally with real
    INT8 QAT and an RL-learned early-exit policy.

    Previously this function:
      - never called prepare_qat, so the final `convert()` was a no-op and the
        "INT8" checkpoint was actually FP32;
      - never called rl_agent.update(), so the agent was pickled with its
        random-init Q-table and its "learned" thresholds were noise;
      - had no validation loop, so it always saved the LAST epoch, not the best.
    """
    print("[INFO] Training Branchy Quantized SNN (QDSNN) AlexNet...")

    if qat:
        print("[INFO] QDSNN QAT: fusing Conv/Linear+BN and preparing fake-quant "
              "(exit heads stay FP32)")
        model.fuse_model()
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        )
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
            exit_outputs, final_output, _ = model(mixed)
            outputs = exit_outputs + [final_output]

            # Weight the exits as the model declares, rather than a hardcoded
            # [1,1,1] that drowned the final exit in early-exit loss.
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
        val_loss = 0.0
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

                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # --- Save best ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                # Save the FAKE-QUANT (QAT) model -- the exact model this epoch's
                # Val Acc was measured on. Previously (qat=True) we saved a
                # torch.quantization.convert()-ed INT8 copy instead, which was
                # NEVER validated: the destructive INT8 conversion of this hybrid
                # SNN collapsed a 65% model to ~10% (random), but the good
                # pre-convert val acc still gated the save, so a broken checkpoint
                # was written every time. evaluate_quantized_pair.py now loads
                # this fake-quant checkpoint and runs it (8-bit energy comes from
                # op-counting, not from real INT8 kernels), so the number you
                # evaluate is the number you validated.
                torch.save(model.state_dict(), save_path)
                print(f"Fake-quant (QAT) model saved to {save_path}")

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
    # NOTE: post-hoc temperature calibration was removed. It over-sharpened the
    # exit confidence (T2~0.03) and, together with the accumulator normalization,
    # made the model exit every sample at t=1 with near-random predictions. The
    # exit decision now uses raw accumulated confidence + a threshold sweep (the
    # LeNet-proven approach); temperatures stay 1.0.
    return model


def train_standard_alexnet(model, train_loader, val_loader, device, lr, num_epochs=100, save_path=None):
    """Train standard AlexNet or QCFS model."""
    print(f"Using learning rate: {lr:.6f}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Use the learning rate passed to the function
    # Ensure all parameters, including QCFS lambda_param, are optimized
    print("Optimizer will include parameters:", [name for name, param in model.named_parameters() if param.requires_grad])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # Support QuantizedDynamicSNN returning (outputs, exit_idx)
            model_out = model(inputs)
            if isinstance(model_out, tuple):
                outputs, _ = model_out
            else:
                outputs = model_out
            loss = criterion(outputs, labels)
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
                # Support QuantizedDynamicSNN returning (outputs, exit_idx)
                model_out = model(inputs)
                if isinstance(model_out, tuple):
                    outputs, _ = model_out
                else:
                    outputs = model_out
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
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
                print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc

def train_branchy_alexnet(model, train_loader, val_loader, device, num_epochs=100, save_path=None):
    """Train BranchyAlexNet model with multiple exits."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Get exit loss weights from model
    exit_loss_weights = model.exit_loss_weights
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        model.training_mode = True
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # Returns list of outputs from all exits
            
            # Calculate weighted loss from all exits
            loss = 0
            for i, exit_output in enumerate(outputs):
                loss += exit_loss_weights[i] * criterion(exit_output, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy using final exit
            _, predicted = torch.max(outputs[-1].data, 1)
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
        model.training_mode = False
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)  # In inference mode, returns (outputs, exit_points)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
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
                print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc

def train_snn_mlp(model, train_loader, val_loader, device, lr, num_epochs=100, save_path=None):
    """Train SNN MLP model with proper spike encoding - matches LeNet/AlexNet SNN pattern."""
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
    
    def encode_poisson(x, num_steps, gain=2.0):
        """Convert input batch to Poisson spike trains - matches LeNet/AlexNet encoding."""
        # Normalize input to [0,1] range
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = torch.clamp(x * gain, 0, 1)
        
        batch_size = x.size(0)
        # Flatten spatial dimensions for MLP
        x_flat = x.view(batch_size, -1)  # [batch_size, 784] for MNIST
        
        # Generate Poisson spikes: [time_steps, batch_size, features]  
        spike_prob = x_flat.unsqueeze(0).expand(num_steps, -1, -1)
        spikes = torch.rand_like(spike_prob) < spike_prob
        return spikes.float()

    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert images to spike trains [time_steps, batch_size, features]
            spike_trains = encode_poisson(inputs, num_steps, gain=2.0)
            
            optimizer.zero_grad()
            
            # Forward pass through SNN - returns [time_steps, batch_size, num_classes]
            mem_outputs = model(spike_trains)
            
            # MEAN over time (not sum): sum scales logits by T and can drive an
            # unbounded logit runaway; argmax is identical, so accuracy is unchanged.
            outputs = mem_outputs.mean(dim=0)  # [batch_size, num_classes]

            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for SNN stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
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
                spike_trains = encode_poisson(inputs, num_steps, gain=2.0)
                
                # Forward pass through SNN
                mem_outputs = model(spike_trains)
                
                # MEAN over time (matches training readout above).
                outputs = mem_outputs.mean(dim=0)  # [batch_size, num_classes]

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

def main():
    # NOTE: torch.autograd.set_detect_anomaly(True) used to be enabled here
    # unconditionally. It roughly doubles backward-pass cost and is a debugging
    # aid, not something to train with.
    parser = argparse.ArgumentParser(description="Train AlexNet models")
    parser.add_argument("--model", type=str, default="standard_alexnet",
                        choices=["standard_alexnet", "branchy_alexnet", "alexnet_qcfs", "branchy_quantized_snn"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=config["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--mixup_alpha", type=float, default=0.4, help="Alpha value for MixUp augmentation")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"],
                        help="Batch size for training")
    # Add learning rate argument
    parser.add_argument("--lr", type=float, default=config["learning_rate"],
                        help="Learning rate for optimizer")
    parser.add_argument("--output_model", type=str, default=None,
                        help="Path to save the trained model")
    parser.add_argument("--qat", action='store_true', help="Enable Quantization-Aware Training for branchy_quantized_snn")
    parser.add_argument("--use_rl", action='store_true', help='Enable RL early-exit training for branchy_quantized_snn')
    parser.add_argument("--agent_output", type=str, default='results/rl_agent.pkl', help='Path to save RL agent Q-table')
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine","step","none"], help="LR scheduler type")
    parser.add_argument("--quant_steps", type=int, default=5, help="Number of progressive quantization steps")
    parser.add_argument("--teacher_model_path", type=str, default="results/standard_alexnet.pth", help="Path to FP32 teacher model for KD")
    parser.add_argument("--kd_alpha", type=float, default=1.0, help="Alpha weight for CE in KD loss (use <1 to enable KD)")
    parser.add_argument("--kd_temperature", type=float, default=1.0, help="Temperature for KD soft targets")
    parser.add_argument("--rl_confidence_bins", type=int, default=10, help="Number of bins for RL confidence discretization")
    args = parser.parse_args()
    # Log CLI args for reproducibility
    import os, json
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_args_train.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Update config with CLI arguments BEFORE building loaders --
    # get_data_loaders reads config["batch_size"], so without this the
    # --batch_size flag was parsed but silently ignored. At T=32 the QAT
    # backward graph spans all 32 timesteps; batch 128 does not fit in 8 GB.
    config['batch_size'] = args.batch_size
    config['mixup_alpha'] = args.mixup_alpha

    # Get data loaders
    train_loader, val_loader, _, _ = get_data_loaders(distributed=False, augment=True)
    
    # Create model
    if args.model == "standard_alexnet":
        model = AlexNet(num_classes=config["num_classes"]).to(device)
        print("Created standard AlexNet model")
        
        # Train model
        train_standard_alexnet(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr, # Pass learning rate
            num_epochs=args.epochs,
            save_path=args.output_model
        )
    
    elif args.model == "branchy_alexnet":
        model = BranchyAlexNet(num_classes=config["num_classes"]).to(device)
        print("Created BranchyAlexNet model")
        
        # Train model
        train_branchy_alexnet(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            save_path=args.output_model
        )

    elif args.model == "alexnet_qcfs":
        model = AlexNetQCFS(num_classes=config["num_classes"]).to(device)
        print("Created AlexNetQCFS model")

        # Use the standard training function for QCFS model, defaulting save path if not provided
        save_path = args.output_model if args.output_model else "results/alexnet_qcfs.pth"
        train_standard_alexnet(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr,
            num_epochs=args.epochs,
            save_path=save_path
        )

    elif args.model == 'branchy_quantized_snn':
        model = BranchyQuantizedSNN(
            num_classes=config["num_classes"],
            time_window=config["num_steps"],
            beta=config.get("beta", 0.95),
        ).to(device)

        # n_exits must match the model's actual early-exit count (2). It used to
        # be hardcoded to 3, so the agent produced 3 thresholds for 2 exits.
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
            print(f"Created BranchyQuantizedSNN (QDSNN) with RL agent "
                  f"(exits={model.n_exits}, bins={args.rl_confidence_bins})")
        else:
            print("Created BranchyQuantizedSNN (QDSNN) model")

        # NOTE: fusion + prepare_qat now happen inside train_branchy_quantized_snn
        # so that the two always stay together. Fusing here (as before) without
        # ever calling prepare_qat is what made the final convert() a no-op.
        train_branchy_quantized_snn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr,
            num_epochs=args.epochs,
            save_path=args.output_model or 'results/branchy_quantized_snn_qat.pth',
            qat=args.qat,
            rl_agent=agent,
            agent_output=args.agent_output,
        )

def recalibrate_thresholds(model, calib_loader, device, num_batches=None):
    """
    Recalibrate early-exit confidence thresholds using calibration data.
    Returns model with updated model.confidence_thresholds.
    """
    model.eval()
    # Force recalibration to CPU to avoid GPU quantization kernel errors
    import torch
    device = torch.device('cpu')
    model.to(device)

    # Prepare lists for each early exit
    num_exits = model.n_exits if hasattr(model, 'n_exits') else len(model.exit_loss_weights)
    confidences = [[] for _ in range(num_exits)]
    with torch.no_grad():
        for idx, (images, _) in enumerate(calib_loader):
            images = images.to(device)
            # Only calibrate early exits. (The old line called model(images)
            # TWICE and unpacked the QDSNN's 3-tuple as a 2-tuple.)
            out = model(images)
            exit_outputs = out[0] if isinstance(out, tuple) else [out]
            for i, out in enumerate(exit_outputs):
                conf = torch.softmax(out, dim=1).max(1)[0]
                conf_8bit = (conf * 255).round() / 255
                confidences[i].append(conf_8bit.cpu())
            if num_batches and (idx+1) >= num_batches:
                break
    # Compute mean thresholds
    thresholds = [torch.cat(conf).mean().item() for conf in confidences]
    model.confidence_thresholds = thresholds
    print(f"Recalibrated thresholds: {thresholds}")
    return model

if __name__ == "__main__":
    main()