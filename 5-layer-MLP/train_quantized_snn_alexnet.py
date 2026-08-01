"""
Train and evaluate an 8-bit quantized SNN AlexNet using QAT.
- Uses QuantizedSNNAlexNet from models/quantized_snn_alexnet.py
- Supports QAT preparation, training, conversion, and evaluation.
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.quantized_snn_alexnet import (
    QuantizedSNNAlexNet,
    prepare_quantized_snn_alexnet_for_qat,
    convert_quantized_snn_alexnet,
    calibrate_snn
)

# --- Hyperparameters ---
batch_size = 128
epochs = 20
learning_rate = 1e-3
time_steps = 4  # SNN temporal dimension

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- Model Instantiation ---
# Thresholds are None, to be handled by calibration if used.
model = QuantizedSNNAlexNet(num_classes=10, beta=0.95, thresholds=None, time_steps=time_steps)

# --- Check quantization support ---
if hasattr(torch.backends, 'quantized'):
    print("Quantization engines supported:", torch.backends.quantized.supported_engines)
else:
    print("torch.backends.quantized not found - quantization support might be limited")

# --- Prepare for QAT (Simplified: uses the method from quantized_snn_alexnet.py) ---
# This function now sets the qconfig and calls prepare_qat internally.
# It also handles excluding SNN/Stub layers.
print("Preparing model for QAT using prepare_quantized_snn_alexnet_for_qat...")
model = prepare_quantized_snn_alexnet_for_qat(model)
print("Model prepared for QAT.")

# --- Training Device ---
train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(train_device)
print(f"Training on device: {train_device}")

# --- SKIPPING Fusion for Conv/BN layers ---
# Fusion failing: `qat.Conv2d` and `nn.BatchNorm2d` are not compatible fusing targets.
# Continuing without fusion, which may impact quantization quality but should allow basic execution.
print("SKIPPING fusion step - direct QAT->convert without fusion")

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# --- Dummy input spike encoding ---
def encode_direct(x, num_steps):
    # Repeat input for num_steps (rate encoding)
    # x: [batch, C, H, W] -> [num_steps, batch, C, H, W]
    return x.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)

# --- Training Loop ---
print("Starting training...")
for epoch in range(epochs):
    model.train() # Ensure model is in training mode for QAT
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(train_device), labels.to(train_device)
        spike_train = encode_direct(images, time_steps)  # [T, B, C, H, W]
        optimizer.zero_grad()
        all_step_outputs = model(spike_train)  # [T, B, Classes]
        
        losses_over_time = []
        for t_step in range(time_steps):
            loss_t = criterion(all_step_outputs[t_step], labels)
            losses_over_time.append(loss_t)
        loss = torch.stack(losses_over_time).mean()
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        if (i + 1) % 100 == 0: # Print progress every 100 batches
            print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Avg Batch Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} finished. Average Loss: {epoch_loss:.4f}")

# --- Fuse Conv+BN layers (recommended for quantized inference) ---
# print("Fusing model after training...")
# model.train() # Ensure model is in train mode for QAT module fusion
# model = fuse_model(model)
# model.eval() # Switch back to eval mode after fusion, before convert

# --- Inspect QConfig before conversion (Optional, can be verbose) ---
# print("-" * 30)
# print("Inspecting QConfig before conversion:")
# try:
#     first_conv_layer = model.features_seq[0] 
#     if hasattr(first_conv_layer, 'qconfig') and first_conv_layer.qconfig:
#         # ... (simplified inspection or remove if too verbose)
#         print(f"QConfig found on {type(first_conv_layer)}: Activation obs: {type(first_conv_layer.qconfig.activation)}, Weight obs: {type(first_conv_layer.qconfig.weight)}")
#     else:
#         print("model.features_seq[0] does not have a 'qconfig' attribute or it is None.")
# except Exception as e:
#     print(f"Error inspecting QConfig: {e}")
# print("-" * 30)

# --- Convert to Quantized Inference Model ---
print("Moving model to CPU before conversion...")
model.to('cpu') # Ensure model is on CPU before conversion for CPU-targetted quantization
print("Converting to quantized model...")
model = convert_quantized_snn_alexnet(model) # convert_quantized_snn_alexnet already does model.eval()
print("Model converted to quantized version.")

# --- Evaluate Quantized Model on CPU ---
print("--- Evaluating quantized model on CPU (required for QUInt8 ops) ---")
model.eval() # Ensure it's in eval mode
correct = 0
with torch.no_grad():
    for images, labels in test_loader: # test_loader should use batch_size
        images, labels = images.to('cpu'), labels.to('cpu')
        try:
            spike_train = encode_direct(images, time_steps) # Ensure spike_train is on CPU
            outputs = model(spike_train) # [T, B, Classes]
            
            if outputs.dim() == 3 and outputs.size(0) == time_steps:
                outputs = outputs.mean(dim=0) 
            elif outputs.dim() != 2: 
                print(f"Warning: Unexpected output dimensions after model call: {outputs.shape}")
                continue
                
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        except Exception as e:
            print(f"Error during inference: {e}")
            break

accuracy = 100. * correct / len(test_loader.dataset)
print(f"Quantized SNN AlexNet Test Accuracy (CPU): {accuracy:.2f}%")

# --- Save the quantized model ---
try:
    torch.save(model.state_dict(), 'quantized_snn_alexnet_qat.pth')
    print("Saved quantized SNN AlexNet model to quantized_snn_alexnet_qat.pth")
except Exception as e:
    print(f"Error saving model: {e}")
