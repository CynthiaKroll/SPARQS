import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from config import config

# Define the surrogate gradient function
spike_grad = surrogate.atan()

class LeNetSNN(nn.Module):
    def __init__(self, num_classes=10, beta=0.95, thresholds=None, time_steps=None):
        super(LeNetSNN, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        # Time window. Defaults to config.NUM_STEPS so this model trains and is
        # counted at the SAME T as the quantized pair -- a hardcoded 32 here
        # (while the pair ran at config["num_steps"]) is exactly the kind of
        # silent train/eval mismatch that made the SNN look bad.
        self.time_steps = time_steps if time_steps is not None else config["num_steps"]
        
        # Define default thresholds based on calibration
        default_thresholds = {
            'features.2': 1.0 / 20.0,
            'features.6': 1.0 / 25.0,
            'classifier.2': 1.0 / 15.0,
            'classifier.5': 1.0 / 18.0,
            'classifier.7': 1.0
        }
        
        # Use provided thresholds if available, otherwise use defaults
        if thresholds is None:
            thresholds = default_thresholds
        print(f"Using SNN thresholds: {thresholds}")
        print(f"Using SNN beta: {self.beta}")
        
        # Feature extractor using Conv -> BatchNorm -> LIF -> AvgPool
        self.features = nn.Sequential(
            # First layer
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('features.2', 1.0), 
                     spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Replaced MaxPool with AvgPool
            
            # Second layer
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('features.6', 1.0), 
                     spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Replaced MaxPool with AvgPool
        )
        
        # Classifier using Linear -> BatchNorm -> LIF
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten features before classifier
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.2', 1.0), 
                     spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'),
            
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.5', 1.0), 
                     spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'),
            
            nn.Linear(84, num_classes),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.7', 1.0), 
                     spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism='subtract')
        )
    
    def forward(self, spike_train):
        # Accept either a raw image batch [B, C, H, W] or a spike train
        # [T, B, C, H, W]. A raw batch is repeat-encoded over self.time_steps
        # (the same direct encoding used during training). Without this, the
        # op/energy counters -- which feed a 4-D image like they do for the
        # quantized models -- crashed on LeNetSNN and their exception was
        # swallowed, so this model's AC ops/energy silently came back as 0/NaN.
        if spike_train.dim() == 4:
            spike_train = spike_train.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)

        # Get num_steps from the input spike train
        num_steps = spike_train.size(0)

        # Initialize/Reset hidden states for all LIF neurons
        batch_size = spike_train.size(1)
        device = spike_train.device
        
        # Reset hidden states and set initial membrane potential v(0) = threshold / 2
        for layer in self.modules():
            if isinstance(layer, snn.Leaky):
                # Reset internal state
                layer.reset_hidden()
                try:
                    # Get the threshold
                    threshold_val = layer.threshold
                    if isinstance(threshold_val, torch.Tensor):
                        threshold_val = threshold_val.to(device)
                    
                    # Calculate initial potential
                    initial_potential = threshold_val / 2.0
                    
                    # Set initial membrane potential
                    if hasattr(layer, 'mem') and layer.mem is not None:
                        layer.mem = torch.ones_like(layer.mem) * initial_potential
                    else:
                        print(f"Warning: layer.mem is None after reset_hidden() for {layer}. Cannot explicitly set v(0)=theta/2.")
                except Exception as e:
                    print(f"Error setting initial membrane potential for {layer}: {e}")
        
        # Record spike outputs from the final layer
        mem_out_rec = []
        
        # Time-step simulation loop
        for step in range(num_steps):
            # Get input spikes for the current time step
            spk_input = spike_train[step]
            
            # Process features
            x = self.features[0](spk_input)  # Conv2d
            x = self.features[1](x)          # BatchNorm2d
            x = self.features[2](x)          # snn.Leaky
            x = self.features[3](x)          # AvgPool2d
            
            x = self.features[4](x)          # Conv2d
            x = self.features[5](x)          # BatchNorm2d
            x = self.features[6](x)          # snn.Leaky
            x = self.features[7](x)          # AvgPool2d
            
            # Process classifier
            x = self.classifier[0](x)        # Flatten
            x = self.classifier[1](x)        # Linear
            x = self.classifier[2](x)        # BatchNorm1d
            x = self.classifier[3](x)        # snn.Leaky
            
            x = self.classifier[4](x)        # Linear
            x = self.classifier[5](x)        # BatchNorm1d
            x = self.classifier[6](x)        # snn.Leaky
            
            x = self.classifier[7](x)        # Linear
            final_layer_output = self.classifier[8](x)  # snn.Leaky with output=True
            
            # Store the membrane potential for this time step
            mem_out_rec.append(final_layer_output[0])
        
        # Stack recorded membrane potentials along time dimension
        mem_out_rec = torch.stack(mem_out_rec, dim=0)
        
        # Return the recorded membrane potentials
        return mem_out_rec  # Shape: [num_steps, batch_size, num_classes]