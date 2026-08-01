import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from utils import encode_poisson_rate, DEFAULT_SPIKE_GAIN

# Define the surrogate gradient function (arctan as specified)
spike_grad = surrogate.atan()

class MLPSNN(nn.Module):
    """
    5-layer MLP SNN model exactly matching LeNet/AlexNet SNN pattern for MNIST dataset.
    Uses manual layer-by-layer forward pass and proper membrane potential initialization.
    """
    def __init__(self, num_classes=10, beta=0.95, thresholds=None, input_size=1*28*28, time_steps=32):
        super(MLPSNN, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.input_size = input_size
        self.time_steps = time_steps

        # Define STABLE thresholds - all set to 1.0 for consistent learning like successful SNNs
        default_thresholds = {
            'features.2': 1.0,      # Set to 1.0
            'features.6': 1.0,      # Set to 1.0  
            'features.10': 1.0,     # Set to 1.0
            'features.13': 1.0,     # Set to 1.0
            'features.16': 1.0,     # Set to 1.0
            'classifier.4': 1.0,    # Set to 1.0
            'classifier.8': 1.0,    # Set to 1.0
            'classifier.10': 1.0    # Keep as 1.0
        }
        
        # Use provided thresholds if available, otherwise use defaults
        if thresholds is None:
            thresholds = default_thresholds
        print(f"Using SNN thresholds: {thresholds}")
        print(f"Using SNN beta: {self.beta}")

        # Feature extractor layers - defined individually like LeNet/AlexNet
        # Layer 1
        self.feat_linear1 = nn.Linear(input_size, 4096)
        self.feat_bn1 = nn.BatchNorm1d(4096)
        self.feat_lif1 = snn.Leaky(beta=self.beta, threshold=thresholds.get('features.2', 1.0), 
                                   spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract')
        self.feat_drop1 = nn.Dropout(0.25)

        # Layer 2
        self.feat_linear2 = nn.Linear(4096, 2048)
        self.feat_bn2 = nn.BatchNorm1d(2048)
        self.feat_lif2 = snn.Leaky(beta=self.beta, threshold=thresholds.get('features.6', 1.0), 
                                   spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract')
        self.feat_drop2 = nn.Dropout(0.25)

        # Layer 3
        self.feat_linear3 = nn.Linear(2048, 1024)
        self.feat_bn3 = nn.BatchNorm1d(1024)
        self.feat_lif3 = snn.Leaky(beta=self.beta, threshold=thresholds.get('features.10', 1.0), 
                                   spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract')

        # Layer 4
        self.feat_linear4 = nn.Linear(1024, 512)
        self.feat_bn4 = nn.BatchNorm1d(512)
        self.feat_lif4 = snn.Leaky(beta=self.beta, threshold=thresholds.get('features.13', 1.0), 
                                   spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract')

        # Layer 5
        self.feat_linear5 = nn.Linear(512, 256)
        self.feat_bn5 = nn.BatchNorm1d(256)
        self.feat_lif5 = snn.Leaky(beta=self.beta, threshold=thresholds.get('features.16', 1.0), 
                                   spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract')
        self.feat_drop5 = nn.Dropout(0.25)

        # Classifier layers - defined individually like LeNet/AlexNet
        self.class_flatten = nn.Flatten()
        self.class_drop1 = nn.Dropout(p=0.5)
        self.class_linear1 = nn.Linear(256, 4096)
        self.class_bn1 = nn.BatchNorm1d(4096)
        self.class_lif1 = snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.4', 1.0), 
                                    spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract')

        self.class_drop2 = nn.Dropout(p=0.5)
        self.class_linear2 = nn.Linear(4096, 4096)
        self.class_bn2 = nn.BatchNorm1d(4096)
        self.class_lif2 = snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.8', 1.0), 
                                    spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract')

        self.class_linear3 = nn.Linear(4096, num_classes)
        self.class_lif3 = snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.10', 1.0), 
                                    spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism='subtract')

    def forward(self, x):
        """
        Forward pass that handles different input formats and uses proper spike encoding.
        
        Args:
            x: Either regular input [batch_size, features] or spike train [time_steps, batch_size, features]
        
        Returns:
            Membrane potentials over time [time_steps, batch_size, num_classes]
        """
        # Handle input format
        if x.dim() == 2:  # Regular input [batch_size, features]
            batch_size = x.size(0)
            x = x.view(batch_size, -1)  # Flatten if needed
            # Convert to proper spike train using Poisson encoding
            spike_train = self._encode_poisson(x, self.time_steps)
        elif x.dim() == 3:  # Already spike train format [time_steps, batch_size, features]
            spike_train = x
        else:
            raise ValueError(f"Invalid input dimension: {x.dim()}. Expected 2D or 3D tensor.")
        
        return self._forward_spike_train(spike_train)

    def _encode_poisson(self, x, num_steps, gain=DEFAULT_SPIKE_GAIN):
        """Canonical Poisson rate encoding (see utils.encode_poisson_rate)."""
        return encode_poisson_rate(x, num_steps, gain)

    def _forward_spike_train(self, spike_train):
        """
        Forward pass with spike train input.
        Manual layer-by-layer execution for proper state handling.
        
        Args:
            spike_train: Pre-encoded spike train [time_steps, batch_size, features]
        
        Returns:
            Membrane potentials over time [time_steps, batch_size, num_classes]
        """
        # Get num_steps from the input spike train
        num_steps = spike_train.size(0)
        batch_size = spike_train.size(1)
        device = spike_train.device

        # Initialize/Reset hidden states and set initial membrane potential v(0) = threshold / 2
        # This matches the LeNet/AlexNet implementation exactly
        for layer in self.modules():
            if isinstance(layer, snn.Leaky):
                # Reset internal state
                layer.reset_hidden()
                try:
                    # Get the threshold
                    threshold_val = layer.threshold
                    if isinstance(threshold_val, torch.Tensor):
                        threshold_val = threshold_val.to(device)

                    # Calculate initial potential v(0) = threshold / 2
                    initial_potential = threshold_val / 2.0

                    # Set initial membrane potential
                    if hasattr(layer, 'mem') and layer.mem is not None:
                        layer.mem = torch.ones_like(layer.mem) * initial_potential
                    else:
                        print(f"Warning: layer.mem is None after reset_hidden() for {layer}. Cannot explicitly set v(0)=theta/2.")
                except Exception as e:
                    print(f"Error setting initial membrane potential for {layer}: {e}")

        # Record membrane potentials from the final layer
        mem_out_rec = []

        # Time-step simulation loop - MANUAL LAYER-BY-LAYER like LeNet/AlexNet
        for step in range(num_steps):
            # Get input spikes for the current time step
            spk_input = spike_train[step]
            
            # Flatten the input for MLP (ensure correct batch dimension)
            if len(spk_input.shape) > 2:
                spk_input = spk_input.view(spk_input.size(0), -1)
            elif len(spk_input.shape) == 1:
                spk_input = spk_input.unsqueeze(0)

            # --- Feature Extraction (manual layer-by-layer) ---
            # Layer 1
            x1 = self.feat_linear1(spk_input)
            x1 = self.feat_bn1(x1)
            spk1 = self.feat_lif1(x1)
            x1 = self.feat_drop1(spk1)

            # Layer 2  
            x2 = self.feat_linear2(x1)
            x2 = self.feat_bn2(x2)
            spk2 = self.feat_lif2(x2)
            x2 = self.feat_drop2(spk2)

            # Layer 3
            x3 = self.feat_linear3(x2)
            x3 = self.feat_bn3(x3)
            spk3 = self.feat_lif3(x3)

            # Layer 4
            x4 = self.feat_linear4(spk3)
            x4 = self.feat_bn4(x4)
            spk4 = self.feat_lif4(x4)

            # Layer 5
            x5 = self.feat_linear5(spk4)
            x5 = self.feat_bn5(x5)
            spk5 = self.feat_lif5(x5)
            feat_out = self.feat_drop5(spk5)

            # --- Classification (manual layer-by-layer) ---
            flat_spk = self.class_flatten(feat_out)
            
            # Classifier Layer 1
            drop1 = self.class_drop1(flat_spk)
            fc1 = self.class_linear1(drop1)
            bn1 = self.class_bn1(fc1)
            spk_c1 = self.class_lif1(bn1)

            # Classifier Layer 2
            drop2 = self.class_drop2(spk_c1)
            fc2 = self.class_linear2(drop2)
            bn2 = self.class_bn2(fc2)
            spk_c2 = self.class_lif2(bn2)

            # Final Layer
            fc3 = self.class_linear3(spk_c2)
            final_layer_output = self.class_lif3(fc3)

            # snntorch Leaky with output=True returns (spk, mem): element [0] is
            # the SPIKE train, not the membrane potential. Recording spikes here
            # means downstream sum(dim=0) classifies by output spike COUNTS
            # (standard rate-coded readout).
            mem_out_rec.append(final_layer_output[0])

        # Stack recorded membrane potentials along time dimension
        mem_out_rec = torch.stack(mem_out_rec, dim=0)

        # Return the recorded membrane potentials
        return mem_out_rec  # Shape: [num_steps, batch_size, num_classes]

    # Legacy method for compatibility with existing training code
    def forward_spike_train(self, spike_train):
        """Forward method that accepts pre-encoded spike trains."""
        return self.forward(spike_train)
