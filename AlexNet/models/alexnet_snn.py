import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

spike_grad = surrogate.atan()

class AlexNetSNN(nn.Module):
    def __init__(self, num_classes=10, beta=0.95, thresholds=None): 
        super(AlexNetSNN, self).__init__()
        self.num_classes = num_classes
        self.beta = beta 

        default_thresholds = {
            'features.2': 1.0 / 26.451233 if 26.451233 > 1e-6 else 1.0,
            'features.6': 1.0 / 32.194092 if 32.194092 > 1e-6 else 1.0,
            'features.10': 1.0 / 21.991165 if 21.991165 > 1e-6 else 1.0,
            'features.13': 1.0 / 27.098610 if 27.098610 > 1e-6 else 1.0,
            'features.16': 1.0 / 13.538292 if 13.538292 > 1e-6 else 1.0,
            'classifier.4': 1.0 / 7.520811 if 7.520811 > 1e-6 else 1.0, 
            'classifier.8': 1.0 / 15.907306 if 15.907306 > 1e-6 else 1.0, 
            'classifier.10': 1.0 
        }
        if thresholds is None:
            thresholds = default_thresholds
        print(f"Using SNN thresholds: {thresholds}")
        print(f"Using SNN beta: {self.beta}")


        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('features.2', 1.0), spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'), # Ensure reset-by-subtraction
            nn.AvgPool2d(kernel_size=2, stride=2), # Replaced MaxPool with AvgPool

            # Layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('features.6', 1.0), spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'), # Ensure reset-by-subtraction
            nn.AvgPool2d(kernel_size=2, stride=2), # Replaced MaxPool with AvgPool

            # Layer 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('features.10', 1.0), spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'), # Ensure reset-by-subtraction

            # Layer 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('features.13', 1.0), spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'), # Ensure reset-by-subtraction

            # Layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('features.16', 1.0), spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'), # Ensure reset-by-subtraction
            nn.AvgPool2d(kernel_size=2, stride=2) # Replaced MaxPool with AvgPool
        )

        
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(p=0.5), 
            nn.Linear(256 * 4 * 4, 4096),
            nn.BatchNorm1d(4096),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.4', 1.0), spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'), # Ensure reset-by-subtraction

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.8', 1.0), spike_grad=spike_grad, init_hidden=True, reset_mechanism='subtract'), # Ensure reset-by-subtraction

            nn.Linear(4096, num_classes),
            
            snn.Leaky(beta=self.beta, threshold=thresholds.get('classifier.10', 1.0), spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism='subtract') # Ensure reset-by-subtraction
        )

    def forward(self, spike_train):
        
        num_steps = spike_train.size(0)

        batch_size = spike_train.size(1)
        device = spike_train.device

        for layer in self.modules():
            if isinstance(layer, snn.Leaky):
                layer.reset_hidden()
                
                try:
                    
                    threshold_val = layer.threshold
                    if isinstance(threshold_val, torch.Tensor):
                        threshold_val = threshold_val.to(device)

                    
                    initial_potential = threshold_val / 2.0

                    
                    if hasattr(layer, 'mem') and layer.mem is not None:
                        
                        layer.mem = torch.ones_like(layer.mem) * initial_potential
                    else:
                        
                        print(f"Warning: layer.mem is None after reset_hidden() for {layer}. Cannot explicitly set v(0)=theta/2. Relying on internal init.")
                        

                except Exception as e:
                    print(f"Error setting initial membrane potential for {layer}: {e}")

        # Record spike outputs from the final layer
        mem_out_rec = []

        for step in range(num_steps):
            spk_input = spike_train[step]

            mem_states_feat = {} 

            current_input = spk_input 

            # --- Feature Extraction ---
            # Layer 1
            cur1 = self.features[0](current_input) # Conv2d
            bn1 = self.features[1](cur1)           # BatchNorm2d
            spk1 = self.features[2](bn1)
            pool1 = self.features[3](spk1)         # AvgPool2d

            # Layer 2
            cur2 = self.features[4](pool1)
            bn2 = self.features[5](cur2)
            spk2 = self.features[6](bn2)
            pool2 = self.features[7](spk2)

            # Layer 3
            cur3 = self.features[8](pool2)
            bn3 = self.features[9](cur3)
            spk3 = self.features[10](bn3)

            # Layer 4
            cur4 = self.features[11](spk3)
            bn4 = self.features[12](cur4)
            spk4 = self.features[13](bn4)

            # Layer 5
            cur5 = self.features[14](spk4)
            bn5 = self.features[15](cur5)
            spk5 = self.features[16](bn5)
            pool5 = self.features[17](spk5) # Final feature output (spikes)

            flat_spk = self.classifier[0](pool5) # Flatten

            # Layer 6 (FC1)
            drop6 = self.classifier[1](flat_spk) # Dropout (applied to spikes here)
            fc6 = self.classifier[2](drop6)      # Linear
            bn6 = self.classifier[3](fc6)        # BatchNorm1d
            spk6 = self.classifier[4](bn6) # snn.Leaky

            # Layer 7 (FC2)
            drop7 = self.classifier[5](spk6)     # Dropout
            fc7 = self.classifier[6](drop7)      # Linear
            bn7 = self.classifier[7](fc7)        # BatchNorm1d
            spk7 = self.classifier[8](bn7) # snn.Leaky

            # Layer 8 (Output)
            fc8 = self.classifier[9](spk7)      
            
            final_layer_output = self.classifier[10](fc8) 

            mem_out_rec.append(final_layer_output[0])

        mem_out_rec = torch.stack(mem_out_rec, dim=0)

        return mem_out_rec 
