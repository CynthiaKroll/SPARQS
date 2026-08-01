import torch
import torch.nn as nn
import torch.ao.quantization as quant # Added for potential QAT later

# Quantization Clip-Floor-Shift Activation Function (Paper Implementation)
# â.h(z) = λ · clip(⌊z·L/λ + ½⌋/L, 0, 1)
class FloorSTE(torch.autograd.Function):
    """Straight-Through Estimator for the floor function."""
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient is 1 (straight-through)
        return grad_output

floor_ste = FloorSTE.apply

class QCFS(nn.Module):
    def __init__(self, L=4, lambda_init=6.0): # L: quantization levels, lambda_init: initial threshold
        super().__init__()
        if L <= 0:
            raise ValueError("Quantization level L must be positive")
        if lambda_init <= 0:
             print(f"Warning: Initial lambda {lambda_init} is not positive. Using abs(lambda_init).")
             lambda_init = abs(lambda_init)
        self.L = L
        # Lambda is a learnable, per-layer threshold parameter
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)))
        self.epsilon = 1e-6 # For safe division

    def forward(self, z):
        # Ensure lambda is positive during forward pass
        # Using torch.clamp directly on the parameter is often preferred over abs()
        # as it behaves better with optimizers. Clamping during the forward pass.
        lambda_val = torch.clamp(self.lambda_param, min=self.epsilon)

        # Calculate the term inside floor: z * L / lambda + 0.5
        scaled_z = z * self.L / lambda_val + 0.5

        # Apply floor with the straight-through estimator. This MUST be floor_ste,
        # not torch.floor: torch.floor has zero gradient everywhere, so it blocks
        # ALL gradient through every QCFS activation -- freezing every conv/linear
        # before it at random init and pinning training at ~10% (loss = ln 10).
        # The FloorSTE.backward passes the gradient straight through (as if floor
        # were identity), which is the entire point of the QCFS formulation.
        floored_z = floor_ste(scaled_z)

        # Apply clipping: clip(value / L, 0, 1)
        clipped_z = torch.clamp(floored_z / self.L, 0.0, 1.0)

        # Final activation output: lambda * clipped_value
        output = lambda_val * clipped_z

        # The STE is applied via floor_ste, so we return the calculated output directly.
        # The gradient will flow back through floor_ste according to its backward definition.
        return output

    def extra_repr(self): # For better printing of the module
        # Display the current value of lambda during printing
        return f'L={self.L}, lambda={self.lambda_param.data.item():.4f}'


class AlexNetQCFS(nn.Module):
    # L_value and initial_lambda can be passed or configured
    def __init__(self, num_classes=10, L_value=4, initial_lambda=6.0):
        super(AlexNetQCFS, self).__init__()
        print(f"Initializing AlexNetQCFS with L={L_value}, initial_lambda={initial_lambda}")
        # Define features layers - Replaced MaxPool with AvgPool
        self.features = nn.Sequential(
            # Layer 1: Conv -> BN -> QCFS -> AvgPool
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Input: 3x32x32 -> Output: 64x32x32
            nn.BatchNorm2d(64),
            QCFS(L=L_value, lambda_init=initial_lambda), # QCFS activation
            nn.AvgPool2d(kernel_size=2, stride=2),                 # Output: 64x16x16

            # Layer 2: Conv -> BN -> QCFS -> AvgPool
            nn.Conv2d(64, 192, kernel_size=3, padding=1),          # Output: 192x16x16
            nn.BatchNorm2d(192),
            QCFS(L=L_value, lambda_init=initial_lambda),
            nn.AvgPool2d(kernel_size=2, stride=2),                 # Output: 192x8x8

            # Layer 3: Conv -> BN -> QCFS
            nn.Conv2d(192, 384, kernel_size=3, padding=1),         # Output: 384x8x8
            nn.BatchNorm2d(384),
            QCFS(L=L_value, lambda_init=initial_lambda),

            # Layer 4: Conv -> BN -> QCFS
            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # Output: 256x8x8
            nn.BatchNorm2d(256),
            QCFS(L=L_value, lambda_init=initial_lambda),

            # Layer 5: Conv -> BN -> QCFS -> AvgPool
            nn.Conv2d(256, 256, kernel_size=3, padding=1),         # Output: 256x8x8
            nn.BatchNorm2d(256),
            QCFS(L=L_value, lambda_init=initial_lambda),
            nn.AvgPool2d(kernel_size=2, stride=2),                 # Output: 256x4x4
        )
        # Define classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flatten features before classifier
            nn.Dropout(p=0.5), # Keep dropout as in original AlexNet
            nn.Linear(256 * 4 * 4, 4096),
            nn.BatchNorm1d(4096),
            QCFS(L=L_value, lambda_init=initial_lambda),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            QCFS(L=L_value, lambda_init=initial_lambda),

            nn.Linear(4096, num_classes), # Final output layer, no QCFS here
        )
        # Initialize weights after defining layers
        self._initialize_weights()

    def forward(self, x):
        # Removed input_norm layer
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten the features
        x = self.classifier(x)
        return x

    # Add weight initialization method
    def _initialize_weights(self):
        print("Initializing weights (Kaiming for Conv, Normal for Linear, Constants for BN). No lambda-based scaling applied here.")
        # Note: The paper doesn't explicitly mention scaling weights based on initial lambda
        # during initialization. It focuses on learning lambda during training.
        # We will use standard initializations.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization is suitable for layers followed by ReLU-like activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Standard initialization for linear layers
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Example usage (optional, for verification)
# if __name__ == '__main__':
#     model = AlexNetQCFS(num_classes=10)
#     print(model)
#     # Prepare for QAT (example)
#     # model.qconfig = quant.get_default_qat_qconfig('fbgemm')
#     # quant.prepare_qat(model, inplace=True)
#     # print("\nModel prepared for QAT:")
#     # print(model)