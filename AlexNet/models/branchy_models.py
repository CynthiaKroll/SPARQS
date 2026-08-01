import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import RL agent for dynamic inference
from .rl_environment import QLearningAgent

# --- Early Exit Block (ANN version) ---
class EarlyExitBlock(nn.Module):
    """ Early exit block with BatchNorm for improved convergence. """
    def __init__(self, in_channels, num_classes):
        super(EarlyExitBlock, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((in_channels // 2) * 4 * 4, 256),
            nn.BatchNorm1d(256),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """ Standard forward pass for ANN. """
        x = self.shared_conv(x)
        x = self.classifier(x)
        return x

# --- BranchyAlexNet (Improved Version) ---
class BranchyAlexNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        """
        Branchy AlexNet with 2 early exits (after the first conv block and after the middle layers).
        The final output layer acts as Exit 3.
        """
        super(BranchyAlexNet, self).__init__()
        self.num_classes = num_classes
        self.training_mode = True  # Custom flag to control forward pass behavior
        self.exit_loss_weights = [0.15, 0.15, 0.70] # Weights for training loss
        self.n_exits = 2  # Number of early exits
        self.rl_agent = QLearningAgent(n_exits=self.n_exits) # Agent with 2 early exit decision points
        self.confidence_thresholds = [0.8, 0.9]  # Default confidence thresholds for exits 1 and 2

        # Early Exit 1: after first conv block
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.exit1 = EarlyExitBlock(64, num_classes) # Output: 64 channels

        # Combined middle layers (acts as input to Exit 2)
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Added BatchNorm
            nn.ReLU(inplace=True),
        )
        self.exit2 = EarlyExitBlock(256, num_classes) # Output: 256 channels

        # Final layers (leading to Exit 3)
        self.final_features = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # Output: 256 channels

        # Adaptive pool before final classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Final classifier (Exit 3)
        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        """ Main forward pass. Switches based on training_mode attribute. """
        if self.training_mode:
            return self._forward_training(x)
        else:
            return self._forward_inference(x)

    def _forward_training(self, x):
        """ Forward pass during training, returning all exit outputs. """
        outputs = []
        x1 = self.features1(x)
        outputs.append(self.exit1(x1))

        x2 = self.features2(x1)
        outputs.append(self.exit2(x2))

        x_final = self.final_features(x2)
        x_final = self.adaptive_pool(x_final)
        x_final = x_final.view(x_final.size(0), -1) # Flatten
        outputs.append(self.output_layer(x_final))

        # Returns [exit1_out, exit2_out, final_out]
        return outputs

    def _forward_inference(self, x):
        """ 
        Forward pass during inference, using confidence thresholds for early exiting.
        
        The confidence thresholds can be set directly or derived from the RL agent.
        """
        device = x.device
        batch_size = x.size(0)
        final_outputs = torch.zeros(batch_size, self.num_classes, device=device)
        # Exit points: 1, 2, or 3 (final) - Note: 1-based indexing
        exit_points = torch.zeros(batch_size, dtype=torch.int, device=device)
        
        # Get confidence thresholds from RL agent if available
        if hasattr(self, 'rl_agent') and self.rl_agent is not None:
            # Use RL agent's learned thresholds if available
            thresholds = self.rl_agent.get_confidence_thresholds()
            if thresholds and len(thresholds) == self.n_exits:
                self.confidence_thresholds = thresholds
        
        # Debug flag check
        debug_confidence = hasattr(self, 'debug_confidence') and self.debug_confidence
        
        # OPTIMIZATION: More efficient batch processing
        # Process all samples through first stage
        x1 = self.features1(x)
        exit1_output = self.exit1(x1)
        
        # Calculate simple confidence for Exit 1 (using only max probability - much faster)
        softmax_output1 = torch.softmax(exit1_output, dim=1)
        confidence1, predictions1 = torch.max(softmax_output1, dim=1)
        
        # Make exit decisions for Exit 1 based on confidence threshold
        exit_mask = confidence1 >= self.confidence_thresholds[0]
        
        # Debug confidence values if requested
        if debug_confidence:
            print(f"Exit 1 confidence stats - Min: {confidence1.min().item():.4f}, Max: {confidence1.max().item():.4f}, Mean: {confidence1.mean().item():.4f}")
            print(f"Confidence threshold for Exit 1: {self.confidence_thresholds[0]}")
            print(f"Samples exiting at Exit 1: {torch.sum(exit_mask).item()}/{batch_size}")
        
        # Process exiting samples at Exit 1
        if exit_mask.any():
            # Store outputs for exiting samples
            exiting_indices = torch.where(exit_mask)[0]
            final_outputs[exiting_indices] = exit1_output[exit_mask]
            exit_points[exiting_indices] = 1  # Exit 1
            
            # Debug exiting samples
            if debug_confidence:
                print(f"Exit 1: {len(exiting_indices)} samples exiting out of {batch_size}")
        
        # Process through Exit 2 only for remaining samples
        non_exit_mask = ~exit_mask
        if non_exit_mask.any():
            # OPTIMIZATION: Only process non-exiting samples through second stage
            remaining_samples = x1[non_exit_mask]
            
            x2 = self.features2(remaining_samples)
            exit2_output = self.exit2(x2)
            
            # Calculate simple confidence for Exit 2
            softmax_output2 = torch.softmax(exit2_output, dim=1)
            confidence2, predictions2 = torch.max(softmax_output2, dim=1)
            
            # Make exit decisions for Exit 2 based on confidence threshold
            exit_mask2 = confidence2 >= self.confidence_thresholds[1]
            
            # Debug confidence values
            if debug_confidence:
                print(f"Exit 2 confidence stats - Min: {confidence2.min().item():.4f}, Max: {confidence2.max().item():.4f}, Mean: {confidence2.mean().item():.4f}")
                print(f"Confidence threshold for Exit 2: {self.confidence_thresholds[1]}")
                print(f"Samples exiting at Exit 2: {torch.sum(exit_mask2).item()}/{torch.sum(non_exit_mask).item()}")
            
            # Process exiting samples at Exit 2
            if exit_mask2.any():
                # Get original indices of samples exiting at Exit 2
                remaining_indices = torch.where(non_exit_mask)[0]
                exiting_indices = remaining_indices[exit_mask2]
                
                # Store outputs
                final_outputs[exiting_indices] = exit2_output[exit_mask2]
                exit_points[exiting_indices] = 2  # Exit 2
                
                # Debug exiting samples
                if debug_confidence:
                    print(f"Exit 2: {len(exiting_indices)} samples exiting out of {len(remaining_indices)} remaining")
            
            # Process remaining samples through final exit
            final_exit_mask = ~exit_mask2
            if final_exit_mask.any():
                # Get samples continuing to final exit
                final_samples = x2[final_exit_mask]
                
                # Process through final stage
                x_final = self.final_features(final_samples)
                x_final = self.adaptive_pool(x_final)
                x_final = x_final.view(x_final.size(0), -1)  # Flatten
                final_output = self.output_layer(x_final)
                
                # Get original indices of samples reaching final exit
                remaining_indices = torch.where(non_exit_mask)[0]
                final_indices = remaining_indices[final_exit_mask]
                
                # Store outputs
                final_outputs[final_indices] = final_output
                exit_points[final_indices] = 3  # Final exit
                
                # Debug remaining samples
                if debug_confidence:
                    print(f"Final exit: {len(final_indices)} samples out of {batch_size} original")
        
        # Return the final computed outputs and the exit points taken
        return final_outputs, exit_points
    
    def prepare_for_inference(self, confidence_thresholds=None):
        """Prepare model for inference by setting training_mode to False."""
        self.training_mode = False
        if confidence_thresholds is not None:
            self.confidence_thresholds = confidence_thresholds
        return self

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
