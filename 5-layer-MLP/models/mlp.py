import torch.nn as nn

class MLP(nn.Module):
    """
    5-layer MLP model equivalent to AlexNet structure but for MNIST dataset.
    This model follows the same pattern of layers, activation functions,
    normalization, and dropout as the original AlexNet but uses fully
    connected layers throughout instead of convolutional layers.
    """
    def __init__(self, num_classes=10, input_size=1*28*28):
        super(MLP, self).__init__()
        # Layer sizes designed to roughly match the computational complexity
        # and parameter count progression of AlexNet
        self.features = nn.Sequential(
            # Layer 1 (equivalent to first conv+pool in AlexNet)
            nn.Linear(input_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),  # Similar effect to pooling in reducing information

            # Layer 2 (equivalent to second conv+pool in AlexNet)
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),  # Similar effect to pooling

            # Layer 3 (equivalent to third conv in AlexNet)
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            # Layer 4 (equivalent to fourth conv in AlexNet)
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Layer 5 (equivalent to fifth conv+pool in AlexNet)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),  # Similar effect to pooling
        )
        
        # Classifier (equivalent to AlexNet classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x
