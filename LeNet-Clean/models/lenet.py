import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # 28x28 -> 28x28
            nn.BatchNorm2d(6),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            
            # Second convolutional layer
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # 14x14 -> 10x10
            nn.BatchNorm2d(16),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10x10 -> 5x5
        )
        
        self.classifier = nn.Sequential(
            # First fully connected layer
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),  # Added BatchNorm
            nn.ReLU(inplace=True),
            
            # Second fully connected layer
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  # Added BatchNorm
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x