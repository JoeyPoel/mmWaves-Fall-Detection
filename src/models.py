"""
Unified PyTorch Model Architectures across all benchmarks.
"""
import torch
import torch.nn as nn

class Radar4DCNN(nn.Module):
    """2D Spatial-Temporal CNN for 4D mmWave Radar Point Clouds."""
    def __init__(self, in_channels=4, num_classes=2):
        super(Radar4DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

class ResNet18Spectrogram(nn.Module):
    """ResNet-18 model for Micro-Doppler Spectrograms (Rep 1)."""
    def __init__(self, in_channels=3, num_classes=2):
        super(ResNet18Spectrogram, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        feat = torch.flatten(self.features(x), 1)
        return self.fc(feat)

# Alias for Rep 2
ResNet18Projections = ResNet18Spectrogram

class PointNetFallDetector(nn.Module):
    """1D PointNet model for Native 3D Point Set Sequences (Rep 3)."""
    def __init__(self, in_channels=5, num_classes=2):
        super(PointNetFallDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        global_feat = torch.max(x, 2, keepdim=True)[0].view(x.size(0), -1)
        out = self.relu(self.fc1(self.dropout(global_feat)))
        return self.fc2(out)
