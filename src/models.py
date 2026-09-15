"""
Unified PyTorch Model Architectures for mmWave Radar Fall Detection.
Includes:
- ResNet-18 adaptors for Representation 1 (Doppler-Time Map) & Representation 2 (2D Orthogonal Projections)
- PointNet++ architecture for Representation 3 (Native 3D Point Sets)
- Radar4DCNN baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18Adaptor(nn.Module):
    """
    Standard ResNet-18 architecture adapted for 2D radar representations (32x32 grids).
    Supports 3-channel input representations (Rep 1 Doppler-Time and Rep 2 Orthogonal Projections).
    """
    def __init__(self, in_channels=3, num_classes=2):
        super(ResNet18Adaptor, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

# Aliases for Representation 1 & Representation 2
ResNet18Spectrogram = ResNet18Adaptor
ResNet18Projections = ResNet18Adaptor


class PointNetPlusPlusFallDetector(nn.Module):
    """
    PointNet++ Architecture for 3D Point Set Sequences (Representation 3).
    Extracts spatial-geometric point features per frame using multi-scale Set Abstraction,
    followed by 1D temporal convolution modeling movement dynamics over time.
    """
    def __init__(self, in_channels=5, num_classes=2):
        super(PointNetPlusPlusFallDetector, self).__init__()
        
        # PointNet Set Abstraction / Spatial Point Feature Extractor (per frame)
        self.sa1_conv1 = nn.Conv1d(in_channels, 64, 1)
        self.sa1_conv2 = nn.Conv1d(64, 128, 1)
        self.sa2_conv1 = nn.Conv1d(128, 256, 1)
        self.sa2_conv2 = nn.Conv1d(256, 512, 1)
        
        self.bn_sa1 = nn.BatchNorm1d(128)
        self.bn_sa2 = nn.BatchNorm1d(512)

        # 1D Temporal Convolution across 10 temporal frames
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, 5, 320) where 320 = 10 frames * 32 points
        B = x.size(0)
        # Reshape to (B * 10, 5, 32)
        x_frames = x.view(B * 10, 5, 32)

        # Point feature extraction
        f1 = F.relu(self.sa1_conv1(x_frames))
        f1 = F.relu(self.bn_sa1(self.sa1_conv2(f1))) # (B*10, 128, 32)

        f2 = F.relu(self.sa2_conv1(f1))
        f2 = F.relu(self.bn_sa2(self.sa2_conv2(f2))) # (B*10, 512, 32)

        # Global Max Pooling per frame (across 32 points)
        frame_feat = torch.max(f2, dim=2)[0] # (B*10, 512)

        # Reshape back to temporal sequence: (B, 512, 10)
        temp_seq = frame_feat.view(B, 10, 512).permute(0, 2, 1) # (B, 512, 10)

        # Temporal sequence modeling
        temp_feat = self.temporal_conv(temp_seq).squeeze(-1) # (B, 128)

        # Binary classification logits
        return self.fc(temp_feat)

PointNetFallDetector = PointNetPlusPlusFallDetector

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
