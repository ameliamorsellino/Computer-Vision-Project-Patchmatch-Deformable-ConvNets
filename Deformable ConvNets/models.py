import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DeformableConvBlock(nn.Module):
    """
    It has:
    - A conv layer that predicts the offsets (2 * kernel_h * kernel_w channels)
    - DeformConv2d
    - BatchNorm + ReLU
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        # layer to predict offests
        # output: 2 * kH * kW (offset x,y for each position kernel)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # initialize offset at zero so that it starts as a standard convolution
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        
        # deformable convolution
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 1) predict offsets from festure map
        offsets = self.offset_conv(x)
        # 2) apply deformable convolution with learned offsets
        out = self.deform_conv(x, offsets)
        out = self.bn(out)
        out = self.relu(out)
        return out, offsets  # returns offsets for visualization


class StandardConvBlock(nn.Module):
    """Standard convolutional block used for comparison"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class StandardCNN(nn.Module):
    """Baseline CNN using standard convolution layers"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            StandardConvBlock(1, 32, 3, padding=1),
            StandardConvBlock(32, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            StandardConvBlock(32, 64, 3, padding=1),
            StandardConvBlock(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            StandardConvBlock(64, 128, 3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_name(self):
        return "StandardCNN"


class DeformableCNN(nn.Module):
    """
    CNN with Deformable Convolutions.
    Standard convolution layers in the final stages are replaced with DeformableConvBlock like in the paper
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # first standard layers
        self.conv1 = StandardConvBlock(1, 32, 3, padding=1)
        self.conv2 = StandardConvBlock(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # deformable conv in last layers
        self.deform_conv1 = DeformableConvBlock(32, 64, 3, padding=1)
        self.deform_conv2 = DeformableConvBlock(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.deform_conv3 = DeformableConvBlock(64, 128, 3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.saved_offsets = {}
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        
        x, off1 = self.deform_conv1(x)
        x, off2 = self.deform_conv2(x)
        self.saved_offsets['deform1'] = off1
        self.saved_offsets['deform2'] = off2
        
        x = self.pool2(x)
        
        x, off3 = self.deform_conv3(x)
        self.saved_offsets['deform3'] = off3
        
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def get_name(self):
        return "DeformableCNN"


def count_parameters(model):
    """Counts total params and trainable params"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    x = torch.randn(2, 1, 28, 28)
    
    std_model = StandardCNN()
    def_model = DeformableCNN()
    
    print("StandardCNN")
    out = std_model(x)
    total, trainable = count_parameters(std_model)
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {total:,} (trainable: {trainable:,})")
    
    print("\nDeformableCNN")
    out = def_model(x)
    total, trainable = count_parameters(def_model)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {total:,} (trainable: {trainable:,})")
    print(f"Offset shapes: {[(k, v.shape) for k, v in def_model.saved_offsets.items()]}")