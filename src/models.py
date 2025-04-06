import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
    
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder blocks
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        
        # Decoder blocks
        self.dec2 = UNetBlock(128 + 64, 64)
        self.dec1 = UNetBlock(64, 32)
        
        # Final layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        x = self.enc2(x)
        
        # Decoder with skip connections
        x = nn.functional.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)
        
        x = self.dec1(x)
        x = self.final(x)
        
        # Output non-negative depth values
        x = torch.sigmoid(x)*10
        
        return x
    