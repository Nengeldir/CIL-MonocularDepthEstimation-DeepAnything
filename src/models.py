import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import pad_to_multiple_of_14
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

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
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.MP = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.MP(x))
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AdvancedUNEt(nn.Module):
    def __init__(self):
        super(AdvancedUNEt, self).__init__()

        self.inc = (ConvBlock(3, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))

        factor = 2

        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor))
        self.up2 = (Up(512, 256 // factor))
        self.up3 = (Up(256, 128 // factor))
        self.up4 = (Up(128, 64))

        self.final = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final(x)
        x = torch.sigmoid(x)*10
        return x

class DepthAnythingSmallPretrained(nn.Module):
    def __init__(self):
        super(DepthAnythingSmallPretrained, self).__init__()
        # Load the Depth-Anything-Small Model with Pre-trained weights
        self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load('Depth_Anything_V2/depth_anything_v2_vits.pth'))
        return

    def forward(self, x):
        # Pad the input to be a multiple of 14 for Depth-Anything to work
        x = pad_to_multiple_of_14(x)

        # Inverse the output from depth-anything, since we want higher values for further away objects
        x = -1 * self.model(x).unsqueeze(1)

        # Resize the image again
        x = nn.functional.interpolate(
            x,
            size=(426, 560),  # Match height and width of targets
            mode='bilinear',
            align_corners=True
        )
        # Normalize depth-anything's output to be between [0, 10] since we only need relative positions anyways
        x_min = x.amin(dim=[2, 3], keepdim=True)
        x_max = x.amax(dim=[2, 3], keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)  # Normalize to [0, 1]
        x = x * 10  # Multiply by 10
        return x
