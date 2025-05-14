import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(Down, self).__init__()
        self.MP = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)

    def forward(self, x):
        return self.conv(self.MP(x))
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)
    
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
    def __init__(self, dropout_rate=0.2):
        super(AdvancedUNEt, self).__init__()

        self.inc = (ConvBlock(3, 64, dropout_rate))
        self.down1 = (Down(64, 128, dropout_rate))
        self.down2 = (Down(128, 256, dropout_rate))
        self.down3 = (Down(256, 512, dropout_rate))

        factor = 2

        self.down4 = (Down(512, 1024 // factor, dropout_rate))

        """
        self.transformer_proj = nn.Conv2d(512, 512, 1) # Linear Projection
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=5
        )
        """

        self.up1 = (Up(1024, 512 // factor, dropout_rate))
        self.up2 = (Up(512, 256 // factor, dropout_rate))
        self.up3 = (Up(256, 128 // factor, dropout_rate))
        self.up4 = (Up(128, 64))

        self.final = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        """
        x5_proj = self.transformer_proj(x5)  # [B, 512, H, W]
        b, c, h, w = x5_proj.shape
        x5_flat = x5_proj.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        x5_trans = self.transformer_encoder(x5_flat)    # [B, HW, C]
        x5_out = x5_trans.permute(0, 2, 1).reshape(b, c, h, w)
        """

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final(x)
        x = torch.sigmoid(x)*10
        return x

        
    