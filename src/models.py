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
    def __init__(self, in_channels, out_channels, resizing):
        super(Down, self).__init__()
        self.MP = nn.MaxPool2d(resizing)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.MP(x))
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, resizing):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=resizing, mode='bilinear', align_corners=True)
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

class AdvancedUNet(nn.Module):
    def __init__(self):
        super(AdvancedUNet, self).__init__()

        self.inc = (ConvBlock(3, 64))
        self.down1 = (Down(64, 128, 2))
        self.down2 = (Down(128, 256, 2))
        self.down3 = (Down(256, 512, 2))

        factor = 2

        self.down4 = (Down(512, 1024 // factor, 2))
        self.up1 = (Up(1024, 512 // factor, 2))
        self.up2 = (Up(512, 256 // factor, 2))
        self.up3 = (Up(256, 128 // factor, 2))
        self.up4 = (Up(128, 64, 2))

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

class AdvancedUNetPooling(nn.Module):
    def __init__(self):
        super(AdvancedUNetPooling, self).__init__()

        self.inc = (ConvBlock(3, 64))
        self.down1 = (Down(64, 128, 8))
        self.down2 = (Down(128, 256, 4))
        self.down3 = (Down(256, 512, 4))

        factor = 2

        self.down4 = (Down(512, 1024 // factor, 2))
        self.up1 = (Up(1024, 512 // factor, 2))
        self.up2 = (Up(512, 256 // factor, 4))
        self.up3 = (Up(256, 128 // factor, 4))
        self.up4 = (Up(128, 64, 8))

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
        x = torch.sigmoid(x) * 10
        return x


class UncertaintyUNEt(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(UncertaintyUNEt, self).__init__()

        self.inc = (ConvBlock(3, 64, dropout_rate))
        self.down1 = (Down(64, 128, dropout_rate))
        self.down2 = (Down(128, 256, dropout_rate))
        self.down3 = (Down(256, 512, dropout_rate))

        factor = 2

        self.down4 = (Down(512, 1024 // factor, dropout_rate))

        self.up1 = (Up(1024, 512 // factor, dropout_rate))
        self.up2 = (Up(512, 256 // factor, dropout_rate))
        self.up3 = (Up(256, 128 // factor, dropout_rate))
        self.up4 = (Up(128, 64, dropout_rate))

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        self.up_uncertainty1 = (Up(1024, 512 // factor, dropout_rate))
        self.up_uncertainty2 = (Up(512, 256 // factor, dropout_rate))
        self.up_uncertainty3 = (Up(256, 128 // factor, dropout_rate))
        self.up_uncertainty4 = (Up(128, 64, dropout_rate))

        self.final_uncertainty = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1_up = self.up1(x5, x4)
        x2_up = self.up2(x1_up, x3)
        x3_up = self.up3(x2_up, x2)
        x4_up = self.up4(x3_up, x1)
        xfinal_up = self.final(x4_up)
        xfinal_up = torch.sigmoid(xfinal_up)*10

        x1_uncertainty = self.up_uncertainty1(x5.detach(),x4.detach())
        x2_uncertainty = self.up_uncertainty2(x1_uncertainty, x3.detach())
        x3_uncertainty = self.up_uncertainty3(x2_uncertainty, x2.detach())
        x4_uncertainty = self.up_uncertainty4(x3_uncertainty, x1.detach())
        xfinal_uncertainty = self.final_uncertainty(x4_uncertainty)
        xfinal_uncertainty = torch.sigmoid(xfinal_uncertainty)
        return xfinal_up, xfinal_uncertainty
