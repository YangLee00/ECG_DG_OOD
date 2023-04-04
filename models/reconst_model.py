import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // reduction, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // reduction, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    

    
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(UNetBlock, self).__init__()

        self.downsample = downsample

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)  # Replace ReLU with LeakyReLU
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)  # Replace ReLU with LeakyReLU
        )

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        if downsample:
            self.down = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        if self.downsample:
            x_down = self.down(x)
            return x, x_down
        else:
            return x    
    
    
# class UNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, downsample=True):
#         super(UNetBlock, self).__init__()

#         self.downsample = downsample

#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.ca = ChannelAttention(out_channels)
#         self.sa = SpatialAttention()

#         if downsample:
#             self.down = nn.MaxPool1d(2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)

#         x = self.ca(x) * x
#         x = self.sa(x) * x

#         if self.downsample:
#             x_down = self.down(x)
#             return x, x_down
#         else:
#             return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = UNetBlock(1, 8)
        self.down2 = UNetBlock(8, 16)
        self.down3 = UNetBlock(16, 32)
        
        self.middle = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),  # Replace ReLU with LeakyReLU
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),  # Replace ReLU with LeakyReLU
        )

        self.up3 = UNetBlock(64+32, 32, downsample=False)
        self.up2 = UNetBlock(32+16, 16, downsample=False)
        self.up1 = UNetBlock(16+8, 8, downsample=False)

        self.out_conv = nn.Conv1d(8, 1, kernel_size=1)

    def forward(self, x):
        x=x.transpose(2,1)
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)

        x = self.middle(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = torch.cat([x3, x], dim=1)
        x = self.up3(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = torch.cat([x2, x], dim=1)
        x = self.up2(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = torch.cat([x1, x], dim=1)
        x = self.up1(x)

        x = self.out_conv(x)
        x=x.transpose(2,1)
        return x