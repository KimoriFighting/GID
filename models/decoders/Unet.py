import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels+skip_channels, out_channels)


    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if x2 != None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, n_classes):
        super(UNet, self).__init__()
        self.n_classes = n_classes

        self.up1 = Up(in_channels[0], out_channels[0], mid_channels[0])
        self.up2 = Up(in_channels[1], out_channels[1], mid_channels[1])
        self.up3 = Up(in_channels[2], out_channels[2], mid_channels[2])
        self.up4 = Up(in_channels[3], out_channels[3], mid_channels[3])
        self.outc = OutConv(out_channels[3], n_classes)

    def forward(self, x, mid=None):
        if mid != None:
            # 1024 -> 512   512
            x = self.up1(x, mid[0])
            # 512 -> 256   256
            x = self.up2(x, mid[1])
            # 256 -> 64    64
            x = self.up3(x, mid[2])
        else:
            # 1024 -> 512   512
            x = self.up1(x)
            # 512 -> 256   256
            x = self.up2(x)
            # 256 -> 64    64
            x = self.up3(x)
        # 64 -> 16     0
        x = self.up4(x)
        logits = self.outc(x)
        return logits

class UNet_idea(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, n_classes):
        super(UNet_idea, self).__init__()
        self.n_classes = n_classes

        self.up1 = Up(in_channels[0], out_channels[0], mid_channels[0])
        self.up2 = Up(in_channels[1], out_channels[1], mid_channels[1])
        self.up3 = Up(in_channels[2], out_channels[2], mid_channels[2])
        self.up4 = Up(in_channels[3], out_channels[3], mid_channels[3])
        self.outc = OutConv(out_channels[3], n_classes)

    def forward(self, x, mid=None):
        if mid != None:
            # 1024 -> 512   512
            x = self.up1(x, mid[0])
            # 512 -> 256   256
            x = self.up2(x, mid[1])
            # 256 -> 64    64
            x = self.up3(x, mid[2])
        else:
            # 1024 -> 512   512
            x = self.up1(x)
            # 512 -> 256   256
            x = self.up2(x)
            # 256 -> 64    64
            x = self.up3(x)
        # 64 -> 16     0
        y = self.up4(x)
        logits = self.outc(y)
        return logits, x
