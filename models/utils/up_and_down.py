import torch
import torch.nn as nn
from torch.nn.functional import interpolate

class CB(nn.Module):
    def __init__(self, input_channels, output_channels, min):
        super(CB, self).__init__()
        if min == 1:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        elif min == 2:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False)
        elif min == 4:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=4, padding=2, bias=False)
        else:
            raise ValueError("min has not this value!!")

        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(inter_channels),
                                  nn.ReLU(),
                                  nn.Dropout(0.1, False),
                                  nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.conv(x)
        return x

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, image_size,
                 up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(FCNHead, self).__init__()
        self._up_kwargs = up_kwargs
        self.image_size = image_size

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        x = self.conv(x)
        # x = interpolate(x, self.image_size, **self._up_kwargs)
        return x