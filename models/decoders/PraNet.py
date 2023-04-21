import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

class RFB_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(RFB_kernel, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = RFB_kernel(in_channel, out_channel, 3)
        self.branch2 = RFB_kernel(in_channel, out_channel, 5)
        self.branch3 = RFB_kernel(in_channel, out_channel, 7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x

class PPD(nn.Module):
    def __init__(self, channel):
        super(PPD, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        self.conv_upsample1 = conv(channel, channel, 3)
        self.conv_upsample2 = conv(channel, channel, 3)
        self.conv_upsample3 = conv(channel, channel, 3)
        self.conv_upsample4 = conv(channel, channel, 3)
        self.conv_upsample5 = conv(2 * channel, 2 * channel, 3)

        self.conv_concat2 = conv(2 * channel, 2 * channel, 3)
        self.conv_concat3 = conv(3 * channel, 3 * channel, 3)
        self.conv4 = conv(3 * channel, 3 * channel, 3)
        self.conv5 = conv(3 * channel, 9, 1, bn=False, bias=True)

    def forward(self, f1, f2, f3):
        f1x2 = self.upsample(f1, f2.shape[-2:])
        f1x4 = self.upsample(f1, f3.shape[-2:])
        f2x2 = self.upsample(f2, f3.shape[-2:])

        f2_1 = self.conv_upsample1(f1x2) * f2
        f3_1 = self.conv_upsample2(f1x4) * self.conv_upsample3(f2x2) * f3

        f1_2 = self.conv_upsample4(f1x2)
        f2_2 = torch.cat([f2_1, f1_2], 1)
        f2_2 = self.conv_concat2(f2_2)

        f2_2x2 = self.upsample(f2_2, f3.shape[-2:])
        f2_2x2 = self.conv_upsample5(f2_2x2)

        f3_2 = torch.cat([f3_1, f2_2x2], 1)
        f3_2 = self.conv_concat3(f3_2)

        f3_2 = self.conv4(f3_2)
        out = self.conv5(f3_2)

        return f3_2, out

class reverse_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3, n_classes=9):
        super(reverse_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, n_classes, 3 if kernel_size == 3 else 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        rmap = -1 * (torch.sigmoid(map)) + 1

        x = rmap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out

class PraNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, in_channels, out_channels, mid_channels, n_classes):
        super(PraNet, self).__init__()
        # self.resnet = res2net50_v1b_26w_4s(pretrained=opt.pretrained, output_stride=opt.output_stride)

        self.context2 = RFB(in_channels[2], 32)
        self.context3 = RFB(in_channels[1], 32)
        self.context4 = RFB(in_channels[0], 32)

        self.decoder = PPD(32)

        self.attention2 = reverse_attention(in_channels[2], 64, 2, 3, n_classes)
        self.attention3 = reverse_attention(in_channels[1], 64, 2, 3, n_classes)
        self.attention4 = reverse_attention(in_channels[0], 256, 3, 5, n_classes)

    def forward(self, x, mid=None):
        base_size = x.shape[-2:]
        x4 = x
        x3 = mid[0]
        x2 = mid[1]

        x2_context = self.context2(x2)
        x3_context = self.context3(x3)
        x4_context = self.context4(x4)

        _, a5 = self.decoder(x4_context, x3_context, x2_context)
        out5 = F.interpolate(a5, size=base_size, mode='bilinear', align_corners=False)

        _, a4 = self.attention4(x4, a5)
        out4 = F.interpolate(a4, size=base_size, mode='bilinear', align_corners=False)

        _, a3 = self.attention3(x3, a4)
        out3 = F.interpolate(a3, size=base_size, mode='bilinear', align_corners=False)

        _, a2 = self.attention2(x2, a3)
        out2 = F.interpolate(a2, size=base_size, mode='bilinear', align_corners=False)

        return out2


if __name__ == '__main__':
    ras = PraNet()
    input_tensor = torch.randn(1, 3, 352, 352)

    out = ras(input_tensor)
    print(out.shape)