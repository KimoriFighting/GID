import torch.nn as nn
import torch

# class Paraphraser(nn.Module):
#     def __init__(self,in_planes, planes, stride=1):
#         super(Paraphraser, self).__init__()
#         self.leakyrelu = nn.LeakyReLU(0.1)
#   #      self.bn0 = nn.BatchNorm2d(in_planes)
#         self.conv0 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
#   #      self.bn1 = nn.BatchNorm2d(planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#   #      self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#   #      self.bn0_de = nn.BatchNorm2d(planes)
#         self.deconv0 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#   #      self.bn1_de = nn.BatchNorm2d(in_planes)
#         self.deconv1 = nn.ConvTranspose2d(planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
#   #      self.bn2_de = nn.BatchNorm2d(in_planes)
#         self.deconv2 = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
#
# #### Mode 0 - throw encoder and decoder (reconstruction)
# #### Mode 1 - extracting teacher factors
#     def forward(self, x, mode):
#         if mode == 0:
#             ## encoder
#             out = self.leakyrelu((self.conv0(x)))
#             out = self.leakyrelu((self.conv1(out)))
#             out = self.leakyrelu((self.conv2(out)))
#             ## decoder
#             out = self.leakyrelu((self.deconv0(out)))
#             out = self.leakyrelu((self.deconv1(out)))
#             out = self.leakyrelu((self.deconv2(out)))
#
#         if mode == 1:
#             out = self.leakyrelu((self.conv0(x)))
#             out = self.leakyrelu((self.conv1(out)))
#             out = self.leakyrelu((self.conv2(out)))
#
#         ## only throw decoder
#         if mode == 2:
#             out = self.leakyrelu((self.deconv0(x)))
#             out = self.leakyrelu((self.deconv1(out)))
#             out = self.leakyrelu((self.deconv2(out)))
#         return out

class Paraphraser(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, padding=1, stride=1):
        super(Paraphraser, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # inter_channels = in_planes // 4
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_planes, inter_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(),
        #     nn.Dropout(0.1, False),
        #     nn.Conv2d(inter_channels, planes, 1))

        self.deconv = nn.ConvTranspose2d(planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)


#### Mode 0 - throw encoder and decoder (reconstruction)
#### Mode 1 - extracting teacher factors
    def forward(self, x, mode):
        if mode == 0:
            # out = self.conv(x)
            out = self.relu(self.bn(self.conv(x)))
            out = self.deconv(out)

        if mode == 1:
            out = self.relu(self.bn(self.conv(x)))
        return out

# class Translator(nn.Module):
#     def __init__(self,in_planes, planes, stride=1):
#         super(Translator, self).__init__()
#         self.leakyrelu = nn.LeakyReLU(0.1)
#   #      self.bn0 = nn.BatchNorm2d(in_planes)
#         self.conv0 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
#   #     self.bn1 = nn.BatchNorm2d(planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#   #     self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#
#     def forward(self, x):
#         out = self.leakyrelu((self.conv0(x)))
#         out = self.leakyrelu((self.conv1(out)))
#         out = self.leakyrelu((self.conv2(out)))
#         return out

class Translator(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1):
        super(Translator, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # inter_channels = in_planes // 4
        # self.conv = nn.Sequential(nn.Conv2d(in_planes, inter_channels,  kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        #                           nn.BatchNorm2d(inter_channels),
        #                           nn.ReLU(),
        #                           nn.Dropout(0.1, False),
        #                           nn.Conv2d(inter_channels, planes, 1))

    def forward(self, x):
        return self.conv(x)

from models.backbones.transformer import Encoder, CONFIGS
import numpy as np
class TransEn(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1, config=CONFIGS['ViT-L_16']):
        super(TransEn, self).__init__()
        config.transformer["num_layers"] = 1
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = Encoder(config, False, [])
        # inter_channels = in_planes // 4
        # self.conv = nn.Sequential(nn.Conv2d(in_planes, inter_channels,  kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        #                           nn.BatchNorm2d(inter_channels),
        #                           nn.ReLU(),
        #                           nn.Dropout(0.1, False),
        #                           nn.Conv2d(inter_channels, planes, 1))

    def take_atten(self, hidden_states):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2).contiguous()
        encoded, attn_weights, mid_encodeds = self.encoder(x)
        encoded = self.take_atten(encoded)
        return encoded

class Sim(nn.Module):
    def __init__(self, in_planes=1024, planes=64):
        super(Sim, self).__init__()
        self.simulation1 = Paraphraser(in_planes, planes)
        self.simulation2 = Paraphraser(in_planes, planes)
        self.simulation3 = Paraphraser(in_planes, planes)

    def forward(self, x, model_num):
        mid_sim = []
        mid_sim.append(self.simulation1(x[1], model_num))
        mid_sim.append(self.simulation1(x[2], model_num))
        mid_sim.append(self.simulation1(x[3], model_num))

        return mid_sim

if __name__ == "__main__":
    batch_size, num_classes, h, w = 1, 1024, 16, 16
    x = torch.autograd.Variable(torch.randn(batch_size, 1024, h, w))
    # model = Translator(1024, 1024)
    model = TransEn()
    outputs = model(x)
    # print(outputs.shape)
    for i in outputs:
        print(i.shape)
    from thop import profile, clever_format
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print("%.2fM" % (flops / 1e6), "%.2fM" % (params / 1e6))

