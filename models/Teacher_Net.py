import math

# from model.vgg import VGG
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
from models.backbones import *
from models.decoders import *
from models.utils import *
from scipy import ndimage
import models.backbones.transformer_configs as configs
from torch.nn.functional import interpolate

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}

# class CT(nn.Module):
#     def __init__(self, input_size, output_size):
#         self.ct= nn.Sequential(nn.Conv2d(input_size, input_size, 1),
#                                   nn.Conv2d(config.hidden_size, 256, 1),
#                                   nn.Conv2d(config.hidden_size, 64, 1))


# ViT-B-16:[3, 6, 9]  L： [9, 14, 19]
class TUNet(nn.Module):
    def __init__(self, num_classes=9, vt_type='ViT-L_16', img_size=224, middle=[9, 14, 19]):
        super(TUNet, self).__init__()
        config = CONFIGS[vt_type]
        self.num_classes = num_classes
        self.backbone = Transformer(config=config, img_size=img_size, middle=middle)
        self.conv = nn.Sequential(nn.Conv2d(config.hidden_size, 512, 1),
                                  nn.Conv2d(config.hidden_size, 256, 1),
                                  nn.Conv2d(config.hidden_size, 64, 1))
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

        self.decoder = UNet([config.hidden_size, 512, 256, 64], [512, 256, 64, 16], [512, 256, 64, 0], num_classes)
        # self.decoder = PraNet([config.hidden_size, 512, 256, 64], [512, 256, 64, 16], [512, 256, 64, 0], num_classes)
        # self.decoder = Cup([512, 256, 128, 64], [256,128,64,16], [512, 256, 64, 0], num_classes)
        # self.decoder = AttU_Net([config.hidden_size, 512, 256, 64], [512, 256, 64, 16], [512, 256, 64, 0], num_classes)
        # self.decoder = PraNet()
        # self.decoder = U_NetDF(selfeat=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)  # 将它变为rgb三通道
        # fea:[1, 1024, 14, 14]  mid:[1, 512, 28, 28] [1, 256, 56, 56] [1, 64, 112, 112]
        feature, mid = self.backbone(x)

        fea_lst = []
        fea_lst.append(feature)
        for i in mid:
            fea_lst.append(i)

        mid_tmp = []

        mid_tmp.append(interpolate(self.conv[0](mid[0]), (28, 28), **self._up_kwargs))
        mid_tmp.append(interpolate(self.conv[1](mid[1]), (56, 56), **self._up_kwargs))
        mid_tmp.append(interpolate(self.conv[2](mid[2]), (112, 112), **self._up_kwargs))

        y = self.decoder(feature, mid_tmp)

        return y, fea_lst

    def load_from(self, weights):
        self.backbone.load_from(np.load(weights))

class TUNet2(nn.Module):
    def __init__(self, num_classes=9, vt_type='ViT-L_16', img_size=224, middle=[3,6,9]):
        super(TUNet2, self).__init__()
        config = CONFIGS[vt_type]
        self.num_classes = num_classes
        self.backbone = Transformer(config=config, img_size=img_size, middle=middle)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)  # 将它变为rgb三通道
        # fea:[1, 1024, 14, 14]  mid:[1, 512, 28, 28] [1, 256, 56, 56] [1, 64, 112, 112]
        feature, mid = self.backbone(x)

        fea_lst = []
        fea_lst.append(feature)
        for i in mid:
            fea_lst.append(i)

        return feature, fea_lst

    def load_from(self, weights):
        self.backbone.load_from(np.load(weights))

# class TUNet(nn.Module):
#     def __init__(self, num_classes=9, vt_type='ViT-L_16', img_size=224, middle=[9, 14, 19]):
#         super(TUNet, self).__init__()
#         config = CONFIGS[vt_type]
#         self.num_classes = num_classes
#         self.backbone = Transformer(config=config, img_size=img_size, middle=middle)
#
#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)  # 将它变为rgb三通道
#         # fea:[1, 1024, 14, 14]  mid:[1, 512, 28, 28] [1, 256, 56, 56] [1, 64, 112, 112]
#         feature, mid = self.backbone(x)
#
#         fea_lst = []
#         fea_lst.append(feature)
#         for i in mid:
#             fea_lst.append(i)
#
#         return 1, fea_lst
#
#     def load_from(self, weights):
#         self.backbone.load_from(np.load(weights))

if __name__ == "__main__":
    batch_size, num_classes, h, w = 1, 4, 224, 224
    x = torch.autograd.Variable(torch.randn(1, 3, h, w))

    model = TUNet(vt_type='ViT-L_16', num_classes=4)
    # y, mid_y = model(x)
    # print(y.shape)
    # print("..........")
    # for i in mid_y:
    #     print(i.shape)
    # for i in mid:
    #     print(i.shape)
    from thop import profile, clever_format
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, (x,))
    macs, params = clever_format([flops, params], "%.2f")
    # 61.20G   312.45M
    print(macs, " ", params)


