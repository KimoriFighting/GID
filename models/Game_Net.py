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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}

# ViT-B-16:[3, 6, 9]  L： [9, 14, 19]
class GameNet(nn.Module):
    def __init__(self, num_classes=9, vt_type='ViT-L_16', img_size=224, middle=[3,6,9]):
        super(GameNet, self).__init__()
        config = CONFIGS[vt_type]
        self.num_classes = num_classes
        self.backbone = Transformer(config=config, img_size=img_size, middle=middle)
        self.backbone2 = ResNetV2()
        self.conv = nn.Sequential(nn.Conv2d(config.hidden_size, 512, 1),
                                  nn.Conv2d(config.hidden_size, 256, 1),
                                  nn.Conv2d(config.hidden_size, 64, 1))
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

        self.decoder = UNet([config.hidden_size*2, 512*2, 256*2, 64*2], [512*2, 256*2, 64*2, 16*2], [512*2, 256*2, 64*2, 0], num_classes)
        # self.decoder = PraNet([config.hidden_size*2, 512*2, 256*2, 64*2], [512*2, 256*2, 64*2, 16*2], [512*2, 256*2, 64*2, 0], num_classes)
        # self.decoder = Cup([config.hidden_size*2, 512*2, 256*2, 64*2], [512*2, 256*2, 64*2, 16*2], [512*2, 256*2, 64*2, 0], num_classes)
        # self.decoder = AttU_Net([config.hidden_size*2, 512*2, 256*2, 64*2], [512*2, 256*2, 64*2, 16*2], [512*2, 256*2, 64*2, 0], num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)  # 将它变为rgb三通道
        # fea:[1, 1024, 14, 14]  mid:[1, 512, 28, 28] [1, 256, 56, 56] [1, 64, 112, 112]
        feature, mid = self.backbone(x)
        feature2, mid2 = self.backbone2(x)

        feature = torch.cat([feature, feature2], dim=1)

        fea_lst = []
        fea_lst.append(feature)
        for i in mid:
            fea_lst.append(i)

        mid_tmp = []

        mid_tmp.append(interpolate(self.conv[0](mid[0]), (32, 32), **self._up_kwargs))
        mid_tmp.append(interpolate(self.conv[1](mid[1]), (64, 64), **self._up_kwargs))
        mid_tmp.append(interpolate(self.conv[2](mid[2]), (128, 128), **self._up_kwargs))

        for i in range(len(mid_tmp)):
            mid_tmp[i] = torch.cat([mid_tmp[i], mid2[i]], dim=1)

        y = self.decoder(feature, mid_tmp)

        return y, fea_lst

    def load_from(self, weights):
        self.backbone.load_from(np.load(weights))

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels+1, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

if __name__ == "__main__":
    batch_size, num_classes, h, w = 1, 9, 224, 224
    x = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))

    model = GameNet(vt_type='ViT-L_16')
    y, mid_y = model(x)
    print(y.shape)
    print("..........")
    for i in mid_y:
        print(i.shape)
    # for i in mid:
    #     print(i.shape)
    # from thop import profile, clever_format
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, (x,))
    # macs, params = clever_format([flops, params], "%.2f")
    # # 61.20G   312.45M
    # print(macs, " ", params)


