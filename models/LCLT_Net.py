import math

# from model.vgg import VGG
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
from .backbones import *
from models.decoders import *
from models.utils import *
from scipy import ndimage

from models.Simulation_module import Translator, TransEn
from models.backbones.transformer import CONFIGS

class LCLT_Net(nn.Module):
    def __init__(self, backbone=None, decoder=None, num_classes=9, img_size=224, middle_dims=[1024, 512, 256, 64], vt_type='ViT-L_16'):
        super(LCLT_Net, self).__init__()

        config = CONFIGS[vt_type]
        self.backbone = backbone
        self.num_classes = num_classes
        # self.backbone = resnet50()
        self.backbone = ResNetV2()
        self.decoder = UNet([1024, 512, 256, 64], [512,256,64,16], [512,256,64,0], num_classes)
        # self.decoder = Cup([512, 256, 128, 64], [256, 128, 64, 16], [512,256,64,0], num_classes)
        # self.decoder = AttU_Net([config.hidden_size, 512, 256, 64], [512, 256, 64, 16], [512, 256, 64, 0], num_classes)
        # self.decoder = PraNet()
        # self.decoder = U_NetDF(selfeat=True)

        # 额外添加

        self.trans0 = TransEn(in_planes=1024, planes=config.hidden_size, config=config)
        self.trans1 = TransEn(in_planes=512, planes=config.hidden_size, kernel_size = 3, stride = 2, padding = 1, config=config)
        # in_planes, planes, kernel_size = 3, stride = 1, padding = 1, config = CONFIGS['ViT-L_16']
        #     Translator(512, config.hidden_size, 3, 2, 1)
        self.trans2 = TransEn(in_planes=256, planes=config.hidden_size, kernel_size = 5, stride = 4, padding = 2, config=config)
            # Translator(256, config.hidden_size, 5, 4, 2)
        self.trans3 = TransEn(in_planes=64, planes=config.hidden_size, kernel_size = 9, stride = 8, padding = 2, config=config)
            # Translator(64, config.hidden_size, 9, 8, 2)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)  # 将它变为rgb三通道
        # fea:[1, 1024, 14, 14]  mid:[1, 512, 28, 28] [1, 256, 56, 56] [1, 64, 112, 112]
        feature, mid = self.backbone(x)
        y = self.decoder(feature, mid)

        mid_y = []
        mid_y.append(self.trans0(feature))
        mid_y.append(self.trans1(mid[0]))
        mid_y.append(self.trans2(mid[1]))
        mid_y.append(self.trans3(mid[2]))

        return y, mid_y

    def load_from(self, weights):
        self.backbone.load_from(np.load(weights))
        # model_dict = self.state_dict()
        # pretrained_dict = torch.load(weights)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)



class Student_Net(nn.Module):
    def __init__(self,  num_classes=9):
        super(Student_Net, self).__init__()
        self.backbone = ResNetV2()
        self.decoder = UNet_idea([1024, 512, 256, 64], [512, 256, 64, 16], [512, 256, 64, 0], num_classes)
        self.up = Up(64, 16, 0)
        self.predict = OutConv(16, num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)  # 将它变为rgb三通道
        # fea:[1, 1024, 14, 14]  mid:[1, 512, 28, 28] [1, 256, 56, 56] [1, 64, 112, 112]
        feature, mid = self.backbone(x)
        ans, tmp = self.decoder(feature, mid)
        tmp = self.up(tmp)
        # tmp = torch.cat(tmp, ans], dim=1)
        y = self.predict(tmp)
        ans = ans+y

        return ans, 1


if __name__ == "__main__":
    batch_size, num_classes, h, w = 3, 9, 224, 224
    x = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))

    model = LCLT_Net()
    outputs, mid_outputs = model(x)
    print(outputs.shape)
    for i in mid_outputs:
        print(i.shape)
    # model.load_from('/mnt/disk/yongsen/model/resnet/resnet50.pth')
    # model.load_from('/mnt/disk/yongsen/model/vit/imageNet21k/imagenet21k_ViT-L_16.npz')
    # model.load_from('/mnt/disk/yongsen/model/resnet/resnet50.pth',
                    # '/mnt/disk/haoli/vit_models/imagenet21k/imagenet21k_ViT-B_32.npz')
    # from thop import profile, clever_format
    #
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, (x,))
    # # print('flops: ', flops, 'params: ', params)
    # # print("%.2fM" % (flops / 1e6), "%.2fM" % (params / 1e6))
    # # print(" params| %.2f flops| %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位，
    # macs, params = clever_format([flops, params], "%.2f")
    # # 8.08G   5.03M   10.14G   5.88M
    # print(macs, " ", params)


