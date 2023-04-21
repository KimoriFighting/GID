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

class Student(nn.Module):
    def __init__(self, backbone=None, decoder=None, num_classes=9, img_size=224, middle_dims=[1024, 512, 256, 64]):
        super(Student, self).__init__()
        self.backbone = ResNetV2()
        self.decoder = UNet([1024, 512, 256, 64], [512,256,64,16], [512,256,64,0], num_classes)
        # self.decoder = Cup([512, 256, 128, 64], [256, 128, 64, 16], [512, 256, 64, 0], num_classes)
        # self.decoder = AttU_Net([1024, 512, 256, 64], [512, 256, 64, 16], [512, 256, 64, 0], num_classes)
        # self.decoder = PraNet()
        # self.decoder = U_NetDF(selfeat=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)  # 将它变为rgb三通道
        # fea:[1, 1024, 14, 14]  mid:[1, 512, 28, 28] [1, 256, 56, 56] [1, 64, 112, 112]
        feature, mid = self.backbone(x)
        y = self.decoder(feature, mid)

        return y, mid

    def load_from(self, weights):
        self.backbone.load_from(np.load(weights))
        # model_dict = self.state_dict()
        # pretrained_dict = torch.load(weights)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)

if __name__ == "__main__":
    batch_size, num_classes, h, w = 3, 9, 224, 224
    x = torch.autograd.Variable(torch.randn(1, 3, h, w))

    model = Student()
    outputs, mid_outputs = model(x)
    print(outputs.shape)
    # print(lateral_map_3.shape)
    # print(lateral_map_4.shape)
    # print(lateral_map_5.shape)

    # for i in mid_outputs:
    #     print(i.shape)
    # model.load_from('/mnt/disk/yongsen/model/resnet/resnet50.pth')
    # model.load_from('/mnt/disk/yongsen/model/vit/imageNet21k/imagenet21k_ViT-L_16.npz')
    # model.load_from('/mnt/disk/yongsen/model/resnet/resnet50.pth',
                    # '/mnt/disk/haoli/vit_models/imagenet21k/imagenet21k_ViT-B_32.npz')
    from thop import profile, clever_format

    x = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, (x,))
    # print('flops: ', flops, 'params: ', params)
    # print("%.2fM" % (flops / 1e6), "%.2fM" % (params / 1e6))
    # print(" params| %.2f flops| %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位，
    macs, params = clever_format([flops, params], "%.2f")
    # 8.08G   5.03M   10.14G   5.88M
    print(macs, " ", params)