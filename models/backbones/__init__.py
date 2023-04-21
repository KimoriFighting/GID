from .transformer import Transformer
from models.backbones.resnetV2 import ResNetV2
from models.backbones.resnet import resnet50

def re_backbone(backbone):
    if backbone == 'Resnet':
        return ResNetV2()
    elif backbone == 'Transformer':
        return Transformer()
