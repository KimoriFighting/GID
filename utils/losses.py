import torch
import torch.nn as nn
from torch.nn.modules.loss import KLDivLoss
import torch.nn.functional as F

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.kl = KLDivLoss(size_average=False, reduce=True)

    def forward(self, inputs, target):
        sum = inputs.shape[-1]*inputs.shape[-2]
        log_input = F.log_softmax(inputs)
        target = F.softmax(target)
        return self.kl(log_input, target)/sum

import numpy as np

def value_to_RGB(value):
    a = [None] * len(value)
    for i in range(len(value)):
        a[i] = [None] * len(value[i])
        for j in range(len(value[i])):
            # 设置对应类别的颜色
            if value[i][j] == 0:
                # lv zhibei
                a[i][j] = [0, 0, 0]
            elif value[i][j] == 1:
                # hui  daolu
                a[i][j] = [255, 255, 255]
            elif value[i][j] == 2:
                # lan fangwu
                a[i][j] = [0, 139, 139]
            elif value[i][j] == 3:
                # heliu
                a[i][j] = [0, 0, 128]
            elif value[i][j] == 4:
                # heliu
                a[i][j] = [255, 255, 0]
            elif value[i][j] == 5:
                # heliu
                a[i][j] = [139, 101, 139]
            elif value[i][j] == 6:
                # heliu
                a[i][j] = [255, 139, 193]
            elif value[i][j] == 7:
                # heliu
                a[i][j] = [255, 0, 0]
            elif value[i][j] == 8:
                # heliu
                a[i][j] = [238, 154, 0]
            elif value[i][j] == 9:
                # heliu
                a[i][j] = [139, 62, 47]
    a = np.array(a)
    return a

from PIL import Image
# 传入一个2D的分类结果
def draw_pit(img, path):
    if len(img.shape) == 4:
        img = img.squeeze(0).squeeze(0)
    elif len(img.shape) == 3:
        img = img.squeeze(0)
    from matplotlib import pyplot as plt
    img = value_to_RGB(img)
    # im = Image.fromarray(img, 'RGB')
    # im.save(f'D:\Study\论文\picture\{index}.png')

    plt.imshow(img, interpolation='nearest')

    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
    # plt.show()


class AvgpoolLoss(nn.Module):
    def __init__(self):
        super(AvgpoolLoss, self).__init__()
        self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.kl = KLLoss()

    def forward(self, inputs, target):
        inputs = self.avgpool(inputs.float())
        inputs = self.avgpool(inputs)
        inputs = self.avgpool(inputs)
        target = self.avgpool(target.float())
        target = self.avgpool(target)
        target = self.avgpool(target)
        # image = torch.argmax(torch.softmax(target, dim=1), dim=1).squeeze(0).cpu()
        # draw_pit(image[])
        # print('11100')
        return self.kl(inputs, target)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

if __name__ == "__main__":
    batch_size, num_classes, h, w = 3, 9, 224, 224
    x = torch.autograd.Variable(torch.randn(batch_size, 9, h, w))
    y = torch.autograd.Variable(torch.randn(batch_size, 9, h, w))
    model = AvepoolLoss()
    outputs = model(x, y)
    print(outputs)
    # for i in mid_outputs:
    #     print(i.shape)