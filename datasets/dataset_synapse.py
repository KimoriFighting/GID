import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2 as cv2
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def random_bright(im, delta=32):
    if random.random() < 0.5:
        delta = random.uniform(-delta, delta)
        im += delta
        im = im.clip(min=0, max=255)
    return im

def random_contrast(im, lower=0.5, upper=1.5):
    if random.random() < 0.5:
        alpha = random.uniform(lower, upper)
        im *= alpha
        im = im.clip(min=0, max=255)
    return im

# def random_saturation(im, lower=0.5, upper=1.5):
#     if random.random() < 0.5:
#         im[:, :, 1] *= random.uniform(lower, upper)
#     return im

def random_gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        image:原始图像
        mean : 均值
        var : 方差,越大，噪声越大
    '''
    if random.random() < 0.5:
        image = image / 255  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
        noise = np.random.normal(mean, var ** 0.5, image.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
        image = image + noise#将噪声和原始图像进行相加得到加噪后的图像
        image = image*255
        image = image.clip(min=0, max=255)
    return image

def random_gaussianBlur(image):
    '''
        添加高斯噪声
        image:原始图像
        mean : 均值
        var : 方差,越大，噪声越大
    '''
    if random.random() < 0.5:
        image = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=0, sigmaY=0)
        image = image.clip(min=0, max=255)
    return image

def random_gamma(image):
    if random.random() < 0.5:
        image = cv2.convertScaleAbs(image)
        fgamma = 120
        image = np.power((image / 255.0), fgamma) * 255.0
    return image

def random_clahe(image):
    if random.random() < 0.5:
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))
        image = clahe.apply(image)
    return image

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # image = image * 255
        # image = random_bright(image)
        # image = random_contrast(image)
        # # image = random_saturation(image)
        # image = random_gasuss_noise(image)
        # image = random_gaussianBlur(image)
        # # image = random_gamma(image)
        # # image = random_clahe(image)
        # image = image / 255

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
