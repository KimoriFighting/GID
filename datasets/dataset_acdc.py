import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


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


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

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


class Acdc_dataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform  # using transform in torch!
        data = h5py.File("/mnt/disk/yongsen/Semantic_Segmentation/pro_ACDC/preproc_data/data_2D_size_256_256_res_1.36719_1.36719.hdf5", 'r')
        self.split = split
        self.images_train = data['images_train']
        self.labels_train = data['masks_train']
        self.images_val = data['images_test']
        self.labels_val = data['masks_test']

    def __len__(self):
        if self.split == 'train':
            return len(self.images_train)
        else:
            return len(self.images_val)

    def __getitem__(self, idx):
        if self.split == "train":
            image, label = self.images_train[idx], self.labels_train[idx]
        else:
            image, label = self.images_val[idx], self.labels_val[idx]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        # sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
