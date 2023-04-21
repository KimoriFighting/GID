import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch
import random
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

class resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
        if 'label' in sample.keys():
            sample['label'] = sample['label'].resize(self.size, Image.BILINEAR)
        if 'mask' in sample.keys():
            sample['mask'] = sample['mask'].resize(self.size, Image.BILINEAR)

        return sample

class random_scale_crop:
    def __init__(self, range=[0.75, 1.25]):
        self.range = range

    def __call__(self, sample):
        scale = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]
        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'label', 'contour']:
                    base_size = sample[key].size

                    scale_size = tuple((np.array(base_size) * scale).round().astype(int))
                    sample[key] = sample[key].resize(scale_size)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

        return sample

class random_flip:
    def __init__(self, lr=True, ud=True):
        self.lr = lr
        self.ud = ud

    def __call__(self, sample):
        lr = np.random.random() < 0.5 and self.lr is True
        ud = np.random.random() < 0.5 and self.ud is True

        for key in sample.keys():
            if key in ['image', 'label', 'contour']:
                sample[key] = np.array(sample[key])
                if lr:
                    sample[key] = np.fliplr(sample[key])
                if ud:
                    sample[key] = np.flipud(sample[key])
                sample[key] = Image.fromarray(sample[key])

        return sample

class random_rotate:
    def __init__(self, range=[0, 360], interval=1):
        self.range = range
        self.interval = interval

    def __call__(self, sample):
        rot = (np.random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'label', 'contour']:
                    base_size = sample[key].size

                    sample[key] = sample[key].rotate(rot, expand=True)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

        return sample

class random_image_enhance:
    def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, sample):
        image = sample['image']
        np.random.shuffle(self.enhance_method)

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(image)
                factor = float(1 + np.random.random() / 10)
                image = enhancer.enhance(factor)
        sample['image'] = image

        return sample

class random_dilation_erosion:
    def __init__(self, kernel_range):
        self.kernel_range = kernel_range

    def __call__(self, sample):
        gt = sample['label']
        gt = np.array(gt)
        key = np.random.random()
        # kernel = np.ones(tuple([np.random.randint(*self.kernel_range)]) * 2, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(*self.kernel_range),) * 2)
        if key < 1 / 3:
            gt = cv2.dilate(gt, kernel)
        elif 1 / 3 <= key < 2 / 3:
            gt = cv2.erode(gt, kernel)

        sample['label'] = Image.fromarray(gt)

        return sample

class random_gaussian_blur:
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        if np.random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        sample['image'] = image

        return sample

class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, gt = sample['image'], sample['label']

        sample['image'] = np.array(image, dtype=np.float32)
        sample['label'] = np.array(gt, dtype=np.float32)

        return sample

class normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, gt = sample['image'], sample['label']
        image /= 255
        image -= self.mean
        image /= self.std

        gt /= 255
        sample['image'] = image
        sample['label'] = gt

        return sample

class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, gt = sample['image'], sample['label']

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        gt = torch.from_numpy(gt)
        gt = gt

        sample['image'] = image
        sample['label'] = gt.long()

        return sample

class HemangiomaDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, opt):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        self.transform = opt

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        original_size = gt.size[::-1]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        sample = self.transform({'image': image, 'label': gt})

        sample['name'] = name
        sample['original_size'] = original_size
        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            # if img.size == gt.size:
            images.append(img_path)
            gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return self.size

class HemangiomaDataset_val(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, opt):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.filter_files()
        self.size = len(self.images)

        self.transform = opt

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])

        original_size = image.size[::-1]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        image = self.transform(image)
        sample = {'image': image}

        sample['name'] = name
        sample['original_size'] = original_size
        return sample

    def filter_files(self):
        images = []
        for img_path in self.images:
            img = Image.open(img_path)
            # if img.size == gt.size:
            images.append(img_path)
        self.images = images

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return self.size

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    db_train = HemangiomaDataset(r'/mnt/disk/yongsen/Semantic_Segmentation/hemangioma/img',
                                 r'/mnt/disk/yongsen/Semantic_Segmentation/hemangioma/mask',
                                 opt=transforms.Compose([
                                    resize(size=(224, 224)),
                                     random_scale_crop(range=(0.75, 1.25)),
                                     random_flip(lr=True, ud=True),
                                     random_rotate(range=(0, 359)),
                                     random_image_enhance(),
                                     random_dilation_erosion(kernel_range=(2, 5)),
                                     tonumpy(),
                                     normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                     totensor(),
                                 ])
                                )
    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=0,
                             pin_memory=True)
    for sampled_batch in enumerate(trainloader):
        print(1)