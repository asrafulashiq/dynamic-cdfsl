from typing import List, MutableSequence
import random
from PIL import ImageFilter
import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import cv2


class TransformLoader:
    def __init__(self,
                 image_size=224,
                 normalize_param=dict(mean=IMAGENET_DEFAULT_MEAN,
                                      std=IMAGENET_DEFAULT_STD),
                 normalize_type='imagenet',
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        if normalize_type == 'cifar10':
            self.normalize_param = dict(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
        elif normalize_type == 'cifar100':
            self.normalize_param = dict(mean=[0.5071, 0.4867, 0.4408],
                                        std=[0.2675, 0.2565, 0.2761])
        else:  # imagenet
            self.normalize_param = dict(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    def parse_transform(self, transform_type):
        if transform_type == 'RandomColorJitter':
            return torchvision.transforms.RandomApply(
                [torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8)
        elif transform_type == 'RandomGrayscale':
            return torchvision.transforms.RandomGrayscale(p=0.2)
        elif transform_type == 'RandomGaussianBlur':
            return torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])],
                                                      p=0.5)
        elif transform_type == 'RandomCrop':
            return torchvision.transforms.RandomCrop(self.image_size,
                                                     padding=4)
        elif transform_type == 'RandomResizedCrop':
            return torchvision.transforms.RandomResizedCrop(self.image_size,
                                                            scale=(0.2, 1.))
        elif transform_type == 'CenterCrop':
            return torchvision.transforms.CenterCrop(self.image_size)
        elif transform_type == 'Resize_up':
            return torchvision.transforms.Resize(
                [int(self.image_size * 1.15),
                 int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return torchvision.transforms.Normalize(**self.normalize_param)
        elif transform_type == 'Resize':
            return torchvision.transforms.Resize(
                [int(self.image_size),
                 int(self.image_size)])
        else:
            method = getattr(torchvision.transforms, transform_type)
            return method()

    def get_composed_transform(self, aug=False):
        if isinstance(aug, MutableSequence) or isinstance(aug, tuple):
            transform_list = list(aug)
        elif isinstance(aug, str) and "," in aug:
            aug = aug.split(",")
            return self.get_composed_transform(aug)
        else:
            if aug == 'MoCo' or aug == 'moco' or aug == 'strong':
                transform_list = [
                    'RandomResizedCrop', 'RandomColorJitter',
                    'RandomGrayscale', 'RandomGaussianBlur',
                    'RandomHorizontalFlip', 'ToTensor', 'Normalize'
                ]
            elif aug is True or aug == 'true' or aug == 'weak' or aug == 'randaug':
                transform_list = [
                    'RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor',
                    'Normalize'
                ]
            elif aug == 'train':
                transform_list = [
                    'RandomResizedCrop', 'RandomColorJitter',
                    'RandomHorizontalFlip', 'ToTensor', 'Normalize'
                ]
            elif aug == 'none':
                transform_list = ['Resize', 'ToTensor', 'Normalize']
            elif aug == 'few_shot_query':
                transform_list = [
                    'RandomResizedCrop', 'RandomColorJitter',
                    'RandomHorizontalFlip', 'ToTensor', 'Normalize'
                ]
            else:
                transform_list = [
                    'Resize_up', 'CenterCrop', 'ToTensor', 'Normalize'
                ]

            if aug in ('weak_strong', 'strong_strong', 'weak_weak',
                       'weak_randaug', 'weak_strong_strong', 'strong_weak'):
                augs = aug.split('_')
                tfms = [self.get_composed_transform(aug=ag) for ag in augs]
                transform = MultiTransform(tfms)
                return transform

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = torchvision.transforms.Compose(transform_funcs)

        if aug in ['MoCo', 'moco']:
            transform = TwoCropsTransform(transform)
        return transform


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, another_transform=None):
        self.base_transform = base_transform
        if another_transform is None:
            self.another_transform = base_transform
        else:
            self.another_transform = another_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.another_transform(x)
        return [q, k]


class MultiTransform(object):
    def __init__(self, transforms: list):
        self.base_transforms = transforms

    def __call__(self, x):
        out = []
        for tfm in self.base_transforms:
            out.append(tfm(x))
        return out


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussianBlurCV(object):

    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max -
                     self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample,
                                      (self.kernel_size, self.kernel_size),
                                      sigma)

        return sample
