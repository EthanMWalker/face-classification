#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
from mahayat/SimCLR-2
Copyright (c) 2020 Thalles Silva
"""

import torchvision.transforms as transforms
import cv2
import numpy as np

np.random.seed(0)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class TrainDataAugmentation(object):
    def __init__(self,s,input_shape):
        self.s = s
        self.input_shape = input_shape
        
    def augment(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])+1),
                                              transforms.ToTensor()])
        return data_transforms

class TuneDataAugmentation(object):
    def __init__(self,s,input_shape):
        self.s = s
        self.input_shape = input_shape
        
    def augment(self):
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.input_shape[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        return data_transforms

class TuneDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x = self.transform(sample)
        return x
    
class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
