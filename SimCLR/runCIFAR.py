#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:16:11 2021

@author: becca
"""


import torchvision as tv
from Augment import DataAugmentation, SimCLRDataTransform
from Models import SimCLR

s = 1
input_shape = (96,96,3)
batch_size = 1
n_classes= 10
in_channels = 3
d_rep = 128
temperature = .5

# get transform pipeline
transform = DataAugmentation(s, input_shape)
augment = transform.augment()

# load data
trainset = tv.datasets.CIFAR10(root='/home/becca/Document/git/face-classification/TestData',
                               train=True,download=False,transform=SimCLRDataTransform(augment))



simclr = SimCLR()
data = simclr.load_data(trainset,s,input_shape)
model,losses = simclr.train(data, temperature,n_epochs=3,ckpt_path='CIFAR10.tar')



