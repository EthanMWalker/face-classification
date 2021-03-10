#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:16:11 2021

@author: becca
"""

import torch
import torchvision as tv
from torchvision.transforms import transforms
from Augment import DataAugmentation, SimCLRDataTransform
from Models import SimCLR
import matplotlib.pyplot as plt

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
trainset = tv.datasets.CIFAR10(
  root='TestData', train=True, download=True,
  transform=SimCLRDataTransform(augment)
)

# perform training
simclr = SimCLR(batch_size=40)
data = simclr.load_data(trainset,s,input_shape)
model,losses = simclr.train(
  data, temperature, n_epochs=5, ckpt_path='CIFAR10.tar'
)

plt.plot(losses)
plt.title('train losses')
plt.savefig('train_losses')
plt.clf()


# fine tuning transform
transform = transforms.Compose(
  [transforms.ToTensor(), transforms.Normalize((.5,.5,.5),(.5,.5,.5))]
)
# dataset for finetuning
tuneset = tv.datasets.CIFAR10(
  root='TestData', train=True, download=True,
  transform=transform
)


# make simclr with the pretrained model
simclr = SimCLR(batch_size=10)
simclr.load_model('CIFAR10.tar')

# do the fine tuning
data = simclr.load_data(tuneset, s, input_shape)
model, losses = simclr.fine_tune(data, 'CIFAR10-tune.tar', n_epochs=10)

plt.plot(losses)
plt.title('fine tuning losses')
plt.savefig('tune_losses')
plt.clf()

