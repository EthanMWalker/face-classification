#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:16:11 2021

@author: becca
"""


import torchvision as tv
from Augment import DataAugmentation, SimCLRDataTransform
from Models import SimCLR, Validate


def trainCIFAR():
    s = 1
    input_shape = (96,96,3)
    temperature = .5
    
    # get transform pipeline
    transform = DataAugmentation(s, input_shape)
    augment = transform.augment()
    
    # load data
    trainset = tv.datasets.CIFAR10(root='TrainData',
                                   train=True,download=False,transform=SimCLRDataTransform(augment))
    
    
    
    simclr = SimCLR()
    data = simclr.load_data(trainset,s,input_shape)
    model,losses = simclr.train(data, temperature,n_epochs=1,ckpt_path='CIFAR10.tar')
    return model, losses

def validateCIFAR(modelpath):
    s = 1
    input_shape = (96,96,3)  
    
    # get transform pipeline
    transform = DataAugmentation(s, input_shape)
    augment = transform.augment()
    
    # load data
    testdata = tv.datasets.CIFAR10(root='TestData',
                                   train=False,download=False,transform=SimCLRDataTransform(augment))
    
    simclr = Validate()
    testloader = simclr.load_data(testdata,s,input_shape)
    i_accuracy, j_accuracy =  simclr.validate_labels(modelpath,testloader)
    return i_accuracy, j_accuracy


if __name__ == "__main__":
#    model, losses = trainCIFAR()
    i_accuracy, j_accuracy = validateCIFAR('CIFAR10.tar81')