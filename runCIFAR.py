#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:16:11 2021

@author: becca
"""

import torch
import torchvision as tv
from torchvision.transforms import transforms

from SimCLR.Augment import TrainDataAugmentation, TuneDataAugmentation,\
  SimCLRDataTransform, TuneDataTransform
from SimCLR.Makers import Train, FineTune, Validate, SimCLR
from SimCLR.Models import ResNetSimCLR

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def test_finetune():
  # fine tuning transform
  transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((.5,.5,.5),(.5,.5,.5))]
  )
  # dataset for finetuning
  tuneset = tv.datasets.CIFAR10(
    root='Data', train=True, download=True,
    transform=transform
  )

  # make fine tuning object
  simclr = FineTune(
    mlp_layers=3, batch_size=10,
    blocks_sizes=[2**i for i in [5,6,7,8]],
    blocks_layers=[2,2,2,2]
  )

  # do the fine tuning
  data = simclr.load_data(tuneset)
  model, losses = simclr.fine_tune(data, 'chkpt/CIFAR10-tune-test.tar', n_epochs=10)

  return model, losses


def trainCIFAR():
  s = 1
  input_shape = (96,96,3)
  temperature = .5
  
  # get transform pipeline
  transform = TrainDataAugmentation(s, input_shape)
  augment = transform.augment()
  
  # load data
  trainset = tv.datasets.CIFAR10(
    root='Data', train=True, download=False, 
    transform=SimCLRDataTransform(augment)
  )
  
  simclr = Train()
  data = simclr.load_data(trainset)
  model,losses = simclr.train(data, temperature,n_epochs=1,ckpt_path='CIFAR10.tar')
  return model, losses

def validateCIFAR(modelpath):
  s = 1
  input_shape = (96,96,3)  
  
  # get transform pipeline
  transform = TuneDataAugmentation(s, input_shape)
  augment = transform.augment()
  
  # load data
  testdata = tv.datasets.CIFAR10(root='TestData',
                                  train=False,download=False,transform=SimCLRDataTransform(augment))
  
  simclr = Validate()
  testloader = simclr.load_data(testdata)
  accuracy =  simclr.validate_labels(modelpath,testloader)
  return accuracy

def end_to_end():
  s=1
  input_shape = (96,96,3)  

  # set up training data
  train_trans = TrainDataAugmentation(s, input_shape)
  train_aug = train_trans.augment()

  train_data = tv.datasets.CIFAR10(
    root='Data', train=True, download=True, 
    transform=SimCLRDataTransform(train_aug)
  )

  # set up tuning data
  tune_trans = TuneDataAugmentation(s, input_shape)
  tune_aug = tune_trans.augment()

  tune_data = tv.datasets.CIFAR10(
    root='Data', train=True, download=True, 
    transform=TuneDataTransform(tune_aug)
  )

  # set up val data
  val_data = tv.datasets.CIFAR10(
    root='Data', train=False, download=True, 
    transform=TuneDataTransform(tune_aug)
  )

  # create the base model
  model = ResNetSimCLR(in_channels=3, n_classes=10, mlp_layers=2)
  simclr = SimCLR(model)
  print(f'Our model has {simclr.trainer.num_params:,} parameters')

  results = simclr.full_model_maker(
    train_data, tune_data, val_data, n_cycles=10, train_epochs=30, tune_epochs=20,
    train_path='chkpt/train.tar', tune_path='chkpt/tune.tar'
  )

  return results

if __name__ == "__main__":
#    model, losses = trainCIFAR()
  # accuracy = validateCIFAR('CIFAR10.tar81')
  model, train_loss, tune_loss, accuracy, actual, predicted = end_to_end()

  plt.plot(train_loss)
  plt.title('train loss')
  plt.savefig('vis/train_losses.png')
  plt.clf()

  plt.plot(tune_loss)
  plt.title('tune losses')
  plt.savefig('vis/tune_losses.png')
  plt.clf()

  plt.plot(accuracy)
  plt.title('accuracy')
  plt.savefig('vis/accuracy.png')
  plt.clf()


  classes = [
    'airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 
    'horses', 'ships', 'trucks'
  ]

  matrix = confusion_matrix(actual, predicted, labels=[0,1,2,3,4,5,6,7,8,9])

  figure = plt.figure(figsize=(16,9))
  ax = figure.add_subplot(111)


  disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
  disp.plot(ax=ax)

  plt.savefig('vis/confusion_matrix.png')
  plt.clf()
