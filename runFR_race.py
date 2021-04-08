#!/usr/bin/env python3
import torch
import torchvision as tv
from torchvision.transforms import transforms
from torch.utils.data import random_split

from SimCLR.Augment import TrainDataAugmentation, TuneDataAugmentation,\
  SimCLRDataTransform, TuneDataTransform
from SimCLR.Makers import SimCLR
from SimCLR.Models import ResNetSimCLR

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  
import pickle 

def train_race_fr():
  s=1
  input_shape = (128,128,3)  

  # set up training data
  train_trans = TrainDataAugmentation(s, input_shape)
  train_aug = train_trans.augment()

  train_data = tv.datasets.ImageFolder(
    root='Data/unlabeled', transform=SimCLRDataTransform(train_aug)
  )

  # set up tuning and validation data
  tune_trans = TuneDataAugmentation(s, input_shape)
  tune_aug = tune_trans.augment()

  tune_val_data = tv.datasets.ImageFolder(
    root='Data/supervised', transform=TuneDataTransform(tune_aug)
  )

  tune_data, val_data = random_split(tune_val_data, [7112, 16595])

  # create the base model
  saved = torch.load('chkpt/race_train.tar')
  trained_model = model = ResNetSimCLR(
    in_channels=3, n_classes=4, mlp_layers=2, blocks_layers=[3,3,3,3]
  )
  trained_model.load_state_dict(saved['model_state_dict'])

  simclr = SimCLR(trained_model)
  print(f'Our model has {simclr.trainer.num_params:,} parameters')

  results = simclr.full_model_maker(
    train_data, tune_data, val_data, n_cycles=1, train_epochs=21, tune_epochs=20,
    train_path='chkpt/race_train.tar', tune_path='chkpt/race_tune.tar'
  )

  return results

def finish_the_job():
  s=1
  input_shape = (128,128,3)  

  # set up training data
  train_trans = TrainDataAugmentation(s, input_shape)
  train_aug = train_trans.augment()

  train_data = tv.datasets.ImageFolder(
    root='Data/unlabeled', transform=SimCLRDataTransform(train_aug)
  )

  # set up tuning and validation data
  tune_trans = TuneDataAugmentation(s, input_shape)
  tune_aug = tune_trans.augment()

  tune_val_data = tv.datasets.ImageFolder(
    root='Data/supervised', transform=TuneDataTransform(tune_aug)
  )

  tune_data, val_data = random_split(tune_val_data, [7112, 16595])

  saved = torch.load('chkpt/race_train.tar')
  trained_model = model = ResNetSimCLR(
    in_channels=3, n_classes=4, mlp_layers=2, blocks_layers=[3,3,3,3]
  )
  trained_model.load_state_dict(saved['model_state_dict'])

  simclr = SimCLR(trained_model)

  model = simclr.trainer.return_model()
  simclr.make_tuner(model)

  tune_epochs = 50
  tune_path = 'chkpt/race_tune.tar'

  model, tune_losses = simclr.tune(tune_data, tune_epochs, tune_path)

  simclr.make_validator(simclr.tuner.model)
  acc, actual, predicted = simclr.validate(val_data)

  results = (model, tune_losses, acc, actual, predicted)

  return results


if __name__ == "__main__":

  model, train_loss, tune_loss, accuracy, actual, predicted = train_race_fr()
  # model, tune_losses, acc, actual, predicted = finish_the_job()

  # pickle everything 
  with open('chkpt/race_fr.pickle', 'wb') as out_file:
    pickle.dump(
      (train_loss, tune_loss, accuracy, actual, predicted),
      out_file
    )
  
  torch.save(
    {'model_state_dict': model.state_dict()}, 'chkpt/final_race_model.tar'
  )

  plt.plot(train_loss)
  plt.title('train loss')
  plt.savefig('vis/race_fr_train_losses.png')
  plt.clf()

  plt.plot(tune_loss)
  plt.title('tune losses')
  plt.savefig('vis/race_fr_tune_losses.png')
  plt.clf()

  # plt.plot(accuracy)
  # plt.title('accuracy')
  # plt.savefig('vis/race_fr_accuracy.png')
  # plt.clf()


  classes = [
    'white', 'black', 'asian', 'indian', 'other'
  ]

  matrix = confusion_matrix(actual, predicted, labels=[0,1,2,3,4])

  figure = plt.figure(figsize=(16,9))
  ax = figure.add_subplot(111)


  disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
  disp.plot(ax=ax)

  plt.title(f'accuracy = {accuracy}')

  plt.savefig('vis/race_fr_confusion_matrix.png')
  plt.clf()
