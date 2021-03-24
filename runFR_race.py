import torch
import torchvision as tv
from torchvision.transforms import transforms

from SimCLR.Augment import TrainDataAugmentation, TuneDataAugmentation,\
  SimCLRDataTransform, TuneDataTransform
from SimCLR.Makers import Train, FineTune, Validate, SimCLR
from SimCLR.Models import ResNetSimCLR

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_race_fr():
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
    train_data, tune_data, val_data, n_cycles=1, train_epochs=101, tune_epochs=20,
    train_path='chkpt/race_train.tar', tune_path='chkpt/race_tune.tar'
  )

  return results


if __name__ == "__main__":

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
