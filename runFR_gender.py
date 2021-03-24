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

def train_gender_fr():
  s=1
  input_shape = (128,128,3)  

  # set up training data
  train_trans = TrainDataAugmentation(s, input_shape)
  train_aug = train_trans.augment()

  train_data = tv.datasets.ImageFolder(
    root='Data/unlabeled', transform=SimCLRDataTransform(train_aug)
  )

  # set up tuning data
  tune_trans = TuneDataAugmentation(s, input_shape)
  tune_aug = tune_trans.augment()

  tune_val_data = tv.datasets.ImageFolder(
    root='Data/labeled', transform=TuneDataTransform(tune_aug)
  )
  tune_data, val_data = random_split(tune_val_data, [7112, 16595])

  # create the base model
  model = ResNetSimCLR(in_channels=3, n_classes=2, mlp_layers=2)
  simclr = SimCLR(model)
  print(f'Our model has {simclr.trainer.num_params:,} parameters')

  results = simclr.full_model_maker(
    train_data, tune_data, val_data, n_cycles=1, train_epochs=31, tune_epochs=20,
    train_path='chkpt/gender_train.tar', tune_path='chkpt/gender_tune.tar'
  )

  return results


if __name__ == "__main__":

  model, train_loss, tune_loss, accuracy, actual, predicted = train_gender_fr()
  torch.save(model.state_dict(),'Results/final_gender.model')
  with open('Results/gender_train_loss.p','wb') as train_loss_file:
      pickle.dump(train_loss, train_loss_file)
  with open('Results/gender_tune_loss.p','wb') as tune_loss_file:
      pickle.dump(tune_loss,tune_loss_file)
  with open('Results/gender_accuracy.p', 'wb') as accuracy_file:
      pickle.dump(accuracy, accuracy_file)
  with open('Results/gender_predicted.p','wb') as predicted_file:
      pickle.dump(predicted, predicted_file)
  with open('Results/gender_actual.p','wb') as actual_file:
      pickle.dump(actual,actual_file)


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
    'male','female'
  ]

  matrix = confusion_matrix(actual, predicted, labels=[0,1])

  figure = plt.figure(figsize=(16,9))
  ax = figure.add_subplot(111)


  disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
  disp.plot(ax=ax)

  plt.savefig('vis/confusion_matrix.png')
  plt.clf()
