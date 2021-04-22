#!/usr/bin/env python3
from SimCLR.Models import RingLossResNet

from tqdm import tqdm
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  random_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_face_data(base_folder, train_set, batch_size=256,):
  transform = transforms.Compose(
    [
      transforms.ToTensor(), 
      transforms.Normalize((127.5,127.5,127.5),(128,128,128))
    ]
  )

  trainset = torchvision.datasets.ImageFolder(
    root=f'{base_folder}/{train_set}', transform=transform
  )
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  testset = torchvision.datasets.ImageFolder(
    root=f'{base_folder}/{train_set}_Test', transform=transform
  )
  testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=2
  )
  
  train_idx = {value: key for key,value in trainset.class_to_idx.items()}
  test_idx = {value: key for key, value in testset.class_to_idx.items()}
  
  if train_idx != test_idx:
    print("Class to index is not the same in training and test sets.")
  
  return trainloader, testloader, train_idx


def test(model, testloader):
  
  correct = 0
  total = 0

  actual = []
  predicted = []
  confs = []

  with tqdm(total=len(testloader)) as prog:
    prog.set_description('Validating')
    with torch.no_grad():
      for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        
        out = model(x, rep_only=True)
        out = F.softmax(out, 1)
        preds = out.argmax(dim=1)
        
        total += y.size(0)
        correct += sum(preds == y)
        
        actual.extend(y.detach().cpu().numpy())
        predicted.extend(preds.detach().cpu().numpy())
        confs.extend(out.max(dim=1)[0].detach().cpu().numpy())



        prog.set_description(f'Validating | accuracy: {correct/total:.3f}')
        prog.update() 

  return (correct/total).item(), actual, predicted, confs

def get_race(actual, idx_to_labels):

  c_index = 2

  # get classes of images
  actual_names = [idx_to_labels[a] for a in actual]
  actual_classes = np.array([int(a.split('_')[c_index]) for a in actual_names])

  return actual_classes


if __name__ == '__main__':
  
  classes = ['White', 'Black', 'Asian', 'Indian', 'Other']
  classification = 'race'
  removed_path = './Data/RaceDatasets'
  N = len(classes)

  for run in ['White20','White30','White40','White50','White90']:
  
    trainloader, testloader, idx_to_label = get_face_data('Data/RaceDatasets/Percents', run, 128)
    n_classes = len(idx_to_label)

    model = RingLossResNet(3, n_classes, .01, blocks_layers=[3,4,6,3]).to(device)

    model.load_state_dict(torch.load(f'chkpt/{run}.tar')['model'])

    # test model on datapoints in the training set
    accuracy, actual, predicted, confs = test(model, testloader)
    races = get_race(actual, idx_to_label)


    df = pd.DataFrame({'name':actual,'race':races,'confidence':confs})

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    df.boxplot('confidence', by='race', ax=ax)
    fig.suptitle(f'Ring Loss normal confidence with {run}')
    plt.savefig(f'vis/{run}_rl_confidence_norm.png')
    plt.clf()


    