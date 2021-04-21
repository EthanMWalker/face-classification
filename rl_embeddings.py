#!/usr/bin/env python3
from SimCLR.Models import RingLossResNet

from tqdm import tqdm
import pickle

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

def get_data(batch_size=128):
  transform = transforms.Compose(
    [
      transforms.ToTensor(), 
      transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    ]
  )

  trainset = torchvision.datasets.CIFAR10(
    root='Data', train=True, download=True, transform=transform
  )
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  testset = torchvision.datasets.CIFAR10(
    root='Data', train=False, download=True, transform=transform
  )
  testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  return trainloader, testloader


def test(model, testloader):
  
  correct = 0
  total = 0

  actual = []
  predicted = []
  embeddings = []
  with tqdm(total=len(testloader)) as prog:
    prog.set_description('Validating')
    with torch.no_grad():
      for x,y in testloader:
        x = x.to(device)
        y = y.to(device)

        out = model(x, rep_only=True)
        preds = out.argmax(dim=1)

        total += y.size(0)
        correct += sum(preds == y)

        actual.extend(y.detach().cpu())
        predicted.extend(preds.detach().cpu())
        embeddings.extend(out.detach().cpu())

        prog.set_description(f'Validating | accuracy: {correct/total:.3f}')
        prog.update()
    
  return correct/total, actual, predicted, embeddings


if __name__ == '__main__':
  
  classes = ['White', 'Black', 'Asian', 'Indian', 'Other']
  classification = 'race'
  removed_path = './Data/RaceDatasets'
  N = len(classes)

  for run in ['White20']:
  
    # trainloader, testloader= get_data(128)

    model = RingLossResNet(3, 10, .01, blocks_layers=[3,4,6,3]).to(device)

    model.load_state_dict(torch.load(f'chkpt/rl_cifar2.tar')['model'])
    print(model.loss.radius)

    # # test model on datapoints in the training set
    # accuracy, actual, predicted, embeddings = test(model, testloader)


    # with open(f'chkpt/rl_embds.pickle', 'wb') as out_file:
    #   pickle.dump(embeddings, out_file)

    