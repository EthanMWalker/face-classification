#!/usr/bin/env python3
from SimCLR.Models import RingLossResNet

from tqdm import tqdm

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


def get_removed_data(removed_folder, batch_size=25):
  transform = transforms.Compose(
    [
      transforms.ToTensor(), 
      transforms.Normalize((127.5,127.5,127.5),(128,128,128))
    ]
  )

  removedset = torchvision.datasets.ImageFolder(
    root=f'{removed_folder}', transform=transform
  )
  removed_dataset, _ = random_split(removedset, (batch_size, len(removedset) - batch_size))
  
  removedloader = torch.utils.data.DataLoader(
    removed_dataset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  return removedloader

def test_removed(model, missingloader):
  incorrect = 0
  total = 0
  labeled_threshold = .05
  confs = np.array([])
  

  with tqdm(total=len(missingloader)) as prog:
    prog.set_description('Validating')
    with torch.no_grad():
      for x,y in missingloader:

        x = x.to(device)
        y = y.to(device)

        out = model(x, rep_only=True)
        out = F.softmax(out, 1)
        max_values = out.max(dim=1)
        confs = np.append(confs, max_values[0].detach().cpu().numpy())
        labeled = max_values[0] >= labeled_threshold
        labeled_idx = max_values[1][labeled]

        prog.update()
        
        if len(labeled_idx):
          incorrect += 1
        total += y.size(0)

  return incorrect, total, confs



if __name__ == '__main__':
  
  classes = ['White', 'Black', 'Asian', 'Indian', 'Other']
  classification = 'race'
  removed_path = './Data/RaceDatasets'
  N = len(classes)

  for run in ['White20', 'White30', 'White40', 'White50', 'White60', 'White90', 'White95', 'White98']:
  
    trainloader, testloader, idx_to_label = get_face_data('Data/RaceDatasets/Percents', run, 128)
    n_classes = len(idx_to_label)

    model = RingLossResNet(3, n_classes, .01, blocks_layers=[3,4,6,3]).to(device)

    model.load_state_dict(torch.load(f'chkpt/{run}.tar')['model'])

    # test with removed images
    removed_folder = [f'{removed_path}/{c}_Missing' for c in classes]
    removed_loader = [get_removed_data(rf, batch_size=25) for rf in removed_folder]

    labeled = []
    total = []
    confs = []

    for loader in removed_loader:
      incorrect, counts, ave_conf = test_removed(model, loader)
      labeled.append(incorrect)
      total.append(counts)
      confs.append(ave_conf)


    plt.boxplot(confs, labels=classes)
    plt.title(f'Confidence of Ring Loss for {run}')
    plt.savefig(f'vis/{run}_rl_confidence_plot.png')
    plt.clf()
    # bar2 = plt.bar(list(range(N)), labeled, tick_label=classes)
    # for i in range(N):
    #   bar = bar2[i]
    #   height = bar.get_height()
    #   plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{labeled[i]}/{total[i]}', ha='center', va='bottom')
    # plt.title(f'Number of people not in dataset that were identified for {run}')
    # plt.savefig(f'vis/{run}_removed.png')
    # plt.clf()