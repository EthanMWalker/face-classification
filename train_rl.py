#!/usr/bin/env python3
from SimCLR.Models import RingLossResNet
from SimCLR.Loss import RingLoss

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_data(batch_size=256):
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
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  return trainloader, testloader



def get_face_data(batch_size=256,train_set='Male'):
  transform = transforms.Compose(
    [
      transforms.ToTensor(), 
      transforms.Normalize((127.5,127.5,127.5),(128,128,128))
    ]
  )

  trainset = torchvision.datasets.ImageFolder(
    root=f'Datasets/{train_set}', transform=transform
  )
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  testset = torchvision.datasets.ImageFolder(
    root='Datasets/Test', transform=transform
  )
  testloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
  )
  
  return trainloader, testloader, {value: key for key, value in testset.class_to_idx.items()}



def train(model, opt, crit, sch, trainloader, n_epochs, filename):

  losses = []

  with tqdm(total=len(trainloader)*n_epochs) as prog:
    for i in range(n_epochs):
      running_loss = 0
      for k,(x,y) in enumerate(trainloader):

        x = x.to(device)
        y = y.to(device)

        out, loss = model(x)
        loss = crit(out,y) + loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()

        prog.set_description(f'epoch: {i} | Loss: {loss.item():.3f}')
        prog.update()

        if k % 10 == 9:
          losses.append(running_loss/10)
          running_loss = 0

      if i > 20:
        sch.step()
      
      if i % 10 == 0:
        torch.save(
          {
            'model':model.state_dict(),
            'opt': opt.state_dict(),
            'epoch': i
          },
          filename
        )

  return model, losses

def test(model, testloader, idx_to_class):
  
  correct = 0
  total = 0

  actual = []
  predicted = []
  class_correct = [0]*N
  class_counts = [0]*N

  with tqdm(total=len(testloader)) as prog:
    prog.set_description('Validating')
    with torch.no_grad():
      for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        
        out = model(x, rep_only=True)
        preds = out.argmax(dim=1)

        total += y.size(0)
        correct_bool = preds == y
        correct += sum(correct_bool)
 
        # get which gender each label belongs to
        y_names = [idx_to_class[i.item()] for i in y]
        class_y = np.array([int(i.split('_')[1]) for i in y_names])
        correct_classes = class_y[correct_bool.cpu()]
        
        for i in range(N):
          class_correct[i] += (correct_classes == i).sum()
          class_counts[i] += (class_y == i).sum()

        actual.extend(y.detach().cpu())

        predicted.extend(preds.detach().cpu())


        prog.set_description(f'Validating | accuracy: {correct/total:.3f}')
        prog.update() 
  
  return correct/total, class_correct, class_counts, actual, predicted


if __name__ == '__main__':
  trainloader, testloader, idx_to_class = get_face_data(train_set='Female')
  
  N = 2 # Number of genders or races
  n_classes = len(idx_to_class)
  model = RingLossResNet(3, n_classes, .01, blocks_layers=[3,4,6,3]).to(device)
  crit = nn.CrossEntropyLoss()
  opt = torch.optim.Adam(model.parameters(), lr=1e-4)
  sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, len(trainloader)
  )

  model, losses = train(
    model, opt, crit, sch,
    trainloader, 3, 'chkpt/faces_female.tar'
  )

  plt.plot(losses)
  plt.title('loss')
  plt.savefig('vis/faces_female_losses.png')
  plt.clf()

  print('\nCalculating accuracy')
  accuracy, class_accuracy, class_counts, actual, predicted = test(model, testloader, idx_to_class)
  
  
  print('\nCalculating Metrics')
  print( accuracy, class_accuracy, class_counts)
  M = [[class_accuracy[0], class_counts[0]],[class_counts[1] ,class_accuracy[1]]] 
  plt.matshow(M)
  plt.title("Confusion Matrix")
  plt.savefig('vis/faces_female_metrics.png')
  plt.clf()


