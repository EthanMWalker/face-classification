#!/usr/bin/env python3
from SimCLR.Models import ResNet
from SimCLR.Loss import AngularSoftmax

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  return trainloader, testloader


def train(model, crit, opt, trainloader, n_epochs, filename):

  losses = []

  with tqdm(total=len(trainloader)*n_epochs) as prog:
    for i in range(n_epochs):
      running_loss = 0
      for x,y in trainloader:

        x = x.to(device)
        y = y.to(device)

        out = model(x)

        loss = crit(out,y)

        opt[0].zero_grad()
        opt[1].zero_grad()

        loss.backward()

        opt[0].step()
        opt[1].step()

        torch.save(
          {
            'model':model.state_dict(),
            'a_sft': crit.state_dict(),
            'opt1': opt[0].state_dict(),
            'opt2': opt[1].state_dict()
          },
          filename
        )

        running_loss += loss.item()

        prog.set_description(f'epoch: {i}. Loss: {loss.item()}')
        prog.update()

        if i % 400 == 399:
          losses.append(running_loss/400)
  
  return model, losses

def test(model, testloader):
  
  correct = 0
  total = 0

  actual = []
  predicted = []
  with torch.nograd():
    for x,y in testloader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)
      preds = torch.max(out.data, 1)

      total += y.size(0)
      correct += (preds == y).sum().item()

      actual.extend(y.detach().cpu())
      predicted.extend(preds.detach().cpu())
  
  return correct/total, actual, predicted


if __name__ == '__main__':

  model = ResNet(3, 10, blocks_layers=[3,4,6,3]).to(device)
  crit = AngularSoftmax(10).to(device)

  opt1 = torch.optim.SGD(crit.parameters(), lr=1e-3)
  opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)

  trainloader, testloader = get_data()

  model, losses = train(
    model, crit, (opt1,opt2), 
    trainloader, 50, 'chkpt/asft_test.tar'
  )

  plt.plot(losses)
  plt.title('loss')
  plt.savefig('vis/asft_cifar_losses.png')
  plt.clf()

  accuracy, actual, predicted = test(model, testloader)

  classes = [
    'plane', 'car', 'bird', 'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck'
  ]

  matrix = confusion_matrix(actual, predicted, labels=[0,1,2,3,4,5,6,7,8,9])

  figure = plt.figure(figsize=(16,9))
  ax = figure.add_subplot(111)
  disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
  disp.plot(ax=ax)
  plt.title(f'accuracy = {accuracy}')
  plt.savefig('vis/asft_cifar_confusion_matrix.png')
  plt.clf()





