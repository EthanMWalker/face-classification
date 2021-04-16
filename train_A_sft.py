#!/usr/bin/env python3
from SimCLR.Models import ResNet, ASoftmaxResNet
from SimCLR.Loss import AngularSoftmax

from tqdm import tqdm
import pickle

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


def train(model, opt, sch, trainloader, n_epochs, filename):

  losses = []

  with tqdm(total=len(trainloader)*n_epochs) as prog:
    for i in range(n_epochs):
      running_loss = 0
      for k,(x,y) in enumerate(trainloader):

        x = x.to(device)
        y = y.to(device)

        out, loss = model(x,y)

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

        embed, out = model(x, y, rep_only=True)
        preds = out.argmax(dim=1)

        total += y.size(0)
        correct += sum(preds == y)

        actual.extend(y.detach().cpu())
        predicted.extend(preds.detach().cpu())
        embeddings.extend(embed.detach().cpu())

        prog.set_description(f'Validating | accuracy: {correct/total:.3f}')
        prog.update()
    
  return correct/total, actual, predicted, embeddings


if __name__ == '__main__':
  trainloader, testloader = get_data()

  model = ASoftmaxResNet(3, 3, 10, blocks_layers=[3,4,6,3]).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-4)
  sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, len(trainloader)
  )

  model, losses = train(
    model, opt, sch,
    trainloader, 30, 'chkpt/asft_test.tar'
  )

  plt.plot(losses)
  plt.title('loss')
  plt.savefig('vis/asft_cifar_losses.png')
  plt.clf()

  accuracy, actual, predicted, embeddings = test(model, testloader)

  classes = [
    'plane', 'car', 'bird', 'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck'
  ]

  matrix = confusion_matrix(actual, predicted, labels=[0,1,2,3,4,5,6,7,8,9])

  figure = plt.figure(figsize=(9,9))
  ax = figure.add_subplot(111)
  disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
  disp.plot(ax=ax)
  plt.title(f'accuracy = {accuracy}')
  plt.savefig('vis/asft_cifar_confusion_matrix.png')
  plt.clf()


  with open('chkpt/asft_embds.pickle', 'wb') as out_file:
    pickle.dump(embeddings, out_file)




