#!/usr/bin/env python3
import torch
import torch.nn as nn
from SimCLR.Models import ResNet, ResNetSimCLR

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((.5,.5,.5),(.5,.5,.5))]
)
trainset = datasets.CIFAR10(
    'Data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=10, shuffle=True, num_workers=2
)

testset = datasets.CIFAR10(
    'Data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=10, shuffle=True, num_workers=2
)

classes = [
    'plane','car','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

def train_model(
    net, trainloader, criterion, optimizer,
    n_epochs, device, clr=None):
    '''
    Trains my resnet using

    Parameters:
        net (ResNet): an untrained resnet
        trainloader (DataLoader): loader for the training data
        criterion: loss function from torch.nn
        optimizer (torch.optim.Optimizer): optimizer object
        device: torch device

    Returns:
        net (ResNet): trained model
        losses (list): data for loss curve 
    '''

    losses = []

    for epoch in tqdm(range(n_epochs)):

        running_loss = 0.
        with tqdm(total=len(trainloader)) as progress:
          for i,data in enumerate(trainloader,0):

              # load the X_train and y_test
              X_train, y_train = data[0].to(device), data[1].to(device)

              optimizer.zero_grad()

              # apply the net to the train data
              h, preds = net(X_train)

              # find the loss
              loss = criterion(preds, y_train)

              # backward
              loss.backward()

              # optimize
              optimizer.step()

              progress.set_description('tune loss:{:.4f}'.format(loss.item()))
              progress.update()

              if clr:
                  clr.step()

              # collect the running loss so that i can see how it's doing
              running_loss += loss.item()
              if i % 40 == 39:
                  # print the man loss
                  losses.append(running_loss/40)
                  running_loss = 0.0
    return net, losses

# make the net
net = ResNetSimCLR(
    3, 10, mlp_layers=3,
    blocks_sizes=[2**i for i in [5,6,7,8]],
    blocks_layers=[2,2,2,2]
)

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


net, loss = train_model(
        net, trainloader, criterion, optimizer, 10, device
)

plt.plot(loss)
plt.savefig('vis/test_test')