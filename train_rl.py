#!/usr/bin/env python3
from SimCLR.Models import RingLossResNet
from SimCLR.Loss import RingLoss

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  random_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


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


def get_results(actual, predicted, idx_to_labels, classification):

  if classification == 'gender':
    c_index = 1
    n_classes = 2
  else:
    c_index = 2
    n_classes = 5

  # get classes of images
  actual_names = [idx_to_labels[a] for a in actual]
  actual_classes = np.array([int(a.split('_')[c_index]) for a in actual_names])


  correct_bool = np.array(actual) == np.array(predicted)
  
  correct_classes = actual_classes[correct_bool]

  correct_counts = [0]*n_classes
  class_counts = [0]*n_classes

  for i in range(N):
    class_counts[i] += (actual_classes == i).sum()
    correct_counts[i] += (correct_classes == i).sum()

  return correct_counts, class_counts


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



def test(model, testloader):
  
  correct = 0
  total = 0

  actual = []
  predicted = []

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
        
        actual.extend(y.detach().cpu().numpy())
        predicted.extend(preds.detach().cpu().numpy())


        prog.set_description(f'Validating | accuracy: {correct/total:.3f}')
        prog.update() 

  return (correct/total).item(), actual, predicted


def test_removed(model, missingloader):
  incorrect = 0
  total = 0
  labeled_threshold = .1

  with tqdm(total=len(testloader)) as prog:
    prog.set_description('Validating')
    with torch.no_grad():
      for x,y in missingloader:

        x = x.to(device)
        y = y.to(device)

        out = model(x, rep_only=True)
        max_values = out.max(dim=1)
        labeled = max_values[0] >= labeled_threshold
        labeled_idx = max_values[1][labeled]
        
        if len(labeled_idx):
          incorrect += 1
        total += y.size(0)

  return incorrect, total


if __name__ == '__main__':
  
  classes = ['Male','Female']
  save_prefix = 'male66'
  classification = 'gender'
  removed_path = 'GenderDatasets'
  trainloader, testloader, idx_to_label = get_face_data('GenderDatasets/Percents', 'Male66')

  N = len(classes)
  n_classes = len(idx_to_label)

  model = RingLossResNet(3, n_classes, .01, blocks_layers=[3,4,6,3]).to(device)
  crit = nn.CrossEntropyLoss()
  opt = torch.optim.Adam(model.parameters(), lr=1e-4)
  sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, len(trainloader)
  )
  
  # train model
  model, losses = train(
    model, opt, crit, sch,
    trainloader, 1, f'chkpt/{save_prefix}.tar'
  )

  plt.plot(losses)
  plt.title('loss')
  plt.savefig(f'vis/{save_prefix}_losses.png')
  plt.clf()

  # test model on datapoints in the training set
  accuracy, actual, predicted = test(model, testloader)
  
  # plot accuracies
  correct_counts, class_counts = get_results(actual, predicted, idx_to_label, classification)
  class_accuracy = [i/j for i,j in zip(correct_counts, class_counts)]
 
  # make arrays for bar plot
  all_accuracies = [accuracy]
  all_accuracies.extend(class_accuracy)
  accuracy_labels = ['Total Accuracy']
  accuracy_labels.extend(classes)
  total_correct, total = sum(correct_counts), sum(class_counts)
  accuracy_counts = [f'{total_correct}/{total}']
  accuracy_counts.extend([f'{i}/{j}' for i,j in zip(correct_counts, class_counts)])
  
  bar1 = plt.bar(list(range(N+1)),all_accuracies, tick_label=accuracy_labels)
  for i in range(N+1):
      bar = bar1[i]
      height = bar.get_height()
      plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{accuracy_counts[i]}', ha='center', va='bottom')
  plt.title(f'Accuracies for {save_prefix}')
  plt.savefig(f'vis/{save_prefix}_accuracy.png')
  plt.clf()

  # Confusion matrix
  M = [[class_accuracy[0], class_counts[1]-class_accuracy[1]],[class_counts[0]-class_accuracy[0] ,class_accuracy[1]]] 
  
  fig = plt.figure()
  ax = plt.gca()
  im = ax.matshow(M, cmap=plt.cm.Blues)
  fig.colorbar(im)
  
  for i in range(N):
    for j in range(N):
      ax.text(x=i,y=j,s=M[i][j], va='center',ha='center')

  ax.set_xticks(np.arange(N))
  ax.set_yticks(np.arange(N))
  ax.set_xticklabels(classes)
  ax.set_yticklabels(['Correct','Incorrect'])
  plt.title('Confusion Matrix')
  plt.ylabel('Predicted')
  plt.xlabel('Actual')
  plt.savefig(f'vis/{save_prefix}_confusion.png')
  plt.clf()


  # test with removed images
  removed_folder = [f'{removed_path}/{c}_Test' for c in classes]
  removed_loader = [get_removed_data(rf, batch_size=25) for rf in removed_folder]

  labeled = []
  total = []
  for loader in removed_loader:
    incorrect, counts = test_removed(model, loader)
    labeled.append(incorrect)
    total.append(counts)

  
  bar2 = plt.bar(list(range(N)), labeled, tick_label=classes)
  for i in range(N):
      bar = bar2[i]
      height = bar.get_height()
      plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{total[i]}', ha='center', va='bottom')
  plt.title(f'Number of people not in dataset that were identified for {save_prefix}')
  plt.savefig(f'vis/{save_prefix}_removed.png')
  plt.clf()

