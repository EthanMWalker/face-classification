#!/usr/bin/env python3
from SimCLR.Models import ASoftmaxResNet

from tqdm import tqdm

import torch
import torch.nn as nn
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

  prediceted_names = [idx_to_label[p] for p in predicted]
  predicted_classes = np.array([int(p.split('_')[c_index]) for p in prediceted_names])


  correct_bool = np.array(actual) == np.array(predicted)
  
  correct_classes = actual_classes[correct_bool]

  correct_counts = [0]*n_classes
  class_counts = [0]*n_classes

  for i in range(N):
    class_counts[i] += (actual_classes == i).sum()
    correct_counts[i] += (correct_classes == i).sum()

  return correct_counts, class_counts, actual_classes, predicted_classes


def train(model, opt, crit, sch, trainloader, n_epochs, filename):

  losses = []

  with tqdm(total=len(trainloader)*n_epochs) as prog:
    for i in range(n_epochs):
      running_loss = 0
      for k,(x,y) in enumerate(trainloader):

        x = x.to(device)
        y = y.to(device)

        out, loss = model(x,y)
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
        
        embed, out = model(x, y, rep_only=True)
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
  labeled_threshold = .05

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
        print(max_values)
        
        if len(labeled_idx):
          incorrect += 1
        total += y.size(0)

  return incorrect, total


if __name__ == '__main__':
  
  classes = ['White', 'Black', 'Asian', 'Indian', 'Other']
  classification = 'race'
  removed_path = 'RaceDatasets'
  N = len(classes)

  for run in ['White20', 'White30', 'White90']:
  
    trainloader, testloader, idx_to_label = get_face_data(
      'Data/RaceDatasets/Percents', run, 128
    )
    n_classes = len(idx_to_label)

    model = ASoftmaxResNet(
      3, n_classes, n_classes, blocks_layers=[3,4,6,3]
    ).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
      opt, len(trainloader)
    )
  
    # train model
    model, losses = train(
      model, opt, crit, sch,
      trainloader, 35, f'chkpt/{run}_asft.tar'
    )

    plt.plot(losses)
    plt.title('loss')
    plt.savefig(f'vis/{run}_asft_losses.png')
    plt.clf()

    # test model on datapoints in the training set
    accuracy, actual, predicted = test(model, testloader)
  
    # plot accuracies
    correct_counts, class_counts, act_dem, pred_dem = get_results(actual, predicted, idx_to_label, classification)
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
    plt.title(f'Accuracies for {run} A softmax')
    plt.savefig(f'vis/{run}_asft_accuracy.png')
    plt.clf()

    matrix = confusion_matrix(act_dem, pred_dem, labels=[0,1,2,3,4])

    figure = plt.figure(figsize=(9,9))
    ax = figure.add_subplot(111)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
    disp.plot(ax=ax)
    plt.title(f'accuracy = {accuracy}')
    plt.savefig(f'vis/{run}_asft_face_confusion_matrix.png')
    plt.clf()




    # # test with removed images
    # removed_folder = [f'{removed_path}/{c}_Test' for c in classes]
    # removed_loader = [get_removed_data(rf, batch_size=25) for rf in removed_folder]

    # labeled = []
    # total = []
    # for loader in removed_loader:
    #   incorrect, counts = test_removed(model, loader)
    #   labeled.append(incorrect)
    #   total.append(counts)

  
    # bar2 = plt.bar(list(range(N)), labeled, tick_label=classes)
    # for i in range(N):
    #   bar = bar2[i]
    #   height = bar.get_height()
    #   plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{labeled[i]}/{total[i]}', ha='center', va='bottom')
    # plt.title(f'Number of people not in dataset that were identified for {run}')
    # plt.savefig(f'vis/{run}_removed.png')
    # plt.clf()