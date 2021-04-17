#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:27:05 2021

@author: becca
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# Modify this part 
infile = 'AugmentedData'  # location of augmented data
classification = 'gender' # either 'gender' or 'race'
classes = ['Male','Female']  # list of classes

n_classes = len(classes)
class_datasets = {c:set() for c in range(n_classes)}
unique_files = set()
dataset = {}
photos = os.listdir(infile)

    
# Get list of photos and separate into classes
files = os.listdir(infile)
for i in files:
    f_groups = i.split('_')
    n, gender, race = f_groups[0][0], f_groups[2], f_groups[3]
    name = '_'.join(f_groups[1:])
    unique_files.add(name) 
    
#    print(name,gender)
    # split into classes
    if classification == 'gender':
        class_datasets[int(gender)].add(name)
    else:
        class_datasets[int(race)].add(name)

#for im in class_datasets[0]:
 #   print(im)

n = len(unique_files)
class_len = [len(class_datasets[i]) for i in range(n_classes)]
print('number of unique people:', n)
print('number in each class:',class_len)
    
# make datasets same number
dataset_n = int(min(class_len)*.95) + 1 # modify if already even
class_data =[train_test_split(list(class_datasets[i]), train_size=dataset_n) for i in range(n_classes)]
print('Lengths of (training, test) sets:', [(len(class_data[i][0]), len(class_data[i][1])) for i in range(n_classes)])

#for im in class_data[0][0]:
#    print(im)
# create equal dataset
equal_data = []
for i in range(n_classes):
    equal_data.extend(train_test_split(class_data[i][0], test_size=.5)[0])
    
print('Length of equal dataset:', len(equal_data))
 
# Split into groups of 75%
if classification == 'gender':
    mostly_female = train_test_split(class_data[1][0], train_size=.75)[0]
    mostly_female.extend(train_test_split(class_data[0][0], train_size=.25)[0])
    mostly_male = train_test_split(class_data[0][0], train_size=.75)[0]
    mostly_male.extend(train_test_split(class_data[1][0], train_size=.25)[0])
    print('Length of mostly female/mostly male datasets:', len(mostly_female), len(mostly_male))

    

# Move files to each location
for d in classes:
    j = classes.index(d)
    print(f'Creating class {d}')
    os.mkdir('Datasets/'+d+'/')
    os.mkdir('Datasets/'+d+'_Test/')

    # move classes
    c = class_data[j]
    for im in c[0]:
        for j in range(16):
            shutil.copyfile(infile+'/'+str(j)+'_'+im,'Datasets/'+d+'/'+str(j)+'_'+im)
    for im in c[1]:
        for j in range(16):
            shutil.copyfile(infile+'/'+str(j)+'_'+im, 'Datasets/'+d+'_Test/'+str(j)+'_'+im)

# move equals
os.mkdir('Datasets/Equal/')
for im in equal_data:
    for j in range(16):
        shutil.copyfile(infile+'/'+str(j)+'_'+im, 'Datasets/Equal/'+str(j)+'_'+im)

# move mostly
if classification == 'gender':
    os.mkdir('Datasets/MostlyFemale/')
    os.mkdir('Datasets/MostlyMale/')

    for im in mostly_female:
        for j in range(16):
            shutil.copyfile(infile+'/'+str(j)+'_'+im, 'Datasets/MostlyFemale/'+str(j)+'_'+im)
    for im in mostly_male:
        for j in range(16):
            shutil.copyfile(infile+'/'+str(j)+'_'+im, 'Datasets/MostlyMale/'+str(j)+'_'+im)

