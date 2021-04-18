#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Create N datasets where each dataset has majority_percent of one class and equal percentage of other classes.


'''
import os
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


# Modify
inpath = 'GenderDatasets'  # location of augmented data
classification = 'gender' # either 'gender' or 'race'
classes = ['Male','Female']#['White','Black', 'Asian', 'Indian','Other']  # list of classes (these must the same as the folders in inpath)
majority_percent = .66   # percent of smallest dataset to use as training
new_folder = 'GenderDatasets/Percents'  # path to save datasets


# Don't need to modify
class_to_idx = {i:c for i,c in enumerate(classes)}
class_files = {c:[] for c in classes}
os.mkdir(new_folder)
N = len(classes)
minority_percent = (1-majority_percent)/(N-1)

files = [os.listdir(f'{inpath}/{i}') for i in classes]
unique_files = [f for sublist in files for f in sublist]


for image in unique_files:
    f_groups = image.split('_')
    gender, race =  int(f_groups[1]), int(f_groups[2])

    if classification == 'gender':
      class_files[class_to_idx[gender]].append(image)
    else:
      class_files[class_to_idx[race]].append(image)


for c in classes:
    folder_name = f'{new_folder}/{c}_{majority_percent}'
    os.mkdir(folder_name)

    for c2 in classes:
      if c2 == c:
          train, test = train_test_split(class_files[c2], train_size=majority_percent)
      else:
          train, test = train_test_split(class_files[c2], train_size=minority_percent)
      for i in train:
          shutil.copytree(f'{inpath}/{c2}/{i}', f'{folder_name}/{i}')



