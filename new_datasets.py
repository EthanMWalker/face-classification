#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Creates a dataset for each class where each dataset has majority_percent of one class and equal percentage of other classes.
Also creates a test dataset with only the images in the datsets
'''

import os
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


def create_datasets(inpath, classes, majority_percent, outpath):

  # intialize variables
  idx_to_class = {i:c for i,c in enumerate(classes)}
  class_files = {c:[] for c in classes}
  minority_percent = ((1-(majority_percent/100))/(majority_percent/100)) / (len(classes) -1 )
  
  print("Minority Percent:", minority_percent/100)

  files = [os.listdir(f'{inpath}/{i}') for i in classes]
  unique_files = [f for sublist in files for f in sublist]
 
  # populate class dictionary with files 
  for image in unique_files:
    f_groups = image.split('_')
    gender, race =  int(f_groups[1]), int(f_groups[2])

    if len(classes) == 2:
      class_files[idx_to_class[gender]].append(image)
    else:
      class_files[idx_to_class[race]].append(image)
  
  # create outpath folder if it does not exist
  if not os.path.isdir(outpath):
    os.mkdir(outpath)

  # create datasets for each class
  for c in classes:
    classpath = f'{outpath}/{c}{majority_percent}'
    testpath = f'{outpath}/{c}{majority_percent}_Test'

    # if remove folders if exist
    if os.path.isdir(classpath):
      shutil.rmtree(classpath)
    if os.path.isdir(testpath):
      shutil.rmtree(testpath)

    # make base folders
    os.mkdir(classpath)
    os.mkdir(testpath)

    # add data from each class to dataset
    for c2 in classes:
      
      # if majority class, use all data
      if c2 == c or minority_percent==1:
          train = class_files[c2]
      else:
          train, _ = train_test_split(class_files[c2], train_size=minority_percent)
      # move data to training and test datasets
      for i in train:
          shutil.copytree(f'{inpath}/{c2}/{i}', f'{classpath}/{i}')
          shutil.copytree(f'{inpath}/{c2}_Test/{i}', f'{testpath}/{i}')


if __name__ == '__main__':
 
  inpath = 'GenderDatasets'  # location of augmented data
  classes = ['Male','Female']#['White','Black', 'Asian', 'Indian','Other']  # list of classes (these must the same as the folders in inpath)
  outpath = 'GenderDatasets/Percents'  # path to save datasets
  percents = [50, 60, 70, 80, 90] #[20, 30, 40, 50, 60, 70, 80] # percent of majority dataset to use as training

  for majority_percent in percents:
    create_datasets(inpath, classes, majority_percent, outpath)
