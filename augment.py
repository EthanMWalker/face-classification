#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Creates base datasets that can be used to create datasets with set representations.
Creates a folder for each class with 16 augmented images per photo for training.
Creates a folder of test data with one augmented image and original (resized) image per photo in training set.
Creates a folder of missing data with one augmented image and original (resized) image per photo not in training set.
'''

import os
import shutil
import albumentations as A
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


def get_data(inpath, classes, remove_n):

  n_classes = len(classes)
  class_datasets = {c:set() for c in range(n_classes)}
  unique_files = []
  
  # get list of photos and separate into classes
  files = os.listdir(inpath)
  for image in files:
    f_groups = image.split('_')
    age, gender, race = f_groups[0], f_groups[1], f_groups[2]
    name = image.split('.')[0]

    # remove young kids
    if int(age) > 10: 
      unique_files.append(name) 
    
      # split into classes
      if len(classes) == 2:
        class_datasets[int(gender)].add(name)
      else:
        class_datasets[int(race)].add(name)

  class_dataset_n = [len(class_datasets[i]) for i in range(n_classes)]
  print('Number of unique people:', len(set(unique_files)))
  print('Number in each class:', class_dataset_n)
    
  
  # make class datasets same number
  smallest_class_n = min(class_dataset_n) - remove_n
  class_remove_n = [i - smallest_class_n for i in class_dataset_n]
  class_data = [train_test_split(list(class_datasets[i]), test_size=class_remove_n[i]) for i in range(n_classes)]
  print('Lengths of (training, removed) sets:', [(len(class_data[i][0]), len(class_data[i][1])) for i in range(n_classes)])

  return class_data



def augment_data(inpath, classes, class_data, copy_n, outpath):

  # Define transform pipeline
  transform = A.Compose([
           A.OneOf([
             A.RandomCrop(100,100,p=1),
             A.Resize(100,100,p=1),
             A.Sequential([
                 A.RandomCrop(150,150,p=1),
                 A.Resize(100,100,p=1)
             ])
           ],p=1),
           A.HorizontalFlip(p=.5),
           A.OneOf([
             A.MedianBlur(blur_limit=3, p=.8),
             A.Blur(blur_limit=5,p=.8)
           ]),
           A.CoarseDropout(max_holes=4,p=.3),
           A.Downscale(p=.3),
           A.OneOf([
             A.RGBShift(p=.7),
             A.ColorJitter(p=.7),
             A.ToGray(p=.5)
           ]),
           A.OneOf([
             A.MultiplicativeNoise(p=.75),
             A.ISONoise(p=.75)
           ])
  ])

  # define resize for original image
  resize = A.Compose([
                    A.Resize(100,100,p=1)
                    ])

  # remove outpath folder if it exists
  if os.path.isdir(outpath):
    shutil.rmtree(outpath)

  # make main directory
  os.mkdir(outpath)

  # create each class folder and populate with augmented data
  for d in classes:
    print(f'Creating class {d}')

    # Create class folders
    c_outpath = f'{outpath}/{d}'
    testpath = f'{outpath}/{d}_Test'
    missingpath = f'{outpath}/{d}_Missing'
    os.mkdir(c_outpath)
    os.mkdir(testpath)
    os.mkdir(missingpath)

    # move data
    j = classes.index(d)
    c = class_data[j]

    # augment and move training data
    for im in c[0]:
      image = np.asarray(Image.open(f'{inpath}/{im}.jpg.chip.jpg'))
      im_name = im.split('.')[0]

      # Create augmented data
      os.mkdir(f'{c_outpath}/{im_name}')
      for j in range(copy_n):
        aug_image = Image.fromarray(transform(image=image)['image'])
        aug_image = aug_image.save(f'{c_outpath}/{im_name}/{j}_{im_name}.jpg')

      # Add one augmented and original to test dataset
      os.mkdir(f'{testpath}/{im_name}')
      aug_im = Image.fromarray(transform(image=image)['image'])
      aug_im = aug_im.save(f'{testpath}/{im_name}/{im_name}.jpg')
      orig_im = Image.fromarray(resize(image=image)['image'])
      orig_im = orig_im.save(f'{testpath}/{im_name}/{im_name}_orig.jpg')

    # Add removed files to missing dataset
    for im in c[1]:
      image = np.asarray(Image.open(f'{inpath}/{im}.jpg.chip.jpg'))
      im_name = im.split('.')[0]
    
      # save original 
      orig_im = Image.fromarray(resize(image=image)['image'])
      orig_im = orig_im.save(f'{missingpath}/{im_name}.jpg')
   
      # save augmented
      aug_image = Image.fromarray(transform(image=image)['image'])
      aug_imag = aug_image.save(f'{missingpath}/{im_name}.jpg')

 
if __name__ == '__main__':
  
  inpath = 'UTKFace200'      # location of original data
  classes = ['White','Black', 'Asian', 'Indian','Other']  # list of classes
  remove_n = 25              # number of images to remove for missing testing
  copy_n = 16
  outpath = 'RaceDatasets' # folder to save augmented data
  
  class_data = get_data(inpath, classes, remove_n)
  augment_data(inpath, classes, class_data, copy_n, outpath)

