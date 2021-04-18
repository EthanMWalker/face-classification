#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Creates base datasets.
One Test dataset with all available classes and a random assortment of images that are in the training dataset as well as not in the training set.

An augmented dataset with 16 images per photo of training images.

_NotUsed contains the original photos for the test data that is not in the training set.
_Used contains the original photos for the training data.

'''
import os
import shutil
import albumentations as A
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Modify this part 
inpath = 'UTKFace200'  # location of augmented data
classification = 'gender' # either 'gender' or 'race'
classes = ['Male','Female']#['White','Black', 'Asian', 'Indian','Other']  # list of classes
train_percent = .99   # percent of smallest dataset to use as training
test_percent = .95    # percent of training data to keep as test data
main_folder = 'GenderDatasets'

n_classes = len(classes)
class_datasets = {c:set() for c in range(n_classes)}
unique_files = []



# Define transform pipeline
transform = A.Compose([
           A.OneOf([
             A.RandomCrop(100,100,p=1),
             A.Resize(100,100,p=1)
           ],p=1),
           A.HorizontalFlip(p=.5),
           A.OneOf([
             A.MedianBlur(blur_limit=3, p=.8),
             A.Blur(blur_limit=5,p=.8)
           ]),
           A.Downscale(p=.2),
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

    
# Get list of photos and separate into classes
files = os.listdir(inpath)
for image in files:
  f_groups = image.split('_')
  age, gender, race = f_groups[0], f_groups[1], f_groups[2]
  name = image.split('.')[0]


  # remove young kids
  if int(age) > 10: 
    unique_files.append(name) 
    
    # split into classes
    if classification == 'gender':
      class_datasets[int(gender)].add(name)
    else:
      class_datasets[int(race)].add(name)

n = len(unique_files)
class_len = [len(class_datasets[i]) for i in range(n_classes)]
print('number of unique people:', n, len(set(unique_files)))
print('number in each class:',class_len)
    
# make class datasets same number
dataset_n = int(min(class_len)*train_percent) + 1 # modify if already even
class_data =[train_test_split(list(class_datasets[i]), train_size=dataset_n) for i in range(n_classes)]
print('Lengths of (training, test) sets:', [(len(class_data[i][0]), len(class_data[i][1])) for i in range(n_classes)])


# Create Datasets
os.mkdir(main_folder)
os.mkdir(f'{main_folder}/Test')

for d in classes:
  j = classes.index(d)
  outpath = f'{main_folder}/{d}/'
  print(f'Creating class {d}')
  os.mkdir(outpath)
  os.mkdir(f'{main_folder}/{d}_NotUsed/')
  os.mkdir(f'{main_folder}/{d}_Used/')
  

  # move classes
  c = class_data[j]
  for im in c[0]:
    image = np.asarray(Image.open(f'{inpath}/{im}.jpg.chip.jpg'))
    name = im.split('.')[0]

    # save original to _Used in case we need it
    orig_im = Image.fromarray(resize(image=image)['image'])
    orig_im = orig_im.save(f'{main_folder}/{d}_Used/{name}.jpg')

    # Create augmented data
    os.mkdir(f'{outpath}/{name}')
    for j in range(16):
      aug_image = Image.fromarray(transform(image=image)['image'])
      aug_image = aug_image.save(f'{outpath}/{name}/{j}_{name}.jpg')


    # Add some to Test dataset
    os.mkdir(f'{main_folder}/Test/{name}')
    if np.random.random() >= test_percent:
      aug_im = Image.fromarray(transform(image=image)['image'])
      aug_im = aug_im.save(f'{main_folder}/Test/{name}/{name}.jpg')

  # Add test set to Test dataset
  for im in c[1]:
    image = np.asarray(Image.open(f'{inpath}/{im}.jpg.chip.jpg'))
    name = im.split('.')[0]
    
    # save original to _NotUsed in case we need it
    orig_im = Image.fromarray(resize(image=image)['image'])
    orig_im = orig_im.save(f'{main_folder}/{d}_NotUsed/{name}.jpg')
    
    os.mkdir(f'{main_folder}/Test/{name}')
    aug_image = Image.fromarray(transform(image=image)['image'])
    aug_imag = aug_image.save(f'{main_folder}/Test/{name}/{name}.jpg')


