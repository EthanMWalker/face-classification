#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import albumentations as A
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Modify this part 
inpath = 'UTKFace200'  # location of augmented data
classification = 'race' # either 'gender' or 'race'
classes = ['White','Black', 'Asian', 'Indian','Other']  # list of classes
train_percent = .99   # percent of smallest dataset to use as training
test_percent = .95    # percent of training data to keep as test data
main_folder = 'RaceDatasets'

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


# create dataset with equal number of classes
equal_data = []
for i in range(n_classes):
  equal_data.extend(train_test_split(class_data[i][0], test_size=.5)[0])

print('Length of equal dataset:', len(equal_data))
 
# For Gender, split into datasets of 75%/25% split
if classification == 'gender':
  mostly_female = train_test_split(class_data[1][0], train_size=.75)[0]
  mostly_female.extend(train_test_split(class_data[0][0], train_size=.25)[0])
  mostly_male = train_test_split(class_data[0][0], train_size=.75)[0]
  mostly_male.extend(train_test_split(class_data[1][0], train_size=.25)[0])
  print('Length of mostly female/mostly male datasets:', len(mostly_female), len(mostly_male))
    

# Create Datasets
os.mkdir(main_folder)
os.mkdir(f'{main_folder}/Test')
os.mkdir(f'{main_folder}/Equal/')

if classification == 'gender':
    os.mkdir(f'{main_folder}/MostlyMale/')
    os.mkdir(f'{main_folder}/MostlyFemale/')

for i in unique_files:
    os.mkdir(f'{main_folder}/Test/{i}')
    os.mkdir(f'{main_folder}/Equal/{i}')
    
    if classification == 'gender':
        os.mkdir(f'{main_folder}/MostlyFemale/{i}')
        os.mkdir(f'{main_folder}/MostlyMale/{i}')

#Create class datasets
for d in classes:
  j = classes.index(d)
  outpath = f'{main_folder}/{d}/'
  print(f'Creating class {d}')
  os.mkdir(outpath)
  os.mkdir(f'{main_folder}/{d}_NotUsed/')
  os.mkdir(f'{main_folder}/{d}_Used/')
  
  for i in unique_files:
    os.mkdir(f'{main_folder}/{d}/{i}')

#  # move classes
  c = class_data[j]
  for im in c[0]:
    image = np.asarray(Image.open(f'{inpath}/{im}.jpg.chip.jpg'))
    name = im.split('.')[0]

    # save original to _Used in case we need it
    orig_im = Image.fromarray(resize(image=image)['image'])
    orig_im = orig_im.save(f'{main_folder}/{d}_Used/{name}.jpg')

    # Create augmented data
    for j in range(16):
      aug_image = Image.fromarray(transform(image=image)['image'])
      aug_image = aug_image.save(f'{outpath}{name}/{j}_{name}.jpg')

    # Create empty folders to Test for class consistency

    # Add some to Test dataset
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

    aug_image = Image.fromarray(transform(image=image)['image'])
    aug_imag = aug_image.save(f'{main_folder}/Test/{name}/{name}.jpg')

# move equals
print("Making equal dataset")
for im in equal_data:
  name = im.split('.')[0]

  for c in classes:
    if not os.listdir(f'{main_folder}/{c}/{name}'):
      os.rmdir(f'{main_folder}/Equal/{name}')
      shutil.copytree(f'{main_folder}/{c}/{name}', f'{main_folder}/Equal/{name}')


# move mostly
if classification == 'gender':

  print("Making mostly female dataset")
  for im in mostly_female:
    name = im.split('.')[0]

    for c in classes:
      if not os.listdir(f'{main_folder}/{c}/{name}'):
        os.rmdir(f'{main_folder}/MostlyFemale/{name}')
        shutil.copytree(f'{main_folder}/{c}/{name}', f'{main_folder}/MostlyFemale/{name}')

  print("Making mostly male dataset")
  for im in mostly_male:
    name = im.split('.')[0]
    
    for c in classes:
      if not os.listdir(f'{main_folder}/{c}/{name}'):
        os.rmdir(f'{main_folder}/MostlyMale/{name}')
        shutil.copytree(f'{main_folder}/{c}/{name}', f'{main_folder}/MostlyMale/{name}')

