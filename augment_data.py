#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:55:20 2021

@author: becca
"""


import albumentations as A
from PIL import Image
import numpy as np
import os


def augment(inpath, outpath):
    
    # Define transform pipeline
    transform = A.Compose([
           A.OneOf([
                   A.Crop(36,36,164,164,p=1),
                   A.Resize(100,100,p=1)
                   ]),
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
           ]
    )
    
    # define resize for original image
    resize = A.Compose([
                    A.Resize(100,100,p=1)
                    ])
    

    for image in os.listdir(inpath):
        
        # remove children under the age of 10
        if int(image.split('_')[0]) not in list(range(10)):

            # load image
            im = np.asarray(Image.open(inpath + '/'+ image))

            # save original resized image
            im16 = Image.fromarray(resize(image=im)['image'])
            im16 = im16.save(outpath + '/15_' + image)  
            
            # create 15 more transformed images
            for i in range(15):
                aug_image = Image.fromarray(transform(image=im)['image'])
                aug_image = aug_image.save(outpath + '/' + str(i) + '_' + image)
            



if __name__ == "__main__":
    inpath = 'UTKFace200'   # Folder name with files to augment
    outpath = 'AugmentedData'  # Path to augmented data
    augment(inpath, outpath)
