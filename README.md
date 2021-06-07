For the complete reports, see the `pdf` files.

# Using Contrastive Learning Methods (SimCLR) for Deep Semi-Supervised Face Classification

### Rebecca Jones, Ethan Walker

### March 27, 2021

## 1 Overview

Classifying images without labels is a diﬀicult machine learning task. Last year, a new
method, SimCLR, came out that substantially improved on previous state-of-the-art self-
supervised learning.[ 2 ] Using small percentages of labeled data, the authors were able to
achieve almost the same accuracy as supervised learning on some common benchmark datasets.
An updated version, SimCLR2, was released in October 2020 that made some improvements
to SimCLR. Notably, in increased accuracy by using deeper ResNets, scaling from ResNet-
to ResNet-152 and making the projection head three layers instead of two. We apply this
method to a combined dataset of facial images to classify the images by race and gender.

## 2 Method

SimCLR consists of several steps. The first is data augmentation. Each image is copied
and both versions are modified in the following way; crop, flip, color jitter, grayscale, and
gaussian blur. The images are compared using NT-Xent loss.

The goal with this portion is to have a self-supervised portion of the training, where the
model learns to create good representations of the images. Following this step is a fine-tuning
step where the model will then learn to apply those representations to different problems,
race and gender. The fine-tuning portion is performed using a standard cross entropy loss.
During the SimCLR training we used the Layer-wise Adaptive Rate Scaling (LARS)
optimizer with SGD, so that we could increase the batch size [ 6 ]. In our CIFAR10 tests, we
were able to get batch sizes up to 1024.

### 2.1 Architecture

We used a 50-layer ResNet as our encoder, and a 3-layer feed forward network as a projection
head. Although improvements to SimCLR suggest deeper networks, we did not have the GPU
capacity to run a ResNet-152. The projection head that is learned during the NT-Xent loss
portion is discarded during the fine tuning phase and a new projection head is created. This
allows the projection head to be more easily trained for different tasks


# Exploring The Effect of Loss Functions on Deep Facial Recognition Algorithms

### Rebecca Jones, Ethan Walker

### April 21, 2021

## 1 Overview

Facial recognition algorithms are well known to be biased against minority groups. Not
only are minority groups misclassified more often, they are also shown to be recognized in
databases in which they are not present more often than majorities. We build on previous
work that has been done on investigating bias, and look at how imbalanced datasets
affect the accuracy and misclassification of different demographics.

## 2 Methods

### 2.1 Architecture

The different implementations of facial recognition algorithms were primarily focused on the
use of different loss functions. So as a base we used a 50 ResNet equipped with the following
different loss functions. We implemented each of these loss functions as described in the
associated papers.
