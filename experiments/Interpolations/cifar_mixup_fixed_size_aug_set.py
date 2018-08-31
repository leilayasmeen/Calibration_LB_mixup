
# This file creates an augmentation set using mixup; this augmentation set should be appended to neural network training sets prior to commencing training.

import os, sys
sys.path.append(os.getcwd())

import random

import numpy as np
import tensorflow as tf
import imageio
from imageio import imsave

import keras

import time
import functools

import sklearn
from sklearn.model_selection import train_test_split

DATASET = 'cifar10' # mnist_256
SETTINGS = '32px_cifar' # mnist_256, 32px_small, 32px_big, 64px_small, 64px_big

OUT_DIR = DATASET + '_mixup_fixed_size_aug_set'

if not os.path.isdir(OUT_DIR):
   os.makedirs(OUT_DIR)
   print "Created directory {}".format(OUT_DIR)
   
from keras.datasets import cifar10
N_CHANNELS = 3
HEIGHT = 32
WIDTH = 32
NUM_CLASSES = 10
(x_train_set, y_train_set), (x_test_set, y_test_set) = cifar10.load_data()
   
x_train_set = x_train_set.transpose(0,3,1,2)
x_test_set = x_test_set.transpose(0,3,1,2)
    
seed = 333
x_train_set, x_dev_set, y_train_set, y_dev_set = train_test_split(x_train_set, y_train_set, test_size=0.1, random_state=seed)

from keras.utils import np_utils           
x_augmentation_set = np.zeros((1, N_CHANNELS, HEIGHT, WIDTH)) 
y_augmentation_set = np.zeros((1, 1, NUM_CLASSES)) 

# Function to translate numeric images into plots
def color_grid_vis(X, nh, nw, save_path):
  # from github.com/Newmu
  X = X.transpose(0,2,3,1)
  h, w = X[0].shape[:2]
  img = np.zeros((h*nh, w*nw, 3))
  for n, x in enumerate(X):
    j = n/nw
    i = n%nw
    img[j*h:j*h+h, i*w:i*w+w, :] = x
    imsave(OUT_DIR + '/' + save_path, img)            
                
numsamples = 45000
            
x_train_set_array = np.array(x_train_set)
y_train_set_array = np.array(y_train_set)  

for imagenum in range(numsamples):
    pvals = np.random.beta(0.2, 0.2, 1)
                    
    imageindices = random.sample(range(x_train_set.shape[0]),2)
    imageindex1 = imageindices[0]
    imageindex2 = imageindices[1]
                    
    # Draw the corresponding images and labels from the training data
    image1 = x_train_set[imageindex1,:]
    image2 = x_train_set[imageindex2,:]  
    label1 = y_train_set[imageindex1,:]
    label2 = y_train_set[imageindex2,:]
                
    # Reshape
    xarray1 = image1.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
    xarray2 = image2.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
    label1 = label1.reshape(1, 1)
    label2 = label2.reshape(1, 1)
                    
    # Save the original images
    #print "Saving original samples"
    #color_grid_vis(
    #    xarray1,
    #    1,
    #    1,
    #    'original_1_classes{}and{}_num{}.png'.format(label1,label2,imagenum)
    #    )
    #color_grid_vis(
    #    xarray2,
    #    1,
    #    1,
    #    'original_2_classes{}and{}_num{}.png'.format(label1,label2,imagenum)
    #    )      
               
    # Change labels to matrix form before performing interpolations
    y1 = np_utils.to_categorical(label1, NUM_CLASSES) 
    y2 = np_utils.to_categorical(label2, NUM_CLASSES) 
                     
    # Combine the arrays and labels
    for p in pvals:
        new_xarray = np.multiply(p,xarray1) + np.multiply((1-p),xarray2)
        new_label = np.multiply(p,y1) + np.multiply((1-p),y2)
        new_label = new_label.reshape(1,1,NUM_CLASSES)

        x_augmentation_set = np.concatenate((x_augmentation_set, new_xarray), axis=0)
        y_augmentation_set = np.concatenate((y_augmentation_set, new_label), axis=0)
               
x_augmentation_array = np.delete(x_augmentation_set, (0), axis=0)
y_augmentation_array = np.delete(y_augmentation_set, (0), axis=0)
            
x_augmentation_array = x_augmentation_array.astype(np.uint8)

np.save(OUT_DIR + '/' + 'x_augmentation_array_mixup', x_augmentation_array) 
np.save(OUT_DIR + '/' + 'y_augmentation_array_mixup', y_augmentation_array) 
