# This file evaluates Neural Network predictions, given a set of trained weights. 
# It saves these predictions in logit form.
# This script is based on Markus Kangsepp's implementation.
# The lines which need to be edited from experiment to experiment have been outlined below. 
# This is for the experiment in which 4,500 Latent Blended pairs were created with SLI, mean
# latent codes, Beta(0.2,0.2) weighting, and a single mixed image per pair of parent images.
# The ResNet model is originally from:
# https://github.com/BIGBALLON/cifar-10-cnn/blob/master/4_Residual_Network/ResNet_keras.py

import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
import pickle

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model


stack_n            = 18            
num_classes10      = 10
num_classes100     = 100
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 200

# Edit the numerator below to equal total training set size (45,000 + aug_set_size)
iterations         = 49500 // batch_size 
weight_decay       = 0.0001
mean = [125.307, 122.95, 113.865]  # Pre-calculated 
std  = [62.9932, 62.0887, 66.7048] # Pre-calculated

# Set a seed for reproducibility
seed = 333

# Load in the model weights corresponding to the discriminator in question
weights_file_10 = "resnet_110_45kclip_augmean_beta_largesample_4500.h5"

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 150:
        return 0.01
    return 0.001

# Define the ResNet
def residual_network(img_input,classes_num=10,stack_n=5):
    
    # Define residual blocks
    def residual_block(intput,out_channel,increase=False):
        if increase:
            stride = (2,2)
        else:
            stride = (1,1)

        pre_bn   = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel,kernel_size=(3,3),
                        strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1   = BatchNormalization()(conv_1)
        relu1  = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel,kernel_size=(3,3),
                        strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1,1),
                                strides=(2,2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput,conv_2])
        return block

    # total layers = stack_n * 3 * 2 + 2
    # stack_n = 5 by default, total layers = 32
    # Input dimensions: 32x32x3 
    # Output dimensions: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),
               strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # Input dimensions: 32x32x16
    # Output dimensions: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x,16,False)

    # Input dimensions: 32x32x16
    # Output dimensions: 16x16x32
    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)
    
    # Input dimensions: 16x16x32
    # Output dimensions: 8x8x64
    x = residual_block(x,64,True)
    for _ in range(1,stack_n):
        x = residual_block(x,64,False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes_num,activation='softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x


if __name__ == '__main__':

    print("Evaluating model for the test set")
      
    # Load in the CIFAR-10 test set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_test = keras.utils.to_categorical(y_test, num_classes10)
    
    # Load in the augmentation set for the experiment being run.
    # The filename should be adjusted appropriately based on the experiment.
    x_train_additions = np.load('x_augmentation_array.npy')
    y_train_additions = np.load('y_augmentation_array.npy')

    # Split the CIFAR-10 dataset into training, validation, and test sets
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, 
                                                          y_train, 
                                                          test_size=0.1, 
                                                          random_state=seed)  
    
    # Add the augmentation set to the training set
    x_train_additions = x_train_additions.transpose(0,2,3,1)
    y_train45 = keras.utils.to_categorical(y_train45, num_classes10)
    y_train_additions = y_train_additions.reshape(-1, num_classes10)
    x_train45 = np.concatenate((x_train45, x_train_additions),axis=0)
    y_train45 = np.concatenate((y_train45, y_train_additions),axis=0)
    
    # Pre-process colors, as specified in the paper
    img_mean = x_train45.mean(axis=0)  # per-pixel mean
    img_std = x_train45.std(axis=0)
    x_train45 = (x_train45-img_mean)/img_std
    x_val = (x_val-img_mean)/img_std
    x_test = (x_test-img_mean)/img_std
    
    # Assemble the neural network. Adjust line 166 to include appropriate filename.
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = residual_network(img_input,num_classes10,stack_n)
    model    = Model(img_input, output)    
    evaluate_model(model, weights_file_10, x_test, y_test, 
                   bins = 15, verbose = True, 
                   pickle_file = "probs_resnet110_c10clip_augmean_beta_largesample_4500",
                   x_val = x_val, y_val = y_val)
    
