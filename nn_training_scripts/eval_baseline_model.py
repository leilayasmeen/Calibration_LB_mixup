# This file obtains predictions, and saves them in logit form, for the baseline ResNet-110.
# It is based off a script implemented by Markus Kangsepp: https://github.com/markus93/NN_calibration

import keras
import numpy as np
from keras.datasets import cifar10, cifar100
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
iterations         = 45000 // batch_size
weight_decay       = 0.0001
mean = [125.307, 122.95, 113.865]  
std  = [62.9932, 62.0887, 66.7048]
seed = 333

# Load in the model weights for the baseline ResNet-110
weights_file_10 = "resnet_110_45kclip.h5"

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
                        strides=stride,padding='same', kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1   = BatchNormalization()(conv_1)
        relu1  = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        if increase:
            projection = Conv2D(out_channel, kernel_size=(1,1), strides=(2,2),
                                padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput,conv_2])
        return block

    # total layers = stack_n * 3 * 2 + 2
    # stack_n = 5 by default, total layers = 32
    # Input dimensions: 32x32x3 
    # Output dimensions: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
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

    x = Dense(classes_num,activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

if __name__ == '__main__':

    # Load in the CIFAR-10 test set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_test = keras.utils.to_categorical(y_test, num_classes10)
    
    # Split into training, validation, and test sets
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, 
                                                          test_size=0.1, random_state=seed)  
    
    # Pre-process colors as specified in the paper
    img_mean = x_train45.mean(axis=0)  
    img_std = x_train45.std(axis=0)
    x_train45 = (x_train45-img_mean)/img_std
    x_val = (x_val-img_mean)/img_std
    x_test = (x_test-img_mean)/img_std
    
    # Assemble the ResNet
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = residual_network(img_input,num_classes10,stack_n)
    model    = Model(img_input, output)    
    evaluate_model(model, weights_file_10, x_test, y_test, 
                   bins = 15, verbose = True, 
                   pickle_file = "probs_resnet110_c10clip", 
                   x_val = x_val, y_val = y_val)
    
