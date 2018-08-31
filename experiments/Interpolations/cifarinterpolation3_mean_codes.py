
"""
This file was used to create mixed examples using spherical linear interpolation of mean latent codes, using constant weighting and four mixed examples per parent pair.

This file is derived from: https://github.com/igul222/PixelVAE

PixelVAE: A Latent Variable Model for Natural Images
Ishaan Gulrajani, Kundan Kumar, Faruk Ahmed, Adrien Ali Taiga, Francesco Visin, David Vazquez, Aaron Courville
"""

import os, sys
sys.path.append(os.getcwd())

N_GPUS = 2

import random
import tflib as lib
import tflib.sampling_loop_cifar_filter_3
import tflib.ops.kl_unit_gaussian
import tflib.ops.kl_gaussian_gaussian
import tflib.ops.conv2d
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.ops.embedding

import tflib.cifar
import tflib.cifar_256

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

OUT_DIR = DATASET + '_interpolation3_final_filter_3_mean'

if not os.path.isdir(OUT_DIR):
   os.makedirs(OUT_DIR)
   print "Created directory {}".format(OUT_DIR)

if SETTINGS == 'mnist_256':
    
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # two_level uses Enc1/Dec1 for the bottom level, Enc2/Dec2 for the top level
    # one_level uses EncFull/DecFull for the bottom (and only) level
    MODE = 'one_level'

    # Whether to treat pixel inputs to the model as real-valued (as in the 
    # original PixelCNN) or discrete (gets better likelihoods).
    EMBED_INPUTS = True

    # Turn on/off the bottom-level PixelCNN in Dec1/DecFull
    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 32
    DIM_1        = 16
    DIM_2        = 32
    DIM_3        = 32
    DIM_4        = 64
    LATENT_DIM_2 = 128
    NUM_CLASSES = 10

    ALPHA1_ITERS = 5000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.0
    BETA_ITERS = 1000

    # In Dec2, we break each spatial location into N blocks (analogous to channels
    # in the original PixelCNN) and model each spatial location autoregressively
    # as P(x)=P(x0)*P(x1|x0)*P(x2|x0,x1)... In my experiments values of N > 1
    # actually hurt performance. Unsure why; might be a bug.
    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 2*500,
        'stop_after': 500*500,
        'callback_every': 10*500
    }

    LR = 1e-3

    LR_DECAY_AFTER = TIMES['stop_after']
    LR_DECAY_FACTOR = 1.

    BATCH_SIZE = 100 
    N_CHANNELS = 1
    HEIGHT = 28
    WIDTH = 28

    # These aren't actually used for one-level models but some parts
    # of the code still depend on them being defined.
    LATENT_DIM_1 = 64
    LATENTS1_HEIGHT = 7
    LATENTS1_WIDTH = 7

elif SETTINGS == '32px_small':
    MODE = 'two_level'

    EMBED_INPUTS = True

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 128
    DIM_1        = 64
    DIM_2        = 128
    DIM_3        = 256
    LATENT_DIM_1 = 64
    DIM_PIX_2    = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    ALPHA1_ITERS = 2000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.00
    BETA_ITERS = 1000

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 1000,
        'stop_after': 200000,
        'callback_every': 20000
    }

    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = 1e-1

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 32
    WIDTH = 32

    LATENTS1_HEIGHT = 8
    LATENTS1_WIDTH = 8

elif SETTINGS == '32px_big':

    MODE = 'two_level'

    EMBED_INPUTS = False

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 256
    DIM_1        = 128
    DIM_2        = 256
    DIM_3        = 512
    LATENT_DIM_1 = 128
    DIM_PIX_2    = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    ALPHA1_ITERS = 2000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.00
    BETA_ITERS = 1000

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 1000,
        'stop_after': 300000,
        'callback_every': 20000
    }

    VANILLA = False
    LR = 1e-3

    LR_DECAY_AFTER = 300000
    LR_DECAY_FACTOR = 1e-1

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 32
    WIDTH = 32
    LATENTS1_HEIGHT = 8
    LATENTS1_WIDTH = 8

elif SETTINGS == '64px_small':
    MODE = 'two_level'

    EMBED_INPUTS = True

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 128
    DIM_0        = 64
    DIM_1        = 64
    DIM_2        = 128
    LATENT_DIM_1 = 64
    DIM_PIX_2    = 256
    DIM_3        = 256
    DIM_4        = 512
    LATENT_DIM_2 = 512

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 10000,
        'stop_after': 200000,
        'callback_every': 50000
    }

    VANILLA = False
    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = .1

    ALPHA1_ITERS = 2000
    ALPHA2_ITERS = 10000
    KL_PENALTY = 1.0
    BETA_ITERS = 1000

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 64
    WIDTH = 64
    LATENTS1_WIDTH = 16
    LATENTS1_HEIGHT = 16

elif SETTINGS == '64px_big':
    MODE = 'two_level'

    EMBED_INPUTS = True

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 384
    DIM_0        = 192
    DIM_1        = 256
    DIM_2        = 512
    LATENT_DIM_1 = 64
    DIM_PIX_2    = 512
    DIM_3        = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 10000,
        'stop_after': 400000,
        'callback_every': 50000
    }

    VANILLA = False
    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = .5

    ALPHA1_ITERS = 1000
    ALPHA2_ITERS = 10000
    KL_PENALTY = 1.00
    BETA_ITERS = 500

    BATCH_SIZE = 48
    N_CHANNELS = 3
    HEIGHT = 64
    WIDTH = 64
    LATENTS1_WIDTH = 16
    LATENTS1_HEIGHT = 16

elif SETTINGS=='64px_big_onelevel':

    # two_level uses Enc1/Dec1 for the bottom level, Enc2/Dec2 for the top level
    # one_level uses EncFull/DecFull for the bottom (and only) level
    MODE = 'one_level'

    # Whether to treat pixel inputs to the model as real-valued (as in the 
    # original PixelCNN) or discrete (gets better likelihoods).
    EMBED_INPUTS = True

    # Turn on/off the bottom-level PixelCNN in Dec1/DecFull
    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 384
    DIM_0        = 192
    DIM_1        = 256
    DIM_2        = 512
    DIM_3        = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    ALPHA1_ITERS = 50000
    ALPHA2_ITERS = 50000
    KL_PENALTY = 1.0
    BETA_ITERS = 1000

    # In Dec2, we break each spatial location into N blocks (analogous to channels
    # in the original PixelCNN) and model each spatial location autoregressively
    # as P(x)=P(x0)*P(x1|x0)*P(x2|x0,x1)... In my experiments values of N > 1
    # actually hurt performance. Unsure why; might be a bug.
    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 10000,
        'stop_after': 400000,
        'callback_every': 50000
    }
    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = 0.5

    BATCH_SIZE = 48
    N_CHANNELS = 3
    HEIGHT = 64
    WIDTH = 64

    # These aren't actually used for one-level models but some parts
    # of the code still depend on them being defined.
    LATENT_DIM_1 = 64
    LATENTS1_HEIGHT = 7
    LATENTS1_WIDTH = 7

elif SETTINGS=='32px_cifar':

    from keras.datasets import cifar10
    (x_train_set, y_train_set), (x_test_set, y_test_set) = cifar10.load_data()
   
    x_train_set = x_train_set.transpose(0,3,1,2)
    x_test_set = x_test_set.transpose(0,3,1,2)
    
    seed = 333
    x_train_set, x_dev_set, y_train_set, y_dev_set = train_test_split(x_train_set, y_train_set, test_size=0.1, random_state=seed)

    # two_level uses Enc1/Dec1 for the bottom level, Enc2/Dec2 for the top level
    # one_level uses EncFull/DecFull for the bottom (and only) level
    MODE = 'one_level'

    # Whether to treat pixel inputs to the model as real-valued (as in the 
    # original PixelCNN) or discrete (gets better likelihoods).
    EMBED_INPUTS = True

    # Turn on/off the bottom-level PixelCNN in Dec1/DecFull
    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 192 
    DIM_0        = 96 
    DIM_1        = 128 
    DIM_2        = 256 
    DIM_3        = 256 
    DIM_4        = 256 
    LATENT_DIM_2 = 256 

    ALPHA1_ITERS = 50000
    ALPHA2_ITERS = 50000
    KL_PENALTY = 1.0
    BETA_ITERS = 1000

    # In Dec2, we break each spatial location into N blocks (analogous to channels
    # in the original PixelCNN) and model each spatial location autoregressively
    # as P(x)=P(x0)*P(x1|x0)*P(x2|x0,x1)... In my experiments values of N > 1
    # actually hurt performance. Unsure why; might be a bug.
    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 10000,
        'stop_after': 400000,
        'callback_every': 50000
    }
    
    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = 0.5

    BATCH_SIZE = 50 
    N_CHANNELS = 3
    HEIGHT = 32 
    WIDTH = 32 
   
    NUM_CLASSES = 10

    # These aren't actually used for one-level models but some parts
    # of the code still depend on them being defined.
    LATENT_DIM_1 = 32 
    LATENTS1_HEIGHT = 7
    LATENTS1_WIDTH = 7

if DATASET == 'mnist_256':
    train_data, dev_data, test_data = lib.mnist_256.load(BATCH_SIZE, BATCH_SIZE) 
elif DATASET == 'lsun_32':
    train_data, dev_data = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=True)
elif DATASET == 'lsun_64':
    train_data, dev_data = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=False)
elif DATASET == 'imagenet_64':
    train_data, dev_data = lib.small_imagenet.load(BATCH_SIZE)
elif DATASET == 'cifar10':
    train_data, dev_data, test_data = lib.cifar_256.load(BATCH_SIZE) 

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    bn_is_training = tf.placeholder(tf.bool, shape=None, name='bn_is_training')
    bn_stats_iter = tf.placeholder(tf.int32, shape=None, name='bn_stats_iter')
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    all_images = tf.placeholder(tf.int32, shape=[None, N_CHANNELS, HEIGHT, WIDTH], name='all_images')
    all_latents1 = tf.placeholder(tf.float32, shape=[None, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH], name='all_latents1')

    split_images = tf.split(all_images, len(DEVICES), axis=0)
    split_latents1 = tf.split(all_images, len(DEVICES), axis=0)

    tower_cost = []
    tower_outputs1_sample = []

    for device_index, (device, images, latents1_sample) in enumerate(zip(DEVICES, split_images, split_latents1)):
        with tf.device(device):

            def nonlinearity(x):
                return tf.nn.elu(x)

            def pixcnn_gated_nonlinearity(a, b):
                return tf.sigmoid(a) * tf.tanh(b)

            def SubpixelConv2D(*args, **kwargs):
                kwargs['output_dim'] = 4*kwargs['output_dim']
                output = lib.ops.conv2d.Conv2D(*args, **kwargs)
                output = tf.transpose(output, [0,2,3,1])
                output = tf.depth_to_space(output, 2)
                output = tf.transpose(output, [0,3,1,2])
                return output

            def ResidualBlock(name, input_dim, output_dim, inputs, filter_size, mask_type=None, resample=None, he_init=True):
                """
                resample: None, 'down', or 'up'
                """
                if mask_type != None and resample != None:
                    raise Exception('Unsupported configuration')

                if resample=='down':
                    conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
                    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
                    conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
                elif resample=='up':
                    conv_shortcut = SubpixelConv2D
                    conv_1        = functools.partial(SubpixelConv2D, input_dim=input_dim, output_dim=output_dim)
                    conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
                elif resample==None:
                    conv_shortcut = lib.ops.conv2d.Conv2D
                    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
                    conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
                else:
                    raise Exception('invalid resample value')

                if output_dim==input_dim and resample==None:
                    shortcut = inputs # Identity skip-connection
                else:
                    shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, mask_type=mask_type, he_init=False, biases=True, inputs=inputs)

                output = inputs
                if mask_type == None:
                    output = nonlinearity(output)
                    output = conv_1(name+'.Conv1', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init, weightnorm=False)
                    output = nonlinearity(output)
                    output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init, weightnorm=False, biases=False)
                    if device_index == 0:
                        output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2,3], output, bn_is_training, bn_stats_iter)
                    else:
                        output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2,3], output, bn_is_training, bn_stats_iter, update_moving_stats=False)
                else:
                    output = nonlinearity(output)
                    output_a = conv_1(name+'.Conv1A', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
                    output_b = conv_1(name+'.Conv1B', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
                    output = pixcnn_gated_nonlinearity(output_a, output_b)
                    output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)

                return shortcut + output

            def Enc1(images):
                output = images

                if WIDTH == 64:
                    if EMBED_INPUTS:
                        output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_0, filter_size=1, inputs=output, he_init=False)
                        output = ResidualBlock('Enc1.InputRes0', input_dim=DIM_0, output_dim=DIM_0, filter_size=3, resample=None, inputs=output)
                        output = ResidualBlock('Enc1.InputRes', input_dim=DIM_0, output_dim=DIM_1, filter_size=3, resample='down', inputs=output)
                    else:
                        output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)
                        output = ResidualBlock('Enc1.InputRes', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample='down', inputs=output)
                else:
                    if EMBED_INPUTS:
                        output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)
                    else:
                        output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)


                output = ResidualBlock('Enc1.Res1Pre', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('Enc1.Res1Pre2', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('Enc1.Res1', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, resample='down', inputs=output)
                if LATENTS1_WIDTH == 16:
                    output = ResidualBlock('Enc1.Res4Pre', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res4', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res4Post', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    mu_and_sigma = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=DIM_2, output_dim=2*LATENT_DIM_1, filter_size=1, inputs=output, he_init=False)
                else:
                    output = ResidualBlock('Enc1.Res2Pre', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res2Pre2', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res2', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', inputs=output)
                    output = ResidualBlock('Enc1.Res3Pre', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res3Pre2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res3Pre3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    mu_and_sigma = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=DIM_3, output_dim=2*LATENT_DIM_1, filter_size=1, inputs=output, he_init=False)

                return mu_and_sigma, output

            def Dec1(latents, images):
                output = tf.clip_by_value(latents, -50., 50.)

                if LATENTS1_WIDTH == 16:
                    output = lib.ops.conv2d.Conv2D('Dec1.Input', input_dim=LATENT_DIM_1, output_dim=DIM_2, filter_size=1, inputs=output, he_init=False)
                    output = ResidualBlock('Dec1.Res1A', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res1B', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res1C', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                else:
                    output = lib.ops.conv2d.Conv2D('Dec1.Input', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=1, inputs=output, he_init=False)
                    output = ResidualBlock('Dec1.Res1', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res1Post', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res1Post2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res2', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', inputs=output)
                    output = ResidualBlock('Dec1.Res2Post', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res2Post2', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)

                output = ResidualBlock('Dec1.Res3', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, resample='up', inputs=output)
                output = ResidualBlock('Dec1.Res3Post', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('Dec1.Res3Post2', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)

                if WIDTH == 64:
                    output = ResidualBlock('Dec1.Res4', input_dim=DIM_1, output_dim=DIM_0, filter_size=3, resample='up', inputs=output)
                    output = ResidualBlock('Dec1.Res4Post', input_dim=DIM_0, output_dim=DIM_0, filter_size=3, resample=None, inputs=output)

                if PIXEL_LEVEL_PIXCNN:

                    if WIDTH == 64:
                        if EMBED_INPUTS:
                            masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_0, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                        else:
                            masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS, output_dim=DIM_0, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                    else:
                        if EMBED_INPUTS:
                            masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                        else:
                            masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

                    # Make the variance of output and masked_images (roughly) match
                    output /= 2

                    # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
                    output = tf.concat([masked_images, output], axis=1)

                    if WIDTH == 64:
                        output = ResidualBlock('Dec1.Pix2Res', input_dim=2*DIM_0, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                        output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                        output = ResidualBlock('Dec1.Pix4Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    else:
                        output = ResidualBlock('Dec1.Pix2Res', input_dim=2*DIM_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                        output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)

                    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)

                else:

                    if WIDTH == 64:
                        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_0, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)
                    else:
                        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_1, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)

                return tf.transpose(
                    tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
                    [0,2,3,4,1]
                )

            def Enc2(h1):
                output = h1

                if LATENTS1_WIDTH == 16:
                    output = ResidualBlock('Enc2.Res0', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Enc2.Res1Pre', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Enc2.Res1Pre2', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Enc2.Res1', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', he_init=True, inputs=output)

                output = ResidualBlock('Enc2.Res2Pre', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2Pre2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2Pre3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res1A', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, resample='down', he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2PreA', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2Post', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)

                output = tf.reshape(output, [-1, 4*4*DIM_4])
                output = lib.ops.linear.Linear('Enc2.Output', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM_2, inputs=output)

                return output

            def Dec2(latents, targets):
                output = tf.clip_by_value(latents, -50., 50.)
                output = lib.ops.linear.Linear('Dec2.Input', input_dim=LATENT_DIM_2, output_dim=4*4*DIM_4, inputs=output)

                output = tf.reshape(output, [-1, DIM_4, 4, 4])

                output = ResidualBlock('Dec2.Res1Pre', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res1Post', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, resample='up', he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res3Post', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res3Post2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res3Post3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)

                if LATENTS1_WIDTH == 16:
                    output = ResidualBlock('Dec2.Res3Post5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Res3Post6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Res3Post7', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Res3Post8', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)

                if HIGHER_LEVEL_PIXCNN:

                    if LATENTS1_WIDTH == 16:
                        masked_targets = lib.ops.conv2d.Conv2D('Dec2.Pix1', input_dim=LATENT_DIM_1, output_dim=DIM_2, filter_size=5, mask_type=('a', PIX_2_N_BLOCKS), he_init=False, inputs=targets)
                    else:
                        masked_targets = lib.ops.conv2d.Conv2D('Dec2.Pix1', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=5, mask_type=('a', PIX_2_N_BLOCKS), he_init=False, inputs=targets)

                    # Make the variance of output and masked_targets roughly match
                    output /= 2

                    output = tf.concat([masked_targets, output], axis=1)

                    if LATENTS1_WIDTH == 16:
                        output = ResidualBlock('Dec2.Pix2Res', input_dim=2*DIM_2, output_dim=DIM_PIX_2, filter_size=3, mask_type=('b', PIX_2_N_BLOCKS), he_init=True, inputs=output)
                    else:
                        output = ResidualBlock('Dec2.Pix2Res', input_dim=2*DIM_3, output_dim=DIM_PIX_2, filter_size=3, mask_type=('b', PIX_2_N_BLOCKS), he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Pix3Res', input_dim=DIM_PIX_2, output_dim=DIM_PIX_2, filter_size=3, mask_type=('b', PIX_2_N_BLOCKS), he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Pix4Res', input_dim=DIM_PIX_2, output_dim=DIM_PIX_2, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=True, inputs=output)

                    output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_PIX_2, output_dim=2*LATENT_DIM_1, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=False, inputs=output)

                else:

                    if LATENTS1_WIDTH == 16:
                        output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_2, output_dim=2*LATENT_DIM_1, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=False, inputs=output)
                    else:
                        output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_3, output_dim=2*LATENT_DIM_1, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=False, inputs=output)

                return output

            # Only for 32px_cifar, 64px_big_onelevel, and MNIST. Needs modification for others.
            def EncFull(images):
                output = images

                if WIDTH == 32: #64 
                    if EMBED_INPUTS:
                        output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_0, filter_size=1, inputs=output, he_init=False)
                    else:
                        output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS, output_dim=DIM_0, filter_size=1, inputs=output, he_init=False)

                    output = ResidualBlock('EncFull.Res1', input_dim=DIM_0, output_dim=DIM_0, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res2', input_dim=DIM_0, output_dim=DIM_1, filter_size=3, resample='down', inputs=output)
                    output = ResidualBlock('EncFull.Res3', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res4', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res5', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, resample='down', inputs=output)
                    output = ResidualBlock('EncFull.Res6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res7', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res8', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', inputs=output)
                    output = ResidualBlock('EncFull.Res9', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res10', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res11', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, resample='down', inputs=output)
                    output = ResidualBlock('EncFull.Res12', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res13', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, inputs=output)
                    output = tf.reshape(output, [-1, 2*2*DIM_4])
                    output = lib.ops.linear.Linear('EncFull.Output', input_dim=2*2*DIM_4, output_dim=2*LATENT_DIM_2, initialization='glorot', inputs=output)
                else:
                    if EMBED_INPUTS:
                        output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)
                    else:
                        output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)

                    output = ResidualBlock('EncFull.Res1', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res2', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, resample='down', inputs=output)
                    output = ResidualBlock('EncFull.Res3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', inputs=output)
                    output = ResidualBlock('EncFull.Res5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('EncFull.Res6', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = tf.reduce_mean(output, reduction_indices=[2,3])
                    output = lib.ops.linear.Linear('EncFull.Output', input_dim=DIM_3, output_dim=2*LATENT_DIM_2, initialization='glorot', inputs=output)

                return output

            # Only for 32px_CIFAR, 64px_big_onelevel and MNIST. Needs modification for others.
            def DecFull(latents, images):
                output = tf.clip_by_value(latents, -50., 50.)

                if WIDTH == 32: 
                    output = lib.ops.linear.Linear('DecFull.Input', input_dim=LATENT_DIM_2, output_dim=2*2*DIM_4, initialization='glorot', inputs=output)
                    output = tf.reshape(output, [-1, DIM_4, 2, 2])
                    output = ResidualBlock('DecFull.Res2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res3', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res4', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, resample='up', he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res6', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res7', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res8', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res9', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res10', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, resample='up', he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res11', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res12', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res13', input_dim=DIM_1, output_dim=DIM_0, filter_size=3, resample='up', he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res14', input_dim=DIM_0, output_dim=DIM_0, filter_size=3, resample=None, he_init=True, inputs=output)
                else:
                    output = lib.ops.linear.Linear('DecFull.Input', input_dim=LATENT_DIM_2, output_dim=DIM_3, initialization='glorot', inputs=output)
                    output = tf.reshape(tf.tile(tf.reshape(output, [-1, DIM_3, 1]), [1, 1, 49]), [-1, DIM_3, 7, 7])
                    output = ResidualBlock('DecFull.Res2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res4', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res5', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res6', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, resample='up', he_init=True, inputs=output)
                    output = ResidualBlock('DecFull.Res7', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, he_init=True, inputs=output)

                if WIDTH == 32: #64:
                    dim = DIM_0
                else:
                    dim = DIM_1

                if PIXEL_LEVEL_PIXCNN:

                    if EMBED_INPUTS:
                        masked_images = lib.ops.conv2d.Conv2D('DecFull.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=dim, filter_size=3, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                    else:
                        masked_images = lib.ops.conv2d.Conv2D('DecFull.Pix1', input_dim=N_CHANNELS, output_dim=dim, filter_size=3, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

                    # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
                    output = tf.concat([masked_images, output], axis=1)

                    output = ResidualBlock('DecFull.Pix2Res', input_dim=2*dim,   output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    output = ResidualBlock('DecFull.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    output = ResidualBlock('DecFull.Pix4Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    if WIDTH != 32:
                        output = ResidualBlock('DecFull.Pix5Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)

                    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)
                  
                else:

                    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=dim, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)

                return tf.transpose(
                    tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
                    [0,2,3,4,1]
                )

            def split(mu_and_logsig):
                mu, logsig = tf.split(mu_and_logsig, 2, axis=1)
                sig = 0.5 * (tf.nn.softsign(logsig)+1)
                logsig = tf.log(sig)
                return mu, logsig, sig
         
            def clamp_logsig_and_sig(logsig, sig):
                # Early during training (see BETA_ITERS), stop sigma from going too low
                floor = 1. - tf.minimum(1., tf.cast(total_iters, 'float32') / BETA_ITERS)
                log_floor = tf.log(floor)
                return tf.maximum(logsig, log_floor), tf.maximum(sig, floor)


            scaled_images = (tf.cast(images, 'float32') - 128.) / 64.
            if EMBED_INPUTS:
                embedded_images = lib.ops.embedding.Embedding('Embedding', 256, DIM_EMBED, images)
                embedded_images = tf.transpose(embedded_images, [0,4,1,2,3])
                embedded_images = tf.reshape(embedded_images, [-1, DIM_EMBED*N_CHANNELS, HEIGHT, WIDTH])

            if MODE == 'one_level':

                # Layer 1

                if EMBED_INPUTS:
                    mu_and_logsig1 = EncFull(embedded_images)
                else:
                    mu_and_logsig1 = EncFull(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                eps = tf.random_normal(tf.shape(mu1))
                #latents1 = mu1 + (eps * sig1)
                latents1 = mu1 

                if EMBED_INPUTS:
                    outputs1 = DecFull(latents1, embedded_images)
                else:
                    outputs1 = DecFull(latents1, scaled_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.reshape(outputs1, [-1, 256]),
                        labels=tf.reshape(images, [-1])
                    )
                )

                # Assembly

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA1_ITERS) * KL_PENALTY

                kl_cost_1 = tf.reduce_mean(
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu1, 
                        logsig1,
                        sig1
                    )
                )

                kl_cost_1 *= float(LATENT_DIM_2) / (N_CHANNELS * WIDTH * HEIGHT)

                cost = reconst_cost + (alpha * kl_cost_1)
                  
            elif MODE == 'two_level':
                # Layer 1

                if EMBED_INPUTS:
                    mu_and_logsig1, h1 = Enc1(embedded_images)
                else:
                    mu_and_logsig1, h1 = Enc1(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                if mu1.get_shape().as_list()[2] != LATENTS1_HEIGHT:
                    raise Exception("LATENTS1_HEIGHT doesn't match mu1 shape!")
                if mu1.get_shape().as_list()[3] != LATENTS1_WIDTH:
                    raise Exception("LATENTS1_WIDTH doesn't match mu1 shape!")

                eps = tf.random_normal(tf.shape(mu1))
                latents1 = mu1 + (eps * sig1)

                if EMBED_INPUTS:
                    outputs1 = Dec1(latents1, embedded_images)
                    outputs1_sample = Dec1(latents1_sample, embedded_images)
                else:
                    outputs1 = Dec1(latents1, scaled_images)
                    outputs1_sample = Dec1(latents1_sample, scaled_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.reshape(outputs1, [-1, 256]),
                        labels=tf.reshape(images, [-1])
                    )
                )

                # Layer 2

                mu_and_logsig2 = Enc2(h1)
                mu2, logsig2, sig2 = split(mu_and_logsig2)

                eps = tf.random_normal(tf.shape(mu2))
                latents2 = mu2 + (eps * sig2)

                outputs2 = Dec2(latents2, latents1)

                mu1_prior, logsig1_prior, sig1_prior = split(outputs2)
                logsig1_prior, sig1_prior = clamp_logsig_and_sig(logsig1_prior, sig1_prior)
                mu1_prior = 2. * tf.nn.softsign(mu1_prior / 2.)

                # Assembly

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha1 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA1_ITERS) * KL_PENALTY
                alpha2 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA2_ITERS) * alpha1# * KL_PENALTY

                kl_cost_1 = tf.reduce_mean(
                    lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
                        mu1, 
                        logsig1,
                        sig1,
                        mu1_prior,
                        logsig1_prior,
                        sig1_prior
                    )
                )

                kl_cost_2 = tf.reduce_mean(
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu2, 
                        logsig2,
                        sig2
                    )
                )

                kl_cost_1 *= float(LATENT_DIM_1 * LATENTS1_WIDTH * LATENTS1_HEIGHT) / (N_CHANNELS * WIDTH * HEIGHT)
                kl_cost_2 *= float(LATENT_DIM_2) / (N_CHANNELS * WIDTH * HEIGHT)

                cost = reconst_cost + (alpha1 * kl_cost_1) + (alpha2 * kl_cost_2)
                  
            tower_cost.append(cost)
            if MODE == 'two_level':
                tower_outputs1_sample.append(outputs1_sample)

    full_cost = tf.reduce_mean(
        tf.concat([tf.expand_dims(x, 0) for x in tower_cost], axis=0), 0
    )
   
    if MODE == 'two_level':
        full_outputs1_sample = tf.concat(tower_outputs1_sample, axis=0)    
        
    # Sampling

    if MODE == 'one_level':

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits = tf.reshape(tf.slice(outputs1, tf.stack([0, ch_sym, y_sym, x_sym, 0]), tf.stack([-1, 1, 1, 1, -1])), [-1, 256])
        dec1_fn_out = tf.multinomial(logits, 1)[:, 0]
          
        def dec1_fn(_latents, _targets, _ch, _y, _x):
            return session.run(dec1_fn_out, feed_dict={latents1: _latents, images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})
        def enc_fn(_images):
            return session.run(latents1, feed_dict={images: _images, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})

        sample_fn_latents1 = np.random.normal(size=(1, LATENT_DIM_2)).astype('float32')
        
        def generate_and_save_samples(tag):
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
                
            numsamples = 25
            pvals = np.linspace(0.2, 0.8, num=4)
            #pvals = np.linspace(0.2, 0.8, num=1)
            p_set = np.zeros(1)
            
            x_train_set_array = np.array(x_train_set)
            y_train_set_array = np.array(y_train_set)  

            for imagenum in range(numsamples):
                for class1 in range(NUM_CLASSES-1): # goes up to class 8
                  idx1 = np.asarray(np.where(np.equal(class1, y_train_set))[0])
                  x_trainsubset1 = x_train_set_array[idx1,:]
                  y_trainsubset1 = y_train_set_array[idx1,:]
                  x_trainsubset1 = x_trainsubset1.reshape(-1, N_CHANNELS, HEIGHT, WIDTH) 
                  y_trainsubset1 = y_trainsubset1.reshape(-1, 1)
                  
                  for class2 in range(class1+1, NUM_CLASSES):
                    idx2 = np.asarray(np.where(np.equal(class2, y_train_set))[0])
                    x_trainsubset2 = x_train_set_array[idx2,:]
                    y_trainsubset2 = y_train_set_array[idx2,:]
                    x_trainsubset2 = x_trainsubset2.reshape(-1, N_CHANNELS, HEIGHT, WIDTH) 
                    y_trainsubset2 = y_trainsubset2.reshape(-1, 1)
                    
                    imageindex1 = random.sample(range(x_trainsubset1.shape[0]),1)
                    imageindex2 = random.sample(range(x_trainsubset2.shape[0]),1)
                    
                    # Draw the corresponding images and labels from the training data
                    image1 = x_trainsubset1[imageindex1,:]
                    image2 = x_trainsubset2[imageindex2,:]  
                    label1 = y_trainsubset1[imageindex1,:]
                    label2 = y_trainsubset2[imageindex2,:]
                
                    # Reshape
                    image1 = image1.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
                    image2 = image2.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
                    label1 = label1.reshape(1, 1)
                    label2 = label2.reshape(1, 1)
                    
                    # Save the original images
                    print "Saving original samples"
                    color_grid_vis(
                        image1, 
                        1, 
                        1, 
                        'original_1_classes{}and{}_num{}.png'.format(class1,class2,imagenum)
                    )
                    color_grid_vis(
                        image2, 
                        1, 
                        1, 
                        'original_2_classes{}and{}_num{}.png'.format(class1,class2,imagenum)
                    )
                      
                    # Encode the images
                    image_code1 = enc_fn(image1)
                    image_code2 = enc_fn(image2)
               
                    # Change labels to matrix form before performing interpolations
                    label1 = np_utils.to_categorical(label1, NUM_CLASSES) 
                    label2 = np_utils.to_categorical(label2, NUM_CLASSES) 
                    
                    # Define slert function
                    #def slerp(val, low, high):
                    #  omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
                    #  so = np.sin(omega)
                    #  if so == 0:
                    #    return (1.0-val) * low + val * high # L'Hopital's rule/LERP
                    #return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
                    # From https://github.com/soumith/dcgan.torch/issues/14
               
                    # Find angle between the two latent codes
                    vec1 = image_code1/np.linalg.norm(image_code1)
                    vec2 = image_code2/np.linalg.norm(image_code2)
                    vec2 = np.transpose(vec2)
                    omega = np.arccos(np.clip(np.dot(vec1, vec2), -1, 1))
                    so = np.sin(omega) 
                     
                    # Combine the latent codes
                    for p in pvals:
                      if so == 0:
                        new_code = (1.0-p) * image_code1 + p * image_code2
                      else:
                        new_code = np.sin((1.0-p)*omega) / so * image_code1 + np.sin(p*omega) / so * image_code2
                        
                      new_label = np.multiply(p,label1) + np.multiply((1-p),label2)
                     
                      new_label = new_label.reshape(1,1,NUM_CLASSES)

                      samples = np.zeros(
                        (1, N_CHANNELS, HEIGHT, WIDTH), 
                        dtype='int32'
                      )

                      print "Generating samples"
                      for y in xrange(HEIGHT):
                        for x in xrange(WIDTH):
                              for ch in xrange(N_CHANNELS):
                                  next_sample = dec1_fn(new_code, samples, ch, y, x) 
                                  samples[:,ch,y,x] = next_sample
                            
                      x_augmentation_set = np.concatenate((x_augmentation_set, samples), axis=0)
                      y_augmentation_set = np.concatenate((y_augmentation_set, new_label), axis=0)
                      p_set = np.append(p_set,p)
                
                      color_grid_vis(
                        samples, 
                        1, 
                        1, 
                        'interpolation1_classes{}and{}_pval{}_num{}.png'.format(class1,class2,p,imagenum)
                      )
                    
                # Sample two unique image indices from different classes
                #classindices = random.sample(range(0,NUM_CLASSES),2)
                #idx1 = np.where(np.equal(classindices[0],y_train_set))
                #idx2 = np.where(np.equal(classindices[1],y_train_set))
                
                #x_train_set_array = np.array(x_train_set)
                #y_train_set_array = np.array(y_train_set)
                
                #x_trainsubset1 = x_train_set_array[idx1,:]
                #x_trainsubset2 = x_train_set_array[idx2,:]
                #y_trainsubset1 = y_train_set_array[idx1,:]
                #y_trainsubset2 = y_train_set_array[idx2,:]
                
                #x_trainsubset1 = x_trainsubset1.reshape(-1, N_CHANNELS, HEIGHT, WIDTH) 
                #x_trainsubset2 = x_trainsubset2.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
                #y_trainsubset1 = y_trainsubset1.reshape(-1, 1)
                #y_trainsubset2 = y_trainsubset2.reshape(-1, 1) 
                
                #imageindex1 = random.sample(range(x_trainsubset1.shape[0]),1)
                #imageindex2 = random.sample(range(x_trainsubset2.shape[0]),1)
                
                # Draw the corresponding images and labels from the training data
                #image1 = x_trainsubset1[imageindex1,:]
                #image2 = x_trainsubset2[imageindex2,:]  
                #label1 = y_trainsubset1[imageindex1,:]
                #label2 = y_trainsubset2[imageindex2,:]
                
                # Reshape
                #image1 = image1.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
                #image2 = image2.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
                #label1 = label1.reshape(1, 1)
                #label2 = label2.reshape(1, 1)
  
            x_augmentation_array = np.delete(x_augmentation_set, (0), axis=0)
            y_augmentation_array = np.delete(y_augmentation_set, (0), axis=0)
            p_set = np.delete(p_set, (0), axis=0)
            
            x_augmentation_array = x_augmentation_array.astype(np.uint8)

            np.save(OUT_DIR + '/' + 'x_augmentation_array_mean', x_augmentation_array) 
            np.save(OUT_DIR + '/' + 'y_augmentation_array_mean', y_augmentation_array)   
            np.save(OUT_DIR + '/' + 'p_set_array', p_set)
                
    # Run

    if MODE == 'one_level':
        prints=[
            ('alpha', alpha), 
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1)
        ]

    decayed_lr = tf.train.exponential_decay(
        LR,
        total_iters,
        LR_DECAY_AFTER,
        LR_DECAY_FACTOR,
        staircase=True
    )

    lib.sampling_loop_cifar_filter_3.sampling_loop( 
        session=session,
        inputs=[total_iters, all_images],
        inject_iteration=True,
        bn_vars=(bn_is_training, bn_stats_iter),
        cost=full_cost,
        stop_after=TIMES['stop_after'],
        prints=prints,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
        test_data=dev_data,
        callback=generate_and_save_samples,
        callback_every=TIMES['callback_every'],
        test_every=TIMES['test_every'],
        save_checkpoints=True
    )
