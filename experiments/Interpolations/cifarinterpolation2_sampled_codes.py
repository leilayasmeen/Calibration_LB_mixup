
"""
This file was used to create mixed examples using simple linear interpolation of sampled latent codes between closely related classes, using constant weighting and four mixed examples per parent pair.

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

OUT_DIR = DATASET + '_interpolation2_final_filter_3_sampled'

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
                latents1 = mu1 + (eps * sig1)
                #latents1 = mu1 

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
        
        def generate_and_save_samples(tag):
            from keras.utils import np_utils
            import itertools
            
            x_augmentation_set = np.zeros((1, N_CHANNELS, HEIGHT, WIDTH))
            y_augmentation_set = np.zeros((1, 1, NUM_CLASSES)) 
            
            ##################################################################
            
            (x_train_set, y_train_set), (x_test_set, y_test_set) = cifar10.load_data()
   
            x_train_set = x_train_set.transpose(0,3,1,2)
            x_test_set = x_test_set.transpose(0,3,1,2)
    
            seed = 333
            x_train_set, x_dev_set, y_train_set, y_dev_set = train_test_split(x_train_set, y_train_set, test_size=0.1, random_state=seed)
            
            all_latents = np.zeros((1,LATENT_DIM_2)).astype('float32')
            
            x_train_set_sub = x_train_set
            y_train_set_sub = y_train_set
        
            # Reshape image files
            x_train_set_sub = x_train_set_sub.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
            y_train_set_sub = y_train_set_sub.reshape(-1, 1)
            print "Reshaped loaded images."
         
            # Encode all images
            print "Encoding images"
            for j in range(x_train_set_sub.shape[0]):
               latestlatents = enc_fn(x_train_set_sub[j,:].reshape(1, N_CHANNELS, HEIGHT, WIDTH))
               latestlatents = latestlatents.reshape(-1,LATENT_DIM_2)
               all_latents = np.concatenate((all_latents, latestlatents), axis=0)
        
            all_latents = np.delete(all_latents, (0), axis=0)
         
            # Find means of latent vectors, by class
            print "Finding class means"
            classmeans = np.zeros((NUM_CLASSES, LATENT_DIM_2)).astype('float32')
            for k in range(NUM_CLASSES):
               idk = np.asarray(np.where(np.equal(y_train_set_sub,k))[0])
               all_latents_groupk = all_latents[idk,:]
               classmeans[k,:] = np.mean(all_latents_groupk, axis=0)
      
            # Find the two pairs of classes that are closest to each other
            # Find all pairs
            print "Finding pairs"
            pairs = np.array(list(itertools.combinations(range(NUM_CLASSES),2)))
            num_pairs = pairs.shape[0]
         
            # Find distances between the members of each pair
            meandist = np.zeros((num_pairs)).astype('float64')
            classarray = np.arange(NUM_CLASSES)
            for m in range(num_pairs):
                  aidx = np.asarray(np.where(np.equal(classarray,pairs[m,0])))
                  a = np.asarray(classmeans[aidx,:])
                  #print "first pair is"
                  #print pairs[m,:]
                  #print "first mean is "
                  #print a
                  bidx = np.asarray(np.where(np.equal(classarray,pairs[m,1])))
                  b = np.asarray(classmeans[bidx,:])
                  #print "second mean is"
                  #print b
                  #print b.shape
                  #a = np.delete(a, -1, axis=1)
                  #b = np.delete(b, -1, axis=1)
                  a = a.reshape(1, LATENT_DIM_2)
                  b = b.reshape(1, LATENT_DIM_2)
                  #c = np.subtract(a,b)
                  #print c
                  meandist[m] = np.linalg.norm(a-b)
                  #meandist[m] = np.sqrt(np.dot(c, np.transpose(c)))
            #print "mean distances are"
            #print meandist
            
            # Sort distances between pairs and find the five smallest
            sorteddistances = np.sort(meandist)
            closestdistance = sorteddistances[0]
            secondclosestdistance = sorteddistances[1]
            thirdclosestdistance = sorteddistances[2]
            #fourthclosestdistance = sorteddistances[3]
            #fifthclosestdistance = sorteddistances[4]
            #print "closest distances"
            #print closestdistance
            #print secondclosestdistance
            #print thirdclosestdistance
      
            # Draw out the pairs corresponding to these distances
            closestidx = np.asarray(np.where(np.equal(meandist, closestdistance))[0])
            secondclosestidx = np.asarray(np.where(np.equal(meandist, secondclosestdistance))[0])
            thirdclosestidx = np.asarray(np.where(np.equal(meandist, thirdclosestdistance))[0])
            #fourthclosestidx = np.asarray(np.where(np.equal(meandist, fourthclosestdistance))[0])
            #fifthclosestidx = np.asarray(np.where(np.equal(meandist, fifthclosestdistance))[0])
            #print "closest ids"
            #print closestidx
            #print secondclosestidx
            #print thirdclosestidx
            #print "now for the pairs themselves"
            closestpair = pairs[closestidx,:]
            secondclosestpair = pairs[secondclosestidx,:]
            thirdclosestpair = pairs[thirdclosestidx,:]
            #fourthclosestpair = pairs[fourthclosestidx,:]
            #fifthclosestpair = pairs[fifthclosestidx,:]
            #print closestpair
            #print secondclosestpair
            #print thirdclosestpair
         
            #classpairs = np.concatenate((closestpair, secondclosestpair, thirdclosestpair, fourthclosestpair, fifthclosestpair), axis=0)
            classpairs = np.concatenate((closestpair, secondclosestpair, thirdclosestpair), axis=0)
            ##################################################################
            
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
                
            numsamples = 1500
            pvals = np.linspace(0.2, 0.8, num=4)
            #pvals = np.linspace(0.2, 0.8, num=1)
            
            x_train_set_array = np.array(x_train_set)
            y_train_set_array = np.array(y_train_set)  

            for imagenum in range(numsamples):
              # Sample unique image indices from class pairs. Images will be interpolated in pairs. Pairs are listed in order.
              classindices = classpairs
 
              idx1 = np.asarray(np.where(np.equal(classindices[0,0],y_train_set))[0])
              idx2 = np.asarray(np.where(np.equal(classindices[0,1],y_train_set))[0])
              idx3 = np.asarray(np.where(np.equal(classindices[1,0],y_train_set))[0])
              idx4 = np.asarray(np.where(np.equal(classindices[1,1],y_train_set))[0])
              idx5 = np.asarray(np.where(np.equal(classindices[2,0],y_train_set))[0])
              idx6 = np.asarray(np.where(np.equal(classindices[2,1],y_train_set))[0])
              #idx7 = np.asarray(np.where(np.equal(classindices[3,0],y_train_set))[0])
              #idx8 = np.asarray(np.where(np.equal(classindices[3,1],y_train_set))[0])
              #idx9 = np.asarray(np.where(np.equal(classindices[4,0],y_train_set))[0])
              #idx10 = np.asarray(np.where(np.equal(classindices[4,1],y_train_set))[0])
                
              x_train_array = np.array(x_train_set)
              y_train_array = np.array(y_train_set)
                
              x_trainsubset1 = x_train_array[idx1,:]
              x_trainsubset2 = x_train_array[idx2,:]
              x_trainsubset3 = x_train_array[idx3,:]
              x_trainsubset4 = x_train_array[idx4,:]
              x_trainsubset5 = x_train_array[idx5,:]
              x_trainsubset6 = x_train_array[idx6,:]
              #x_trainsubset7 = x_train_array[idx7,:]
              #x_trainsubset8 = x_train_array[idx8,:]
              #x_trainsubset9 = x_train_array[idx9,:]
              #x_trainsubset10 = x_train_array[idx10,:] 
               
              y_trainsubset1 = y_train_array[idx1,:]
              y_trainsubset2 = y_train_array[idx2,:]
              y_trainsubset3 = y_train_array[idx3,:]
              y_trainsubset4 = y_train_array[idx4,:]
              y_trainsubset5 = y_train_array[idx5,:]
              y_trainsubset6 = y_train_array[idx6,:]
              #y_trainsubset7 = y_train_array[idx7,:]
              #y_trainsubset8 = y_train_array[idx8,:]
              #y_trainsubset9 = y_train_array[idx9,:]
              #y_trainsubset10 = y_train_array[idx10,:]
                
              x_trainsubset1 = x_trainsubset1.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              x_trainsubset2 = x_trainsubset2.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              x_trainsubset3 = x_trainsubset3.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              x_trainsubset4 = x_trainsubset4.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              x_trainsubset5 = x_trainsubset5.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              x_trainsubset6 = x_trainsubset6.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              #x_trainsubset7 = x_trainsubset7.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              #x_trainsubset8 = x_trainsubset8.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              #x_trainsubset9 = x_trainsubset9.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
              #x_trainsubset10 = x_trainsubset10.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)              
               
              y_trainsubset1 = y_trainsubset1.reshape(-1, 1)
              y_trainsubset2 = y_trainsubset2.reshape(-1, 1)
              y_trainsubset3 = y_trainsubset3.reshape(-1, 1)
              y_trainsubset4 = y_trainsubset4.reshape(-1, 1)
              y_trainsubset5 = y_trainsubset5.reshape(-1, 1)
              y_trainsubset6 = y_trainsubset6.reshape(-1, 1)
              #y_trainsubset7 = y_trainsubset7.reshape(-1, 1)
              #y_trainsubset8 = y_trainsubset8.reshape(-1, 1)
              #y_trainsubset9 = y_trainsubset9.reshape(-1, 1)
              #y_trainsubset10 = y_trainsubset10.reshape(-1, 1) 

              imageindex1 = random.sample(range(x_trainsubset1.shape[0]),1)
              imageindex2 = random.sample(range(x_trainsubset2.shape[0]),1)
              imageindex3 = random.sample(range(x_trainsubset3.shape[0]),1)
              imageindex4 = random.sample(range(x_trainsubset4.shape[0]),1)
              imageindex5 = random.sample(range(x_trainsubset5.shape[0]),1)
              imageindex6 = random.sample(range(x_trainsubset6.shape[0]),1)
              #imageindex7 = random.sample(range(x_trainsubset7.shape[0]),1)
              #imageindex8 = random.sample(range(x_trainsubset8.shape[0]),1)
              #imageindex9 = random.sample(range(x_trainsubset9.shape[0]),1)
              #imageindex10 = random.sample(range(x_trainsubset10.shape[0]),1) 

              # Draw the corresponding images and labels from the training data
              image1 = x_trainsubset1[imageindex1,:]
              image2 = x_trainsubset2[imageindex2,:]  
              image3 = x_trainsubset3[imageindex3,:]
              image4 = x_trainsubset4[imageindex4,:]
              image5 = x_trainsubset5[imageindex5,:]
              image6 = x_trainsubset6[imageindex6,:]
              #image7 = x_trainsubset7[imageindex7,:]
              #image8 = x_trainsubset8[imageindex8,:]
              #image9 = x_trainsubset9[imageindex9,:]
              #image10 = x_trainsubset10[imageindex10,:]            
            
              label1 = y_trainsubset1[imageindex1,:]
              label2 = y_trainsubset2[imageindex2,:]
              label3 = y_trainsubset3[imageindex3,:]
              label4 = y_trainsubset4[imageindex4,:]
              label5 = y_trainsubset5[imageindex5,:]
              label6 = y_trainsubset6[imageindex6,:]
              #label7 = y_trainsubset7[imageindex7,:]
              #label8 = y_trainsubset8[imageindex8,:]
              #label9 = y_trainsubset9[imageindex9,:]
              #label10 = y_trainsubset10[imageindex10,:]
            
              # Reshape
              image1 = image1.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              image2 = image2.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              image3 = image3.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              image4 = image4.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              image5 = image5.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              image6 = image6.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              #image7 = image7.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              #image8 = image8.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              #image9 = image9.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
              #image10 = image10.reshape(1, N_CHANNELS, HEIGHT, WIDTH)               
               
              label1 = label1.reshape(1, 1)
              label2 = label2.reshape(1, 1)
              label3 = label3.reshape(1, 1)
              label4 = label4.reshape(1, 1)
              label5 = label5.reshape(1, 1)
              label6 = label6.reshape(1, 1) 
              #label7 = label7.reshape(1, 1)
              #label8 = label8.reshape(1, 1)
              #label9 = label9.reshape(1, 1)
              #label10 = label10.reshape(1, 1)  

              # Save original images
              print "Saving original samples"
              color_grid_vis(
                 image1, 
                 1, 
                 1,
                 'originalclass{}_classes{}and{}_num{}.png'.format(classindices[0,0],classindices[0,1],classindices[0,0],imagenum)
              )
              color_grid_vis(
                 image2,
                 1,
                 1,
                 'originalclass{}_classes{}and{}_num{}.png'.format(classindices[0,0],classindices[0,1],classindices[0,1],imagenum)
              )
              color_grid_vis(
                 image3,
                 1,
                 1,
                 'originalclass{}_classes{}and{}_num{}.png'.format(classindices[1,0],classindices[1,1],classindices[1,0],imagenum)
              ) 
            
              color_grid_vis(
                 image4,
                 1,
                 1,
                 'originalclass{}_classes{}and{}_num{}.png'.format(classindices[1,0],classindices[1,1],classindices[1,1],imagenum)
              ) 
              color_grid_vis(
                 image5,
                 1,
                 1,
                 'originalclass{}_classes{}and{}_num{}.png'.format(classindices[2,0],classindices[2,1],classindices[2,0],imagenum)
              ) 
            
              color_grid_vis(
                 image6,
                 1,
                 1,
                 'originalclass{}_classes{}and{}_num{}.png'.format(classindices[2,0],classindices[2,1],classindices[2,1],imagenum)
              )   
              #color_grid_vis(
              #   image7,
              #   1,
              #   1,
              #   'originalclass{}_classes{}and{}_num{}.png'.format(classindices[3,0],classindices[3,1],classindices[3,0],imagenum)
              #) 
            
              #color_grid_vis(
              #   image8,
              #   1,
              #   1,
              #   'originalclass{}_classes{}and{}_num{}.png'.format(classindices[3,0],classindices[3,1],classindices[3,1],imagenum)
              #) 
              #color_grid_vis(
              #   image9,
              #   1,
              #   1,
              #   'originalclass{}_classes{}and{}_num{}.png'.format(classindices[4,0],classindices[4,1],classindices[4,0],imagenum)
              #) 
            
              #color_grid_vis(
              #   image10,
              #   1,
              #   1,
              #   'originalclass{}_classes{}and{}_num{}.png'.format(classindices[4,0],classindices[4,1],classindices[4,1],imagenum)
              #)               
               
              # Encode the images
              image_code1 = enc_fn(image1)
              image_code2 = enc_fn(image2)
              image_code3 = enc_fn(image3)
              image_code4 = enc_fn(image4)
              image_code5 = enc_fn(image5)
              image_code6 = enc_fn(image6)
              #image_code7 = enc_fn(image7)
              #image_code8 = enc_fn(image8)
              #image_code9 = enc_fn(image9)
              #image_code10 = enc_fn(image10)           
                  
              # Change the labels to matrix form before performing interpolations
              label1 = np_utils.to_categorical(label1, NUM_CLASSES) 
              label2 = np_utils.to_categorical(label2, NUM_CLASSES)
              label3 = np_utils.to_categorical(label3, NUM_CLASSES) 
              label4 = np_utils.to_categorical(label4, NUM_CLASSES)
              label5 = np_utils.to_categorical(label5, NUM_CLASSES) 
              label6 = np_utils.to_categorical(label6, NUM_CLASSES)
              #label7 = np_utils.to_categorical(label7, NUM_CLASSES) 
              #label8 = np_utils.to_categorical(label8, NUM_CLASSES)
              #label9 = np_utils.to_categorical(label9, NUM_CLASSES) 
              #label10 = np_utils.to_categorical(label10, NUM_CLASSES)
            
              # Combine the latent codes using p
              for p in pvals:
                  new_code12 = np.multiply(p,image_code1) + np.multiply((1-p),image_code2)
                  new_label12 = np.multiply(p,label1) + np.multiply((1-p),label2)
                  new_code34 = np.multiply(p,image_code3) + np.multiply((1-p),image_code4)
                  new_label34 = np.multiply(p,label3) + np.multiply((1-p),label4)
                  new_code56 = np.multiply(p,image_code5) + np.multiply((1-p),image_code6)
                  new_label56 = np.multiply(p,label5) + np.multiply((1-p),label6)
                  #new_code78 = np.multiply(p,image_code7) + np.multiply((1-p),image_code8)
                  #new_label78 = np.multiply(p,label7) + np.multiply((1-p),label8)
                  #new_code910 = np.multiply(p,image_code9) + np.multiply((1-p),image_code10)
                  #new_label910 = np.multiply(p,label9) + np.multiply((1-p),label10) 

                  # Reshape the new labels to enable saving in the proper format for the neural networks later on
                  new_label12 = new_label12.reshape(1,1,NUM_CLASSES)
                  new_label34 = new_label34.reshape(1,1,NUM_CLASSES)
                  new_label56 = new_label56.reshape(1,1,NUM_CLASSES)
                  #new_label78 = new_label78.reshape(1,1,NUM_CLASSES)
                  #new_label910 = new_label910.reshape(1,1,NUM_CLASSES)
                  
                  samples12 = np.zeros(
                     (1, N_CHANNELS, HEIGHT, WIDTH), 
                     dtype='int32'
                  )
                
                  samples34 = np.zeros(
                     (1, N_CHANNELS, HEIGHT, WIDTH), 
                     dtype='int32'
                  )
                  
                  samples56 = np.zeros(
                     (1, N_CHANNELS, HEIGHT, WIDTH), 
                     dtype='int32'
                  )
                  
                  #samples78 = np.zeros(
                  #   (1, N_CHANNELS, HEIGHT, WIDTH), 
                  #   dtype='int32'
                  #)
                  
                  #samples910 = np.zeros(
                  #   (1, N_CHANNELS, HEIGHT, WIDTH), 
                  #   dtype='int32'
                  #)
                  
                  print "Generating samples"
                  for y in xrange(HEIGHT):
                     for x in xrange(WIDTH):
                        for ch in xrange(N_CHANNELS):
                           next_sample12 = dec1_fn(new_code12, samples12, ch, y, x) 
                           samples12[:,ch,y,x] = next_sample12
                            
                  for y in xrange(HEIGHT):
                     for x in xrange(WIDTH):
                        for ch in xrange(N_CHANNELS):
                           next_sample34 = dec1_fn(new_code34, samples34, ch, y, x) 
                           samples34[:,ch,y,x] = next_sample34
                           
                  for y in xrange(HEIGHT):
                     for x in xrange(WIDTH):
                        for ch in xrange(N_CHANNELS):
                           next_sample56 = dec1_fn(new_code56, samples56, ch, y, x) 
                           samples56[:,ch,y,x] = next_sample56
  
                  #for y in xrange(HEIGHT):
                  #   for x in xrange(WIDTH):
                  #      for ch in xrange(N_CHANNELS):
                  #         next_sample78 = dec1_fn(new_code78, samples78, ch, y, x) 
                  #         samples78[:,ch,y,x] = next_sample78
                           
                  #for y in xrange(HEIGHT):
                  #   for x in xrange(WIDTH):
                  #      for ch in xrange(N_CHANNELS):
                  #         next_sample910 = dec1_fn(new_code910, samples910, ch, y, x) 
                  #         samples910[:,ch,y,x] = next_sample910
                           
                  x_augmentation_set = np.concatenate((x_augmentation_set, samples12), axis=0)
                  x_augmentation_set = np.concatenate((x_augmentation_set, samples34), axis=0)
                  x_augmentation_set = np.concatenate((x_augmentation_set, samples56), axis=0)
                  #x_augmentation_set = np.concatenate((x_augmentation_set, samples78), axis=0)
                  #x_augmentation_set = np.concatenate((x_augmentation_set, samples910), axis=0)
                  
                  y_augmentation_set = np.concatenate((y_augmentation_set, new_label12), axis=0)
                  y_augmentation_set = np.concatenate((y_augmentation_set, new_label34), axis=0)
                  y_augmentation_set = np.concatenate((y_augmentation_set, new_label56), axis=0)
                  #y_augmentation_set = np.concatenate((y_augmentation_set, new_label78), axis=0)
                  #y_augmentation_set = np.concatenate((y_augmentation_set, new_label910), axis=0)
                
                  print "Saving samples and their corresponding tags"
                  color_grid_vis(
                     samples12, 
                     1, 
                     1, 
                     'interpolation2_classes{}and{}_pval{}_num{}.png'.format(classindices[0,0],classindices[0,1],p,imagenum)
                  )
                  color_grid_vis(
                     samples34, 
                     1, 
                     1, 
                     'interpolation2_classes{}and{}_pval{}_num{}.png'.format(classindices[1,0],classindices[1,1],p,imagenum)
                  )
                  color_grid_vis(
                     samples56, 
                     1, 
                     1, 
                     'interpolation2_classes{}and{}_pval{}_num{}.png'.format(classindices[2,0],classindices[2,1],p,imagenum)
                  ) 
                  #color_grid_vis(
                  #   samples78, 
                  #   1, 
                  #   1, 
                  #   'interpolation2_classes{}and{}_pval{}_num{}.png'.format(classindices[3,0],classindices[3,1],p,imagenum)
                  #)
                  #color_grid_vis(
                  #   samples910, 
                  #   1, 
                  #   1, 
                  #   'interpolation2_classes{}and{}_pval{}_num{}.png'.format(classindices[4,0],classindices[4,1],p,imagenum)
                  #)  

            x_augmentation_array = np.delete(x_augmentation_set, (0), axis=0)
            y_augmentation_array = np.delete(y_augmentation_set, (0), axis=0)
            
            x_augmentation_array = x_augmentation_array.astype(np.uint8)

            np.save(OUT_DIR + '/' + 'x_augmentation_array_sampled', x_augmentation_array) 
            np.save(OUT_DIR + '/' + 'y_augmentation_array_sampled', y_augmentation_array)   
                
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
