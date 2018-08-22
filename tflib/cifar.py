import numpy

import os
import urllib
import gzip
import cPickle as pickle
import keras
from keras.datasets import cifar10
import sklearn
from sklearn.model_selection import train_test_split

def cifar_generator(images, targets, batch_size, n_labelled): # LEILAEDIT: changed "data" to "images, targets"
    #images, targets = data

    images = images.astype('float32')
    targets = targets.astype('int32')
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 3*32*32)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, n_labelled=None):
    #filepath = '/tmp/mnist.pkl.gz'
    #url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    #if not os.path.isfile(filepath):
    #    print "Couldn't find MNIST dataset in /tmp, downloading..."
    #    urllib.urlretrieve(url, filepath)

    #with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
    #    train_data, dev_data, test_data = pickle.load(f)
    (x_traincifar, y_traincifar), (x_testcifar, y_testcifar) = cifar10.load_data()
    x_traincifar = x_traincifar.transpose(0,3,1,2)
    x_testcifar = x_testcifar.transpose(0,3,1,2)
    x_devcifar = x_testcifar
    y_devcifar = y_testcifar
    
    #seed = 333
    #x_traincifar, x_devcifar, y_traincifar, y_devcifar = train_test_split(x_traincifar, y_traincifar, test_size=0.1, random_state=seed)
    
    return (
        #cifar_generator(train_data, batch_size, n_labelled), 
        #cifar_generator(dev_data, test_batch_size, n_labelled), 
        #cifar_generator(test_data, test_batch_size, n_labelled)
        cifar_generator(x_traincifar, y_traincifar, batch_size, n_labelled),
        cifar_generator(x_devcifar, y_devcifar, batch_size, n_labelled),
        cifar_generator(x_testcifar, y_testcifar, batch_size, n_labelled)
    )
