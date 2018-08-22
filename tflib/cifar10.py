import sys
sys.modules['theano'] = None

import numpy as np
import pdb

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
# from fuel.transformers.image import RandomFixedSizeCrop
from sklearn.model_selection import train_test_split

PATH = 'cifar10.hdf5'

from scipy.misc import imsave
def color_grid_vis(X, nh, nw, save_path):
    # from github.com/Newmu
    X = X.transpose(0,2,3,1)
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    imsave(save_path, img)


def _make_stream(stream, bs):
    def new_stream():
      result = np.empty((bs, 32, 32, 3), dtype='int32')
      pdb.set_trace()
      for (imb,) in stream.get_epoch_iterator():
        for i, img in enumerate(imb):
          result[i] =  img[:32, :32, :]                
        yield (result.transpose(0,3,1,2),)
    return new_stream

def load(batch_size=128): #LEILAEDIT: took out downsampling, changed validation data to CIFAR test set, changed tr_data in line 55 to val_data
    
    tr_data = H5PYDataset(PATH, which_sets=('train',))
    val_data = H5PYDataset(PATH, which_sets=('test',))

    ntrain = tr_data.num_examples
    nval = val_data.num_examples

    print "ntrain {}, nval {}".format(ntrain, nval)

    tr_scheme = ShuffledScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    # te_scheme = SequentialScheme(examples=ntest, batch_size=batch_size)
    # te_stream = DataStream(te_data, iteration_scheme=te_scheme)

    val_scheme = SequentialScheme(examples=nval, batch_size=batch_size)
    val_stream = DataStream(val_data, iteration_scheme=val_scheme)

    return _make_stream(tr_stream, batch_size), _make_stream(val_stream, batch_size)
