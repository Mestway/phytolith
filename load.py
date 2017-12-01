from __future__ import absolute_import 
from __future__ import print_function

import numpy as np
from scipy.misc import imresize

import os
import h5py
import parse
import re

from pprint import pprint


def process_label(label_segments):
    """ Given the label of an object, clean it up into subfamily, tribe and genus """

    processed = []

    # subfamily
    r = parse.parse('{}- Subfamily', label_segments[0])
    if r is not None:
        processed.append(r[0].strip().lower()) 
    else:
        raise RuntimeError('unable to get subfamily name from %r' % dirname)

    # tribe
    r = parse.parse('{}- Tribe', label_segments[1])
    if r is not None:
        processed.append(r[0].strip().lower()) 
    else:
        processed.append(label_segments[1].strip().lower()) 

    # genus
    s = label_segments[2].replace('Copy of ', '')
    idx_first_nonletter = re.search('[^a-zA-Z]', s).start()
    processed.append(s[:idx_first_nonletter].lower())

    return processed


def load_data(file_name, img_size) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') 
    item = file 
    data = []
    load_into_tree(item, img_size, collector=data)
    file.close()
    return data
 
def load_into_tree(g, img_size, prefix=[], collector=[]) :
    """Prints the input file/group/dataset (g) name and begin iterations on its content
        Args:
            g: a h5py object
            img_size: a tuple (width, height) representing the target img size
            prefix: the label prefix of the current object
            collector: a list used to collect data
    """
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        
        for key, val in dict(g).iteritems() :
            subg = val
            #print(offset, key) #,"   ", subg.name #, val, subg.len(), type(subg),
            if len(dict(g)) == 1:
                new_prefix = prefix
            else:
                new_prefix = prefix + [key]
            load_into_tree(subg, img_size, new_prefix, collector)

    elif isinstance(g, h5py.Dataset):
        #print('(Dataset)', g.name, '    len =', g.shape) #, g.dtype
        data = g
        if data.shape[0] > 100 and data.shape[1] > 100:
            if len(data.shape) == 3:
                X = np.mean(data, -1)
            else:
                X = data

            # pad them into square images
            h_pad = (0, 0)
            w_pad = (0, 0)

            if X.shape[0] > X.shape[1]:
                delta = X.shape[0] - X.shape[1]
                w_pad = (int(delta / 2), int(delta / 2))
            if X.shape[1] > X.shape[0]:
                delta = X.shape[1] - X.shape[1]
                h_pad = (int(delta / 2), int(delta / 2))

            X = np.pad(X, pad_width=(w_pad, h_pad), mode='constant', constant_values=0)
            X = imresize(X, img_size)

            collector.append((X, process_label(prefix)))
 
if __name__ == "__main__" :
    data = load_data(os.path.join("data", "meandata.hdf5"), img_size=(32,32))
    pprint(data)
    
