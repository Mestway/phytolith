from __future__ import absolute_import 
from __future__ import print_function

import numpy as np
from scipy.misc import imresize

import os
import h5py
import parse
import re

from pprint import pprint
import matplotlib.pyplot as plt


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

def load_data(file_name, img_size=None, to_XY=False) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') 
    item = file 
    data = []
    load_into_tree(item, img_size, prefix=[], collector=data)
    file.close()

    train_data = []
    dev_data = []
    test_data = []

    def split_fn(l):
        return l[:int(0.2 * len(l))], l[int(0.2 * len(l)):int(0.3 * len(l))], l[int(0.3 * len(l)):]

    class_dict = {}
    for d in data:
        label = "_".join(d[1])
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(d)

    # split evenly for each classes
    np.random.seed(789)
    for k in class_dict:
        indices = np.random.permutation(len(class_dict[k]))
        l = [class_dict[k][i] for i in indices]
        test_l, dev_l, train_l = split_fn(l)
        
        test_data.extend(test_l)
        dev_data.extend(dev_l)
        train_data.extend(train_l)
    np.random.seed(None)

    return train_data, dev_data, test_data

 
def load_into_tree(g, img_size, prefix, collector) :
    """Prints the input file/group/dataset (g) name and begin iterations on its content
        Args:
            g: a h5py object
            img_size: a tuple (width, height) representing the target img size
            prefix: the label prefix of the current object
            collector: a list used to collect data
    """
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        
        dg = dict(g)

        for key in dg:
            subg = dg[key]
            #print(offset, key) #,"   ", subg.name #, val, subg.len(), type(subg),
            if len(dict(g)) == 1:
                new_prefix = prefix
            else:
                new_prefix = prefix + [key]
            load_into_tree(subg, img_size, new_prefix, collector)

    elif isinstance(g, h5py.Dataset):

        X = g

        if img_size is not None:

            num_stack = img_size[0]
            # we need to resize stack
            stride = int(len(g) / num_stack)
            X = []
            for i in range(num_stack):
                l = i * stride
                r = (i + 1) * stride if i != num_stack - 1 else len(g)
                X.append(np.mean(g[l:r], 0))

            # we need to resize images
            X = np.array([imresize(x, (img_size[1], img_size[2])) for x in X])

        collector.append((np.array(X), process_label(prefix)))


if __name__ == "__main__" :
    train_data, dev_data, test_data = load_data(os.path.join("data", "zoomeddata_32_64_64.hdf5"), img_size=None)
    #pprint(data)
    #print(len(data))
    sl = float(len(train_data) + len(dev_data) + len(test_data))
    print(sl)
    print(len(train_data) / sl )
    print(len(dev_data) / sl)
    print(len(test_data) / sl)
    print(len(train_data))
    print(len(dev_data))
    print(len(test_data))
    
