from __future__ import absolute_import 
from __future__ import print_function

import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import json
import tensorflow as tf

from load import load_data

from cnn import CNNModel
from linear import LinearModel

np.random.seed(999)

params = json.load(open('params.json'))

def main(input_dir="data"):

    for data_file in ["meandata.hdf5"]:#, "mindata.hdf5", "maxdata.hdf5"]:

        data = load_data(os.path.join(input_dir, data_file), img_size=(params["width"], params["height"]))

        Xs = [d[0] for d in data]
        ys = [d[1] for d in data]

        rand_indices = np.random.permutation(len(Xs))

        Xs = [Xs[i] for i in rand_indices]
        ys = [ys[i] for i in rand_indices]

        #for i in range(5):
        #    plt.imshow(Xs[i])
        #    plt.show()

        Xs = np.array([X.flatten() for X in Xs])
        ys = ["_".join(y) for y in ys]

        all_labels = list(set(ys))
        ys = [all_labels.index(y) for y in ys]

        split_indices = [int(0.6 * len(Xs)), int(0.8 * len(Xs))]

        X_train = Xs[:split_indices[0]]
        y_train = ys[:split_indices[0]]

        X_dev = Xs[split_indices[0] + 1: split_indices[1]]
        y_dev = ys[split_indices[0] + 1: split_indices[1]]

        #X_test = Xs[split_indices[1] + 1:]
        #y_test = ys[split_indices[1] + 1:]

        #model = CNNModel(params, len(yset))
        model = LinearModel(params, len(set(ys)))

        graph = model.build_and_train(X_train, y_train, X_dev, y_dev)

if __name__ == '__main__':
    main()