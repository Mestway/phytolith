from __future__ import absolute_import 
from __future__ import print_function

import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import json
import tensorflow as tf

from load import load

from cnn import CNNModel
from linear import LinearModel

np.random.seed(999)

params = json.load(open('params.json'))


def main(input_dir="data"):

    for data_file in ["meandata.hdf5"]:#, "mindata.hdf5", "maxdata.hdf5"]:

        Xs, ys = load(os.path.join(input_dir, data_file), params["width"], params["height"])

        rand_indices = np.random.permutation(len(Xs)-1)

        Xs = [Xs[i] for i in rand_indices]
        ys = [ys[i] for i in rand_indices]

        #for i in range(5):
        #    plt.imshow(Xs[i])
        #    plt.show()

        Xs = np.array([X.flatten() for X in Xs])

        yset = list(set(ys))
        y_dict = {}
        for i in range(len(yset)):
            y_dict[yset[i]] = i
        ys = np.array([y_dict[y] for y in ys])

        pprint(y_dict)

        split_indices = [int(0.6 * len(Xs)), int(0.8 * len(Xs))]

        X_train = Xs[:split_indices[0]]
        y_train = ys[:split_indices[0]]

        X_dev = Xs[split_indices[0] + 1: split_indices[1]]
        y_dev = ys[split_indices[0] + 1: split_indices[1]]

        #X_test = Xs[split_indices[1] + 1:]
        #y_test = ys[split_indices[1] + 1:]

        #model = CNNModel(params, len(yset))
        model = LinearModel(params, len(yset))

        graph = model.build_and_train(X_train, y_train, X_dev, y_dev)

if __name__ == '__main__':
    main()