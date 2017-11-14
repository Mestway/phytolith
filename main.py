from __future__ import absolute_import 
from __future__ import print_function

import os
from pprint import pprint
import matplotlib.pyplot as plt


import tensorflow as tf

from load import load


class linear_model(object):

	def __init__(X_train, y_train, X_dev, y_dev):

		self.X_train = X_train
		self.y_train = y_train


def main(input_dir="data"):

    for data_file in ["meandata.hdf5"]: #, "mindata.hdf5", "meandata.hdf5"]:

        Xs, ys = load(os.path.join(input_dir, data_file))

        #for i in range(5):
        #    plt.imshow(Xs[i])
        #    plt.show()

        Xs = [X.flatten() for X in Xs]

        split_indices = [int(0.7 * len(Xs)), int(0.8 * len(Xs))]

        X_train = Xs[:split_indices[0]]
        y_train = ys[:split_indices[0]]

        X_dev = Xs[split_indices[0] + 1: split_indices[1]]
        y_dev = ys[split_indices[0] + 1: split_indices[1]]

        X_test = Xs[split_indices[1] + 1:]
        y_test = ys[split_indices[1] + 1:]


if __name__ == '__main__':
    main()