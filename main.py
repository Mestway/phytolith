from __future__ import absolute_import 
from __future__ import print_function

import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import json
import tensorflow as tf

from load import load

np.random.seed(999)

params = json.load(open('params.json'))

class LinearModel(object):

    def __init__(self, params, num_classes):

        self.X_dim = params["width"] * params["height"]
        self.num_classes = num_classes

        self.learning_rate = params["learning_rate"]
        self.training_epochs = params["training_epochs"]
        self.eta = 0.01
        self.batch_size = params["batch_size"]


    def build_and_train(self, X_train, y_train, X_dev, y_dev):
        batch_size = self.batch_size
        training_epochs = self.training_epochs
        
        graph = tf.Graph()
        with graph.as_default(), tf.variable_scope("graph", dtype=tf.float32):

            # tf Graph Input
            train_input = tf.placeholder(tf.float32, [batch_size, self.X_dim])
            train_label = tf.placeholder(tf.int32, [batch_size]) 

            W = tf.get_variable("W", initializer=tf.random_uniform([self.X_dim, self.num_classes], -1.0, 1))
            b = tf.Variable(tf.zeros([self.num_classes]))

            logits = tf.matmul(train_input, W) + b

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32), train_label), tf.float32))

            # Minimize error using cross entropy
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label, logits=logits) + self.eta * tf.norm(W, ord=1)
            
            # Gradient Descent
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            with tf.Session() as sess:

                # Run the initializer
                sess.run(tf.global_variables_initializer())

                # Training cycle
                for epoch in range(training_epochs):
                    
                    train_loss = 0.
                    train_acc = 0.

                    total_batch = len(X_train) // batch_size
                    
                    # Loop over all batches
                    for i in range(total_batch):
                        
                        batch_xs = X_train[i * batch_size : (i + 1) * batch_size]
                        batch_ys = y_train[i * batch_size : (i + 1) * batch_size]

                        # Run optimization op (backprop) and cost op (to get loss value)
                        tr_loss, tr_acc, _ = sess.run([loss, accuracy, optimizer], 
                                                     feed_dict={ train_input: batch_xs,
                                                                 train_label: batch_ys })
                        # Compute average loss
                        train_loss += tr_loss / total_batch
                        train_acc += tr_acc / total_batch

                    dev_loss = 0.
                    dev_acc = 0.

                    for i in range(len(X_dev)):
                        d_loss, d_acc = sess.run([loss, accuracy], 
                                                  feed_dict={ train_input: [X_dev[i]],
                                                              train_label: [y_dev[i]] })
                        # Compute average loss
                        dev_loss += d_loss / len(X_dev)
                        dev_acc += d_acc / len(X_dev)

                    # Display logs per epoch step
                    print("===== Epoch {} =====".format(epoch))
                    print("train_acc: {}".format(train_acc))
                    print("train_loss: {}".format(train_loss))
                    print("dev_acc: {}".format(dev_acc))
                    print("dev_loss: {}".format(dev_loss))


def main(input_dir="data"):

    for data_file in ["meandata.hdf5", "mindata.hdf5", "maxdata.hdf5"]:

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

        split_indices = [int(0.7 * len(Xs)), int(0.8 * len(Xs))]

        X_train = Xs[:split_indices[0]]
        y_train = ys[:split_indices[0]]

        X_dev = Xs[split_indices[0] + 1: split_indices[1]]
        y_dev = ys[split_indices[0] + 1: split_indices[1]]

        #X_test = Xs[split_indices[1] + 1:]
        #y_test = ys[split_indices[1] + 1:]

        linear_model = LinearModel(params, len(yset))
        graph = linear_model.build_and_train(X_train, y_train, X_dev, y_dev)

if __name__ == '__main__':
    main()