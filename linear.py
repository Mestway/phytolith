import tensorflow as tf
import numpy as np

class LinearModel(object):

    def __init__(self, params, num_classes):

        self.X_dim = params["width"] * params["height"]
        self.num_classes = num_classes

        self.learning_rate = params["learning_rate"]
        self.training_epochs = params["training_epochs"]
        self.lbd = 10000
        self.batch_size = params["batch_size"]


    def build_and_train(self, X_train, y_train, X_dev, y_dev):
        training_epochs = self.training_epochs
        
        graph = tf.Graph()
        with graph.as_default(), tf.variable_scope("graph", dtype=tf.float32):

            # tf Graph Input
            train_input = tf.placeholder(tf.float32, [None, self.X_dim])
            train_label = tf.placeholder(tf.int32, [None]) 

            W = tf.get_variable("W", initializer=tf.random_uniform([self.X_dim, self.num_classes], -1.0, 1))
            b = tf.Variable(tf.zeros([self.num_classes]))

            logits = tf.matmul(train_input, W) + b

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32), train_label), tf.float32))

            # Minimize error using cross entropy
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label, logits=logits) #+ self.lbd * tf.norm(W, ord=1)

            # Gradient Descent
            optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

            with tf.Session() as sess:

                # Run the initializer
                sess.run(tf.global_variables_initializer())

                # Training cycle
                for epoch in range(training_epochs):
                    
                    batch_size = self.batch_size

                    train_loss = 0.
                    train_acc = 0.
                    total_batch = len(X_train) // batch_size
                    
                    # Loop over all batches
                    for i in range(total_batch + 1):
                        
                        if i < total_batch:
                            batch_xs = X_train[i * batch_size : (i + 1) * batch_size]
                            batch_ys = y_train[i * batch_size : (i + 1) * batch_size]
                        else:
                            if (i * batch_size  == len(X_train)):
                                break

                            batch_xs = X_train[i * batch_size : ]
                            batch_ys = y_train[i * batch_size : ]

                        # Run optimization op (backprop) and cost op (to get loss value)
                        tr_loss, tr_acc, _ = sess.run([loss, accuracy, optimizer], 
                                                     feed_dict={ train_input: batch_xs,
                                                                 train_label: batch_ys })

                        # Compute average loss
                        train_loss += np.mean(tr_loss) / total_batch
                        train_acc += tr_acc / total_batch

                    dev_loss = 0.
                    dev_acc = 0.

                    for i in range(len(X_dev)):
                        d_loss, d_acc = sess.run([loss, accuracy], 
                                                  feed_dict={ train_input: [X_dev[i]],
                                                              train_label: [y_dev[i]] })
                        # Compute average loss
                        dev_loss += np.mean(d_loss) / len(X_dev)
                        dev_acc += d_acc / len(X_dev)

                    # Display logs per epoch step
                    print("===== Epoch {} =====".format(epoch))
                    print("train_acc: {}".format(train_acc))
                    print("train_loss: {}".format(train_loss))
                    print("dev_acc: {}".format(dev_acc))
                    print("dev_loss: {}".format(dev_loss))
