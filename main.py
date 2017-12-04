from __future__ import absolute_import 
from __future__ import print_function

import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import json
import argparse

from load import load_data
from linear_model import LinearModel
import classifier

np.random.seed(999)




def batch_data(data, batch_size):
    """ Given X, y, batch them """

    batch_num = int(np.ceil(len(data) / float(batch_size)))

    data_out = []
    for i in range(batch_num):
        l = i * batch_size
        r = min((i + 1) * batch_size, len(data))

        data_out.append((np.array([d[0] for d in data[l:r]], dtype=np.float32),
                         np.array([d[1] for d in data[l:r]], dtype=np.int64)))

    return data_out


def main(input_dir="data", cuda_enable=False):

    # load params
    params = json.load(open('params.json'))
    if cuda_enable:
        params = json.load(open('params_cuda.json'))

    for data_file in ["zoomeddata_32_64_64.hdf5"]:

        d_train, d_dev, d_test = load_data(os.path.join(input_dir, data_file),
                                           img_size=(params["width"], params["height"]),
                                           num_stack=params["num_stack"])

        vocab = {
            "subfamily": [],
            "tribe": [],
            "genus": []
        }
        for d in d_train:
            vocab["subfamily"].append(d[1][0])
            vocab["tribe"].append(d[1][1])
            vocab["genus"].append(d[1][2])
        for k in vocab:
            vocab[k] = list(set(vocab[k]))


        def process_data(data, tree_label=False):
            rand_indices = np.random.permutation(len(data))
            if not tree_label:
                data_out = [(data[i][0], vocab["genus"].index(data[i][1][2])) 
                            for i in rand_indices]
            else:
                data_out = [(data[i][0], 
                             np.array([vocab["subfamily"].index(data[i][1][0]), 
                                       vocab["tribe"].index(data[i][1][1]), 
                                       vocab["genus"].index(data[i][1][2])])) 
                            for i in rand_indices]
            return data_out


        train_data = batch_data(process_data(d_train), batch_size=params["batch_size"])
        dev_data = batch_data(process_data(d_dev), batch_size=params["batch_size"])

        net = LinearModel(params, len(vocab["genus"]))

        train_acc, dev_acc = classifier.train(net, train_data, dev_data, params, cuda_enable)

        ## show images
        #if examples_to_demo > 0:
        #    show_data = [d_dev[i] for i in np.random.randint(len(dev_data), size=examples_to_demo)]
        #    show_data = batch_data(process_data(show_data), batch_size=1) 
        #    classifier.test(net, show_data, cuda_enable, show_example=True)

        with open("result.log", "w") as f:
            f.write("train_acc, dev_acc\n")
            for i in range(len(train_acc)):
                f.write("{:.3f}".format(train_acc[i]) + ", " + "{:.3f}".format(dev_acc[i]) + "\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Phytolith classifier')
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true')

    args = parser.parse_args()

    main(input_dir="data", output_dir="out", cuda_enable=args.cuda)
