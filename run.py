from __future__ import absolute_import 
from __future__ import print_function

import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import json
import argparse
from pprint import pprint

from load import load_data
from models import *
from enhance_data import gen_synthetic
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


def main(params, input_dir="data", output_dir="out", cuda_enable=False):

    for data_file in ["zoomeddata_32_64_64.hdf5"]:

        d_train, d_dev, d_test = load_data(os.path.join(input_dir, data_file), img_size=params["img_size"])

        # build the vocabulary for images
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

        mean_image = np.mean([d[0] for d in d_train])
        std_image = np.std([d[0] for d in d_train])

        def normalize_fn(X):
            return ((X - mean_image) / 255.)

        def process_data(data, tree_label=False):
            rand_indices = np.random.permutation(len(data))
            if not tree_label:
                data_out = [(normalize_fn(data[i][0]), vocab["genus"].index(data[i][1][2])) 
                            for i in rand_indices]
            else:
                data_out = [(normalize_fn(data[i][0]),
                             np.array([vocab["subfamily"].index(data[i][1][0]), 
                                       vocab["tribe"].index(data[i][1][1]), 
                                       vocab["genus"].index(data[i][1][2])])) 
                            for i in rand_indices]
            return data_out

        if params["add_synthetic_data"]:
            synthesized = gen_synthetic(d_train)
            print("{} synthetic examples added.".format(len(synthesized)))
        else:
            synthesized = []

        tree_label = params["tree_prediction"]

        train_data = batch_data(process_data(d_train + synthesized, tree_label), batch_size=params["batch_size"])
        dev_data = batch_data(process_data(d_dev, tree_label), batch_size=params["batch_size"])
        test_data = batch_data(process_data(d_test, tree_label), batch_size=params["batch_size"])

        #net = LinearModel(params, len(vocab["genus"]))
        #net = SimpleNet(params, len(vocab["genus"]))
    
        for model_class in [LinearModel, SimpleNet, SimpleConvNet, ConvNet]:
            print(">>>>>>>>>> {} <<<<<<<<<<".format(model_class))

            if tree_label:
                num_classes = len(vocab["subfamily"]) + len(vocab["tribe"]) + len(vocab["genus"])
            else:
                num_classes = len(vocab["genus"])

            net = model_class(params, num_classes)
            
            train_acc, dev_acc = classifier.train(net, train_data, dev_data, vocab, params, cuda_enable)

            for test_index in [0, 1, 2]:
                test_acc = classifier.test(net, test_data, vocab, cuda_enable, tree_label, test_index)
                print("[Test accuracy for label {} is {}]".format(test_index, test_acc))

            ## show images
            #if examples_to_demo > 0:
            #    show_data = [d_dev[i] for i in np.random.randint(len(dev_data), size=examples_to_demo)]
            #    show_data = batch_data(process_data(show_data), batch_size=1) 
            #    classifier.test(net, show_data, cuda_enable, show_example=True)

            with open(os.path.join(output_dir, "result_{}.log".format(model_class.__name__)), "w") as f:
                f.write("train_acc, dev_acc\n")
                for i in range(len(train_acc)):
                    f.write("{:.3f}".format(train_acc[i]) + ", " + "{:.3f}".format(dev_acc[i]) + "\n")


def sample_param_from_space(param_space):

    params = {}
    for k in param_space:
        if type(param_space[k]) is list:
            l = param_space[k]
            params[k] = l[np.random.randint(len(l))]
        if type(param_space[k]) is dict:
            params[k] = sample_param_from_space(param_space[k])
    return params


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Phytolith classifier')
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true')
    parser.add_argument('--param-search', dest='param_search', default=False, action='store_true')
    parser.add_argument('--mult', dest='mult', default=False, action='store_true')

    args = parser.parse_args()

    if args.param_search:
        param_space = json.load(open("param_space.json"))

        for i in range(100):
            print("[Start {}]".format(i))
            # sample parameter
            params = sample_param_from_space(param_space)
            pprint(params)
            
            time_stamp = datetime.now().strftime("%b_%d_%f").lower()
            out_dir = os.path.join("output", "out_{}".format(time_stamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            main(params, input_dir="data", output_dir=out_dir, cuda_enable=args.cuda)
            print("[End {}]".format(i))
    else:
        num_runs = 1
        if args.mult:
            num_runs = 10

        for i in range(num_runs):
            # load params
            params = json.load(open('params.json'))
            if args.cuda:
                params = json.load(open('params_cuda.json'))

            out_dir = "output"
            time_stamp = datetime.now().strftime("%b_%d_%f").lower()
            out_dir = os.path.join("output", "out_{}".format(time_stamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            main(params, input_dir="data", output_dir=out_dir, cuda_enable=args.cuda)

