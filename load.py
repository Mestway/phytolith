from __future__ import absolute_import 
from __future__ import print_function

import h5py
import matplotlib.pyplot as plt

import os
import numpy as np

from scipy.misc import imresize

from pprint import pprint

import json

params = json.load(open('params.json'))

def load(file_name):

    f = h5py.File(file_name, "r")

    Xs = []
    ys = []

    max_width = 512
    max_height = 512

    def load_handler(name):
        data = f[name]
        try:
            if data.shape[0] > 100 and data.shape[1] > 100:

                if len(data.shape) == 3:
                    X = np.mean(data, -1)
                else:
                    X = data

                w_pad = max_width - X.shape[0]
                w_pad = (int(w_pad / 2), int(w_pad - w_pad / 2))
                
                h_pad = max_height - X.shape[1]
                h_pad = (int(h_pad / 2), int(h_pad - h_pad / 2))

                X = np.pad(X, pad_width=(w_pad, h_pad), mode='constant', constant_values=0)

                if params["width"] < max_width or params["height"] < max_height:
                    X = imresize(X, (params["width"], params["height"]))

                segs = name.split('/')
                for seg in segs:
                    if "Subfamily" in seg:
                        family = seg.split('-')[0].strip()
                    if "Tribe" in seg:
                        tribe = seg.split('-')[0].strip()

                Xs.append(X)
                ys.append(family)
        except:
            pass

    f.visit(load_handler)

    return Xs, ys


