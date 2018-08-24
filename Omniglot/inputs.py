import os
import re
import random
import tensorflow as tf
import sys
from PIL import Image
import numpy as np

def get_data():
    train, valid, test = [], [], []
    topdir = os.path.join('images_background_small2', 'Japanese_(katakana)')
    regexp = re.compile(r'character(\d+)')

    # os.walk: directory search
    for dirpath, _, files in os.walk(topdir, followlinks=True):
        match = regexp.search(dirpath)
        if match is None:
            continue

        # .group: 1-st group
        label = int(match.group(1))-1

        data = [(label, os.path.join(dirpath, file)) for file in files]
        random.shuffle(data)
        num_train = int(len(data)*0.5)
        num_valid = int(len(data)*0.3)
        # array[:A]: 0<=array<A, array[A:]: A<=array<last
        train += data[:num_train]
        valid += data[num_train:(num_train+num_valid)]
        test += data[(num_train+num_valid):]
    return train, valid, test

def make_batch(data_list):
    labels, paths, images = [], [], []
    for data in data_list:
        labels.append(data[0])
        paths.append(data[1])

        bak = Image.open(data[1])
        bak = np.array(bak, dtype=np.float32)
        bak = bak.reshape([bak.shape[0], bak.shape[1] ,1])
        mean = np.mean(bak)
        stddev = np.std(bak)
        adjust_stddev = np.max([stddev, 1.0/np.sqrt(bak.shape[0]*bak.shape[1])])
        bak = (bak-mean)/adjust_stddev
        images.append(bak)

    images = np.array(images)

    return images, labels
