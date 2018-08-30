import os
import re
import sys
import random
from PIL import Image
import numpy as np

import tensorflow as tf
from functools import partial

import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import multiply, add, Input, Dense, Flatten, Conv2D, MaxPooling2D, Reshape, Conv2DTranspose
from tensorflow.contrib.keras.api.keras import backend, callbacks
from tensorflow.contrib.keras.api.keras.utils import Progbar
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras import backend as K
from tensorflow.python.keras.layers.merge import _Merge

from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

_BATCH_SIZE=64
_NOISE_DIM=10
_WIDTH=32
_HEIGHT=32

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
        num_train = int(len(data))
        # array[:A]: 0<=array<A, array[A:]: A<=array<last
        train += data[:num_train]
    return train

def make_batch(data_list):
    labels, paths, images = [], [], []
    for data in data_list:
        labels.append(data[0])
        paths.append(data[1])

        bak = Image.open(data[1])
        bak = bak.resize((_WIDTH, _HEIGHT), Image.ANTIALIAS)
        bak = np.array(bak, dtype=np.float32)
        bak = bak.reshape([_WIDTH,_HEIGHT,1])
        bak = (bak-0.5)*2.0
        images.append(bak)

    images = np.array(images)

    return images, labels

def generator():
    model = Sequential()
    model.add(Dense(4*4*36, input_shape=(_NOISE_DIM,)))
    model.add(LeakyReLU())
    model.add(Reshape((4,4,36)))

    model.add(Conv2DTranspose(24, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(16, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(1, kernel_size=(5,5), activation='tanh', strides=(2,2), padding='same'))
    return model

def discriminator():
    model = Sequential()
    model.add(Conv2D(1, kernel_size=(5,5), strides=(2,2), padding='same', input_shape=(_WIDTH, _HEIGHT, 1)))
    model.add(LeakyReLU())
    model.add(Conv2D(16, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1))
    return model

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        eps = K.random_uniform((_BATCH_SIZE,1,1,1))
        return (eps*inputs[0]+(1-eps)*inputs[1])


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def gradient_penalty(y_true, y_pred, interpolate, lamb):
    grad = K.gradients(y_pred, interpolate)[0]
    norm = K.square(grad)
    norm_sum = K.sum(norm,axis=np.arange(1,len(norm.shape)))
    l2_norm = K.sqrt(norm_sum)

    gp_reg = lamb*K.square(1-l2_norm)
    return K.mean(gp_reg)

def main():

    batch_size = _BATCH_SIZE
    noise_dim = _NOISE_DIM
    lamb = 10.0

    train = get_data()
    train_images, train_labels = make_batch(train)

    gen = generator()
    dis = discriminator()
    gen.summary()
    dis.summary()

    dis_opt = Adam(lr=1.0e-4, beta_1=0.0, beta_2=0.9)
    gen_opt = Adam(lr=1.0e-4, beta_1=0.0, beta_2=0.9)


    gen.trainable = True
    dis.trainable = False
    gen_inputs = Input(shape=(noise_dim,))
    gen_outputs = gen(gen_inputs)
    dis_outputs = dis(gen_outputs)
    gen_model = Model(inputs=[gen_inputs],outputs=[dis_outputs])
    gen_model.compile(loss=wasserstein_loss,optimizer=gen_opt)
    gen_model.summary()


    gen.trainable = False
    dis.trainable = True
    real_inputs = Input(shape=train_images.shape[1:])
    dis_real_outputs = dis(real_inputs)

    fake_inputs = Input(shape=(noise_dim,))
    gen_fake_outputs = gen(fake_inputs)
    dis_fake_outputs = dis(gen_fake_outputs)

    interpolate = RandomWeightedAverage()([real_inputs,gen_fake_outputs])
    dis_interpolate_outputs = dis(interpolate)

    gp_reg = partial(gradient_penalty,interpolate=interpolate,lamb=lamb)
    #gp_reg.__name__ = 'gradient_penalty'

    dis_model = Model(inputs=[real_inputs, fake_inputs],\
    outputs=[dis_real_outputs, dis_fake_outputs,dis_interpolate_outputs])

    dis_model.compile(loss=[wasserstein_loss,wasserstein_loss,gp_reg],optimizer=dis_opt)
    dis_model.summary()


    max_epoch = 10001
    max_train_only_dis = 5
    minibatch_size = batch_size*max_train_only_dis
    max_loop = int(train_images.shape[0]/minibatch_size)

    real = np.zeros((batch_size,train_images.shape[1],train_images.shape[2],train_images.shape[3]), dtype=np.float32)
    minibatch_train_images = np.zeros((minibatch_size,train_images.shape[1],train_images.shape[2],train_images.shape[3]), dtype=np.float32)

    progbar = Progbar(target=max_epoch)
    real_label = [-1]*batch_size
    fake_label = [1]*batch_size
    dummy_label = [0]*batch_size
    for epoch in range(max_epoch):

        np.random.shuffle(train_images)
        for loop in range(max_loop):

            minibatch_train_images = train_images[loop*minibatch_size:(loop+1)*minibatch_size]
            for train_only_dis in range(max_train_only_dis):

                real = minibatch_train_images[train_only_dis*batch_size:(train_only_dis+1)*batch_size]
                noise = np.random.uniform(-1,1,(batch_size,noise_dim)).astype(np.float32)
                dis_loss = dis_model.train_on_batch([real,noise],[real_label,fake_label,dummy_label])

            noise = np.random.uniform(-1,1,(batch_size,noise_dim)).astype(np.float32)
            gen_loss = gen_model.train_on_batch(noise,real_label)

        progbar.add(1, values=[("dis_loss",dis_loss[0]), ("gen_loss", gen_loss)])
        if epoch%100 == 0:
            noise = np.random.uniform(-1,1,(batch_size,10)).astype(np.float32)
            fake = gen.predict(noise)
            tmp = [r.reshape(-1,32) for r in fake]
            tmp = np.concatenate(tmp,axis=1)
            img = ((tmp/2.0+0.5)*255.0).astype(np.uint8)
            Image.fromarray(img).save("generate/%d.png"%(epoch))

    backend.clear_session()

if __name__ == '__main__':
    main()
