import numpy as np
import tensorflow as tf

import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras import backend, callbacks


import urllib.request
import sys
import os

IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

def download_file():

    with open(IRIS_TRAINING, 'wb') as f:
        f.write(urllib.request.urlopen(IRIS_TRAINING_URL).read())

    with open(IRIS_TEST, 'wb') as f:
        f.write(urllib.request.urlopen(IRIS_TEST_URL).read())

def network(num_classes):
    model = Sequential()
    model.add(Dense(10,activation='relu',input_shape=(4,)))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    return model

def main():

    if os.path.isfile(IRIS_TRAINING)==0:
        download_file()

    training_data = np.loadtxt('./iris_training.csv',delimiter=',',skiprows=1)
    test_data = np.loadtxt('./iris_test.csv',delimiter=',',skiprows=1)

    # split training and validation
    validation_data, training_data = np.split(training_data,[int(training_data.shape[0]/3)])

    # A[:-1] is to slice excepe last element
    # A[-1] is the last element
    train_X = training_data[:,:-1]
    train_y = training_data[:,-1]

    valid_X = validation_data[:,:-1]
    valid_y = validation_data[:,-1]

    test_X = test_data[:,:-1]
    test_y = test_data[:,-1]


    num_classes = 3
    train_y = keras.utils.to_categorical(train_y,num_classes)
    valid_y = keras.utils.to_categorical(valid_y,num_classes)
    test_y = keras.utils.to_categorical(test_y,num_classes)

    model = network(num_classes)

    model.summary()

    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    # Callbacks definition
    tensorboard = callbacks.TensorBoard(log_dir='./logs/', histogram_freq=1)
    callback_list = [tensorboard]

    # verbose: display mode: 0:no display, 1: progress bar
    history = model.fit(train_X,train_y,batch_size=20,epochs=2000,verbose=0,validation_data = (valid_X, valid_y),callbacks=callback_list)

    score = model.evaluate(test_X,test_y,verbose=0)

    # score[0]: loss, score[1]: accuracy
    print('Loss:', score[0])
    print('Accuracy:', score[1])

    backend.clear_session()

if __name__ == '__main__':
    main()
