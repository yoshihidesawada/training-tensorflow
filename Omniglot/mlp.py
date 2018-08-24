import inputs
import tensorflow as tf
import os
import sys

import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras import backend, callbacks



def network(num_classes):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu',input_shape=(105,105,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(24, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(36, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


def main():
    train, valid, test = inputs.get_data()
    train_images, train_labels = inputs.make_batch(train)
    valid_images, valid_labels = inputs.make_batch(valid)
    test_images, test_labels = inputs.make_batch(test)

    num_classes = 47
    train_y = keras.utils.to_categorical(train_labels,num_classes)
    valid_y = keras.utils.to_categorical(valid_labels,num_classes)
    test_y = keras.utils.to_categorical(test_labels,num_classes)

    model = network(num_classes)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    # verbose: display mode: 0:no display, 1: progress bar
    batch_size = 32
    history = model.fit(train_images,train_y,batch_size=32,\
    epochs=10,verbose=1,validation_data = (valid_images, valid_y))
    score = model.evaluate(test_images,test_y,verbose=0)

    # score[0]: loss, score[1]: accuracy
    print('Loss:', score[0])
    print('Accuracy:', score[1])

    backend.clear_session()

if __name__ == '__main__':
    main()
