from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import datetime


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def gpu():
    '''
    Basic Multi GPU computation example using TensorFlow library.
    Author: Aymeric Damien
    Project: https://github.com/aymericdamien/TensorFlow-Examples/
    '''

    '''
    This tutorial requires your machine to have 1 GPU
    "/cpu:0": The CPU of your machine.
    "/gpu:0": The first GPU of your machine
    '''

    import numpy as np
    import tensorflow as tf
    import datetime

    # Processing Units logs
    log_device_placement = True

    # Num of multiplications to perform
    n = 10

    '''
    Example: compute A^n + B^n on 2 GPUs
    Results on 8 cores with 2 GTX-980:
     * Single GPU computation time: 0:00:11.277449
     * Multi GPU computation time: 0:00:07.131701
    '''
    # Create random large matrix
    A = np.random.rand(10000, 10000).astype('float32')
    B = np.random.rand(10000, 10000).astype('float32')

    # Create a graph to store results
    c1 = []
    c2 = []

    def matpow(M, n):
        if n < 1:  # Abstract cases where n < 1
            return M
        else:
            return tf.matmul(M, matpow(M, n - 1))

    '''
    Single GPU computing
    '''
    with tf.device('/gpu:0'):
        a = tf.placeholder(tf.float32, [10000, 10000])
        b = tf.placeholder(tf.float32, [10000, 10000])
        # Compute A^n and B^n and store results in c1
        c1.append(matpow(a, n))
        c1.append(matpow(b, n))

    with tf.device('/cpu:0'):
        sum = tf.add_n(c1)  # Addition of all elements in c1, i.e. A^n + B^n

    t1_1 = datetime.datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
        # Run the op.
        sess.run(sum, {a: A, b: B})
    t2_1 = datetime.datetime.now()

    print("Single GPU computation time: " + str(t2_1 - t1_1))


def xor():
    # Define our training input and output data with type 16 bit float
    # Each input maps to an output

    X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float16)
    Y = tf.constant([[0], [1], [1], [0]], dtype=tf.float16)

    # Create a new Sequential Model
    model = keras.Sequential()

    # Add our layers

    model.add(layers.Dense(
        2000,  # Amount of Neurons
        input_dim=2,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        10000,  # Amount of Neurons
        input_dim=2000,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        10000,  # Amount of Neurons
        input_dim=10000,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        2000,  # Amount of Neurons
        input_dim=10000,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        2000,  # Amount of Neurons
        input_dim=2000,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        1,  # Amount of Neurons. We want one output
        activation='sigmoid'  # Use sigmoid because we want to output a binary classification
    ))

    # Compile our layers into a model

    model.compile(
        loss='mean_squared_error',  # The loss function that is being minimized
        optimizer='adam',  # Our optimization function
        metrics=['binary_accuracy']  # Metrics are different values that you want the model to track while training
    )

    # Our function to take in two numerical inputs and output the relevant boolean
    def cleanPredict(a, b):
        inputTens = tf.constant([[a, b]])
        return round(model.predict(inputTens)[0][0]) == 1  # model.predict(input) yields a 2d tensor

    print(cleanPredict(1, 0))  # Will yield a random value because model isn't yet trained

    model.fit(
        X,  # Input training data
        Y,  # Output training data
        epochs=300,  # Amount of iterations we want to train for
        verbose=0  # Amount of detail you want shown in terminal while training
    )

    print(cleanPredict(1, 0))  # Should Yield True


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


startTime = datetime.now()
xor()
print("Time taken:", datetime.now() - startTime)

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
