#!/usr/bin/env python

#tensorflow/examples/courses/udacity_intro_to_tensorflow_for_deep_learning/l04c01_image_classification_with_cnns.ipynb

# Image Classification with Convolutional Neural Networks
import tensorflow as tf

import os
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from clothes import *


### Build the model
#
#Building the neural network requires configuring the layers of the model, then
#compiling the model.
#
#### Setup the layers
#The basic building block of a neural network is the *layer*. A layer extracts a
#representation from the data fed into it. Hopefully, a series of connected
#layers results in a representation that is meaningful for the problem at hand.
#
#Much of deep learning consists of chaining together simple layers. Most layers,
#like `tf.keras.layers.Dense`, have internal parameters which are adjusted
#("learned") during training.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#This network layers are:
#
#* **"convolutions"** `tf.keras.layers.Conv2D and MaxPooling2D`— Network start
#with two pairs of Conv/MaxPool. The first layer is a Conv2D filters (3,3) being
#applied to the input image, retaining the original image size by using padding,
#and creating 32 output (convoluted) images (so this layer creates 32 convoluted
#                                            images of the same size as input).
#After that, the 32 outputs are reduced in size using a MaxPooling2D (2,2) with
#a stride of 2. The next Conv2D also has a (3,3) kernel, takes the 32 images as
#input and creates 64 outputs which are again reduced in size by a MaxPooling2D
#layer. So far in the course, we have described what a Convolution does, but we
#haven't yet covered how you chain multiples of these together. We will get back
#to this in lesson 4 when we use color images. At this point, it's enough if you
#understand the kind of operation a convolutional filter performs
#
#* **output** `tf.keras.layers.Dense` — A 128-neuron, followed by 10-node
#*softmax* layer. Each node represents a class of clothing. As in the previous
#layer, the final layer takes input from the 128 nodes in the layer before it,
#and outputs a value in the range `[0, 1]`, representing the probability that
#the image belongs to that class. The sum of all 10 node values is 1.
#
#> Note: Using `softmax` activation and `SparseCategoricalCrossentropy()` has
#issues and which are patched by the `tf.keras` model. A safer approach, in
#general, is to use a linear output (no activation function) with
#`SparseCategoricalCrossentropy(from_logits=True)`.


#### Compile the model
#
#Before the model is ready for training, it needs a few more settings. These are
#added during the model's *compile* step:
#
#* *Loss function* — An algorithm for measuring how far the model's outputs are
#from the desired output. The goal of training is this measures loss.  
#* *Optimizer* —An algorithm for adjusting the inner parameters of the model in
#order to minimize loss.  
#* *Metrics* —Used to monitor the training and testing
#steps. The following example uses *accuracy*, the fraction of the images that
#are correctly classified.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

### Train the model
#
#First, we define the iteration behavior for the train dataset:
#1. Repeat forever by specifying `dataset.repeat()` (the `epochs` parameter described
#                                          below limits how long we perform
#                                          training).
#2. The
#`dataset.shuffle(60000)` randomizes the order so our model cannot learn
#anything from the order of the examples.
#3. And `dataset.batch(32)` tells
#`model.fit` to use batches of 32 images and labels when updating the model
#variables.
#
#Training is performed by calling the `model.fit` method: 1. Feed the training
#data to the model using `train_dataset`.  2. The model learns to associate
#images and labels.  3. The `epochs=5` parameter limits training to 5 full
#iterations of the training dataset, so a total of 5 * 60000 = 300000 examples.
#
#(Don't worry about `steps_per_epoch`, the requirement to have this flag will soon be removed.)

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

saved = 'cnn.h5'
if os.path.exists(saved):
    model = tf.keras.models.load_model(saved)
else:
    model.fit(train_dataset, epochs=10,
             verbose=2, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
    model.save(saved)

#As the model trains, the loss and accuracy metrics are displayed. This model
#reaches an accuracy of about 0.97 (or 97%) on the training data.

## Evaluate accuracy
#Next, compare how the model performs on the test dataset. Use all examples we
#have in the test dataset to assess accuracy.

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

#As it turns out, the accuracy on the test dataset is smaller than the accuracy
#on the training dataset. This is completely normal, since the model was trained
#on the `train_dataset`. When the model sees images it has never seen during
#training, (that is, from the `test_dataset`), we can expect performance to go
#down. 

show_results(model, test_dataset)

# And, as before, the model predicts a label of 6 (shirt).

# # Exercises

# Experiment with different models and see how the accuracy results differ. In
# particular change the following parameters: *   Set training epochs set to 1 *
# Number of neurons in the Dense layer following the Flatten one. For example, go
# really low (e.g. 10) in ranges up to 512 and see how accuracy changes *   Add
# additional Dense layers between the Flatten and the final Dense(10), experiment
# with different units in these layers *   Don't normalize the pixel values, and
# see the effect that has


# Remember to enable GPU to make everything run faster
# (Runtime -> Change runtime type -> Hardware accelerator -> GPU).
# Also, if you run into trouble, simply reset the entire environment and start
# from the beginning: *   Edit -> Clear all outputs *   Runtime -> Reset all
# runtimes
