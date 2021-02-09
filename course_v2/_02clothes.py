#!/usr/bin/env python3

#tensorflow/examples/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb


import tensorflow as tf

# Helper libraries
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from clothes import *

## Build the model
#Building the neural network requires configuring the layers of the model, then
#compiling the model.

### Setup the layers
#
#The basic building block of a neural network is the *layer*. A layer extracts a
#representation from the data fed into it. Hopefully, a series of connected
#layers results in a representation that is meaningful for the problem at hand.
#
#Much of deep learning consists of chaining together simple layers. Most layers,
#like `tf.keras.layers.Dense`, have internal parameters which are adjusted
#("learned") during training.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#This network has three layers:
#
#* **input** `tf.keras.layers.Flatten` — This layer transforms the images from a
#2d-array of 28 $\times$ 28 pixels, to a 1d-array of 784 pixels (28\*28). Think
#of this layer as unstacking rows of pixels in the image and lining them up.
#This layer has no parameters to learn, as it only reformats the data.
#
#* **"hidden"** `tf.keras.layers.Dense`— A densely connected layer of 128
#neurons. Each neuron (or node) takes input from all 784 nodes in the previous
#layer, weighting that input according to hidden parameters which will be
#learned during training, and outputs a single value to the next layer.
#
#* **output**  `tf.keras.layers.Dense` — A 128-neuron, followed by 10-node
#*softmax* layer. Each node represents a class of clothing. As in the previous
#layer, the final layer takes input from the 128 nodes in the layer before it,
#and outputs a value in the range `[0, 1]`, representing the probability that
#the image belongs to that class. The sum of all 10 node values is 1.
#
#> Note: Using `softmax` activation and `SparseCategoricalCrossentropy()` has
#issues and which are patched by the `tf.keras` model. A safer approach, in
#general, is to use a linear output (no activation function) with
#`SparseCategoricalCrossentropy(from_logits=True)`.
#
#
#### Compile the model
#
#Before the model is ready for training, it needs a few more settings. These are
#added during the model's *compile* step:
#
#
#* *Loss function* — An algorithm for measuring how far the model's outputs are
#from the desired output. The goal of training is this measures loss.  *
#*Optimizer* —An algorithm for adjusting the inner parameters of the model in
#order to minimize loss.  * *Metrics* —Used to monitor the training and testing
#steps. The following example uses *accuracy*, the fraction of the images that
#are correctly classified.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

### Train the model
#
#First, we define the iteration behavior for the train dataset:
#1. Repeat forever by specifying `dataset.repeat()` (the `epochs` parameter described below limits how long we perform training).
#2. The `dataset.shuffle(60000)` randomizes the order so our model cannot learn anything from the order of the examples.
#3. And `dataset.batch(32)` tells `model.fit` to use batches of 32 images and labels when updating the model variables.
#
#Training is performed by calling the `model.fit` method:
#1. Feed the training data to the model using `train_dataset`.
#2. The model learns to associate images and labels.
#3. The `epochs=5` parameter limits training to 5 full iterations of the training dataset, so a total of 5 * 60000 = 300000 examples.
#
#(Don't worry about `steps_per_epoch`, the requirement to have this flag will soon be removed.)

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(
  train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE),verbose=2)

#As the model trains, the loss and accuracy metrics are displayed. This model
#reaches an accuracy of about 0.88 (or 88%) on the training data.

## Evaluate accuracy
#Next, compare how the model performs on the test dataset. Use all examples we
#have in the test dataset to assess accuracy.

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32),verbose=2)
print('Accuracy on test dataset:', test_accuracy)

#As it turns out, the accuracy on the test dataset is smaller than the accuracy
#on the training dataset. This is completely normal, since the model was trained
#on the `train_dataset`. When the model sees images it has never seen during
#training, (that is, from the `test_dataset`), we can expect performance to go
#down. 

show_results(model, test_dataset)

#And, as before, the model predicts a label of 6 (shirt).

## Exercises
#
#Experiment with different models and see how the accuracy results differ. In
#particular change the following parameters:
#*   Set training epochs set to 1 *
#Number of neurons in the Dense layer following the Flatten one. For example, go
#really low (e.g. 10) in ranges up to 512 and see how accuracy changes
#*   Add
#additional Dense layers between the Flatten and the final `Dense(10)`,
#experiment with different units in these layers
#*   Don't normalize the pixel
#values, and see the effect that has
#
#
#Remember to enable GPU to make everything run faster 
#(Runtime -> Change runtime type -> Hardware accelerator -> GPU).
#Also, if you run into trouble, simply reset the entire environment and start
#from the beginning:
#*   Edit -> Clear all outputs *   Runtime -> Reset all runtimes
