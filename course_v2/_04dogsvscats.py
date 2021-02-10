#!/usr/bin/env python3

#courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb

# Dogs vs Cats Image Classification Without Image Augmentation

# In this tutorial, we will discuss how to classify images into pictures of cats
# or pictures of dogs. We'll build an image classifier using
# `tf.keras.Sequential` model and load data using
# `tf.keras.preprocessing.image.ImageDataGenerator`.

# ## Specific concepts that will be covered:
# In the process, we will build practical experience and develop intuition around the following concepts

# * Building _data input pipelines_ using the
# `tf.keras.preprocessing.image.ImageDataGenerator` class — How can we
# efficiently work with data on disk to interface with our model?
# * _Overfitting_ - what is it, how to identify it?

# **Before you begin**
# Before running the code in this notebook, reset the runtime by going to
# **Runtime -> Reset all runtimes** in the menu above. If you have been working
# through several notebooks, this will help you avoid reaching Colab's memory
# limits.

# Let's start by importing required packages:

# *   os — to read files and directory structure
# *   numpy — for some matrix math outside of TensorFlow
# *   matplotlib.pyplot — to plot the graph and display images in our training and validation data


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('TkAgg')

import os
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Data Loading
# To build our image classifier, we begin by downloading the dataset. The dataset
# we are using is a filtered version of <a
# href="https://www.kaggle.com/c/dogs-vs-cats/data" target="_blank">Dogs vs.
# Cats</a> dataset from Kaggle (ultimately, this dataset is provided by Microsoft
                              # Research).
# In previous Colabs, we've used <a href="https://www.tensorflow.org/datasets"
# target="_blank">TensorFlow Datasets</a>, which is a very easy and convenient
# way to use datasets. In this Colab however, we will make use of the class
# `tf.keras.preprocessing.image.ImageDataGenerator` which will read data from
# disk. We therefore need to directly download *Dogs vs. Cats* from a URL and
# unzip it to the Colab filesystem.

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
# puts it into ~/.keras/datasets 

# We'll now assign variables with the proper file path for the training and validation sets.

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

### Understanding our data

#Let's look at how many cats and dogs images we have in our training and validation directory

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# Setting Model Parameters

# For convenience, we'll set up variables that will be used later while
# pre-processing our dataset and training our network.

BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

# Data Preparation 
# Images must be formatted into appropriately pre-processed floating point
# tensors before being fed into the network. The steps involved in preparing
# these images are:
# 1. Read images from the disk
# 2. Decode contents of these images and convert it into proper grid format as
# per their RGB content
# 3. Convert them into floating point tensors
# 4. Rescale the tensors from values between 0 and 255 to values between 0 and 1,
# as neural networks prefer to deal with small input values.
# Fortunately, all these tasks can be done using the class
# **tf.keras.preprocessing.image.ImageDataGenerator**.
# We can set this up in a couple of lines of code.

train_image_generator      = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

# After defining our generators for training and validation images,
# **flow_from_directory** method will load images from the disk, apply rescaling,
# and resize them using single line of code.

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                              class_mode='binary')

### Visualizing Training images
# We can visualize our training images by getting a batch of images from the
# training generator, and then plotting a few of them using `matplotlib`.

sample_training_images, _ = next(train_data_gen) 

# The `next` function returns a batch from the dataset. One batch is a tuple of
# (*many images*, *many labels*). For right now, we're discarding the labels
# because we just want to look at the images.

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])  # Plot images 0-4

# Model Creation
## Define the model
# The model consists of four convolution blocks with a max pool layer in each of
# them. Then we have a fully connected layer with 512 units, with a `relu`
# activation function. The model will output class probabilities for two classes
# — dogs and cats — using `softmax`. 

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

### Compile the model
# As usual, we will use the `adam` optimizer. Since we output a softmax
# categorization, we'll use `sparse_categorical_crossentropy` as the loss
# function. We would also like to look at training and validation accuracy on
# each epoch as we train our network, so we are passing in the metrics argument.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

### Model Summary
# Let's look at all the layers of our network using **summary** method.
model.summary()

### Train the model

EPOCHS = 100
saved = '../data/catsvsdogs.h5'
historied = saved+'.history.pickle'
if os.path.exists(saved):
    model = tf.keras.models.load_model(saved)
    if os.path.exists(historied):
        with open(historied,'rb') as h:
            history = pickle.load(h)
    else:
        history = None
else:
    history = model.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))),
        verbose=2
    )
    model.save(saved)
    with open(historied,'wb') as h:
        pickle.dump(history,h)

if history:
    acc = history.history['accuracy']

    ### Visualizing results of the training
    #We'll now visualize the results we get after training our network.

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('../data/foo.png')
    plt.show()

# As we can see from the plots, training accuracy and validation accuracy are off
# by large margin and our model has achieved only around **70%** accuracy on the
# validation set (depending on the number of epochs you trained for).

# This is a clear indication of overfitting. Once the training and validation
# curves start to diverge, our model has started to memorize the training data
# and is unable to perform well on the validation data.
