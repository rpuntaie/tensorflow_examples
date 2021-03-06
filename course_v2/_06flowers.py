#!/usr/bin/env python3

#courses/udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>

import matplotlib
matplotlib.use('TkAgg')

import os
import matplotlib.pyplot as plt
import numpy as np

import glob
import shutil
from six.moves import cPickle as pickle

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')


# The dataset we downloaded contains images of 5 types of flowers:

# 1. Rose
# 2. Daisy
# 3. Dandelion
# 4. Sunflowers
# 5. Tulips

# So, let's create the labels for these 5 classes: 

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# Also, the dataset we have downloaded has following directory structure. n
# <pre style="font-size: 10.0pt; font-family: Arial; line-height: 2; letter-spacing: 1.0pt;" >
# <b>flower_photos</b>
# |__ <b>daisy</b>
# |__ <b>dandelion</b>
# |__ <b>roses</b>
# |__ <b>sunflowers</b>
# |__ <b>tulips</b>
# </pre>
#
# As you can see there are no folders containing training and validation data.
# Therefore, we will have to create our own training and validation set. Let's
# write some code that will do this.
#
# The code below creates a `train` and a `val` folder each containing 5 folders
# (one for each type of flower). It then moves the images from the original
# folders to these new folders such that 80% of the images go to the training set
# and 20% of the images go into the validation set. In the end our directory will
# have the following structure:
#
# <pre style="font-size: 10.0pt; font-family: Arial; line-height: 2; letter-spacing: 1.0pt;" >
# <b>flower_photos</b>
# |__ <b>daisy</b>
# |__ <b>dandelion</b>
# |__ <b>roses</b>
# |__ <b>sunflowers</b>
# |__ <b>tulips</b>
# |__ <b>train</b>
#     |______ <b>daisy</b>: [1.jpg, 2.jpg, 3.jpg ....]
#     |______ <b>dandelion</b>: [1.jpg, 2.jpg, 3.jpg ....]
#     |______ <b>roses</b>: [1.jpg, 2.jpg, 3.jpg ....]
#     |______ <b>sunflowers</b>: [1.jpg, 2.jpg, 3.jpg ....]
#     |______ <b>tulips</b>: [1.jpg, 2.jpg, 3.jpg ....]
#  |__ <b>val</b>
#     |______ <b>daisy</b>: [507.jpg, 508.jpg, 509.jpg ....]
#     |______ <b>dandelion</b>: [719.jpg, 720.jpg, 721.jpg ....]
#     |______ <b>roses</b>: [514.jpg, 515.jpg, 516.jpg ....]
#     |______ <b>sunflowers</b>: [560.jpg, 561.jpg, 562.jpg .....]
#     |______ <b>tulips</b>: [640.jpg, 641.jpg, 642.jpg ....]
# </pre>
#
# Since we don't delete the original folders, they will still be in our
# `flower_photos` directory, but they will be empty. The code below also prints
# the total number of flower images we have for each type of flower. 

def maybemove(a, b):
  try:
    shutil.move(a, b)
  except shutil.Error:
    pass

for cl in classes:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  num_train = int(round(len(images)*0.8))
  train, val = images[:num_train], images[num_train:]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    maybemove(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    maybemove(v, os.path.join(base_dir, 'val', cl))

round(len(images)*0.8)

# For convenience, let us set up the path for the training and validation sets

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# # Data Augmentation
# 
# Overfitting generally occurs when we have small number of training examples.
# One way to fix this problem is to augment our dataset so that it has sufficient
# number of training examples. Data augmentation takes the approach of generating
# more training data from existing training samples, by augmenting the samples
# via a number of random transformations that yield believable-looking images.
# The goal is that at training time, your model will never see the exact same
# picture twice. This helps expose the model to more aspects of the data and
# generalize better.
# 
# In **tf.keras** we can implement this using the same **ImageDataGenerator**
# class we used before. We can simply pass different transformations we would
# want to our dataset as a form of arguments and it will take care of applying it
# to the dataset during our training process. 
# 
# ## Experiment with Various Image Transformations
# 
# In this section you will get some practice doing some basic image
# transformations. Before we begin making transformations let's define our
# `batch_size` and our image size. Remember that the input to our CNN are images
# of the same size. We therefore have to resize the images in our dataset to the
# same size.

batch_size = 100
IMG_SHAPE = 150 


image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE)
                                                )

# Let's take 1 sample image from our training examples and repeat it 5 times so
# that the augmentation can be applied to the same image 5 times over randomly,
# to see the augmentation in action.

# This function will plot images in the form of a grid with 1 row and 5 columns
# where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))

# Let's take 1 sample image from our training examples and repeat it 5 times so
# that the augmentation can be applied to the same image 5 times over randomly,
# to see the augmentation in action.

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE, IMG_SHAPE)
                                                )

# Let's take 1 sample image from our training examples and repeat it 5 times so
# that the augmentation can be applied to the same image 5 times over randomly,
# to see the augmentation in action.

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# ### Put It All Together
# 
# In the cell below, use ImageDataGenerator to create a transformation that
# rescales the images by 255 and that applies:
# 
# - random 45 degree rotation
# - random zoom of up to 50%
# - random horizontal flip
# - width shift of 0.15
# - height shift of 0.15
#
# Then use the `.flow_from_directory` method to apply the above transformation to
# the images in our training set. Make sure you indicate the batch size, the path
# to the directory of the training images, the target size for the images, to
# shuffle the images, and to set the class mode to `sparse`.

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


train_data_gen = image_gen_train.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE),
                                                class_mode='sparse'
                                                )

# Let's visualize how a single image would look like 5 different times, when we
# pass these augmentations randomly to our dataset. 

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# ### Create a Data Generator for the Validation Set
#
# Generally, we only apply data augmentation to our training examples. So, in the
# cell below, use ImageDataGenerator to create a transformation that only
# rescales the images by 255. Then use the `.flow_from_directory` method to apply
# the above transformation to the images in our validation set. Make sure you
# indicate the batch size, the path to the directory of the validation images,
# the target size for the images, and to set the class mode to `sparse`. Remember
# that it is not necessary to shuffle the images in the validation set. 

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')

# Create the CNN
# 
# In the cell below, create a convolutional neural network that consists of 3
# convolution blocks. Each convolutional block contains a `Conv2D` layer followed
# by a max pool layer. The first convolutional block should have 16 filters, the
# second one should have 32 filters, and the third one should have 64 filters.
# All convolutional filters should be 3 x 3. All max pool layers should have a
# `pool_size` of `(2, 2)`.
# 
# After the 3 convolutional blocks you should have a flatten layer followed by a
# fully connected layer with 512 units. The CNN should output class probabilities
# based on 5 classes which is done by the **softmax** activation function. All
# other layers should use a **relu** activation function. You should also add
# Dropout layers with a probability of 20%, where appropriate.


model = Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE,IMG_SHAPE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(5))

# Compile the Model
# 
# In the cell below, compile your model using the ADAM optimizer, the sparse
# cross entropy function as a loss function. We would also like to look at
# training and validation accuracy on each epoch as we train our network, so make
# sure you also pass the metrics argument.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the Model
#
epochs = 80

saved = '../data/flowers.h5'
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
        steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))),
        verbose=2
    )
    model.save(saved)
    with open(historied,'wb') as h:
        pickle.dump(history,h)

# Plot Training and Validation Graphs.
# In the cell below, plot the training and validation accuracy/loss graphs.
epochs_range = range(epochs)
if history:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

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
    plt.show()

