#!/usr/bin/env python3
# https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c03_exercise_convert_model_to_tflite.ipynb
# Train Your Own Model and Convert It to TFLite

# This notebook uses the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
# dataset which contains 70,000 grayscale images in 10 categories. The images
# show individual articles of clothing at low resolution (28 by 28 pixels), as
# seen here:
# Fashion MNIST is intended as a drop-in replacement for the classic
# [MNIST](http://yann.lecun.com/exdb/mnist/) dataset—often used as the "Hello,
# World" of machine learning programs for computer vision. The MNIST dataset
# contains images of handwritten digits (0, 1, 2, etc.) in a format identical to
# that of the articles of clothing we'll use here.
# This uses Fashion MNIST for variety, and because it's a slightly more
# challenging problem than regular MNIST. Both datasets are relatively small and
# are used to verify that an algorithm works as expected. They're good starting
# points to test and debug code.
# We will use 60,000 images to train the network and 10,000 images to evaluate
# how accurately the network learned to classify images. You can access the
# Fashion MNIST directly from TensorFlow. Import and load the Fashion MNIST data
# directly from TensorFlow:

# Setup

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
print(tf.__version__)

# Download Fashion MNIST Dataset
(train_examples, validation_examples, test_examples), info = tfds.load(
    'fashion_mnist', with_info=True, as_supervised=True,
    split=('train[:80%]', 'train[80%:]', 'test'))
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

with open('fashiondata/labels.txt', 'w') as f:
  f.write('\n'.join(class_names))

IMG_SIZE = 28

# Preprocessing data
## Preprocess
# Write a function to normalize and resize the images
def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  image = image / 255.0
  return image, label

# Set the batch size to 32
BATCH_SIZE = 32

## Create a Dataset from images and labels
# Prepare the examples by preprocessing them and then batching them (and
# optionally prefetching them)

asis=tf.autograph.experimental.do_not_convert
# If you wish you can shuffle train set here
train_batches = train_examples.cache().shuffle(
  num_examples//4).batch(BATCH_SIZE).map(asis(format_example)).prefetch(1)
validation_batches = validation_examples.cache().batch(
  BATCH_SIZE).map(asis(format_example)).prefetch(1)
test_batches = test_examples.cache().batch(1).map(asis(format_example))

# Building the model
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 16)        160       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 32)        4640      
_________________________________________________________________
flatten (Flatten)            (None, 3872)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                247872    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 253,322
Trainable params: 253,322
Non-trainable params: 0
"""

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10)
])

# Set the loss and accuracy metrics
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

export_dir = '../data/fashiondata/1'
if not os.path.exists(export_dir):
  model.fit(train_batches, 
            epochs=10,
            validation_data=validation_batches,
            verbose=0)
  tf.saved_model.save(model, export_dir)

mode = "Speed" #@param ["Default", "Storage", "Speed"]
if mode == 'Storage':
  optimization = tf.lite.Optimize.OPTIMIZE_FOR_SIZE
elif mode == 'Speed':
  optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
else:
  optimization = tf.lite.Optimize.DEFAULT

# Invoke the converter to finally generate the TFLite model
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [optimization]
tflite_model = converter.convert()

tflite_model_file = '../data/fashiondata/model.tflite'

with open(tflite_model_file, "wb") as f:
  f.write(tflite_model)

# Test if your model is working

# ==== problem with input format
# https://stackoverflow.com/questions/50764572/how-can-i-test-a-tflite-model-to-prove-that-it-behaves-as-the-original-model-us
# https://stackoverflow.com/questions/50443411/how-to-load-a-tflite-model-in-script/51986982#51986982

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

# ==== continue original tutorial

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Gather results for the randomly sampled test images
predictions = []
test_labels = []
test_images = []

for img, label in test_batches.take(50):
  #img, label = tuple(test_batches.take(1))[0]
  interpreter.set_tensor(input_index, img)
  interpreter.invoke()
  predictions.append(interpreter.get_tensor(output_index))
  test_labels.append(label[0])
  test_images.append(np.array(img))

#@title Utility functions for plotting
# Utilities for plotting

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  img = np.squeeze(img)
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label.numpy():
    color = 'green'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks(list(range(10)), class_names, rotation='vertical')
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array[0], color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array[0])
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')

#@title Visualize the outputs { run: "auto" }
index = 49 #@param {type:"slider", min:1, max:50, step:1}
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(index, predictions, test_labels, test_images)
plt.show()
plot_value_array(index, predictions, test_labels)
plt.show()

# # Download TFLite model and assets
# 
# **NOTE: You might have to run to the cell below twice**
# try:
#   from google.colab import files
#   files.download(tflite_model_file)
#   files.download('../data/fashiondata/labels.txt')
# except:
#   pass
# # Deploying TFLite model
# 
# # Now once you've the trained TFLite model downloaded, you can ahead and deploy
# # this on an Android/iOS application by placing the model assets in the
# # appropriate location.
# 
# # Prepare the test images for download (Optional)
# 
# # !mkdir -p test_images
# 
# from PIL import Image
# 
# for index, (image, label) in enumerate(test_batches.take(50)):
#   image = tf.cast(image * 255.0, tf.uint8)
#   image = tf.squeeze(image).numpy()
#   pil_image = Image.fromarray(image)
#   pil_image.save('test_images/{}_{}.jpg'.format(class_names[label[0]].lower(), index))
# 
# # !ls test_images
# 
# # !zip -qq fmnist_test_images.zip -r test_images/
# 
# try:
#   files.download('fmnist_test_images.zip')
# except:
#   pass
