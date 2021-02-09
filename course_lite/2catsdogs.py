#!/usr/bin/env python3

# https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c02_transfer_learning.ipynb

# Transfer Learning with TensorFlow Hub for TFLite
## Setup 

import os
import pathlib
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

## Select the Hub/TF2 module to use
#Hub modules for TF 1.x won't work here, please use one of the selections provided.

module_selection = ("mobilenet_v2", 224, 1280) #@param ["(\"mobilenet_v2\", 224, 1280)", "(\"inception_v3\", 299, 2048)"] {type:"raw", allow-input: true}
handle_base, pixels, FV_SIZE = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {} and output dimension {}".format(
  MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

## Data preprocessing
# Use [TensorFlow Datasets](http://tensorflow.org/datasets) to load the cats and
# dogs dataset.  This `tfds` package is the easiest way to load pre-defined data.
# If you have your own data, and are interested in importing using it with
# TensorFlow see [loading image data](../load_data/images.ipynb)

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# The `tfds.load` method downloads and caches the data, and returns a
# `tf.data.Dataset` object. These objects provide powerful, efficient methods for
# manipulating data and piping it into your model.
# Since `"cats_vs_dog"` doesn't define standard splits, use the subsplit feature
# to divide it into (train, validation, test) with 80%, 10%, 10% of the data
# respectively.

(train_examples, validation_examples, test_examples), info = tfds.load(
    'cats_vs_dogs',
    split=['train[80%:]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True, 
    as_supervised=True, 
)
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

### Format the Data
# Use the `tf.image` module to format the images for the task.
# Resize the images to a fixes input size, and rescale the input channels

def format_image(image, label):
  image = tf.image.resize(image, IMAGE_SIZE) / 255.0
  return  image, label

#Now shuffle and batch the data

BATCH_SIZE = 32 #@param {type:"integer"}

train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)

#Inspect a batch
for image_batch, label_batch in train_batches.take(1):
  pass
#image_batch.shape

## Defining the model
# All it takes is to put a linear classifier on top of the
# `feature_extractor_layer` with the Hub module.
# For speed, we start out with a non-trainable `feature_extractor_layer`, but you
# can also enable fine-tuning for greater accuracy.

do_fine_tuning = False #@param {type:"boolean"}

#Load TFHub Module

feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,), 
                                   output_shape=[FV_SIZE],
                                   trainable=do_fine_tuning)
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(num_classes)
])
model.summary()

#@title (Optional) Unfreeze some layers
NUM_LAYERS = 7 #@param {type:"slider", min:1, max:50, step:1}
if do_fine_tuning:
  feature_extractor.trainable = True
  for layer in model.layers[-NUM_LAYERS:]:
    layer.trainable = True
else:
  feature_extractor.trainable = False

## Training the model

if do_fine_tuning:
  model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.002, momentum=0.9), 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
else:
  model.compile(
    optimizer='adam', 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

EPOCHS = 5
hist = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    verbose=0)

## Export the model
CATS_VS_DOGS_SAVED_MODEL = "catsdogsdata/1"

#Export the SavedModel
tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)

#! saved_model_cli show --dir catsdogsdata --tag_set serve --signature_def serving_default

loaded = tf.saved_model.load(CATS_VS_DOGS_SAVED_MODEL)

print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
print(infer.structured_input_signature)
print(infer.structured_outputs)

## Convert using TFLite's Converter

#Load the TFLiteConverter with the SavedModel

converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)

### Post-training quantization
# The simplest form of post-training quantization quantizes weights from floating
# point to 8-bits of precision. This technique is enabled as an option in the
# TensorFlow Lite converter. At inference, weights are converted from 8-bits of
# precision to floating point and computed using floating-point kernels. This
# conversion is done once and cached to reduce latency.
# To further improve latency, hybrid operators dynamically quantize activations
# to 8-bits and perform computations with 8-bit weights and activations. This
# optimization provides latencies close to fully fixed-point inference. However,
# the outputs are still stored using floating point, so that the speedup with
# hybrid ops is less than a full fixed-point computation.

converter.optimizations = [tf.lite.Optimize.DEFAULT]

### Post-training integer quantization
# We can get further latency improvements, reductions in peak memory usage, and
# access to integer only hardware accelerators by making sure all model math is
# quantized. To do this, we need to measure the dynamic range of activations and
# inputs with a representative data set. You can simply create an input data
# generator and provide it to our converter.

def representative_data_gen():
  for input_value, _ in test_batches.take(100):
    yield [input_value]

converter.representative_dataset = representative_data_gen

# The resulting model will be fully quantized but still take float input and
# output for convenience.
# Ops that do not have quantized implementations will automatically be left in
# floating point. This allows conversion to occur smoothly but may restrict
# deployment to accelerators that support float. 

### Full integer quantization
# To require the converter to only output integer operations, one can specify:

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

### Finally convert the model

mkdir = lambda p: p.parent.mkdir(parents=True,exist_ok=True) or p
tflite_model = converter.convert()
tflite_model_file = pathlib.Path('catsdogsdata/catsdogs.tflite')
mkdir(tflite_model_file).write_bytes(tflite_model)
##Test the TFLite model using the Python Interpreter

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

from tqdm import tqdm

predictions = []

test_labels, test_imgs = [], []
for img, label in tqdm(test_batches.take(10)):
  interpreter.set_tensor(input_index, img)
  interpreter.invoke()
  predictions.append(interpreter.get_tensor(output_index))
  test_labels.append(label.numpy()[0])
  test_imgs.append(img)

#@title Utility functions for plotting
# Utilities for plotting

class_names = ['cat', 'dog']

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  img = np.squeeze(img)
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


# NOTE: Colab runs on server CPUs. At the time of writing this, TensorFlow Lite
# doesn't have super optimized server CPU kernels. For this reason post-training
# full-integer quantized models  may be slower here than the other kinds of
# optimized models. But for mobile CPUs, considerable speedup can be observed.

#@title Visualize the outputs { run: "auto" }
index = 0 #@param {type:"slider", min:0, max:9, step:1}
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(index, predictions, test_labels, test_imgs)
plt.show()

# # Download the model.
# # **NOTE: You might have to run to the cell below twice**
# 
# labels = ['cat', 'dog']
# with open('catsdogsdata/catsdogslabels.txt', 'w') as f:
#   f.write('\n'.join(labels))
# try:
#   from google.colab import files
#   files.download('catsdogsdata/catsdogs.tflite')
#   files.download('catsdogsdata/catsdogslabels.txt')
# except:
#   pass
# 
# # Prepare the test images for download (Optional)
# # This part involves downloading additional test images for the Mobile Apps only
# # in case you need to try out more samples
# 
# #!mkdir -p test_images
# 
# from PIL import Image
# 
# for index, (image, label) in enumerate(test_batches.take(50)):
#   image = tf.cast(image * 255.0, tf.uint8)
#   image = tf.squeeze(image).numpy()
#   pil_image = Image.fromarray(image)
#   pil_image.save('test_images/{}_{}.jpg'.format(class_names[label[0]], index))
# 
# # !ls test_images
# 
# # !zip -qq cats_vs_dogs_test_images.zip -r test_images/
# 
# try:
#   files.download('cats_vs_dogs_test_images.zip')
# except:
#   pass
