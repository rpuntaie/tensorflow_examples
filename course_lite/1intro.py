#!/usr/bin/env python3

# https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c01_linear_regression.ipynb

#Running TFLite models
## Setup

import pathlib
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

## Create a basic model of the form y = mx + c

# Create a simple Keras model.
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=200, verbose=0)

## Generate a SavedModel

export_dir = '../data/tflite_model/1'
tf.saved_model.save(model, export_dir)

## Convert the SavedModel to TFLite

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('regression/regressionmodel.tflite')
mkdir = lambda p: p.parent.mkdir(parents=True,exist_ok=True) or p
mkdir(tflite_model_file).write_bytes(tflite_model)

## Initialize the TFLite interpreter to try it out
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
inputs, outputs = [], []
for _ in range(100):
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  tflite_results = interpreter.get_tensor(output_details[0]['index'])
  # Test the TensorFlow model on random input data.
  tf_results = model(tf.constant(input_data))
  output_data = np.array(tf_results)
  inputs.append(input_data[0][0])
  outputs.append(output_data[0][0])

## Visualize the model
plt.plot(inputs, outputs, 'r')
plt.show()

## Download the TFLite model file
# try:
#   from google.colab import files
#   files.download(tflite_model_file)
# except:
#   pass
