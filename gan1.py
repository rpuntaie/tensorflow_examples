#!/usr/bin/env python3

"""
UpSampling2D

         1, 2
Input = (3, 4)

          1, 1, 2, 2
Output = (1, 1, 2, 2)
          3, 3, 4, 4
          3, 3, 4, 4
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# define input data
X = np.asarray([[1, 2],
			 [3, 4]])
# show input data for context
print(X)

# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))

# define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.UpSampling2D(input_shape=(2, 2, 1)))
#model.add(tf.keras.layers.UpSampling2D(size=(2, 3)))
#model.add(tf.keras.layers.UpSampling2D(interpolation='bilinear'))
# summarize the model
model.summary()

yhat = model.predict(X)

yhat = yhat.reshape(4,4)


