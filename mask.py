#!/usr/bin/env python

"""
Chain models.

Masking.

Show output of layer.
"""

import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Masking, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model

X_train = np.random.rand(4,3,2)
Dense_unit = 1
dense_reg = 0.01
mdl = Sequential()
mdl.add(Input(shape=(X_train.shape[1],X_train.shape[2]),name='input_feature'))
mdl.add(Masking(mask_value=0,name='masking'))
mdl.add(Dense(Dense_unit,kernel_regularizer=l2(dense_reg),activation='relu',name='output_feature'))
mdl.summary()
#this is the same as chaining models
mdl2mask = Model(inputs=mdl.input,outputs=mdl.get_layer("masking").output)
mdl2mask.compile()
mdl.compile()
maskoutput = mdl2mask.predict(X_train)
mdloutput = mdl.predict(X_train)
print(maskoutput) # print output after/of masking
print(mdloutput) # print output of mdl
print(maskoutput.shape) #(4, 3, 2): masking has the shape of the layer before (input here)
print(mdloutput.shape) #(4, 3, 1): shape of the output of dense

