#!/usr/bin/env python3

# https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide

"""
Welcome to the comprehensive guide for Keras weight pruning.

This page documents various use cases and shows how to use the API for each one. Once you know which APIs you need, find the parameters and the low-level details in the
[API docs](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity).

*  If you want to see the benefits of pruning and what's supported, see the [overview](https://www.tensorflow.org/model_optimization/guide/pruning).
*  For a single end-to-end example, see the [pruning example](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras).

The following use cases are covered:
* Define and train a pruned model.
   * Sequential and Functional.
   * Keras model.fit and custom training loops
* Checkpoint and deserialize a pruned model.
* Deploy a pruned model and see compression benefits.

For configuration of the pruning algorithm, refer to the `tfmot.sparsity.keras.prune_low_magnitude` API docs.

## Setup
For finding the APIs you need and understanding purposes, you can run but skip reading this section.

! pip install -q tensorflow-model-optimization

"""


import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

# %load_ext tensorboard

import tempfile

input_shape = [20]
x_train = np.random.randn(1, 20).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randn(1), num_classes=20)

def setup_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(20, input_shape=input_shape),
      tf.keras.layers.Flatten()
  ])
  return model

def setup_pretrained_weights():
  model = setup_model()

  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
  )

  model.fit(x_train, y_train)

  _, pretrained_weights = tempfile.mkstemp('.tf')

  model.save_weights(pretrained_weights)

  return pretrained_weights

def get_gzipped_model_size(model):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)

  return os.path.getsize(zipped_file)

setup_model()
pretrained_weights = setup_pretrained_weights()

"""
## Define model
### Prune whole model (Sequential and Functional)

**Tips for better model accuracy:**
* Try "Prune some layers" to skip pruning the layers that reduce accuracy the most.
* It's generally better to finetune with pruning as opposed to training from scratch.

To make the whole model train with pruning, apply
`tfmot.sparsity.keras.prune_low_magnitude` to the model.
"""

base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended.
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)
model_for_pruning.summary()

"""
### Prune some layers (Sequential and Functional)

Pruning a model can have a negative effect on accuracy. You can selectively
prune layers of a model to explore the trade-off between accuracy, speed, and
model size.

**Tips for better model accuracy:**
* It's generally better to finetune with pruning as opposed to training from scratch.
* Try pruning the later layers instead of the first layers.
* Avoid pruning critical layers (e.g. attention mechanism).

**More**:
* The `tfmot.sparsity.keras.prune_low_magnitude` API docs provide details on how to vary the pruning configuration per layer.

In the example below, prune only the `Dense` layers.
"""

# Create a base model
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy

# Helper function uses `prune_low_magnitude` to make only the 
# Dense layers train with pruning.
def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer

# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
# to the layers of the model.
model_for_pruning = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_pruning_to_dense,
)

model_for_pruning.summary()


"""
While this example used the type of the layer to decide what to prune, the
easiest way to prune a particular layer is to set its `name` property, and look
for that name in the `clone_function`.
"""

print(base_model.layers[0].name)

"""
#### More readable but potentially lower model accuracy

This is not compatible with fine-tuning with pruning, which is why it may be
less accurate than the above examples which support fine-tuning.

While `prune_low_magnitude` can be applied while defining the initial model,
loading the weights after does not work in the below examples.

**Functional example**
"""

# Use `prune_low_magnitude` to make the `Dense` layer train with pruning.
i = tf.keras.Input(shape=(20,))
x = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(10))(i)
o = tf.keras.layers.Flatten()(x)
model_for_pruning = tf.keras.Model(inputs=i, outputs=o)

model_for_pruning.summary()

#**Sequential example**


# Use `prune_low_magnitude` to make the `Dense` layer train with pruning.
model_for_pruning = tf.keras.Sequential([
  tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(20, input_shape=input_shape)),
  tf.keras.layers.Flatten()
])

model_for_pruning.summary()

"""
### Prune custom Keras layer or modify parts of layer to prune

**Common mistake:** pruning the bias usually harms model accuracy too much.

`tfmot.sparsity.keras.PrunableLayer` serves two use cases:
1. Prune a custom Keras layer
2. Modify parts of a built-in Keras layer to prune.

For an example, the API defaults to only pruning the kernel of the
`Dense` layer. The example below prunes the bias also.

"""

class MyDenseLayer(tf.keras.layers.Dense, tfmot.sparsity.keras.PrunableLayer):

  def get_prunable_weights(self):
    # Prune bias also, though that usually harms model accuracy too much.
    return [self.kernel, self.bias]

# Use `prune_low_magnitude` to make the `MyDenseLayer` layer train with pruning.
model_for_pruning = tf.keras.Sequential([
  tfmot.sparsity.keras.prune_low_magnitude(MyDenseLayer(20, input_shape=input_shape)),
  tf.keras.layers.Flatten()
])

model_for_pruning.summary()


"""
## Train model
### Model.fit
Call the `tfmot.sparsity.keras.UpdatePruningStep` callback during training. 
To help debug training, use the `tfmot.sparsity.keras.PruningSummaries` callback.
"""

# Define the model.
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

log_dir = tempfile.mkdtemp()
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Log sparsity and other metrics in Tensorboard.
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
]

model_for_pruning.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
)

model_for_pruning.fit(
    x_train,
    y_train,
    callbacks=callbacks,
    epochs=2,
)

#docs_infra: no_execute
#%tensorboard --logdir={log_dir}

"""
For non-Colab users, you can see
[the results of a previous run]
(https://tensorboard.dev/experiment/XiNXEBjHQ3Oabc6jRLKiXQ/#scalars&_smoothingWeight=0)
of this code block on [TensorBoard.dev](https://tensorboard.dev/).

### Custom training loop

Call the `tfmot.sparsity.keras.UpdatePruningStep` callback during training. 

To help debug training, use the `tfmot.sparsity.keras.PruningSummaries` callback.
"""

# Define the model.
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

# Boilerplate
loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()
log_dir = tempfile.mkdtemp()
unused_arg = -1
epochs = 2
batches = 1 # example is hardcoded so that the number of batches cannot change.

# Non-boilerplate.
model_for_pruning.optimizer = optimizer
step_callback = tfmot.sparsity.keras.UpdatePruningStep()
step_callback.set_model(model_for_pruning)
log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir) # Log sparsity and other metrics in Tensorboard.
log_callback.set_model(model_for_pruning)

step_callback.on_train_begin() # run pruning callback
for _ in range(epochs):
  log_callback.on_epoch_begin(epoch=unused_arg) # run pruning callback
  for _ in range(batches):
    step_callback.on_train_batch_begin(batch=unused_arg) # run pruning callback

    with tf.GradientTape() as tape:
      logits = model_for_pruning(x_train, training=True)
      loss_value = loss(y_train, logits)
      grads = tape.gradient(loss_value, model_for_pruning.trainable_variables)
      optimizer.apply_gradients(zip(grads, model_for_pruning.trainable_variables))

  step_callback.on_epoch_end(batch=unused_arg) # run pruning callback

#docs_infra: no_execute
#%tensorboard --logdir={log_dir}

"""
For non-Colab users, you can see
[the results of a previous run]
(https://tensorboard.dev/experiment/jDeGzF3xQeSyb7Qir1ZcBQ/#scalars&_smoothingWeight=0)
 of this code block on [TensorBoard.dev](https://tensorboard.dev/).

### Improve pruned model accuracy

First, look at the `tfmot.sparsity.keras.prune_low_magnitude` API docs
to understand what a pruning schedule is and the math of
each type of pruning schedule.

**Tips**:

* Have a learning rate that's not too high or too low when the model is
pruning. Consider the [pruning
schedule](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PruningSchedule)
to be a hyperparameter.

* As a quick test, try experimenting with pruning a model to the final sparsity
at the begining of training by setting `begin_step` to 0 with a
`tfmot.sparsity.keras.ConstantSparsity` schedule. You might get lucky with good
results.

* Do not prune very frequently to give the model time to recover. The [pruning
schedule](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PruningSchedule)
provides a decent default frequency.

* For general ideas to improve model accuracy, look for tips for your use
case(s) under "Define model".

## Checkpoint and deserialize

You must preserve the optimizer step during checkpointing. This means while you
can use Keras HDF5 models for checkpointing, you cannot use Keras HDF5 weights.
"""

base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

_, keras_model_file = tempfile.mkstemp('.h5')

# Checkpoint: saving the optimizer is necessary (include_optimizer=True is the default).
model_for_pruning.save(keras_model_file, include_optimizer=True)

# The above applies generally. The code below is only needed for the HDF5 model
# format (not HDF5 weights and other formats).

# Deserialize model.
with tfmot.sparsity.keras.prune_scope():
  loaded_model = tf.keras.models.load_model(keras_model_file)

loaded_model.summary()

"""
## Deploy pruned model

### Export model with size compression

**Common mistake**: both `strip_pruning` and applying a standard compression
algorithm (e.g. via gzip) are necessary to see the compression
benefits of pruning.
"""

# Define the model.
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

# Typically you train the model here.

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

print("final model")
model_for_export.summary()

print("\n")
print("Size of gzipped pruned model without stripping: %.2f bytes" % (get_gzipped_model_size(model_for_pruning)))
print("Size of gzipped pruned model with stripping: %.2f bytes" % (get_gzipped_model_size(model_for_export)))

"""
### Hardware-specific optimizations

Once different backends [enable pruning to improve
latency]((https://github.com/tensorflow/model-optimization/issues/173)), using
block sparsity can improve latency for certain hardware.

Increasing the block size will decrease the peak sparsity that's achievable for
a target model accuracy. Despite this, latency can still improve.

For details on what's supported for block sparsity, see
the `tfmot.sparsity.keras.prune_low_magnitude` API docs.
"""

base_model = setup_model()

# For using intrinsics on a CPU with 128-bit registers, together with 8-bit
# quantized weights, a 1x16 block size is nice because the block perfectly
# fits into the register.
pruning_params = {'block_size': [1, 16]}
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)

model_for_pruning.summary()
