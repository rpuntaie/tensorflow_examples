#!/usr/bin/env python3
#https://www.tensorflow.org/guide/eager?hl=en

from include_tf import *

## Setup and basic usage

# tf.executing_eagerly() # is default

def eager1():
  a = tf.constant([[1, 2],
                   [3, 4]])
  print(a)
  # Broadcasting support
  b = tf.add(a, 1)
  print(b)
  # Operator overloading is supported
  print(a * b)
  c = np.multiply(a, b)
  print(c)
  # Obtain numpy value from a tensor:
  print(a.numpy())

## Dynamic control flow

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1

## Eager training
### Computing gradients

#Automatic differentiation
def eager2():
  w = tf.Variable([[1.0]])
  with tf.GradientTape() as tape:
    loss = w * w
  grad = tape.gradient(loss, w)
  print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)

### Train a model

#MNIST handwritten digits
def eager3():
  (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
  dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
  dataset = dataset.shuffle(1000).batch(32)
  mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                           input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
  ])
  for images,labels in dataset.take(1):
    print("Logits: ", mnist_model(images[0:1]).numpy())
  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  loss_history = []
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      logits = mnist_model(images, training=True)
      # Add asserts to check the shape of the output.
      tf.debugging.assert_equal(logits.shape, (32, 10))
      loss_value = loss_object(labels, logits)
    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
  def train(epochs):
    for epoch in range(epochs):
      for (batch, (images, labels)) in enumerate(dataset):
        train_step(images, labels)
      print ('Epoch {} finished'.format(epoch))
  train(epochs = 3)
  plt.plot(loss_history)
  plt.xlabel('Batch #')
  plt.ylabel('Loss [entropy]')
  plt.show()

### Variables and optimizers
def eager4():
  class Linear(tf.keras.Model):
    def __init__(self):
      super(Linear, self).__init__()
      self.W = tf.Variable(5., name='weight')
      self.B = tf.Variable(10., name='bias')
    def call(self, inputs):
      return inputs * self.W + self.B
  NUM_EXAMPLES = 2000
  training_inputs = tf.random.normal([NUM_EXAMPLES])
  noise = tf.random.normal([NUM_EXAMPLES])
  training_outputs = training_inputs * 3 + 2 + noise
  # The loss function to be optimized
  def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))
  def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])
  model = Linear()
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
  steps = 300
  for i in range(steps):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 20 == 0:
      print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))
  print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
  print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
  model.save_weights('weights')
  status = model.load_weights('weights')
  x = tf.Variable(10.)
  checkpoint = tf.train.Checkpoint(x=x)
  x.assign(2.)   # Assign a new value to the variables and save.
  checkpoint_path = './data/ckpt/'
  checkpoint.save('./data/ckpt/')
  x.assign(11.)  # Change the variable after saving.
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
  print(x)  # => 2.0

def eager5():
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  checkpoint_dir = './data/ckpt/'
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  root = tf.train.Checkpoint(optimizer=optimizer,model=model)
  root.save(checkpoint_prefix)
  root.restore(tf.train.latest_checkpoint(checkpoint_dir))

### Object-oriented metrics
### Summaries and TensorBoard
def eager6():
  m = tf.keras.metrics.Mean("loss")
  m(0)
  m(5)
  m.result()  # => 2.5
  m([8, 9])
  m.result()  # => 5.5
  logdir = "./data/tb/"
  writer = tf.summary.create_file_writer(logdir)
  steps = 1000
  with writer.as_default():  # or call writer.set_as_default() before the loop.
    for i in range(steps):
      step = i + 1
      # Calculate loss with your real train function.
      loss = 1 - 0.001 * step
      if step % 100 == 0:
        tf.summary.scalar('loss', loss, step=step)
  """
  ls ./data/tb/
  """

## Advanced automatic differentiation topics
### Dynamic models
def eager7():
  def line_search_step(fn, init_x, rate=1.0):
    with tf.GradientTape() as tape:
      # Variables are automatically tracked.
      # But to calculate a gradient from a tensor, you must `watch` it.
      tape.watch(init_x)
      value = fn(init_x)
    grad = tape.gradient(value, init_x)
    grad_norm = tf.reduce_sum(grad * grad)
    init_value = value
    while value > init_value - rate * grad_norm:
      x = init_x - rate * grad
      value = fn(x)
      rate /= 2.0
    return x, value

### Custom gradients
def eager8():
  def log1pexp(x):
    return tf.math.log(1 + tf.exp(x))
  def grad_log1pexp(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      value = log1pexp(x)
    return tape.gradient(value, x)
  grad_log1pexp(tf.constant(0.)).numpy()
  # However, x = 100 fails because of numerical instability.
  grad_log1pexp(tf.constant(100.)).numpy()
  @tf.custom_gradient
  def log1pexp(x):
    e = tf.exp(x)
    def grad(dy):
      return dy * (1 - 1 / (1 + e))
    return tf.math.log(1 + e), grad
  def grad_log1pexp(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      value = log1pexp(x)
    return tape.gradient(value, x)
  grad_log1pexp(tf.constant(0.)).numpy()
  # And the gradient computation also works at x = 100.
  grad_log1pexp(tf.constant(100.)).numpy()

## Performance
def eager9():
  import time
  def measure(x, steps):
    # TensorFlow initializes a GPU the first time it's used, exclude from timing.
    tf.matmul(x, x)
    start = time.time()
    for i in range(steps):
      x = tf.matmul(x, x)
    # tf.matmul can return before completing the matrix multiplication
    # (e.g., can return after enqueing the operation on a CUDA stream).
    # The x.numpy() call below will ensure that all enqueued operations
    # have completed (and will also copy the result to host memory,
    # so we're including a little more than just the matmul operation
    # time).
    _ = x.numpy()
    end = time.time()
    return end - start
  shape = (1000, 1000)
  steps = 200
  print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))
  # Run on CPU:
  with tf.device("/cpu:0"):
    print("CPU: {} secs".format(measure(tf.random.normal(shape), steps)))
  # Run on GPU, if available:
  if tf.config.experimental.list_physical_devices("GPU"):
    with tf.device("/gpu:0"):
      print("GPU: {} secs".format(measure(tf.random.normal(shape), steps)))
  else:
    print("GPU: not found")
  if tf.config.experimental.list_physical_devices("GPU"):
    x = tf.random.normal([10, 10])
    x_gpu0 = x.gpu()
    x_cpu = x.cpu()
    _ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
    _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

def main():
  fizzbuzz(15)
  eager1()
  eager2()
  eager3()
  eager4()
  eager5()
  eager6()
  eager7()
  eager8()
  eager9()

if __name__ == "__main__":
  main()
