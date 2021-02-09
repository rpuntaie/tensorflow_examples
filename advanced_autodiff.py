#!/usr/bin/env python3
# https://www.tensorflow.org/guide/advanced_autodiff?hl=en

from include_tf import *


x = tf.Variable(2.0)
y = tf.Variable(3.0)

def tape1():
  with tf.GradientTape() as t:
    x_sq = x * x
    with t.stop_recording():
      y_sq = y * y
    z = x_sq + y_sq
  grad = t.gradient(z, {'x': x, 'y': y})
  print('dz/dx:', grad['x'])  # 2*x => 4
  print('dz/dy:', grad['y'])  # None

def tape2():
  with tf.GradientTape() as t:
    x_sq = x * x
    t.reset()
    y_sq = y * y
    z = x_sq + y_sq
  grad = t.gradient(z, {'x': x, 'y': y})
  print('dz/dx:', grad['x'])  # None
  print('dz/dy:', grad['y'])  # 2*y => 6

def tape3():
  with tf.GradientTape() as t:
    y_sq = y**2
    z = x**2 + tf.stop_gradient(y_sq)
  grad = t.gradient(z, {'x': x, 'y': y})
  print('dz/dx:', grad['x'])  # 4
  print('dz/dy:', grad['y'])  # None

def tape4():
  @tf.custom_gradient
  def clip_gradients(y):
    def backward(dy):
      return tf.clip_by_norm(dy, 0.5)
    return y, backward
  v = tf.Variable(2.0)
  with tf.GradientTape() as t:
    output = clip_gradients(v * v)
  print(t.gradient(output, v))  # calls "backward", which clips 4 to 2

def tape5():
  x0 = tf.constant(0.0)
  x1 = tf.constant(0.0)
  with tf.GradientTape() as t0, tf.GradientTape() as t1:
    t0.watch(x0)
    t1.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.nn.sigmoid(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)
  print(t0.gradient(ys, x0).numpy())   # cos(x) => 1.0
  print(t1.gradient(ys, x1).numpy())   # sigmoid(x1)*(1-sigmoid(x1)) => 0.25

def tape6():
  x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0
  with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
      y = x * x * x
    # Compute the gradient inside the outer `t2` context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t1.gradient(y, x)
  d2y_dx2 = t2.gradient(dy_dx, x)
  print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
  print('d2y_dx2:', d2y_dx2.numpy())  # 6 * x => 6.0

def tape7():
  x = tf.random.normal([7, 5])
  layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)
  with tf.GradientTape() as t2:
    # The inner tape only takes the gradient with respect to the input,
    # not the variables.
    with tf.GradientTape(watch_accessed_variables=False) as t1:
      t1.watch(x)
      y = layer(x)
      out = tf.reduce_sum(layer(x)**2)
    # 1. Calculate the input gradient.
    g1 = t1.gradient(out, x)
    # 2. Calculate the magnitude of the input gradient.
    g1_mag = tf.norm(g1)
  # 3. Calculate the gradient of the magnitude with respect to the model.
  dg1_mag = t2.gradient(g1_mag, layer.trainable_variables)
  print([var.shape for var in dg1_mag])

def tape8():
  x = tf.linspace(-10.0, 10.0, 200+1)
  delta = tf.Variable(0.0)
  with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x+delta)
  dy_dx = t.jacobian(y, delta)
  #When you take the Jacobian with respect to a scalar the result has the shape
  #of the target, and gives the gradient of the each element with respect to the source:
  print(y.shape)
  print(dy_dx.shape)
  plt.plot(x.numpy(), y, label='y')
  plt.plot(x.numpy(), dy_dx, label='dy/dx')
  plt.legend()
  _ = plt.xlabel('x')
  plt.show()

def tape9():
  x = tf.random.normal([7, 5])
  layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)
  with tf.GradientTape(persistent=True) as t:
    y = layer(x)
  #The shape of the Jacobian of the output with respect to the kernel is those two
  #shapes concatenated together:
  j = t.jacobian(y, layer.kernel)
  print(y.shape,layer.kernel.shape)
  print(j.shape)
  g = t.gradient(y, layer.kernel)
  print('g.shape:', g.shape)
  j_sum = tf.reduce_sum(j, axis=[0, 1])
  print('j_sum.shape:',j_sum.shape)
  delta = tf.reduce_max(abs(g - j_sum)).numpy()
  assert delta < 1e-3
  print('delta:', delta)

def imshow_zero_center(image, **kwargs):
  lim = tf.reduce_max(abs(image))
  plt.imshow(image, vmin=-lim, vmax=lim, cmap='seismic', **kwargs)
  plt.colorbar()
  plt.show()

def tape_hessian():
  x = tf.random.normal([7, 5])
  layer1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
  layer2 = tf.keras.layers.Dense(6, activation=tf.nn.relu)
  with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
      x = layer1(x)
      x = layer2(x)
      loss = tf.reduce_mean(x**2)
    g = t1.gradient(loss, layer1.kernel)
  h = t2.jacobian(g, layer1.kernel)
  print(f'layer.kernel.shape: {layer1.kernel.shape}')
  print(f'h.shape: {h.shape}')
  #To use this Hessian for a Newton's method step, you would first flatten out its
  #axes into a matrix, and flatten out the gradient into a vector:
  n_params = tf.reduce_prod(layer1.kernel.shape)
  g_vec = tf.reshape(g, [n_params, 1])
  h_mat = tf.reshape(h, [n_params, n_params])
  #The Hessian matrix should be symmetric:
  imshow_zero_center(h_mat)
  eps = 1e-3
  eye_eps = tf.eye(h_mat.shape[0])*eps
  #Note: Don't actually invert the matrix.
  # X(k+1) = X(k) - (∇²f(X(k)))^-1 @ ∇f(X(k))
  # h_mat = ∇²f(X(k))
  # g_vec = ∇f(X(k))
  update = tf.linalg.solve(h_mat + eye_eps, g_vec)
  # Reshape the update and apply it to the variable.
  _ = layer1.kernel.assign_sub(tf.reshape(update, layer1.kernel.shape))


def plot_as_patches(j):
  # Reorder axes so the diagonals will each form a contiguous patch.
  j = tf.transpose(j, [1, 0, 3, 2])
  # Pad in between each patch.
  lim = tf.reduce_max(abs(j))
  j = tf.pad(j, [[0, 0], [1, 1], [0, 0], [1, 1]],
             constant_values=-lim)
  # Reshape to form a single image.
  s = j.shape
  j = tf.reshape(j, [s[0]*s[1], s[2]*s[3]])
  imshow_zero_center(j, extent=[-0.5, s[2]-0.5, s[0]-0.5, -0.5])


def tape_batch_jacobian():
  x = tf.random.normal([7, 5])
  layer1 = tf.keras.layers.Dense(8, activation=tf.nn.elu)
  layer2 = tf.keras.layers.Dense(6, activation=tf.nn.elu)
  with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t:
    t.watch(x)
    y = layer1(x)
    y = layer2(y)
  print(y.shape) # [7, 6]
  #The full Jacobian of y with respect to x has a shape of
  #(batch, ins, batch, outs), even if you only want (batch, ins, outs).
  j = t.jacobian(y, x)
  print(j.shape) # [7, 6, 7, 5]
  #If the gradients of each item in the stack are independent, then every
  #(batch, batch) slice of this tensor is a diagonal matrix:
  imshow_zero_center(j[:, 0, :, 0])
  _ = plt.title('A (batch, batch) slice')
  plot_as_patches(j)
  _ = plt.title('All (batch, batch) slices are diagonal')
  #To get the desired result you can sum over the duplicate batch dimension, or
  #else select the diagonals using tf.einsum.
  j_sum = tf.reduce_sum(j, axis=2)
  print(j_sum.shape) #(7, 6, 5)
  j_select = tf.einsum('bxby->bxy', j)
  print(j_select.shape) #(7, 6, 5)
  #It would be much more efficient to do the calculation without the extra
  #dimension in the first place.
  #The GradientTape.batch_jacobian method does exactly that.
  jb = t.batch_jacobian(y, x)
  print(jb.shape) #(7, 6, 5)
  error = tf.reduce_max(abs(jb - j_sum))
  assert error < 1e-3
  print(error.numpy())
  x = tf.random.normal([7, 5])
  layer1 = tf.keras.layers.Dense(8, activation=tf.nn.elu)
  bn = tf.keras.layers.BatchNormalization()
  layer2 = tf.keras.layers.Dense(6, activation=tf.nn.elu)
  with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t:
    t.watch(x)
    y = layer1(x)
    y = bn(y, training=True)
    y = layer2(y)
  j = t.jacobian(y, x)
  print(f'j.shape: {j.shape}') # (7, 6, 7, 5)
  plot_as_patches(j)
  _ = plt.title('These slices are not diagonal')
  _ = plt.xlabel("Don't use `batch_jacobian`")
  #In this case batch_jacobian still runs and returns something with the
  #expected shape, but it's contents have an unclear meaning.
  jb = t.batch_jacobian(y, x)
  print(f'jb.shape: {jb.shape}')


def main():
  tape1()
  tape2()
  tape3()
  tape4()
  tape5()
  tape6()
  tape7()
  tape9()
  tape_hessian()
  tape_batch_jacobian()


if __name__ == "__main__":
  main()
