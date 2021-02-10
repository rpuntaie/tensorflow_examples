#!/usr/bin/env python3

from include_tf import *

class CustomModel1(keras.Model):
    def __init__(self):
        super().__init__(self)
        self.a = tf.Variable([1.0*i for i in range(3)])
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as t:
            y_pred = self.a[2]*x*x+self.a[1]*x+self.a[0]
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

def fitquadratic():
  #tb_callback = keras.callbacks.TensorBoard(log_dir='./data/log_dir', profile_batch='10, 15')
  model = CustomModel1()
  model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  n=500 #todo: how to make this converge faster?
  x = np.random.random((n, ))
  a,b,c = 2,3,4
  y = a*x*x+b*x+c
  model.fit(x,y, epochs=n, verbose=2
            #, callbacks=[tb_callback]
            )
  print((a,b,c),model.a.numpy())

def main():
  fitquadratic()

if __name__ == "__main__":
  main()

