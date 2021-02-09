#!/usr/bin/env python3

#https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

from include_tf import *

class CustomModel1(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def customfit1():
  # Construct and compile an instance of CustomModel1
  inputs = keras.Input(shape=(32,))
  outputs = keras.layers.Dense(1)(inputs)
  model = CustomModel1(inputs, outputs)
  model.compile(optimizer="adam", loss="mse", metrics=["mae"])

  # Just use `fit` as usual
  x = np.random.random((1000, 32))
  y = np.random.random((1000, 1))
  model.fit(x, y, epochs=3)


loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")


class CustomModel2(keras.Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = keras.losses.mean_squared_error(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mae_metric]


def customfit2():
  # Construct an instance of CustomModel2
  inputs = keras.Input(shape=(32,))
  outputs = keras.layers.Dense(1)(inputs)
  model = CustomModel2(inputs, outputs)

  # We don't passs a loss or metrics here.
  model.compile(optimizer="adam")

  # Just use `fit` as usual -- you can use callbacks, etc.
  x = np.random.random((1000, 32))
  y = np.random.random((1000, 1))
  model.fit(x, y, epochs=5)

class CustomModel3(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def customfit3():
  # Construct and compile an instance of CustomModel3
  inputs = keras.Input(shape=(32,))
  outputs = keras.layers.Dense(1)(inputs)
  model = CustomModel3(inputs, outputs)
  model.compile(optimizer="adam", loss="mse", metrics=["mae"])

  # You can now use sample_weight argument
  x = np.random.random((1000, 32))
  y = np.random.random((1000, 1))
  sw = np.random.random((1000, 1))
  model.fit(x, y, sample_weight=sw, epochs=3)

class CustomModel4(keras.Model):
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

def customfit4():

  # Construct an instance of CustomModel4
  inputs = keras.Input(shape=(32,))
  outputs = keras.layers.Dense(1)(inputs)
  model = CustomModel4(inputs, outputs)
  model.compile(loss="mse", metrics=["mae"])

  # Evaluate with our custom test_step
  x = np.random.random((1000, 32))
  y = np.random.random((1000, 1))
  model.evaluate(x, y)



# Create the discriminator
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator
latent_dim = 128
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


def customfit5():

  # Prepare the dataset. We use both the training & test MNIST digits.
  batch_size = 64
  (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
  all_digits = np.concatenate([x_train, x_test])
  all_digits = all_digits.astype("float32") / 255.0
  all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
  dataset = tf.data.Dataset.from_tensor_slices(all_digits)
  dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

  gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
  gan.compile(
      d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
      g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
      loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
  )

  # To limit the execution time, we only train on 100 batches. You can train on
  # the entire dataset. You will need about 20 epochs to get nice results.
  gan.fit(dataset.take(100), epochs=1)


def main():
  customfit1()
  customfit2()
  customfit3()
  customfit4()
  customfit5()

if __name__ == "__main__":
  main()

