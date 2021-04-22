#!/usr/bin/env python3
#https://www.tensorflow.org/guide/estimator
##!!! 20210209 Some problems here

from include_tf import *
import tempfile
import tensorflow_datasets as tfds

def train_input_fn():
  titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
  titanic = tf.data.experimental.make_csv_dataset(
      titanic_file, batch_size=32,
      label_name="survived")
  titanic_batches = (
      titanic.cache().repeat().shuffle(500)
      .prefetch(tf.data.AUTOTUNE))
  return titanic_batches

def estimator1():
  age = tf.feature_column.numeric_column('age')
  cls = tf.feature_column.categorical_column_with_vocabulary_list('class', ['First', 'Second', 'Third']) 
  embark = tf.feature_column.categorical_column_with_hash_bucket('embark_town', 32)
  model_dir = tempfile.mkdtemp()
  model = tf.estimator.LinearClassifier(
      model_dir=model_dir,
      feature_columns=[embark, cls, age],
      n_classes=2
  )
  model = model.train(input_fn=train_input_fn, steps=100)
  result = model.evaluate(train_input_fn, steps=10)
  for key, value in result.items():
    print(key, ":", value)
  for pred in model.predict(train_input_fn):
    for key, value in pred.items():
      print(key, ":", value)
    break
  keras_mobilenet_v2 = tf.keras.applications.MobileNetV2(
      input_shape=(160, 160, 3), include_top=False)
  keras_mobilenet_v2.trainable = False
  estimator_model = tf.keras.Sequential([
      keras_mobilenet_v2,
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(1)
  ])
  estimator_model.compile(
      optimizer='adam',
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=['accuracy'])

def estimator2():
  import tensorflow.compat.v1 as tf_compat
  def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(
      dict(x=inputs, y=labels)).repeat().batch(2)
  class Net(tf.keras.Model):
    """A simple linear model."""
    def __init__(self):
      super(Net, self).__init__()
      self.l1 = tf.keras.layers.Dense(5)
    def call(self, x):
      return self.l1(x)
  def model_fn(features, labels, mode):
    net = Net()
    opt = tf.keras.optimizers.Adam(0.1)
    ckpt = tf.train.Checkpoint(step=tf_compat.train.get_global_step(),
                               optimizer=opt, net=net)
    with tf.GradientTape() as tape:
      output = net(features['x'])
      loss = tf.reduce_mean(tf.abs(output - features['y']))
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    return tf.estimator.EstimatorSpec(
      mode,
      loss=loss,
      train_op=tf.group(opt.apply_gradients(zip(gradients, variables)),
                        ckpt.step.assign_add(1)),
      # Tell the Estimator to save "ckpt" in an object-based format.
      scaffold=tf_compat.train.Scaffold(saver=ckpt))
  tf.keras.backend.clear_session()
  est = tf.estimator.Estimator(model_fn, './tf_estimator_example/')
  est.train(toy_dataset, steps=10)
  opt = tf.keras.optimizers.Adam(0.1)
  net = Net()
  ckpt = tf.train.Checkpoint(
    step=tf.Variable(1, dtype=tf.int64), optimizer=opt, net=net)
  ckpt.restore(tf.train.latest_checkpoint('./tf_estimator_example/'))
  ckpt.step.numpy()  # From est.train(..., steps=10)
  input_column = tf.feature_column.numeric_column("x")
  estimator = tf.estimator.LinearClassifier(feature_columns=[input_column])
  def input_fn():
    return tf.data.Dataset.from_tensor_slices(
      ({"x": [1., 2., 3., 4.]}, [1, 1, 0, 0])).repeat(200).shuffle(64).batch(16)
  estimator.train(input_fn)
  tmpdir = tempfile.mkdtemp()
  serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec([input_column]))
  estimator_base_path = os.path.join(tmpdir, 'from_estimator')
  estimator_path = estimator.export_saved_model(estimator_base_path, serving_input_fn)
  imported = tf.saved_model.load(estimator_path)
  def predict(x):
    example = tf.train.Example()
    example.features.feature["x"].float_list.value.extend([x])
    return imported.signatures["predict"](
      examples=tf.constant([example.SerializeToString()]))
  print(predict(1.5))
  print(predict(3.5))
  mirrored_strategy = tf.distribute.MirroredStrategy()
  config = tf.estimator.RunConfig(
      train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)
  regressor = tf.estimator.LinearRegressor(
      feature_columns=[tf.feature_column.numeric_column('feats')],
      optimizer='SGD',
      config=config)
  def input_fn():
    dataset = tf.data.Dataset.from_tensors(({"feats":[1.]}, [1.]))
    return dataset.repeat(1000).batch(10)
  regressor.train(input_fn=input_fn, steps=10)
  regressor.evaluate(input_fn=input_fn, steps=10)

def main():
  estimator1()
  estimator2()

if __name__ == "__main__":
  main()
