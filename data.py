#!/usr/bin/env python3
#https://www.tensorflow.org/guide/data

from include_tf import *

np.set_printoptions(precision=4)

## Basic mechanics
def data1():
  dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
  dataset
  for elem in dataset:
    print(elem.numpy())
  it = iter(dataset)
  print(next(it).numpy())
  print(dataset.reduce(0, lambda state, value: state + value).numpy())
  ### Dataset structure
  dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
  dataset1.element_spec
  dataset2 = tf.data.Dataset.from_tensor_slices(
     (tf.random.uniform([4]),
      tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
  dataset2.element_spec
  dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
  dataset3.element_spec
  dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))
  dataset4.element_spec
  dataset4.element_spec.value_type
  dataset1 = tf.data.Dataset.from_tensor_slices(
      tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
  for z in dataset1:
    print(z.numpy())
  dataset2 = tf.data.Dataset.from_tensor_slices(
     (tf.random.uniform([4]),
      tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
  dataset2
  dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
  dataset3
  for a, (b,c) in dataset3:
    print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))

## Reading input data
def data2():
  train, test = tf.keras.datasets.fashion_mnist.load_data()
  images, labels = train
  images = images/255
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  dataset
  def count(stop):
    i = 0
    while i<stop:
      yield i
      i += 1
  ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )
  for count_batch in ds_counter.repeat().batch(10).take(10):
    print(count_batch.numpy())
  def gen_series():
    i = 0
    while True:
      size = np.random.randint(0, 10)
      yield i, np.random.normal(size=(size,))
      i += 1
  for i, series in gen_series():
    print(i, ":", str(series))
    if i > 5:
      break
  ds_series = tf.data.Dataset.from_generator(
      gen_series, 
      output_types=(tf.int32, tf.float32), 
      output_shapes=((), (None,)))
  ds_series_batch = ds_series.shuffle(20).padded_batch(10)
  ids, sequence_batch = next(iter(ds_series_batch))
  print(ids.numpy())
  print()
  print(sequence_batch.numpy())

def data3():
  flowers = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)
  img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
  images, labels = next(img_gen.flow_from_directory(flowers))
  print(images.dtype, images.shape)
  print(labels.dtype, labels.shape)
  ds = tf.data.Dataset.from_generator(
      lambda: img_gen.flow_from_directory(flowers), 
      output_types=(tf.float32, tf.float32), 
      output_shapes=([32,256,256,3], [32,5])
  )
  ds.element_spec

def data4():
  fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
  dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
  dataset
  raw_example = next(iter(dataset))
  parsed = tf.train.Example.FromString(raw_example.numpy())
  parsed.features.feature['image/text']
  directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
  file_names = ['cowper.txt', 'derby.txt', 'butler.txt']
  file_paths = [
      tf.keras.utils.get_file(file_name, directory_url + file_name)
      for file_name in file_names
  ]
  dataset = tf.data.TextLineDataset(file_paths)
  for line in dataset.take(5):
    print(line.numpy())
  files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
  lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)
  for i, line in enumerate(lines_ds.take(9)):
    if i % 3 == 0:
      print()
    print(line.numpy())
  titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
  titanic_lines = tf.data.TextLineDataset(titanic_file)
  for line in titanic_lines.take(10):
    print(line.numpy())
  def survived(line):
    return tf.not_equal(tf.strings.substr(line, 0, 1), "0")
  survivors = titanic_lines.skip(1).filter(survived)
  for line in survivors.take(10):
    print(line.numpy())
  titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
  df = pd.read_csv(titanic_file)
  df.head()
  titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))
  for feature_batch in titanic_slices.take(1):
    for key, value in feature_batch.items():
      print("  {!r:20s}: {}".format(key, value))
  titanic_batches = tf.data.experimental.make_csv_dataset(
      titanic_file, batch_size=4,
      label_name="survived")
  for feature_batch, label_batch in titanic_batches.take(1):
    print("'survived': {}".format(label_batch))
    print("features:")
    for key, value in feature_batch.items():
      print("  {!r:20s}: {}".format(key, value))
  titanic_batches = tf.data.experimental.make_csv_dataset(
      titanic_file, batch_size=4,
      label_name="survived", select_columns=['class', 'fare', 'survived'])
  for feature_batch, label_batch in titanic_batches.take(1):
    print("'survived': {}".format(label_batch))
    for key, value in feature_batch.items():
      print("  {!r:20s}: {}".format(key, value))
  titanic_types  = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string] 
  dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types , header=True)
  for line in dataset.take(10):
    print([item.numpy() for item in line])

def data5():
  flowers_root = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)
  flowers_root = pathlib.Path(flowers_root)
  for item in flowers_root.glob("*"):
    print(item.name)
  list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))
  for f in list_ds.take(5):
    print(f.numpy())
  def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label
  labeled_ds = list_ds.map(process_path)
  for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())
  inc_dataset = tf.data.Dataset.range(100)
  dec_dataset = tf.data.Dataset.range(0, -100, -1)
  dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
  batched_dataset = dataset.batch(4)
  for batch in batched_dataset.take(4):
    print([arr.numpy() for arr in batch])
  batched_dataset
  batched_dataset = dataset.batch(7, drop_remainder=True)
  batched_dataset
  dataset = tf.data.Dataset.range(100)
  dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
  dataset = dataset.padded_batch(4, padded_shapes=(None,))
  for batch in dataset.take(2):
    print(batch.numpy())
    print()
  titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
  titanic_lines = tf.data.TextLineDataset(titanic_file)
  def plot_batch_sizes(ds):
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')
    plt.show()
  titanic_batches = titanic_lines.repeat(3).batch(128)
  plot_batch_sizes(titanic_batches)
  titanic_batches = titanic_lines.batch(128).repeat(3)
  plot_batch_sizes(titanic_batches)
  epochs = 3
  dataset = titanic_lines.batch(128)
  for epoch in range(epochs):
    for batch in dataset:
      print(batch.shape)
    print("End of epoch: ", epoch)
  lines = tf.data.TextLineDataset(titanic_file)
  counter = tf.data.experimental.Counter()

  dataset = tf.data.Dataset.zip((counter, lines))
  dataset = dataset.shuffle(buffer_size=100)
  dataset = dataset.batch(20)
  dataset
  n,line_batch = next(iter(dataset))
  print(n.numpy())
  dataset = tf.data.Dataset.zip((counter, lines))
  shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)
  print("Here are the item ID's near the epoch boundary:\n")
  for n, line_batch in shuffled.skip(60).take(5):
    print(n.numpy())
  shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]
  plt.plot(shuffle_repeat, label="shuffle().repeat()")
  plt.ylabel("Mean item ID")
  plt.legend()
  plt.show()
  dataset = tf.data.Dataset.zip((counter, lines))
  shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)
  print("Here are the item ID's near the epoch boundary:\n")
  for n, line_batch in shuffled.skip(55).take(15):
    print(n.numpy())
  repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]
  plt.plot(shuffle_repeat, label="shuffle().repeat()")
  plt.plot(repeat_shuffle, label="repeat().shuffle()")
  plt.ylabel("Mean item ID")
  plt.legend()
  plt.show()
  list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))
  def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-2]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])
    return image, label
  file_path = next(iter(list_ds))
  image, label = parse_image(file_path)
  def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')
    plt.show()
  show(image, label)
  images_ds = list_ds.map(parse_image)
  for image, label in images_ds.take(2):
    show(image, label)
  import scipy.ndimage as ndimage
  def random_rotate_image(image):
    image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
    return image
  image, label = next(iter(images_ds))
  image = random_rotate_image(image)
  show(image, label)
  def tf_random_rotate_image(image, label):
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image, label
  rot_ds = images_ds.map(tf_random_rotate_image)
  for image, label in rot_ds.take(2):
    show(image, label)

def data7():
  fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
  dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
  dataset
  raw_example = next(iter(dataset))
  parsed = tf.train.Example.FromString(raw_example.numpy())
  feature = parsed.features.feature
  raw_img = feature['image/encoded'].bytes_list.value[0]
  img = tf.image.decode_png(raw_img)
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  _ = plt.title(feature["image/text"].bytes_list.value[0])
  raw_example = next(iter(dataset))
  def tf_parse(eg):
    example = tf.io.parse_example(
        eg[tf.newaxis], {
            'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/text': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
        })
    return example['image/encoded'][0], example['image/text'][0]
  img, txt = tf_parse(raw_example)
  print(txt.numpy())
  print(repr(img.numpy()[:20]), "...")
  decoded = dataset.map(tf_parse)
  decoded
  image_batch, text_batch = next(iter(decoded.batch(10)))
  image_batch.shape

def data8():
  range_ds = tf.data.Dataset.range(100000)
  batches = range_ds.batch(10, drop_remainder=True)
  for batch in batches.take(5):
    print(batch.numpy())
  def dense_1_step(batch):
    # Shift features and labels one step relative to each other.
    return batch[:-1], batch[1:]
  predict_dense_1_step = batches.map(dense_1_step)
  for features, label in predict_dense_1_step.take(3):
    print(features.numpy(), " => ", label.numpy())
  batches = range_ds.batch(15, drop_remainder=True)
  def label_next_5_steps(batch):
    return (batch[:-5],   # Take the first 5 steps
            batch[-5:])   # take the remainder
  predict_5_steps = batches.map(label_next_5_steps)
  for features, label in predict_5_steps.take(3):
    print(features.numpy(), " => ", label.numpy())
  feature_length = 10
  label_length = 3
  features = range_ds.batch(feature_length, drop_remainder=True)
  labels = range_ds.batch(feature_length).skip(1).map(lambda labels: labels[:label_length])
  predicted_steps = tf.data.Dataset.zip((features, labels))
  for features, label in predicted_steps.take(5):
    print(features.numpy(), " => ", label.numpy())
  window_size = 5
  windows = range_ds.window(window_size, shift=1)
  for sub_ds in windows.take(5):
    print(sub_ds)
  for x in windows.flat_map(lambda x: x).take(30):
    print(x.numpy(), end=' ')
  def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)
  for example in windows.flat_map(sub_to_batch).take(5):
    print(example.numpy())
  def make_window_dataset(ds, window_size=5, shift=1, stride=1):
    windows = ds.window(window_size, shift=shift, stride=stride)
    def sub_to_batch(sub):
      return sub.batch(window_size, drop_remainder=True)
    windows = windows.flat_map(sub_to_batch)
    return windows
  ds = make_window_dataset(range_ds, window_size=10, shift = 5, stride=3)
  for example in ds.take(10):
    print(example.numpy())
  dense_labels_ds = ds.map(dense_1_step)
  for inputs,labels in dense_labels_ds.take(3):
    print(inputs.numpy(), "=>", labels.numpy())

def data9():
  zip_path = tf.keras.utils.get_file(
      origin='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip',
      fname='creditcard.zip',
      extract=True)
  csv_path = zip_path.replace('.zip', '.csv')
  creditcard_ds = tf.data.experimental.make_csv_dataset(
      csv_path, batch_size=1024, label_name="Class",
      # Set the column types: 30 floats and an int.
      column_defaults=[float()]*30+[int()])
  def count(counts, batch):
    features, labels = batch
    class_1 = labels == 1
    class_1 = tf.cast(class_1, tf.int32)
    class_0 = labels == 0
    class_0 = tf.cast(class_0, tf.int32)
    counts['class_0'] += tf.reduce_sum(class_0)
    counts['class_1'] += tf.reduce_sum(class_1)
    return counts
  counts = creditcard_ds.take(10).reduce(
      initial_state={'class_0': 0, 'class_1': 0},
      reduce_func = count)
  counts = np.array([counts['class_0'].numpy(),
                     counts['class_1'].numpy()]).astype(np.float32)
  fractions = counts/counts.sum()
  print(fractions)
  negative_ds = (
    creditcard_ds
      .unbatch()
      .filter(lambda features, label: label==0)
      .repeat())
  positive_ds = (
    creditcard_ds
      .unbatch()
      .filter(lambda features, label: label==1)
      .repeat())
  for features, label in positive_ds.batch(10).take(1):
    print(label.numpy())
  balanced_ds = tf.data.experimental.sample_from_datasets(
      [negative_ds, positive_ds], [0.5, 0.5]).batch(10)
  for features, labels in balanced_ds.take(10):
    print(labels.numpy())
  def class_func(features, label):
    return label
  resampler = tf.data.experimental.rejection_resample(
      class_func, target_dist=[0.5, 0.5], initial_dist=fractions)
  resample_ds = creditcard_ds.unbatch().apply(resampler).batch(10)
  balanced_ds = resample_ds.map(lambda extra_label, features_and_label: features_and_label)
  for features, labels in balanced_ds.take(10):
    print(labels.numpy())
  range_ds = tf.data.Dataset.range(20)
  iterator = iter(range_ds)
  ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=iterator)
  manager = tf.train.CheckpointManager(ckpt, '/tmp/my_ckpt', max_to_keep=3)
  print([next(iterator).numpy() for _ in range(5)])
  save_path = manager.save()
  print([next(iterator).numpy() for _ in range(5)])
  ckpt.restore(manager.latest_checkpoint)
  print([next(iterator).numpy() for _ in range(5)])

def data10():
  train, test = tf.keras.datasets.fashion_mnist.load_data()
  images, labels = train
  images = images/255.0
  labels = labels.astype(np.int32)
  fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
  fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)
  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
  ])
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=['accuracy'])
  model.fit(fmnist_train_ds, epochs=2)
  model.fit(fmnist_train_ds.repeat(), epochs=2, steps_per_epoch=20)
  loss, accuracy = model.evaluate(fmnist_train_ds)
  print("Loss :", loss)
  print("Accuracy :", accuracy)
  loss, accuracy = model.evaluate(fmnist_train_ds.repeat(), steps=10)
  print("Loss :", loss)
  print("Accuracy :", accuracy)
  predict_ds = tf.data.Dataset.from_tensor_slices(images).batch(32)
  result = model.predict(predict_ds, steps = 10)
  print(result.shape)
  result = model.predict(fmnist_train_ds, steps = 10)
  print(result.shape)

def main():
  print("data1"*9);data1()
  print("data2"*9);data2()
  print("data3"*9);data3()
  print("data4"*9);data4()
  print("data5"*9);data5()
  print("data7"*9);data7()
  print("data8"*9);data8()
  print("data9"*9);data9()
  print("data10"*9);data10()

if __name__ == "__main__":
  main()
