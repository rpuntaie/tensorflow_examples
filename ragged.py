#!/usr/bin/env python3
#https://www.tensorflow.org/guide/ragged_tensor

from include_tf import *
import math
import google.protobuf.text_format as pbtext
import tempfile

def main():
  digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
  words = tf.ragged.constant([["So", "long"], ["thanks", "for", "all", "the", "fish"]])
  print(tf.add(digits, 3))
  print(tf.reduce_mean(digits, axis=1))
  print(tf.concat([digits, [[5, 3]]], axis=0))
  print(tf.tile(digits, [1, 2]))
  print(tf.strings.substr(words, 0, 2))
  print(tf.map_fn(tf.math.square, digits))
  print(digits[0])       # First row
  print(digits[:, :2])   # First two values in each row.
  print(digits[:, -2:])  # Last two values in each row.
  print(digits + 3)
  print(digits + tf.ragged.constant([[1, 2, 3, 4], [], [5, 6, 7], [8], []]))
  times_two_plus_one = lambda x: x * 2 + 1
  print(tf.ragged.map_flat_values(times_two_plus_one, digits))
  digits.to_list()
  digits.numpy()
  sentences = tf.ragged.constant([
      ["Let's", "build", "some", "ragged", "tensors", "!"],
      ["We", "can", "use", "tf.ragged.constant", "."]])
  print(sentences)
  paragraphs = tf.ragged.constant([
      [['I', 'have', 'a', 'cat'], ['His', 'name', 'is', 'Mat']],
      [['Do', 'you', 'want', 'to', 'come', 'visit'], ["I'm", 'free', 'tomorrow']],
  ])
  print(paragraphs)
  print(tf.RaggedTensor.from_value_rowids(
      values=[3, 1, 4, 1, 5, 9, 2],
      value_rowids=[0, 0, 0, 0, 2, 2, 3]))
  print(tf.RaggedTensor.from_row_lengths(
      values=[3, 1, 4, 1, 5, 9, 2],
      row_lengths=[4, 0, 2, 1]))
  print(tf.RaggedTensor.from_row_splits(
      values=[3, 1, 4, 1, 5, 9, 2],
      row_splits=[0, 4, 4, 6, 7]))
  print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]))  # ok: type=string, rank=2
  print(tf.ragged.constant([[[1, 2], [3]], [[4, 5]]]))        # ok: type=int32, rank=3
  try:
    tf.ragged.constant([["one", "two"], [3, 4]])              # bad: multiple types
  except ValueError as exception:
    print(exception)
  try:
    tf.ragged.constant(["A", ["B", "C"]])                     # bad: multiple nesting depths
  except ValueError as exception:
    print(exception)
  queries = tf.ragged.constant([['Who', 'is', 'Dan', 'Smith'],
                                ['Pause'],
                                ['Will', 'it', 'rain', 'later', 'today']])
  num_buckets = 1024
  embedding_size = 4
  embedding_table = tf.Variable(
      tf.random.truncated_normal([num_buckets, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  # Look up the embedding for each word.
  word_buckets = tf.strings.to_hash_bucket_fast(queries, num_buckets)
  word_embeddings = tf.nn.embedding_lookup(embedding_table, word_buckets)     # ①
  # Add markers to the beginning and end of each sentence.
  marker = tf.fill([queries.nrows(), 1], '#')
  padded = tf.concat([marker, queries, marker], axis=1)                       # ②
  # Build word bigrams & look up embeddings.
  bigrams = tf.strings.join([padded[:, :-1], padded[:, 1:]], separator='+')   # ③
  bigram_buckets = tf.strings.to_hash_bucket_fast(bigrams, num_buckets)
  bigram_embeddings = tf.nn.embedding_lookup(embedding_table, bigram_buckets) # ④
  # Find the average embedding for each sentence
  all_embeddings = tf.concat([word_embeddings, bigram_embeddings], axis=1)    # ⑤
  avg_embedding = tf.reduce_mean(all_embeddings, axis=1)                      # ⑥
  print(avg_embedding)
  tf.ragged.constant([["Hi"], ["How", "are", "you"]]).shape
  print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]).bounding_shape())
  ragged_x = tf.ragged.constant([["John"], ["a", "big", "dog"], ["my", "cat"]])
  ragged_y = tf.ragged.constant([["fell", "asleep"], ["barked"], ["is", "fuzzy"]])
  print(tf.concat([ragged_x, ragged_y], axis=1))
  sparse_x = ragged_x.to_sparse()
  sparse_y = ragged_y.to_sparse()
  sparse_result = tf.sparse.concat(sp_inputs=[sparse_x, sparse_y], axis=1)
  print(tf.sparse.to_dense(sparse_result, ''))
  sentences = tf.constant(
      ['What makes you think she is a witch?',
       'She turned me into a newt.',
       'A newt?',
       'Well, I got better.'])
  is_question = tf.constant([True, False, True, False])
  # Preprocess the input strings.
  hash_buckets = 1000
  words = tf.strings.split(sentences, ' ')
  hashed_words = tf.strings.to_hash_bucket_fast(words, hash_buckets)

"""
#todo: NotImplementedError: Cannot convert a symbolic Tensor (lstm_7/strided_slice:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
# Build the Keras model.
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[None], dtype=tf.int64, ragged=True),
    tf.keras.layers.Embedding(hash_buckets, 16),
    tf.keras.layers.LSTM(32, use_bias=False),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Activation(tf.nn.relu),
    tf.keras.layers.Dense(1)
])
keras_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
keras_model.fit(hashed_words, is_question, epochs=5)
print(keras_model.predict(hashed_words))
def build_tf_example(s):
  return pbtext.Merge(s, tf.train.Example()).SerializeToString()
example_batch = [
  build_tf_example(r'''
    features {
      feature {key: "colors" value {bytes_list {value: ["red", "blue"]} } }
      feature {key: "lengths" value {int64_list {value: [7]} } } }'''),
  build_tf_example(r'''
    features {
      feature {key: "colors" value {bytes_list {value: ["orange"]} } }
      feature {key: "lengths" value {int64_list {value: []} } } }'''),
  build_tf_example(r'''
    features {
      feature {key: "colors" value {bytes_list {value: ["black", "yellow"]} } }
      feature {key: "lengths" value {int64_list {value: [1, 3]} } } }'''),
  build_tf_example(r'''
    features {
      feature {key: "colors" value {bytes_list {value: ["green"]} } }
      feature {key: "lengths" value {int64_list {value: [3, 5, 2]} } } }''')]
feature_specification = {
    'colors': tf.io.RaggedFeature(tf.string),
    'lengths': tf.io.RaggedFeature(tf.int64),
}
feature_tensors = tf.io.parse_example(example_batch, feature_specification)
for name, value in feature_tensors.items():
  print("{}={}".format(name, value))
def print_dictionary_dataset(dataset):
  for i, element in enumerate(dataset):
    print("Element {}:".format(i))
    for (feature_name, feature_value) in element.items():
      print('{:>14} = {}'.format(feature_name, feature_value))
dataset = tf.data.Dataset.from_tensor_slices(feature_tensors)
print_dictionary_dataset(dataset)
batched_dataset = dataset.batch(2)
print_dictionary_dataset(batched_dataset)
unbatched_dataset = batched_dataset.unbatch()
print_dictionary_dataset(unbatched_dataset)
non_ragged_dataset = tf.data.Dataset.from_tensor_slices([1, 5, 3, 2, 8])
non_ragged_dataset = non_ragged_dataset.map(tf.range)
batched_non_ragged_dataset = non_ragged_dataset.apply(
    tf.data.experimental.dense_to_ragged_batch(2))
for element in batched_non_ragged_dataset:
  print(element)
def transform_lengths(features):
  return {
      'mean_length': tf.math.reduce_mean(features['lengths']),
      'length_ranges': tf.ragged.range(features['lengths'])}
transformed_dataset = dataset.map(transform_lengths)
print_dictionary_dataset(transformed_dataset)
@tf.function
def make_palindrome(x, axis):
  return tf.concat([x, tf.reverse(x, [axis])], axis)
make_palindrome(tf.constant([[1, 2], [3, 4], [5, 6]]), axis=1)
make_palindrome(tf.ragged.constant([[1, 2], [3], [4, 5, 6]]), axis=1)
@tf.function(
    input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32)])
def max_and_min(rt):
  return (tf.math.reduce_max(rt, axis=-1), tf.math.reduce_min(rt, axis=-1))
max_and_min(tf.ragged.constant([[1, 2], [3], [4, 5, 6]]))
@tf.function
def increment(x):
  return x + 1
rt = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
cf = increment.get_concrete_function(rt)
print(cf(rt))
keras_module_path = tempfile.mkdtemp()
tf.saved_model.save(keras_model, keras_module_path)
imported_model = tf.saved_model.load(keras_module_path)
imported_model(hashed_words)
class CustomModule(tf.Module):
  def __init__(self, variable_value):
    super(CustomModule, self).__init__()
    self.v = tf.Variable(variable_value)
  @tf.function
  def grow(self, x):
    return x * self.v
module = CustomModule(100.0)
# Before saving a custom model, we must ensure that concrete functions are
# built for each input signature that we will need.
module.grow.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None],
                                                      dtype=tf.float32))

custom_module_path = tempfile.mkdtemp()
tf.saved_model.save(module, custom_module_path)
imported_model = tf.saved_model.load(custom_module_path)
imported_model.grow(tf.ragged.constant([[1.0, 4.0, 3.0], [2.0]]))
x = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
y = tf.ragged.constant([[1, 1], [2], [3, 3, 3]])
print(x + y)
x = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
print(x + 3)
queries = tf.ragged.constant(
    [['Who', 'is', 'George', 'Washington'],
     ['What', 'is', 'the', 'weather', 'tomorrow'],
     ['Goodnight']])
print(queries[1])                   # A single query
print(queries[1, 2])                # A single word
print(queries[1:])                  # Everything but the first row
print(queries[:, :3])               # The first 3 words of each query
print(queries[:, -2:])              # The last 2 words of each query
rt = tf.ragged.constant([[[1, 2, 3], [4]],
                         [[5], [], [6]],
                         [[7]],
                         [[8, 9], [10]]])
print(rt[1])                        # Second row (2-D RaggedTensor)
print(rt[3, 0])                     # First element of fourth row (1-D Tensor)
print(rt[:, 1:3])                   # Items 1-3 of each row (3-D RaggedTensor)
print(rt[:, -1:])                   # Last item of each row (3-D RaggedTensor)
ragged_sentences = tf.ragged.constant([
    ['Hi'], ['Welcome', 'to', 'the', 'fair'], ['Have', 'fun']])
print(ragged_sentences.to_tensor(default_value='', shape=[None, 10]))
x = [[1, 3, -1, -1], [2, -1, -1, -1], [4, 5, 8, 9]]
print(tf.RaggedTensor.from_tensor(x, padding=-1))
print(ragged_sentences.to_sparse())
st = tf.SparseTensor(indices=[[0, 0], [2, 0], [2, 1]],
                     values=['a', 'b', 'c'],
                     dense_shape=[3, 3])
print(tf.RaggedTensor.from_sparse(st))
rt = tf.ragged.constant([[1, 2], [3, 4, 5], [6], [], [7]])
print("python list:", rt.to_list())
print("numpy array:", rt.numpy())
print("values:", rt.values.numpy())
print("splits:", rt.row_splits.numpy())
print("indexed value:", rt[1].numpy())
# x       (2D ragged):  2 x (num_rows)
# y       (scalar)
# result  (2D ragged):  2 x (num_rows)
x = tf.ragged.constant([[1, 2], [3]])
y = 3
print(x + y)
# x         (2d ragged):  3 x (num_rows)
# y         (2d tensor):  3 x          1
# Result    (2d ragged):  3 x (num_rows)
x = tf.ragged.constant(
   [[10, 87, 12],
    [19, 53],
    [12, 32]])
y = [[1000], [2000], [3000]]
print(x + y)
# x      (3d ragged):  2 x (r1) x 2
# y      (2d ragged):         1 x 1
# Result (3d ragged):  2 x (r1) x 2
x = tf.ragged.constant(
    [[[1, 2], [3, 4], [5, 6]],
     [[7, 8]]],
    ragged_rank=1)
y = tf.constant([[10]])
print(x + y)
# x      (3d ragged):  2 x (r1) x (r2) x 1
# y      (1d tensor):                    3
# Result (3d ragged):  2 x (r1) x (r2) x 3
x = tf.ragged.constant(
    [
        [
            [[1], [2]],
            [],
            [[3]],
            [[4]],
        ],
        [
            [[5], [6]],
            [[7]]
        ]
    ],
    ragged_rank=2)
y = tf.constant([10, 20, 30])
print(x + y)
# x      (2d ragged): 3 x (r1)
# y      (2d tensor): 3 x    4  # trailing dimensions do not match
x = tf.ragged.constant([[1, 2], [3, 4, 5, 6], [7]])
y = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)
# x      (2d ragged): 3 x (r1)
# y      (2d ragged): 3 x (r2)  # ragged dimensions do not match.
x = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
y = tf.ragged.constant([[10, 20], [30, 40], [50]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)
# x      (3d ragged): 3 x (r1) x 2
# y      (3d ragged): 3 x (r1) x 3  # trailing dimensions do not match
x = tf.ragged.constant([[[1, 2], [3, 4], [5, 6]],
                        [[7, 8], [9, 10]]])
y = tf.ragged.constant([[[1, 2, 0], [3, 4, 0], [5, 6, 0]],
                        [[7, 8, 0], [9, 10, 0]]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)
rt = tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2],
    row_splits=[0, 4, 4, 6, 7])
print(rt)
rt = tf.RaggedTensor.from_row_splits(
    values=tf.RaggedTensor.from_row_splits(
        values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        row_splits=[0, 3, 3, 5, 9, 10]),
    row_splits=[0, 1, 1, 5])
print(rt)
print("Shape: {}".format(rt.shape))
print("Number of partitioned dimensions: {}".format(rt.ragged_rank))
rt = tf.RaggedTensor.from_nested_row_splits(
    flat_values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    nested_row_splits=([0, 1, 1, 5], [0, 3, 3, 5, 9, 10]))
print(rt)
# shape = [batch, (paragraph), (sentence), (word)]
conversations = tf.ragged.constant(
    [[[["I", "like", "ragged", "tensors."]],
      [["Oh", "yeah?"], ["What", "can", "you", "use", "them", "for?"]],
      [["Processing", "variable", "length", "data!"]]],
     [[["I", "like", "cheese."], ["Do", "you?"]],
      [["Yes."], ["I", "do."]]]])
conversations.shape
assert conversations.ragged_rank == len(conversations.nested_row_splits)
conversations.ragged_rank  # Number of partitioned dimensions.
conversations.flat_values.numpy()
rt = tf.RaggedTensor.from_row_splits(
    values=[[1, 3], [0, 0], [1, 3], [5, 3], [3, 3], [1, 2]],
    row_splits=[0, 3, 4, 6])
print(rt)
print("Shape: {}".format(rt.shape))
print("Number of partitioned dimensions: {}".format(rt.ragged_rank))
print("Flat values shape: {}".format(rt.flat_values.shape))
print("Flat values:\n{}".format(rt.flat_values))
rt = tf.RaggedTensor.from_uniform_row_length(
    values=tf.RaggedTensor.from_row_splits(
        values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        row_splits=[0, 3, 5, 9, 10]),
    uniform_row_length=2)
print(rt)
print("Shape: {}".format(rt.shape))
print("Number of partitioned dimensions: {}".format(rt.ragged_rank))
"""

if __name__ == "__main__":
  main()


