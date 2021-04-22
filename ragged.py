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

if __name__ == "__main__":
  main()


