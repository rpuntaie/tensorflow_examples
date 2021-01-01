#!/usr/bin/env python

# Tokenizing text and creating sequences for sentences
# courses/udacity_intro_to_tensorflow_for_deep_learning/l09c01_nlp_turn_words_into_tokens.ipynb

# This colab shows you how to tokenize text and create sequences for sentences as
# the first stage of preparing text for use with TensorFlow models.

## Import the Tokenizer

# Import the Tokenizer
import io

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

import tensorflow_datasets as tfds

from savefit import *

## Write some sentences
# Feel free to change and add sentences as you like

sentences = [
    'My favorite food is ice cream',
    'do you like ice cream too?',
    'My dog likes ice cream!',
    "your favorite flavor of icecream is chocolate",
    "chocolate isn't good for dogs",
    "your dog, your cat, and your parrot prefer broccoli"
]

## Tokenize the words
# The first step to preparing text to be used in a machine learning model is to
# tokenize the text, in other words, to generate numbers for the words.

# Optionally set the max number of words to tokenize.
# The out of vocabulary (OOV) token represents words that are not in the index.
# Call fit_on_text() on the tokenizer to generate unique numbers for each word
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)


## View the word index
# After you tokenize the text, the tokenizer has a word index that contains
# key-value pairs for all the words and their numbers.
# The word is the key, and the number is the value.
# Notice that the OOV token is the first entry.

# Examine the word index
word_index = tokenizer.word_index
print(word_index)

# Get the number for a given word
print(word_index['favorite'])

# Create sequences for the sentences

# After you tokenize the words, the word index contains a unique number for each
# word. However, the numbers in the word index are not ordered. Words in a
# sentence have an order. So after tokenizing the words, the next step is to
# generate sequences for the sentences.

sequences = tokenizer.texts_to_sequences(sentences)
print (sequences)

# Sequence sentences that contain words that are not in the word index

# Let's take a look at what happens if the sentence being sequenced contains
# words that are not in the word index.
# The Out of Vocabluary (OOV) token is the first entry in the word index. You
# will see it shows up in the sequences in place of any word that is not in the
# word index.

sentences2 = ["I like hot chocolate", "My dogs and my hedgehog like kibble but my squirrel prefers grapes and my chickens like ice cream, preferably vanilla"]

sequences2 = tokenizer.texts_to_sequences(sentences2)
print(sequences2)


# Preparing text to use with TensorFlow models
# courses/udacity_intro_to_tensorflow_for_deep_learning/l09c02_nlp_padding.ipynb

# The high level steps to prepare text to be used in a machine learning model are:

# 1.   Tokenize the words to get numerical values for them
# 2.   Create numerical sequences of the sentences
# 3.   Adjust the sequences to all be the same length.

## Make the sequences all the same length

# Later, when you feed the sequences into a neural network to train a model, the
# sequences all need to be uniform in size. Currently the sequences have varied
# lengths, so the next step is to make them all be the same size, either by
# padding them with zeros and/or truncating them.
# 
# Use f.keras.preprocessing.sequence.pad_sequences to add zeros to the sequences
# to make them all be the same length. By default, the padding goes at the start
# of the sequences, but you can specify to pad at the end.
# 
# You can optionally specify the maximum length to pad the sequences to.
# Sequences that are longer than the specified max length will be truncated. By
# default, sequences are truncated from the beginning of the sequence, but you
# can specify to truncate from the end.
# 
# If you don't provide the max length, then the sequences are padded to match the
# length of the longest sentence.
# 
# For all the options when padding and truncating sequences, see
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences


padded = pad_sequences(sequences)
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)


# Specify a max length for the padded sequences
padded = pad_sequences(sequences, maxlen=15)
print(padded)

# Put the padding at the end of the sequences
padded = pad_sequences(sequences, maxlen=15, padding="post")
print(padded)

# Limit the length of the sequences, you will see some sequences get truncated
padded = pad_sequences(sequences, maxlen=3)
print(padded)

## What happens if some of the sentences contain words that are not in the word index?

# Here's where the "out of vocabulary" token is used. Try generating sequences
# for some sentences that have words that are not in the word index.

# Try turning sentences that contain words that 
# aren't in the word index into sequences.
# Add your own sentences to the test_data
test_data = [
    "my best friend's favorite ice cream flavor is strawberry",
    "my dog's best friend is a manatee"
]
print (test_data)

# Remind ourselves which number corresponds to the
# out of vocabulary token in the word index
print("<OOV> has the number", word_index['<OOV>'], "in the word index.")

# Convert the test sentences to sequences
test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

# Pad the new sequences
padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")

# Notice that "1" appears in the sequence wherever there's a word 
# that's not in the word index
print(padded)


# Tokenize and sequence a bigger corpus of text
# courses/udacity_intro_to_tensorflow_for_deep_learning/l09c03_nlp_prepare_larger_text_corpus.ipynb

# So far, you have written some test sentences and generated a word index and
# then created sequences for the sentences. 

# Now you will tokenize and sequence a larger body of text, specifically reviews
# from Amazon and Yelp. 

## About the dataset

# You will use a dataset containing Amazon and Yelp reviews of products and
# restaurants. This dataset was originally extracted from
# [Kaggle](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set).

# The dataset includes reviews, and each review is labelled as 0 (bad) or 1
# (good). However, in this exercise, you will only work with the reviews, not the
# labels, to practice tokenizing and sequencing the text. 

### Example good reviews:

# *   This is hands down the best phone I've ever had.
# *   Four stars for the food & the guy in the blue shirt for his great vibe & still letting us in to eat !

### Example bad reviews:  

# *   A lady at the table next to us found a live green caterpillar In her salad
# *   If you plan to use this in a car forget about it.

### See more reviews
# Feel free to [download the
              # dataset](https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P)
# from a drive folder belonging to Udacity and open it on your local machine to
# see more reviews.

# Get the corpus of text

# The combined dataset of reviews has been saved in a Google drive belonging to
# Udacity. You can download it from there.

path = tf.keras.utils.get_file('reviews.csv', 'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P')
print (path)


# Each row in the csv file is a separate review.
# The csv file has 2 columns:
# 
# *   **text** (the review)
# *   **sentiment** (0 or 1 indicating a bad or good review)

# Read the csv file
dataset = pd.read_csv(path)

# Review the first few entries in the dataset
dataset.head()

# Get the reviews from the csv file

# Get the reviews from the text column
reviews = dataset['text'].tolist()

# Tokenize the text
# Create the tokenizer, specify the OOV token, tokenize the text, then inspect the word index.

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)


# Generate sequences for the reviews
# Generate a sequence for each review. Set the max length to match the longest
# review. Add the padding zeros at the end of the review for reviews that are not
# as long as the longest one.

sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, padding='post')

# What is the shape of the vector containing the padded sequences?
# The shape shows the number of sequences and the length of each one.
print(padded_sequences.shape)

# What is the first review?
print (reviews[0])

# Show the sequence for the first review
print(padded_sequences[0])

# Try printing the review and padded sequence for other elements.


# Word Embeddings and Sentiment
# courses/udacity_intro_to_tensorflow_for_deep_learning/l09c04_nlp_embeddings_and_sentiment.ipynb

# In this colab, you'll work with word embeddings and train a basic neural
# network to predict text sentiment. At the end, you'll be able to visualize how
# the network sees the related sentiment of each word in the dataset.

## Get the dataset

# We're going to use a dataset containing Amazon and Yelp reviews, with their
# related sentiment (1 for positive, 0 for negative). This dataset was originally
# extracted from
# [here](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set).

# !wget --no-check-certificate -O sentiment.csv https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P

dataset = pd.read_csv('sentiment.csv')

sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

# Separate out the sentences and labels into training and test sets
training_size = int(len(sentences) * 0.8)

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Make labels into numpy arrays for use with the network later
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

## Tokenize the dataset

# Tokenize the dataset, including padding and OOV

vocab_size = 1000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, padding=padding_type, 
                       truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length, 
                               padding=padding_type, truncating=trunc_type)

## Review a Sequence

# Let's quickly take a look at one of the padded sequences to ensure everything
# above worked appropriately.

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])

## Train a Basic Sentiment Model with Embeddings

# Build a basic sentiment network
# Note the embedding layer is first, 
# and the output is only 1 node as it is either 0 or 1 (negative or positive)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
model,history = savefit(model, padded, training_labels_final, epochs=num_epochs,
          validation_data=(testing_padded, testing_labels_final), verbose=0)

## Get files for visualizing the network

# The code below will download two files for visualizing how your network "sees"
# the sentiment related to each word. Head to http://projector.tensorflow.org/
# and load these files, then click the "Sphereize" checkbox.

# First get the weights of the embedding layer
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


# Write out the embedding vectors and metadata
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# Download the files
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

## Predicting Sentiment in New Reviews

# Now that you've trained and visualized your network, take a look below at how
# we can predict sentiment in new reviews the network has never seen before.

# Use the model to predict a review   
fake_reviews = ['I love this phone', 'I hate spaghetti', 
                'Everything was cold',
                'Everything was hot exactly as I wanted', 
                'Everything was green', 
                'the host seated us immediately',
                'they gave us free chocolate cake', 
                'not sure about the wilted flowers on the table',
                'only works when I stand on tippy toes', 
                'does not work when I stand on my head']

print(fake_reviews) 

# Create the sequences
padding_type='post'
sample_sequences = tokenizer.texts_to_sequences(fake_reviews)
fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)           

print('\nHOT OFF THE PRESS! HERE ARE SOME NEWLY MINTED, ABSOLUTELY GENUINE REVIEWS!\n')              

classes = model.predict(fakes_padded)

# The closer the class is to 1, the more positive the review is deemed to be
for x in range(len(fake_reviews)):
  print(fake_reviews[x])
  print(classes[x])
  print('\n')

# Try adding reviews of your own
# Add some negative words (such as "not") to the good reviews and see what happens
# For example:
# they gave us free chocolate cake and did not charge us


# Tweaking the Model
# courses/udacity_intro_to_tensorflow_for_deep_learning/l09c05_nlp_tweaking_the_model.ipynb

# In this colab, you'll investigate how various tweaks to data processing and the
# model itself can impact results. At the end, you'll once again be able to
# visualize how the network sees the related sentiment of each word in the
# dataset.

sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

# Separate out the sentences and labels into training and test sets
training_size = int(len(sentences) * 0.8)

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Make labels into numpy arrays for use with the network later
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

## Tokenize the dataset (with tweaks!)

# Now, we'll tokenize the dataset, but we can make some changes to this from
# before. Previously, we used: 

vocab_size = 1000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'

# How might changing the `vocab_size`, `embedding_dim` or `max_length` affect how
# the model performs?

vocab_size = 500
embedding_dim = 16
max_length = 50
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

## Train a Sentiment Model (with tweaks!)

# We'll use a slightly different model here, using `GlobalAveragePooling1D`
# instead of `Flatten()`.

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
model,history = savefit(model, training_padded, training_labels_final, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels_final),verbose=0)

## Visualize the training graph

# You can use the code below to visualize the training and validation accuracy
# while you try out different tweaks to the hyperparameters and model.

def plot_graphs(history, string):
  if not history:
      return
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

## Get files for visualizing the network

# The code below will download two files for visualizing how your network "sees"
# the sentiment related to each word. Head to http://projector.tensorflow.org/
# and load these files, then click the checkbox to "sphereize" the data.

# Note: You may run into errors with the projection if your `vocab_size` earlier
# was larger than the actual number of words in the vocabulary, in which case
# you'll need to decrease this variable and re-train in order to visualize.

# First get the weights of the embedding layer
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

# Create the reverse word index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Write out the embedding vectors and metadata
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# Download the files
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

## Predicting Sentiment in New Reviews

# Using LSTMs, CNNs, GRUs with a larger dataset
# courses/udacity_intro_to_tensorflow_for_deep_learning/l10c02_nlp_multiple_models_for_predicting_sentiment.ipynb

# In this colab, you use different kinds of layers to see how they affect the
# model.
# You will use the glue/sst2 dataset, which is available through tensorflow_datasets. 
# The General Language Understanding Evaluation (GLUE) benchmark
# (https://gluebenchmark.com/) is a collection of resources for training,
# evaluating, and analyzing natural language understanding systems.
# These resources include the Stanford Sentiment Treebank (SST) dataset that
# consists of sentences from movie reviews and human annotations of their
# sentiment. This colab uses version 2 of the SST dataset.
# The splits are:
# 
# *   train	67,349
# *   validation	872
# 
# and the column headings are:
# 
# *   sentence
# *   label

# For more information about the dataset, see
# [https://www.tensorflow.org/datasets/catalog/glue#gluesst2](https://www.tensorflow.org/datasets/catalog/glue#gluesst2)

# Get the dataset.
# It has 70000 items, so might take a while to download
dataset, info = tfds.load('glue/sst2', with_info=True)
print(info.features)
print(info.features["label"].num_classes)
print(info.features["label"].names)

# Get the training and validation datasets
dataset_train, dataset_validation = dataset['train'], dataset['validation']
dataset_train

# Print some of the entries
for example in dataset_train.take(2):
  review, label = example["sentence"], example["label"]
  print("Review:", review)
  print("Label: %d \n" % label.numpy())

# Get the sentences and the labels
# for both the training and the validation sets
training_reviews = []
training_labels = []

validation_reviews = []
validation_labels = []

# The dataset has 67,000 training entries, but that's a lot to process here!

# If you want to take the entire dataset: WARNING: takes longer!!
# for item in dataset_train.take(-1):

# Take 10,000 reviews
for item in dataset_train.take(10000):
  review, label = item["sentence"], item["label"]
  training_reviews.append(str(review.numpy()))
  training_labels.append(label.numpy())

print ("\nNumber of training reviews is: ", len(training_reviews))

# print some of the reviews and labels
for i in range(0, 2):
  print (training_reviews[i])
  print (training_labels[i])

# Get the validation data
# there's only about 800 items, so take them all
for item in dataset_validation.take(-1):  
  review, label = item["sentence"], item["label"]
  validation_reviews.append(str(review.numpy()))
  validation_labels.append(label.numpy())

print ("\nNumber of validation reviews is: ", len(validation_reviews))

# Print some of the validation reviews and labels
for i in range(0, 2):
  print (validation_reviews[i])
  print (validation_labels[i])


# Tokenize the words and sequence the sentences


# There's a total of 21224 words in the reviews
# but many of them are irrelevant like with, it, of, on.
# If we take a subset of the training data, then the vocab
# will be smaller.

# A reasonable review might have about 50 words or so,
# so we can set max_length to 50 (but feel free to change it as you like)

vocab_size = 4000
embedding_dim = 16
max_length = 50
trunc_type='post'
pad_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_reviews)
word_index = tokenizer.word_index


# Pad the sequences

# Pad the sequences so that they are all the same length
training_sequences = tokenizer.texts_to_sequences(training_reviews)
training_padded = pad_sequences(training_sequences,maxlen=max_length, 
                                truncating=trunc_type, padding=pad_type)

validation_sequences = tokenizer.texts_to_sequences(validation_reviews)
validation_padded = pad_sequences(validation_sequences,maxlen=max_length)

training_labels_final = np.array(training_labels)
validation_labels_final = np.array(validation_labels)

# Create the model using an Embedding

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),  
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Train the model

num_epochs = 20
model,history = savefit(model, training_padded, training_labels_final, epochs=num_epochs, 
                    validation_data=(validation_padded, validation_labels_final),verbose=0)


# Plot the accurracy and loss

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Write a function to predict the sentiment of reviews

# Write some new reviews 

review1 = """I loved this movie"""

review2 = """that was the worst movie I've ever seen"""

review3 = """too much violence even for a Bond film"""

review4 = """a captivating recounting of a cherished myth"""

new_reviews = [review1, review2, review3, review4]


# Define a function to prepare the new reviews for use with a model
# and then use the model to predict the sentiment of the new reviews           

def predict_review(model, reviews):
  # Create the sequences
  padding_type='post'
  sample_sequences = tokenizer.texts_to_sequences(reviews)
  reviews_padded = pad_sequences(sample_sequences, padding=padding_type, 
                                 maxlen=max_length) 
  classes = model.predict(reviews_padded)
  for x in range(len(reviews_padded)):
    print(reviews[x])
    print(classes[x])
    print('\n')

predict_review(model, new_reviews)



# Define a function to train and show the results of models with different layers

def fit_model_and_show_results (model, reviews):
  model.summary()
  model, history = savefit(model, training_padded, training_labels_final, epochs=num_epochs, 
                      validation_data=(validation_padded, validation_labels_final),verbose=0)
  plot_graphs(history, "accuracy")
  plot_graphs(history, "loss")
  predict_review(model, reviews)

# Use a CNN

num_epochs = 30

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(16, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Default learning rate for the Adam optimizer is 0.001
# Let's slow down the learning rate by 10.
learning_rate = 0.0001
model_cnn.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate), 
                  metrics=['accuracy'])

fit_model_and_show_results(model_cnn, new_reviews)

# Use a GRU

num_epochs = 30

model_gru = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.00003 # slower than the default learning rate
model_gru.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])

fit_model_and_show_results(model_gru, new_reviews)

# Add a bidirectional LSTM

num_epochs = 30

model_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.00003
model_bidi_lstm.compile(loss='binary_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate),
                        metrics=['accuracy'])
fit_model_and_show_results(model_bidi_lstm, new_reviews)

# Use multiple bidirectional LSTMs

num_epochs = 30

model_multiple_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, 
                                                       return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.0003
model_multiple_bidi_lstm.compile(loss='binary_crossentropy',
                                 optimizer=tf.keras.optimizers.Adam(learning_rate),
                                 metrics=['accuracy'])
fit_model_and_show_results(model_multiple_bidi_lstm, new_reviews)

# Try some more reviews

# Write some new reviews 

review1 = """I loved this movie"""

review2 = """that was the worst movie I've ever seen"""

review3 = """too much violence even for a Bond film"""

review4 = """a captivating recounting of a cherished myth"""

review5 = """I saw this movie yesterday and I was feeling low to start with,
 but it was such a wonderful movie that it lifted my spirits and brightened 
 my day, you can\'t go wrong with a movie with Whoopi Goldberg in it."""

review6 = """I don\'t understand why it received an oscar recommendation
 for best movie, it was long and boring"""

review7 = """the scenery was magnificent, the CGI of the dogs was so realistic I
 thought they were played by real dogs even though they talked!"""

review8 = """The ending was so sad and yet so uplifting at the same time. 
 I'm looking for an excuse to see it again"""

review9 = """I had expected so much more from a movie made by the director 
 who made my most favorite movie ever, I was very disappointed in the tedious 
 story"""

review10 = "I wish I could watch this movie every day for the rest of my life"

more_reviews = [review1, review2, review3, review4, review5, review6, review7, 
               review8, review9, review10]


print("============================\n","Embeddings only:\n", "============================")
predict_review(model, more_reviews)

print("============================\n","With CNN\n", "============================")
predict_review(model_cnn, more_reviews)

print("===========================\n","With bidirectional GRU\n", "============================")
predict_review(model_gru, more_reviews)

print("===========================\n", "With a single bidirectional LSTM:\n", "===========================")
predict_review(model_bidi_lstm, more_reviews)

print("===========================\n", "With multiple bidirectional LSTM:\n", "==========================")
predict_review(model_multiple_bidi_lstm, more_reviews)


# Constructing a Text Generation Model
# courses/udacity_intro_to_tensorflow_for_deep_learning/l10c03_nlp_constructing_text_generation_model.ipynb

# Using most of the techniques you've already learned, it's now possible to
# generate new text by predicting the next word that follows a given seed word.
# To practice this method, we'll use the [Kaggle Song Lyrics
# Dataset](https://www.kaggle.com/mousehead/songlyrics).

## Import TensorFlow and related functions

## Get the Dataset

# As noted above, we'll utilize the [Song Lyrics
# dataset](https://www.kaggle.com/mousehead/songlyrics) on Kaggle.

# !wget --no-check-certificate https://drive.google.com/uc?id=1LiJFZd41ofrWoBtW-pMYsfz1w8Ny0Bj8 -O songdata.csv

## **First 10 Songs**

# Let's first look at just 10 songs from the dataset, and see how things perform.

### Preprocessing

# Let's perform some basic preprocessing to get rid of punctuation and make
# everything lowercase. We'll then split the lyrics up by line and tokenize the
# lyrics.

def tokenize_corpus(corpus, num_words=-1):
  # Fit a Tokenizer on the corpus
  if num_words > -1:
    tokenizer = Tokenizer(num_words=num_words)
  else:
    tokenizer = Tokenizer()
  tokenizer.fit_on_texts(corpus)
  return tokenizer

import string

def create_lyrics_corpus(dataset, field):
  # Remove all other punctuation
  dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
  # Make it lowercase
  dataset[field] = dataset[field].str.lower()
  # Make it one long string to split by line
  lyrics = dataset[field].str.cat()
  corpus = lyrics.split('\n')
  # Remove any trailing whitespace
  for l in range(len(corpus)):
    corpus[l] = corpus[l].rstrip()
  # Remove any empty lines
  corpus = [l for l in corpus if l != '']

  return corpus

# Read the dataset from csv - just first 10 songs for now
dataset = pd.read_csv('songdata.csv', dtype=str)[:10]
# Create the corpus using the 'text' column containing lyrics
corpus = create_lyrics_corpus(dataset, 'text')
# Tokenize the corpus
tokenizer = tokenize_corpus(corpus)

total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

### Create Sequences and Labels

# After preprocessing, we next need to create sequences and labels. Creating the
# sequences themselves is similar to before with `texts_to_sequences`, but also
# including the use of
# [N-Grams](https://towardsdatascience.com/introduction-to-language-models-n-gram-e323081503d9);
# creating the labels will now utilize those sequences as well as utilize one-hot
# encoding over all potential output words.

sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		sequences.append(n_gram_sequence)

# Pad sequences for equal input length 
max_sequence_len = max([len(seq) for seq in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

# Split sequences between the "input" sequence and "output" predicted word
input_sequences, labels = sequences[:,:-1], sequences[:,-1]
# One-hot encode the labels
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Check out how some of our data is being stored
# The Tokenizer has just a single index per word
print(tokenizer.word_index['know'])
print(tokenizer.word_index['feeling'])
# Input sequences will have multiple indexes
print(input_sequences[5])
print(input_sequences[6])
# And the one hot labels will be as long as the full spread of tokenized words
print(one_hot_labels[5])
print(one_hot_labels[6])

### Train a Text Generation Model

# Building an RNN to train our text generation model will be very similar to the
# sentiment models you've built previously. The only real change necessary is to
# make sure to use Categorical instead of Binary Cross Entropy as the loss
# function - we could use Binary before since the sentiment was only 0 or 1, but
# now there are hundreds of categories.

# From there, we should also consider using *more* epochs than before, as text
# generation can take a little longer to converge than sentiment analysis, *and*
# we aren't working with all that much data yet. I'll set it at 200 epochs here
# since we're only use part of the dataset, and training will tail off quite a
# bit over that many epochs.


model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model,history = savefit(model, input_sequences, one_hot_labels, epochs=200, verbose=0)

### View the Training Graph

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  if not history:
      return
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')

### Generate new lyrics!

# It's finally time to generate some new lyrics from the trained model, and see
# what we get. To do so, we'll provide some "seed text", or an input sequence for
# the model to start with. We'll also decide just how long of an output sequence
# we want - this could essentially be infinite, as the input plus the previous
# output will be continuously fed in for a new output word (at least up to our
                                                          # max sequence length).

seed_text = "im feeling chills"
next_words = 100

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = np.argmax(model.predict(token_list), axis=-1)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)


# Optimizing the Text Generation Model
# courses/udacity_intro_to_tensorflow_for_deep_learning/l10c04_nlp_optimizing_the_text_generation_model.ipynb

## 250 Songs

# Now we've seen a model trained on just a small sample of songs, and how this
# often leads to repetition as you get further along in trying to generate new
# text. Let's switch to using the 250 songs instead, and see if our output
# improves. This will actually be nearly 10K lines of lyrics, which should be
# sufficient.

# Note that we won't use the full dataset here as it will take up quite a bit of
# RAM and processing time, but you're welcome to try doing so on your own later.
# If interested, you'll likely want to use only some of the more common words for
# the Tokenizer, which will help shrink processing time and memory needed 
# (or else you'd have an output array hundreds of thousands of words long).

### Preprocessing

def tokenize_corpus(corpus, num_words=-1):
  # Fit a Tokenizer on the corpus
  if num_words > -1:
    tokenizer = Tokenizer(num_words=num_words)
  else:
    tokenizer = Tokenizer()
  tokenizer.fit_on_texts(corpus)
  return tokenizer

def create_lyrics_corpus(dataset, field):
  # Remove all other punctuation
  dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
  # Make it lowercase
  dataset[field] = dataset[field].str.lower()
  # Make it one long string to split by line
  lyrics = dataset[field].str.cat()
  corpus = lyrics.split('\n')
  # Remove any trailing whitespace
  for l in range(len(corpus)):
    corpus[l] = corpus[l].rstrip()
  # Remove any empty lines
  corpus = [l for l in corpus if l != '']

  return corpus

def tokenize_corpus(corpus, num_words=-1):
  # Fit a Tokenizer on the corpus
  if num_words > -1:
    tokenizer = Tokenizer(num_words=num_words)
  else:
    tokenizer = Tokenizer()
  tokenizer.fit_on_texts(corpus)
  return tokenizer

# Read the dataset from csv - this time with 250 songs
dataset = pd.read_csv('songdata.csv', dtype=str)[:250]
# Create the corpus using the 'text' column containing lyrics
corpus = create_lyrics_corpus(dataset, 'text')
# Tokenize the corpus
tokenizer = tokenize_corpus(corpus, num_words=2000)
total_words = tokenizer.num_words

# There should be a lot more words now
print(total_words)

### Create Sequences and Labels

sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		sequences.append(n_gram_sequence)

# Pad sequences for equal input length 
max_sequence_len = max([len(seq) for seq in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

# Split sequences between the "input" sequence and "output" predicted word
input_sequences, labels = sequences[:,:-1], sequences[:,-1]
# One-hot encode the labels
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

### Train a (Better) Text Generation Model

# With more data, we'll cut off after 100 epochs to avoid keeping you here all
# day. You'll also want to change your runtime type to GPU if you haven't already
# (you'll need to re-run the above cells if you change runtimes).

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model,history = savefit(model, input_sequences, one_hot_labels, epochs=100, verbose=0)

### View the Training Graph

plot_graphs(history, 'accuracy')

### Generate better lyrics!

# This time around, we should be able to get a more interesting output with less
# repetition.

seed_text = "im feeling chills"
next_words = 100
  
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = np.argmax(model.predict(token_list), axis=-1)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)

### Varying the Possible Outputs

# In running the above, you may notice that the same seed text will generate
# similar outputs. This is because the code is currently always choosing the top
# predicted class as the next word. What if you wanted more variance in the
# output? 

# Switching from `model.predict_classes` to `model.predict_proba` will get us all
# of the class probabilities. We can combine this with `np.random.choice` to
# select a given predicted output based on a probability, thereby giving a bit
# more randomness to our outputs.

# Test the method with just the first word after the seed text
seed_text = "im feeling chills"
next_words = 100
  
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
predicted_probs = model.predict(token_list)[0]
predicted = np.random.choice([x for x in range(len(predicted_probs))], 
                             p=predicted_probs)
# Running this cell multiple times should get you some variance in output
print(predicted)

# Use this process for the full output generation
seed_text = "im feeling chills"
next_words = 100
  
for _ in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
  predicted_probs = model.predict(token_list)[0]
  predicted = np.random.choice([x for x in range(len(predicted_probs))],
                               p=predicted_probs)
  output_word = ""
  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break
  seed_text += " " + output_word
print(seed_text)
