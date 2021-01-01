
import tensorflow as tf
# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

### Import the Fashion MNIST dataset
#This guide uses the
#[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
#dataset, which contains 70,000 grayscale images in 10 categories. The images
#show individual articles of clothing at low resolution (28 $\times$ 28 pixels),
#as seen here:
#
#Fashion MNIST is intended as a drop-in replacement for the classic
#[MNIST](http://yann.lecun.com/exdb/mnist/) dataset—often used as the "Hello,
#World" of machine learning programs for computer vision. The MNIST dataset
#contains images of handwritten digits (0, 1, 2, etc) in an identical format to
#the articles of clothing we'll use here.
#
#This guide uses Fashion MNIST for variety, and because it's a slightly more
#challenging problem than regular MNIST. Both datasets are relatively small and
#are used to verify that an algorithm works as expected. They're good starting
#points to test and debug code.
#
#We will use 60,000 images to train the network and 10,000 images to evaluate
#how accurately the network learned to classify images. You can access the
#Fashion MNIST directly from TensorFlow, using the
#[Datasets](https://www.tensorflow.org/datasets) API:

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#Loading the dataset returns metadata as well as a *training dataset* and *test dataset*.
#
#* The model is trained using `train_dataset`.
#* The model is tested against `test_dataset`.
#
#The images are 28 $\times$ 28 arrays, with pixel values in the range 
#`[0, 255]`.
#The *labels* are an array of integers, in the range `[0, 9]`. These correspond
#to the *class* of clothing the image represents:
#
#Each image is mapped to a single label.
#Since the *class names* are not included with the dataset, store them here to
#use later when plotting the images:

class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))

### Explore the data
#
#Let's explore the format of the dataset before training the model. The
#following shows there are 60,000 images in the training set, and 10000 images
#in the test set:

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

## Preprocess the data
#
#The value of each pixel in the image data is an integer in the range `[0,255]`.
#For the model to work properly, these values need to be normalized to the range
#`[0,1]`. So here we create a normalization function, and then apply it to each
#image in the test and train datasets.

def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

### Explore the processed data
#
#Let's plot an image to see what it looks like.

# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
#plt.show()

#Display the first 25 images from the *training set* and display the class name
#below each image. Verify that the data is in the correct format and we're ready
#to build and train the network.

plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(test_dataset.take(25)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
#plt.show()

def show_results(model,test_dataset):

  ## Make predictions and explore
  #With the model trained, we can use it to make predictions about some images.

  for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

  predictions.shape

  #Here, the model has predicted the label for each image in the testing set.
  #Let's take a look at the first prediction:

  predictions[0]

  #A prediction is an array of 10 numbers. These describe the "confidence" of the
  #model that the image corresponds to each of the 10 different articles of
  #clothing. We can see which label has the highest confidence value:

  np.argmax(predictions[0])

  #So the model is most confident that this image is a shirt, or `class_names[6]`.
  #And we can check the test label to see this is correct:

  test_labels[0]

  #We can graph this to look at the full set of 10 class predictions

  def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img[...,0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

  def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

  #Let's look at the 0th image, predictions, and prediction array. 

  i = 0
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(1,2,2)
  plot_value_array(i, predictions, test_labels)
  plt.show()

  i = 12
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(1,2,2)
  plot_value_array(i, predictions, test_labels)
  plt.show()

  #Let's plot several images with their predictions. Correct prediction labels are
  #blue and incorrect prediction labels are red. The number gives the percent 
  #(out of 100)
  #for the predicted label. Note that it can be wrong even when very confident. 

  # Plot the first X test images, their predicted label, and the true label
  # Color correct predictions in blue, incorrect predictions in red

  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
  plt.show()


  #Finally, use the trained model to make a prediction about a single image. 

  # Grab an image from the test dataset
  img = test_images[0]
  print(img.shape)

  #`tf.keras` models are optimized to make predictions on a *batch*, or
  #collection, of examples at once. So even though we're using a single image, we
  #need to add it to a list:

  # Add the image to a batch where it's the only member.

  img = np.array([img])
  print(img.shape)

  #Now predict the image:

  predictions_single = model.predict(img)
  print(predictions_single)

  plot_value_array(0, predictions_single, test_labels)
  _ = plt.xticks(range(10), class_names, rotation=45)

  #`model.predict` returns a list of lists, one for each image in the batch of
  #data. Grab the predictions for our (only) image in the batch:

  np.argmax(predictions_single[0])
