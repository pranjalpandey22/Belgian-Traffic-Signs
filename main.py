# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:24:29 2018

@author: Pranjal
"""
# Impporting the necessary modules

import os
import skimage.data
import numpy as np
import matplotlib.pyplot as plt

# function for loading data and return lists for images and their labels
def load_data(data_directory):
    
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    
    for d in directories:        
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    
    return images, labels

ROOT_PATH = "C:\\Users\\pande\\Desktop\\Projects\\Belgian DNN"

train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns\\Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns\\Testing")

images, labels = load_data(train_data_directory)

'''
# images and labels details
np.array(images).ndim
np.array(images).size
images[0]
np.array(labels).ndim
np.array(labels).size
print(len(set(labels)))
'''

# visualize
plt.hist(labels, 62)
plt.title('Distribution of Traffic Sign Labels')
plt.show()

# random articles chosen
traffic_signs = [300, 2250, 3650, 4000]

# function to view the images
def view_images(images_in):
    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(images_in[traffic_signs[i]], cmap='gray')
        plt.subplots_adjust(wspace=0.5)
        print("shape : {0}, min : {1}, max : {2}".format(images_in[traffic_signs[i]].shape,
                                                         images_in[traffic_signs[i]].min(),
                                                         images_in[traffic_signs[i]].max()))

    plt.show()

view_images(images)


# Printing an image of each class with their counts

unique_labels = set(labels)

plt.figure(figsize=(15, 15))

i = 1

for label in unique_labels:
    image = images[labels.index(label)] 
    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.title("Label {0} ({1})".format(label, labels.count(label)), fontsize=7)
    plt.tick_params(labelsize='small')
    i += 1
    plt.imshow(image)

plt.show()

# Recaling the images

from skimage import transform

# Image rescaled to size 28X28
images28 = [transform.resize(image ,(28, 28)) for image in images]

view_images(images28)

# Conversion to grayscale

from skimage.color import rgb2gray

images28 = np.array(images28)

images28 = rgb2gray(images28)

# View images, but with colormap for grayscale
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap='gray')
    plt.subplots_adjust(wspace=0.5)
    
plt.show()


# Building with TensorFlow

import tensorflow as tf

# Placeholder variables to be used later
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Defining an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label index
correct_pred = tf.argmax(logits, 1)

# Accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''
# Recap of everything
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)
'''

# Running the NN

tf.set_random_seed(1234)

# Starting session
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
    print('EPOCH ', i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
    if i%10 == 0:
        print("Loss: ", loss)
    print("DONE WITH EPOCH")
    
# Predictions
    
import random

# Any 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the correct_pred operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually 
fig = plt.figure(figsize=(10, 10))

for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth: {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap='gray')

plt.show()
    
sess.close()
