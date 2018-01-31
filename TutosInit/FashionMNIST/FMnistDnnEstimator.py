from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# récupération des bases de données
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./FM_data/')


# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
 feature_columns=feature_columns,
 hidden_units=[256, 32],
 optimizer=tf.train.AdamOptimizer(1e-4),
 n_classes=10,
 dropout=0.1,
 model_dir='./FMnistDnnModel'
)

def input(dataset):
 return dataset.images, dataset.labels.astype(np.int32)
 
def getFeatures(dataset):
 return dataset.images

def getLabels(dataset):
 return dataset.labels.astype(np.int32)

 
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
 x={"x": getFeatures(mnist.train)},
 y=getLabels(mnist.train),
 num_epochs=None,
 batch_size=50,
 shuffle=True
)

# Train model.
classifier.train(input_fn=train_input_fn, steps=10000)

# Evaluation sur la base d'apprentissage
train_input_eval_fn = tf.estimator.inputs.numpy_input_fn(
 x={"x": getFeatures(mnist.train)},
 y=getLabels(mnist.train),
 num_epochs=1,
 shuffle=False
)

accuracy_score = classifier.evaluate(input_fn=train_input_eval_fn)["accuracy"]
print("\nLearning Accuracy: {0:f}\n".format(accuracy_score))

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
 x={"x": getFeatures(mnist.test)},
 y=getLabels(mnist.test),
 num_epochs=1,
 shuffle=False
)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

