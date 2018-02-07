from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import shutil

import tensorflow as tf

# Data sets
IRIS_TRAINING = "DATA/trainBase.csv"

IRIS_TEST = "DATA/testBase.csv"

        
# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
  filename=IRIS_TRAINING,
  target_dtype=np.int,
  features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
  filename=IRIS_TEST,
  target_dtype=np.int,
  features_dtype=np.float32)


# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[726])]

# Build 3 layer DNN with 10, 20, 10 units respectively.

# If the model_dir exists, we delete it.
# to avoid accidental multiple trainings.
visuPath = './VisuDnn'
if os.path.exists(visuPath):
  shutil.rmtree(visuPath)
os.makedirs(visuPath)

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[256,100],
                                      n_classes=71,
                                      model_dir=visuPath)
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": np.array(training_set.data)},
  y=np.array(training_set.target),
  num_epochs=None,
  shuffle=True)


# Train model.
classifier.train(input_fn=train_input_fn, steps=15000)


# Save Model


# Evaluation sur la base d'apprentissage
train_input_eval_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": np.array(training_set.data)},
  y=np.array(training_set.target),
  num_epochs=1,
  shuffle=False)


accuracy_score = classifier.evaluate(input_fn=train_input_eval_fn)["accuracy"]
print("\nLearning Accuracy: {0:f}\n".format(accuracy_score))

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": np.array(test_set.data)},
  y=np.array(test_set.target),
  num_epochs=1,
  shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


