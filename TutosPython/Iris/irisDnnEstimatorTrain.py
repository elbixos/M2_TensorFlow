from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import shutil

import tensorflow as tf



# Data sets
IRIS_TRAINING = "IrisDatabase/iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "IrisDatabase/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# If the training and test sets aren't stored locally, download them.
if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
        f.write(raw)

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
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]


# Build 3 layer DNN with 10, 20, 10 units respectively.

# If the model_dir exists, we delete it.
# to avoid accidental multiple trainings.
visuPath = './VisuDnn'
if os.path.exists(visuPath):
  shutil.rmtree(visuPath)
os.makedirs(visuPath)

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=3,
                                      model_dir=visuPath)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": np.array(training_set.data)},
  y=np.array(training_set.target),
  num_epochs=None,
  shuffle=True)


# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)


# Save Model

def serving_input_receiver_fn():
  feature_spec = {'x': tf.FixedLenFeature([4],tf.float32)}
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input_tensors')
  receiver_tensors = {'inputs': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

savePath = './SavedNetworksEstimator/'
if os.path.exists(savePath):
  shutil.rmtree(savePath)
os.makedirs(savePath)
    
classifier.export_savedmodel(savePath, serving_input_receiver_fn)

## Supression du repertoire Timestamp, remplace par lastSave
folderName = os.listdir(savePath)[0]
folderFullName = os.path.join(savePath, folderName)
targetFullName = os.path.join(savePath, 'lastSave')

shutil.move(folderFullName,targetFullName)




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


