from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

from PIL import Image
import PIL.ImageOps

import numpy as np
import shutil

import tensorflow as tf

# récupération des bases de données
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./FM_DATA/')


# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
# If the model_dir exists, we delete it.
# to avoid accidental multiple trainings.
visuPath = './VisuDnn'
if os.path.exists(visuPath):
  shutil.rmtree(visuPath)
os.makedirs(visuPath)

classifier = tf.estimator.DNNClassifier(
 feature_columns=feature_columns,
 hidden_units=[256, 32],
 optimizer=tf.train.AdamOptimizer(1e-4),
 n_classes=10,
 dropout=0.1,
 model_dir=visuPath
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

# Save Model
def serving_input_receiver_fn():
  feature_spec = {'x': tf.FixedLenFeature([784],tf.float32)}
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input_tensors')
  receiver_tensors = {'inputs': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  
# sauvegarde en fin d'apprentissage
savePath = './SavedNetworksEstimator/'
if not os.path.exists(savePath):
    os.makedirs(savePath)
    
classifier.export_savedmodel(savePath, serving_input_receiver_fn)

## Supression du repertoire Timestamp, remplace par lastSave
folderName = os.listdir(savePath)[0]
folderFullName = os.path.join(savePath, folderName)
targetFullName = os.path.join(savePath, 'lastSave')

shutil.move(folderFullName,targetFullName)

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
