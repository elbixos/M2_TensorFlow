from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
  filename=IRIS_TRAINING,
  target_dtype=np.int,
  features_dtype=np.float32)
  
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
  filename=IRIS_TEST,
  target_dtype=np.int,
  features_dtype=np.float32)


# Chargement du réseau
savePath = 'SavedNetworks'
modelName = 'myMonoCouchemodel.ckpt'
savePathFull = os.path.join(savePath, modelName)
print ("ModelFilename ", repr(savePathFull))

metagraphFilename = savePathFull+'.meta'

print ("restoring graph ", repr(metagraphFilename))

with tf.Session() as sess:

  #print ("restoring graph ", metagraphFilename)
  new_saver = tf.train.import_meta_graph(metagraphFilename)
  
  
  print ("restoring variables from latest checkpoint in", savePath)
  new_saver.restore(sess, tf.train.latest_checkpoint(savePath))

  graph = tf.get_default_graph()
  x = graph.get_tensor_by_name("X/X:0")
  y_int= graph.get_tensor_by_name("Y_True/Y_int:0")
  accuracy = graph.get_tensor_by_name("Accuracy/accuracy:0")
  
  print("Resultats en Apprentissage", sess.run(accuracy, feed_dict={x: training_set.data, y_int: training_set.target}))
  print("Résultats en Généralisation", sess.run(accuracy, feed_dict={x: test_set.data, y_int: test_set.target}))
