from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf


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
  
  classe = graph.get_tensor_by_name("Classe/classe:0")
  
  new_samples = np.array(
    [[6.9, 3.2, 4.5, 1.5],
    [4.8, 3.1, 5.0, 1.7],
    [1.9, 6.2, 2.5, 1.5]], dtype=np.float32)

  predictions = sess.run(classe, {x: new_samples})
  
  dicoClasses = ['setosa', 'versicolor', 'virginica']

  for p in predictions :
    print ("je pense que c'est : ",dicoClasses[p])   

  