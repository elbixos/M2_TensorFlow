from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import os
import tensorflow as tf

# récupération des bases de données
from tensorflow.examples.tutorials.mnist import input_data
fashionMnist = input_data.read_data_sets('./FM_DATA/', one_hot=True)


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
  y_ = graph.get_tensor_by_name("Y_True/Y_True:0")
  
  accuracy = graph.get_tensor_by_name("Accuracy/accuracy:0")
  print("Resultats en Apprentissage", sess.run(accuracy, feed_dict={x: fashionMnist.train.images, y_: fashionMnist.train.labels}))
  print("Résultats en Généralisation", sess.run(accuracy, feed_dict={x: fashionMnist.test.images, y_: fashionMnist.test.labels}))