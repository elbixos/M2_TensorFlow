from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from PIL import Image
import PIL.ImageOps

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
  classe = graph.get_tensor_by_name("classe:0")  
  
  ## Prediction sur une image
  # Lecture de l'image, et préparation de l'image 
  imageFilename = '4.jpg'
  imageGray = Image.open(imageFilename).resize((28,28)).convert('L')
  imageInvert =  PIL.ImageOps.invert(imageGray)
  
  imageInvert.save('temp.bmp')
  
  # conversion en vecteur
  a = np.array(imageInvert)
  flat_arr = a.reshape((1, 784))
  
  print("Classe prédite", sess.run(classe, {x: flat_arr}))

