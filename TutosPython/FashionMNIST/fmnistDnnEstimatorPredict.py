from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

from PIL import Image
import PIL.ImageOps

import numpy as np
import tensorflow as tf

from tensorflow.contrib import predictor

with tf.Session() as sess:

  # Chargement du réseau
  basePath = 'SavedNetworksEstimator'
  tmpDir = '1517846831'
  savePathFull = os.path.join(basePath, tmpDir)
  print ("Restoring from ", savePathFull)

  # loading model
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], savePathFull)

  #get the predictor , refer tf.contrib.predicdtor
  predictor= tf.contrib.predictor.from_saved_model(savePathFull)

  #get the input_tensor tensor from the model graph
  # name is input_tensor defined in input_receiver function refer to tf.dnn.classifier
  input_tensor=tf.get_default_graph().get_tensor_by_name("input_tensors:0")


  ## Prediction sur une image
  # Lecture de l'image, et préparation de l'image 
  imageFilename = 'images/bag.jpg'
  imageGray = Image.open(imageFilename).resize((28,28)).convert('L')
  imageInvert =  PIL.ImageOps.invert(imageGray)

  #imageInvert.save('temp.bmp')


  # conversion en vecteur
  a = np.array(imageInvert)
  flat_arr = a.reshape((1, 784)).ravel()
  

  model_input= tf.train.Example(features=tf.train.Features(feature={
					'x': tf.train.Feature(float_list=tf.train.FloatList(value=flat_arr))        
					}))

  #Prepare model input, the model expects a float array to be passed to x
  # check line 28 serving_input_receiver_fn
  model_input=model_input.SerializeToString()
	  
  # calcul de la prediction ... depuis les scores 
  predictions= predictor({"inputs":[model_input]})
  classe_id = np.argmax(predictions["scores"])

  dicoClasses = ["t-shirts", "trousers", "pullovers", "dresses", "coats", "sandals", "shirts", "sneakers", "bags", "ankle boots"]
  print("\n Je pense que c'est : ", dicoClasses[classe_id], " / label : ", classe_id)
  
