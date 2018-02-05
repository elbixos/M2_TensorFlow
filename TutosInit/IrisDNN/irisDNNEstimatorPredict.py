from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf


with tf.Session() as sess:

  # Chargement du réseau
  basePath = 'SavedNetworksEstimator'
  tmpDir = '1517595177'
  savePathFull = os.path.join(basePath, tmpDir)
  print ("Restoring from ", savePathFull)

  # loading model
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], savePathFull)

  #get the predictor , refer tf.contrib.predicdtor
  predictor= tf.contrib.predictor.from_saved_model(savePathFull)

  #get the input_tensor tensor from the model graph
  # name is input_tensor defined in input_receiver function refer to tf.dnn.classifier
  input_tensor=tf.get_default_graph().get_tensor_by_name("input_tensors:0")

  #get the output dict
  # do not forget [] around model_input or else it will complain shape() for Tensor shape(?,)
  # since its of shape(?,) when we trained it

  new_samples = np.array(
  [[6.9, 3.2, 4.5, 1.5],
   [4.8, 3.1, 5.0, 1.7]], dtype=np.float32)
   
  for sample in new_samples:
	  print (sample)
    
	  model_input= tf.train.Example(features=tf.train.Features(feature={
					'x': tf.train.Feature(float_list=tf.train.FloatList(value=sample))        
					}))

	  #Prepare model input, the model expects a float array to be passed to x
	  # check line 28 serving_input_receiver_fn
	  model_input=model_input.SerializeToString()
	  
	  # calcul de la prediction ... depuis les scores 
	  predictions= predictor({"inputs":[model_input]})
	  classe_id = np.argmax(predictions["scores"])
	  
	  dicoClasses = ['setosa', 'versicolor', 'virginica']

	  print ("je pense que c'est : ",dicoClasses[classe_id])


  
