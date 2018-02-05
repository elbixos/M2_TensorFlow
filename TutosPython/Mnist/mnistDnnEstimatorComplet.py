from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

from PIL import Image
import PIL.ImageOps

import numpy as np
import tensorflow as tf

# récupération des bases de données
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data/')


# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]

# sauvegarde en fin d'apprentissage
visuPath = './VisuDnn'
if not os.path.exists(visuPath):
    os.makedirs(visuPath)

# Build 3 layer DNN with 10, 20, 10 units respectively.
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

savePath = './SavedNetworksEstimator/'
    
classifier.export_savedmodel(savePath, serving_input_receiver_fn)



# Evaluation sur la base d'apprentissage
train_input_eval_fn = tf.estimator.inputs.numpy_input_fn(
 x={"x": getFeatures(mnist.train)},
 y=getLabels(mnist.train),
 num_epochs=1,
 shuffle=False
)

accuracy_score = classifier.evaluate(input_fn=train_input_eval_fn)["accuracy"]
print("Learning Accuracy: {0:f}\n".format(accuracy_score))

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
 x={"x": getFeatures(mnist.test)},
 y=getLabels(mnist.test),
 num_epochs=1,
 shuffle=False
)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("Test Accuracy: {0:f}\n".format(accuracy_score))


## Prediction sur une image
# Lecture de l'image, et préparation de l'image 
imageFilename = 'images/flou.jpg'
imageGray = Image.open(imageFilename).resize((28,28)).convert('L')
imageInvert =  PIL.ImageOps.invert(imageGray)

#imageInvert.save('temp.bmp')


# conversion en vecteur
a = np.array(imageInvert)
flat_arr = a.reshape((1, 784))

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": flat_arr},
  num_epochs=1,
  shuffle=False)

predictions = classifier.predict(input_fn=predict_input_fn)

for p in predictions :
    class_id = p['class_ids'][0]
    probability = p['probabilities'][class_id]
    print ("je pense que c'est un : ",class_id, "avec une proba de ",probability )   
