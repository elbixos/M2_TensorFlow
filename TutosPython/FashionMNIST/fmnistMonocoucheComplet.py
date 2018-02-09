from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import PIL.ImageOps
import numpy as np
import os
import shutil

import tensorflow as tf

# récupération des bases de données
from tensorflow.examples.tutorials.mnist import input_data
fashionMnist = input_data.read_data_sets('./FM_DATA/', one_hot=True)


with tf.name_scope('X'):
  # entrées
  x = tf.placeholder(tf.float32, [None, 784], name = "X")

with tf.name_scope('Y_True'):
  # sorties voulues
  y_ = tf.placeholder(tf.float32, [None, 10], name = "Y_True")

# Le modèle
with tf.name_scope("Weights"):
  W = tf.Variable(tf.zeros([784, 10]),name ="W")

with tf.name_scope("Biases"):
  b = tf.Variable(tf.zeros([10]), name = "b")
  
with tf.name_scope("Score"):
  score = tf.matmul(x, W) + b

classe = tf.argmax(score,1)    


# calcul de l'entropie croisée
# Cross entropy version 1 (un peu instable)
with tf.name_scope('softmax'):
  softmax = y = tf.nn.softmax(score)
with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Cross entropy version 2
#with tf.name_scope('cross_entropy'):
#  cross_entropy = tf.reduce_mean(
#      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=score))

# Calcul de la prédiction finale
with tf.name_scope('Accuracy'):
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Choix d'une méthode de minimisation
with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()


# Configuration de TensorBoard
# If the model_dir exists, we delete it.
# to avoid accidental multiple trainings.
visuPath = './VisuMonoCouche'
if os.path.exists(visuPath):
  shutil.rmtree(visuPath)
os.makedirs(visuPath)

writer = tf.summary.FileWriter(visuPath, sess.graph)
tf.summary.scalar('Entropie Croisee', cross_entropy)
tf.summary.scalar('Precision', accuracy)

#tf.summary.scalar('W', W)
merged = tf.summary.merge_all()



init = tf.global_variables_initializer();
sess.run(init);

for i in range(1000):
  batch_xs, batch_ys = fashionMnist.train.next_batch(1000)
  summary, _ = sess.run([merged,train_step], feed_dict={x: batch_xs, y_: batch_ys})
  #print("Resultats en Apprentissage", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

  writer.add_summary(summary, i)
  


print("Resultats en Apprentissage", sess.run(accuracy, feed_dict={x: fashionMnist.train.images, y_: fashionMnist.train.labels}))
print("Résultats en Généralisation", sess.run(accuracy, feed_dict={x: fashionMnist.test.images, y_: fashionMnist.test.labels}))

## Prediction sur une image
# Lecture de l'image, et préparation de l'image 
imageFilename = 'images/tshirt.jpg'
imageGray = Image.open(imageFilename).resize((28,28)).convert('L')
imageInvert =  PIL.ImageOps.invert(imageGray)

#imageInvert.save('temp.bmp')


# conversion en vecteur
a = np.array(imageInvert)
flat_arr = a.reshape((1, 784))

dicoClasses = ["t-shirts", "trousers", "pullovers", "dresses", "coats", "sandals", "shirts", "sneakers", "bags", "ankle boots"]

classIndex = sess.run(classe, {x: flat_arr})
print("\nJe pense que c'est : ", dicoClasses[classIndex[0]], " / label : ", classIndex)


writer.close()
