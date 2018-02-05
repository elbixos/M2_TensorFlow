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


with tf.name_scope('X'):
	# entrées
	x = tf.placeholder(tf.float32, [None, 4], name = "X")

with tf.name_scope('Y_True'):
    # sorties voulues
    y_int = tf.placeholder(tf.uint8, [None], name = "Y_int")
    y_ = tf.one_hot(y_int, depth=3, name = "Y_True")



# Le modèle
with tf.name_scope("Weights"):
	W = tf.Variable(tf.zeros([4, 3]),name ="W")

with tf.name_scope("Biases"):
	b = tf.Variable(tf.zeros([3]), name = "b")
	
with tf.name_scope("Score"):
	score = tf.matmul(x, W) + b



# calcul de l'entropie croisée
# Cross entropy version 1 (un peu instable)
with tf.name_scope('softmax'):
	y = tf.nn.softmax(score)
with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Cross entropy version 2
#with tf.name_scope('cross_entropy'):
#	cross_entropy = tf.reduce_mean(
#      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=score))

# Calcul de la prédiction finale
with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

with tf.name_scope('Classe'):
    classe = tf.argmax(y,1, name="classe")    

# Choix d'une méthode de minimisation
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()



# Configuration de TensorBoard
# If the visu dir exists, we delete it.
# to avoid accidental multiple trainings visualisation
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
  summary, _ = sess.run([merged,train_step], feed_dict={x: training_set.data, y_int: training_set.target})
  #print("Resultats en Apprentissage", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

  writer.add_summary(summary, i)

  
#Create a saver object which will save all the variables
saver = tf.train.Saver()

# sauvegarde en fin d'apprentissage
savePath = 'SavedNetworks/'
modelName = 'myMonoCouchemodel.ckpt'
if not os.path.exists(savePath):
    os.makedirs(savePath)
    
savePathFull = os.path.join(savePath, modelName)

saver.save(sess, savePathFull)  


print("Resultats en Apprentissage", sess.run(accuracy, feed_dict={x: training_set.data, y_int: training_set.target}))
print("Résultats en Généralisation", sess.run(accuracy, feed_dict={x: test_set.data, y_int: test_set.target}))
