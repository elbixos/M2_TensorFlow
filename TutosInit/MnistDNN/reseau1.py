from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# récupération des bases de données
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# entrées
x = tf.placeholder(tf.float32, [None, 784])

# sorties voulues
y_ = tf.placeholder(tf.float32, [None, 10])

# Le modèle
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

# calcul de l'entropie croisée
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# Choix d'une méthode de minimisation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

init = tf.global_variables_initializer();
sess.run(init);

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  
# Calcul de la prédiction finale
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Resultats en Apprentissage", sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
print("Résultats en Généralisation", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
