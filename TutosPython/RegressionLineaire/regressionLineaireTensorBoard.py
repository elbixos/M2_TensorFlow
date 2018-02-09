import tensorflow as tf

## Définition des variables 
W1 = tf.Variable([.3], dtype=tf.float32, name="W1")
W = tf.Variable([.3], dtype=tf.float32, name="W")
b = tf.Variable([-.3], dtype=tf.float32, name="b")
x = tf.placeholder(tf.float32, name="x")

#sortieCalculee = W*x + b
sortieCalculee = W1*tf.square(x) + W*x + b

sortieVoulue = tf.placeholder(tf.float32, name="sortieVoulue")

squared_deltas = tf.square(sortieCalculee - sortieVoulue)
erreur = tf.reduce_sum(squared_deltas)


optimizer = tf.train.GradientDescentOptimizer(0.002)
train = optimizer.minimize(erreur)

sess = tf.Session()

# Configuration de TensorBoard
pathLog="./pathLog/";
writer = tf.summary.FileWriter(pathLog, sess.graph)
tf.summary.scalar('erreur quadratique', erreur)
#tf.summary.scalar('W', W)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess.run(init) # reset values to incorrect defaults.
for i in range(10000):
  print("erreur :", sess.run(erreur,{x: [1, 2, 3, 4], sortieVoulue: [0.0, -1.0, -2.0, -3.0]}))
  summary, _ = sess.run([merged,train], {x: [1, 2, 3, 4], sortieVoulue: [0.001, -1.01, -2.01, -3.001]})
  writer.add_summary(summary, i)
  
print("(W,b finaux) :", sess.run([W1, W, b]))
print("Resultats en Apprentissage :", sess.run(sortieCalculee,{x: [1, 2, 3, 4]}))
print("Resultats en Prediction :", sess.run(sortieCalculee,{x: [1.5, 1.3, 2.1, 4.7]}))


writer.close()
