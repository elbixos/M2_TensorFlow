import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32, name="W")
b = tf.Variable([-.3], dtype=tf.float32, name="b")
x = tf.placeholder(tf.float32, name="x")
sortieCalculee = W*x + b

sortieVoulue = tf.placeholder(tf.float32, name="sortieVoulue")

squared_deltas = tf.square(sortieCalculee - sortieVoulue)
erreur = tf.reduce_sum(squared_deltas)


sess = tf.Session()


init = tf.global_variables_initializer()
sess.run(init)


print(sess.run(sortieCalculee, {x: [1, 2, 3, 4]}))

print("erreur totale :", sess.run(erreur, {x: [1, 2, 3, 4], sortieVoulue: [0, -1, -2, -3]}))



fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

print("erreur totale :", sess.run(erreur, {x: [1, 2, 3, 4], sortieVoulue: [0, -1, -2, -3]}))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(erreur)

pathLog="./pathLog/";
writer = tf.summary.FileWriter(pathLog, sess.graph)
tf.summary.scalar('erreur', erreur)
merged = tf.summary.merge_all()

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  summary, _ = sess.run([merged,train], {x: [1, 2, 3, 4], sortieVoulue: [0, -1, -2, -3]})
  writer.add_summary(summary, i)
  
print("(W,b finaux) :", sess.run([W, b]))


writer.close()