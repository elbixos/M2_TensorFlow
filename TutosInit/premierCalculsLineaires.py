import tensorflow as tf


W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
sortieCalculee = W*x + b

sortieVoulue = tf.placeholder(tf.float32)

squared_deltas = tf.square(sortieCalculee - sortieVoulue)

loss = tf.reduce_sum(squared_deltas)


sess = tf.Session()


init = tf.global_variables_initializer()
sess.run(init)


print(sess.run(sortieCalculee, {x: [1, 2, 3, 4]}))

print(sess.run(loss, {x: [1, 2, 3, 4], sortieVoulue: [0, -1, -2, -3]}))



fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

print(sess.run(loss, {x: [1, 2, 3, 4], sortieVoulue: [0, -1, -2, -3]}))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], sortieVoulue: [0, -1, -2, -3]})

print(sess.run([W, b]))

