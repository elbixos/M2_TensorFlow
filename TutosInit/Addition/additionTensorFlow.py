import tensorflow as tf

x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name='y')

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print (sess.run(y))
