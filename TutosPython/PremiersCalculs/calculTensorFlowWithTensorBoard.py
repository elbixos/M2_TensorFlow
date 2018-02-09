import tensorflow as tf

x = tf.placeholder(tf.float32, name="x")

W = tf.Variable([.3], dtype=tf.float32, name="W")
b = tf.Variable([-.3], dtype=tf.float32, name="b")

y = W*x + b

print(y)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

resu = sess.run(y, {x:2}) 
print(resu)

resu = sess.run(y, {x:[1, 2, 3]}) 
print(resu)

pathLog="./SaveHere/";
writer = tf.summary.FileWriter(pathLog, sess.graph)
writer.close()
