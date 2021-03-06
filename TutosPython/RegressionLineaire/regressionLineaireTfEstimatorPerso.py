import numpy as np
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
def ma_fn(features, labels, mode):
  # Build a linear model and predict values
  W1 = tf.get_variable("W1", [1], dtype=tf.float64)
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  
  y = W1*tf.square(features['x']) + W*features['x'] + b
  #y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.0001)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
                   
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

print("yeba")

estimator = tf.estimator.Estimator(model_fn=ma_fn, model_dir="./modelEstimator")
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0.0, -1.0, -2.0, -3.0])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

# train
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

train_metrics = estimator.evaluate(input_fn=train_input_fn)

print("train metrics: %r"% train_metrics)


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("eval metrics: %r"% eval_metrics)
