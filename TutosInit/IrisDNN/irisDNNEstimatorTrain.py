from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
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

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=3,
                                      model_dir="./Iris_model")
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": np.array(training_set.data)},
  y=np.array(training_set.target),
  num_epochs=None,
  shuffle=True)


# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)


# Save Model

def serving_input_receiver_fn():
  feature_spec = {'x': tf.FixedLenFeature([4],tf.float32)}
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
  x={"x": np.array(training_set.data)},
  y=np.array(training_set.target),
  num_epochs=1,
  shuffle=False)


accuracy_score = classifier.evaluate(input_fn=train_input_eval_fn)["accuracy"]
print("\nLearning Accuracy: {0:f}\n".format(accuracy_score))

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": np.array(test_set.data)},
  y=np.array(test_set.target),
  num_epochs=1,
  shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))



# Classify two new flower samples.
new_samples = np.array(
  [[6.9, 3.2, 4.5, 1.5],
   [4.8, 3.1, 5.0, 1.7]], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": new_samples},
  num_epochs=1,
  shuffle=False)

dicoClasses = ['setosa', 'versicolor', 'virginica']
predictions = classifier.predict(input_fn=predict_input_fn)

for p in predictions :
    class_id = p['class_ids'][0]
    probability = p['probabilities'][class_id]
    print ("je pense que c'est : ",dicoClasses[class_id], "avec une proba de ",probability )   
