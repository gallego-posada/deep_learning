"""
This module implements training and evaluation of a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import os

import cifar10_utils
from mlp_tf import MLP
import pickle

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.0
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'

# Default model name
NAME_DEFAULT = 'mlp'
# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'
# Directory for trained models
SAVE_PATH_DEFAULT = './saved_models/'


# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/programmers_guide/variables
# https://www.tensorflow.org/api_guides/python/contrib.layers#Initializers
WEIGHT_INITIALIZATION_DICT = {'xavier': lambda _: tf.contrib.layers.xavier_initializer(uniform = True),  # Xavier initialisation
                              'normal': lambda scale: tf.random_normal_initializer(stddev = scale), # Initialization from a standard normal
                              'uniform': lambda scale: tf.random_uniform_initializer(minval = -scale, maxval = scale),  # Initialization from a uniform distribution
                              }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/contrib.layers#Regularizers
WEIGHT_REGULARIZER_DICT = {'none': None,  # No regularization
                           'l1': lambda scale: tf.contrib.layers.l1_regularizer(scale = scale),  # L1 regularization
                           'l2': lambda scale: tf.contrib.layers.l2_regularizer(scale = scale)  # L2 regularization
                           }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/nn
ACTIVATION_DICT = {'relu': tf.nn.relu,  # ReLU
                   'elu': tf.nn.elu,  # ELU
                   'tanh': tf.nn.tanh,  # Tanh
                   'sigmoid': tf.nn.sigmoid}  # Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/train
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,  # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,  # Adagrad
                  'adam': tf.train.AdamOptimizer,  # Adam
                  'rmsprop': tf.train.RMSPropOptimizer  # RMSprop
                  }

FLAGS = None

def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
  as you did in the task 1 of this assignment.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  tf.set_random_seed(42)
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  # Import dataset
  cifar10 = cifar10_utils.get_cifar10(data_dir= FLAGS.data_dir)

  # Create session
  tf.reset_default_graph()
  sess = tf.Session()

  # Create MLP object
  mlp = MLP(n_hidden = dnn_hidden_units,
            n_classes = 10,
            is_training = True,
            activation_fn = ACTIVATION_DICT[FLAGS.activation],
            dropout_rate = FLAGS.dropout_rate,
            weight_initializer = WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init](FLAGS.weight_init_scale),
            weight_regularizer = WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg](FLAGS.weight_reg_strength) if WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg] is not None else None,
            input_shape = [32, 32, 3])

  # Setup placeholders for input data and labels
  with tf.name_scope('input'):
      x = tf.placeholder(tf.float32, [None, mlp.input_dim], name='x-input')
      y = tf.placeholder(tf.float32, [None, mlp.n_classes], name='y-input')

  keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
  tf.summary.scalar('keep_prob', keep_prob)

  # TF image summaries
  with tf.name_scope('input_reshape'):
      image_shaped_input = tf.reshape(x, [-1] + mlp.input_shape)
      tf.summary.image('input', image_shaped_input, 10)


  #Define global step and optimizer
  global_step = tf.Variable(0, trainable = False, name = 'global_step')
  optimizer = OPTIMIZER_DICT[FLAGS.optimizer](learning_rate = FLAGS.learning_rate)

  # Define ops
  logits_op = mlp.inference(x, keep_prob)
  loss_op = mlp.loss(logits_op, y)
  accuracy_op = mlp.accuracy(logits_op, y)
  train_op = mlp.train_step(loss_op, {'optimizer': optimizer, 'global_step': global_step})
  conf_mat_op = mlp.confusion_matrix(logits_op, y)
  summary_op = tf.summary.merge_all()

  save_model = FLAGS.save_path is not None
  write_log = FLAGS.log_dir is not None

  # If enabled, set up log writers
  if write_log:
    train_log_path = os.path.join(FLAGS.log_dir, '{}_train'.format(FLAGS.name))
    _check_path(train_log_path)
    train_log_writer = tf.summary.FileWriter(train_log_path, graph = sess.graph)

    test_log_path = os.path.join(FLAGS.log_dir, '{}_test'.format(FLAGS.name))
    _check_path(test_log_path)
    test_log_writer = tf.summary.FileWriter(test_log_path, graph = sess.graph)

  # Run init op
  init_op = tf.global_variables_initializer()
  local_init_op = tf.local_variables_initializer()
  sess.run(fetches=[init_op, local_init_op])

  # Load test data once instead of every time
  x_test, y_test = cifar10.test.images, cifar10.test.labels
  x_test = np.reshape(x_test, (x_test.shape[0], -1))

  tr_stats = []
  test_stats = []

  for tr_step in range(FLAGS.max_steps):
       # Get next batch
       x_tr, y_tr = cifar10.train.next_batch(FLAGS.batch_size)
       x_tr = np.reshape(x_tr, (FLAGS.batch_size, -1))

       tr_feed = {x: x_tr, y: y_tr, keep_prob: 1. - FLAGS.dropout_rate}
       fetches = [train_op, loss_op, accuracy_op]

       # Run train step on training set
       if tr_step % 10 == 0 and write_log:
           fetches += [summary_op]
           _, tr_loss, tr_accuracy, tr_summary = sess.run(fetches = fetches, feed_dict = tr_feed)
           train_log_writer.add_summary(tr_summary, tr_step)
       else:
           _, tr_loss, tr_accuracy = sess.run(fetches = fetches, feed_dict = tr_feed)

       tr_stats += [[tr_step , tr_loss, tr_accuracy]]
       # Print statistics
    #    if tr_step % 50 == 0:
    #        print('Step:{} Loss:{:.4f}, Accuracy:{:.4f}'.format(tr_step, tr_loss, tr_accuracy))

       # Test set evaluation
       if tr_step % 100 == 0 or tr_step == FLAGS.max_steps-1:
           test_feed = {x: x_test, y: y_test, keep_prob: 1.0}
           test_loss, test_accuracy, test_logits, test_confusion_matrix, test_summary = sess.run(
                fetches = [loss_op, accuracy_op, logits_op, conf_mat_op, summary_op],
                feed_dict = test_feed)
           if write_log:
               test_log_writer.add_summary(test_summary, tr_step)

           test_stats += [[tr_step, test_loss, test_accuracy]]

           #print('TEST - Loss:{:.4f}, Accuracy:{:.4f}'.format(test_loss, test_accuracy))
           #print('TEST - Conf Matrix \n {} \n'.format(test_confusion_matrix))

  # Once done with training, close writers
  if write_log:
        train_log_writer.close()
        test_log_writer.close()

  print(tr_step, tr_loss, tr_accuracy, test_loss, test_accuracy)
  pickle.dump( tr_stats, open( "./pickles/" + FLAGS.name + "_train.p", "wb" ) )
  pickle.dump( test_stats, open( "./pickles/" + FLAGS.name + "_test.p", "wb" ) )

  # Save trained model
  if save_model:
        save_dir = os.path.join(FLAGS.save_path, FLAGS.name)
        saver = tf.train.Saver()
        _check_path(save_dir)
        saver.save(sess, save_path = os.path.join(save_dir, 'model.ckpt'))

  ########################
  # END OF YOUR CODE    #
  #######################

def _check_path(path):
    """
    Makes sure path for log and model saving exists
    """
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  # Make directories if they do not exists yet
  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--weight_init', type = str, default = WEIGHT_INITIALIZATION_DEFAULT,
                      help='Weight initialization type [xavier, normal, uniform].')
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg', type = str, default = WEIGHT_REGULARIZER_DEFAULT,
                      help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  parser.add_argument('--activation', type = str, default = ACTIVATION_DEFAULT,
                      help='Activation function [relu, elu, tanh, sigmoid].')
  parser.add_argument('--optimizer', type = str, default = OPTIMIZER_DEFAULT,
                      help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')

  parser.add_argument('--name', type = str, default = NAME_DEFAULT,
                      help = 'Model name for future reference')
  parser.add_argument('--save_path', type = str, default = SAVE_PATH_DEFAULT,
                      help = 'Directory for storing model data')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
