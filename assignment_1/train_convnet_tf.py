from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import numpy as np
import time

import cifar10_utils
from convnet_tf import ConvNet
import pickle


LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'adam'

WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
DROPOUT_RATE_DEFAULT = 0.0
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
DATA_AUGMENTATION_DEFAULT = False

# Default model name
NAME_DEFAULT = 'convnet'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cnn'
CHECKPOINT_DIR_DEFAULT = './checkpoints'


WEIGHT_INITIALIZATION_DICT = {'xavier': lambda _: tf.contrib.layers.xavier_initializer(uniform = True),  # Xavier initialisation
                              'normal': lambda scale: tf.random_normal_initializer(stddev = scale), # Initialization from a standard normal
                              'uniform': lambda scale: tf.random_uniform_initializer(minval = -scale, maxval = scale),  # Initialization from a uniform distribution
                              }

WEIGHT_REGULARIZER_DICT = {'none': None,  # No regularization
                           'l1': lambda scale: tf.contrib.layers.l1_regularizer(scale = scale),  # L1 regularization
                           'l2': lambda scale: tf.contrib.layers.l2_regularizer(scale = scale)  # L2 regularization
                           }

ACTIVATION_DICT = {'relu': tf.nn.relu,  # ReLU
                   'elu': tf.nn.elu,  # ELU
                   'tanh': tf.nn.tanh,  # Tanh
                   'sigmoid': tf.nn.sigmoid}  # Sigmoid

OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,  # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,  # Adagrad
                  'adam': tf.train.AdamOptimizer,  # Adam
                  'rmsprop': tf.train.RMSPropOptimizer  # RMSprop
                  }

FLAGS = None

def train():
  """
  Performs training and evaluation of ConvNet model.

  First define your graph using class ConvNet and its methods. Then define
  necessary operations such as savers and summarizers. Finally, initialize
  your model within a tf.Session and do the training.

  ---------------------------
  How to evaluate your model:
  ---------------------------
  Evaluation on test set should be conducted over full batch, i.e. 10k images,
  while it is alright to do it over minibatch for train set.

  ---------------------------------
  How often to evaluate your model:
  ---------------------------------
  - on training set every print_freq iterations
  - on test set every eval_freq iterations

  ------------------------
  Additional requirements:
  ------------------------
  Also you are supposed to take snapshots of your model state (i.e. graph,
  weights and etc.) every checkpoint_freq iterations. For this, you should
  study TensorFlow's tf.train.Saver class.
  """

  # Set the random seeds for reproducibility. DO NOT CHANGE.
  tf.set_random_seed(42)
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  ########################
  # Import dataset
  cifar10 = cifar10_utils.get_cifar10(data_dir= FLAGS.data_dir)

  # Create session
  tf.reset_default_graph()
  sess = tf.Session()

  # Create MLP object
  conv_net = ConvNet(n_classes = 10,
                     weight_initializer = WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init](FLAGS.weight_init_scale),
                     weight_regularizer = WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg](FLAGS.weight_reg_strength) if WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg] is not None else None)



  # Setup placeholders for input data and labels
  with tf.name_scope('input'):
      x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
      y = tf.placeholder(tf.float32, [None, conv_net.n_classes], name='y-input')

  keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
  #tf.summary.scalar('keep_prob', keep_prob)

  #Define global step and optimizer
  global_step = tf.Variable(0, trainable = False, name = 'global_step')
  optimizer = OPTIMIZER_DICT[FLAGS.optimizer](learning_rate = FLAGS.learning_rate)

  # Define ops
  logits_op = conv_net.inference(x, keep_prob)
  loss_op = conv_net.loss(logits_op, y)
  accuracy_op = conv_net.accuracy(logits_op, y)
  train_op = conv_net.train_step(loss_op, {'optimizer': optimizer, 'global_step': global_step})
  conf_mat_op = conv_net.confusion_matrix(logits_op, y)
  summary_op = tf.summary.merge_all()

  save_model = FLAGS.checkpoint_dir is not None
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

  if FLAGS.data_augmentation:
      img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                        rotation_range = 10,
                        shear_range = 0.1,
                        zoom_range = 0.1,
                        fill_mode = 'nearest',
                        data_format = 'channels_last')

      cifar10_augmented = img_generator.flow(x = cifar10.train.images,
                                 y = cifar10.train.labels,
                                 batch_size = FLAGS.batch_size)

  tr_stats = []
  test_stats = []

  for tr_step in range(FLAGS.max_steps):

       # Get next batch
       if FLAGS.data_augmentation:
           x_tr, y_tr = cifar10_augmented.next()
       else:
           x_tr, y_tr = cifar10.train.next_batch(FLAGS.batch_size)

       tr_feed = {x: x_tr, y: y_tr, keep_prob: 1. - FLAGS.dropout_rate}
       fetches = [train_op, loss_op, accuracy_op]

       # Run train step on training set
       if tr_step % FLAGS.print_freq == 0 and write_log:
           fetches += [summary_op]
           _, tr_loss, tr_accuracy, tr_summary = sess.run(fetches = fetches, feed_dict = tr_feed)
           train_log_writer.add_summary(tr_summary, tr_step)
       else:
           _, tr_loss, tr_accuracy = sess.run(fetches = fetches, feed_dict = tr_feed)

       tr_stats += [[tr_step , tr_loss, tr_accuracy]]

       # Print statistics
       if tr_step % FLAGS.print_freq == 0:
           print('Step:{} Loss:{:.4f}, Accuracy:{:.4f}'.format(tr_step, tr_loss, tr_accuracy))

       # Test set evaluation
       if tr_step % FLAGS.eval_freq == 0 or tr_step == FLAGS.max_steps-1:
           #Use 10 batches to estimate test performance with less variance
           x_test, y_test = cifar10.test.next_batch(10*FLAGS.batch_size)
           test_feed = {x: x_test, y: y_test, keep_prob: 1.0}
           test_loss, test_accuracy, test_logits, test_summary, test_confusion_matrix = sess.run(
                fetches = [loss_op, accuracy_op, logits_op, summary_op, conf_mat_op],
                feed_dict = test_feed)
           if write_log:
               test_log_writer.add_summary(test_summary, tr_step)

           test_stats += [[tr_step, test_loss, test_accuracy]]

           print('TEST - Loss:{:.4f}, Accuracy:{:.4f}'.format(test_loss, test_accuracy))
           #print('TEST - Conf Matrix \n {} \n'.format(test_confusion_matrix))

       # Save checkpoint model
       if tr_step % FLAGS.checkpoint_freq == 0 and  save_model:
             save_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.name)
             saver = tf.train.Saver()
             _check_path(save_dir)
             saver.save(sess, save_path = os.path.join(save_dir, 'model.ckpt'))


  # Once done with training, close writers
  if write_log:
        train_log_writer.close()
        test_log_writer.close()

  print(tr_step, tr_loss, tr_accuracy, test_loss, test_accuracy)
  pickle.dump( tr_stats, open( "./pickles/" + FLAGS.name + "_train.p", "wb" ) )
  pickle.dump( test_stats, open( "./pickles/" + FLAGS.name + "_test.p", "wb" ) )

  # Save final trained model
  if save_model:
        save_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.name)
        saver = tf.train.Saver()
        _check_path(save_dir)
        saver.save(sess, save_path = os.path.join(save_dir, 'model.ckpt'))

  ########################
  # END OF YOUR CODE    #
  ########################

def _check_path(path):
    """
    Makes sure path for log and model saving exists
    """
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)

def initialize_folders():
  """
  Initializes all folders in FLAGS variable.
  """

  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)

  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  print_flags()

  initialize_folders()

  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
  parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
  parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
  parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
  parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')

  parser.add_argument('--optimizer', type = str, default = OPTIMIZER_DEFAULT,
                      help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  parser.add_argument('--weight_init', type = str, default = WEIGHT_INITIALIZATION_DEFAULT,
                      help='Weight initialization type [xavier, normal, uniform].')
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg', type = str, default = WEIGHT_REGULARIZER_DEFAULT,
                      help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')

  parser.add_argument('--data_augmentation', type = bool, default = DATA_AUGMENTATION_DEFAULT,
                    help='Use data augmentation.')

  parser.add_argument('--name', type = str, default = NAME_DEFAULT,
                      help = 'Model name for future reference')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
