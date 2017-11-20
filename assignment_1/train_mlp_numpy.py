"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import pickle
import cifar10_utils
from mlp_numpy import MLP

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DNN_HIDDEN_UNITS_DEFAULT = '100'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model on the whole test set each 100 iterations.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
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
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)

  # Load test data
  x_test, y_test = cifar10.test.images, cifar10.test.labels
  x_test = np.reshape(x_test, [x_test.shape[0], -1])

  batch_size = FLAGS.batch_size
  n_classes = 10
  input_dim = 3 * 32 * 32

  mlp = MLP(n_hidden= dnn_hidden_units,
            n_classes = n_classes,
            weight_decay = FLAGS.weight_reg_strength,
            weight_scale = FLAGS.weight_init_scale,
            input_dim = input_dim,
            learning_rate = FLAGS.learning_rate)

  for tr_step in range(FLAGS.max_steps):

      # Get next batch
      x_tr, y_tr = cifar10.train.next_batch(batch_size)
      # Reshape data for MLP
      x_tr = np.reshape(x_tr, (batch_size, -1))

      # Inference
      tr_logits = mlp.inference(x_tr)

      # Calculate loss and accuracy
      tr_loss = mlp.loss(tr_logits, y_tr)
      tr_accuracy = mlp.accuracy(tr_logits, y_tr)

      if tr_step % 10 == 0:
          print('Step:{} Loss:{:.4f}, Accuracy:{:.4f}'.format(tr_step, tr_loss, tr_accuracy))

      mlp.train_step()

      if tr_step % 100 == 0 or tr_step == FLAGS.max_steps-1:
          # Inference
          test_logits = mlp.inference(x_test)

          # Calculate loss and accuracy
          test_loss = mlp.loss(test_logits, y_test)
          test_accuracy = mlp.accuracy(test_logits, y_test)

          print('TEST - Loss:{:.4f}, Accuracy:{:.4f}'.format(test_loss, test_accuracy))


  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

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
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
