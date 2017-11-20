"""
This module implements a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in Tensorflow.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_hidden, n_classes, is_training = True,
               activation_fn = tf.nn.relu, dropout_rate = 0.,
               weight_initializer = xavier_initializer(),
               weight_regularizer = l2_regularizer(0.001),
               input_shape = [32, 32, 3]):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      is_training: Bool Tensor, it indicates whether the model is in training
                        mode or not. This will be relevant for methods that perform
                        differently during training and testing (such as dropout).
                        Have look at how to use conditionals in TensorFlow with
                        tf.cond.
      activation_fn: callable, takes a Tensor and returns a transformed tensor.
                          Activation function specifies which type of non-linearity
                          to use in every hidden layer.
      dropout_rate: float in range [0,1], presents the fraction of hidden units
                         that are randomly dropped for regularization.
      weight_initializer: callable, a weight initializer that generates tensors
                               of a chosen distribution.
      weight_regularizer: callable, returns a scalar regularization loss given
                               a weight variable. The returned loss will be added to
                               the total loss for training purposes.
    """
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.input_shape = input_shape
    self.input_dim = int(np.prod(self.input_shape))
    self.is_training = is_training
    self.activation_fn = activation_fn
    self.dropout_rate = dropout_rate
    self.weight_initializer = weight_initializer
    self.weight_regularizer = weight_regularizer
    self.bias_initializer = tf.constant_initializer(value = 0.001, dtype = tf.float32)

  def _variable_summaries(self, var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def _fc_layer(self, input_tensor, input_dim, output_dim, layer_name, act, keep_prob):
      with tf.variable_scope(layer_name):
          weights = tf.get_variable(name = 'weights',
                                    shape = [input_dim, output_dim],
                                    dtype = tf.float32,
                                    initializer = self.weight_initializer,
                                    regularizer = self.weight_regularizer)

          with tf.name_scope('weights'):
              self._variable_summaries(weights)

          biases = tf.get_variable(name = 'biases',
                                    shape = [output_dim],
                                    dtype = tf.float32,
                                    initializer = self.bias_initializer)

          with tf.name_scope('biases'):
              self._variable_summaries(biases)


          mmul = tf.matmul(input_tensor, weights)
          #print('MMUL', input_tensor.shape, weights.shape, mmul.shape, biases.shape)

          preactivate = tf.add(mmul, biases, name = 'preacts')
          activations = act(preactivate, name = 'activations')

          tf.summary.histogram('preactivations', preactivate)
          tf.summary.histogram('activations', activations)

          with tf.name_scope('dropout'):
              output = tf.nn.dropout(activations, keep_prob)
              tf.summary.histogram('output', output)

      return output

  def inference(self, x, keep_prob):
    """
    Performs inference given an input tensor. This is the central portion
    of the network. Here an input tensor is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    In order to keep things uncluttered we recommend you (though it's not required)
    to implement a separate function that is used to define a fully connected
    layer of the MLP.

    In order to make your code more structured you can use variable scopes and name
    scopes. You can define a name scope for the whole model, for each hidden
    layer and for output. Variable scopes are an essential component in TensorFlow
    design for parameter sharing.

    You can use tf.summary.histogram to save summaries of the fully connected layer weights,
    biases, pre-activations, post-activations, and dropped-out activations
    for each layer. It is very useful for introspection of the network using TensorBoard.

    Args:
      x: 2D float Tensor of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Pass x as the input of the network
    h = x

    # Do layer-by-layer computation of the hidden layers
    for k in range(len(self.n_hidden)):
        h = self._fc_layer(input_tensor = h,
                           input_dim = h.shape[1],
                           output_dim = self.n_hidden[k],
                           layer_name = 'fc_' + str(k),
                           act = self.activation_fn,
                           keep_prob = keep_prob)

    # Computation for the output layer
    logits = self._fc_layer(h, h.shape[1], self.n_classes, 'fc_' + str(k + 1),
                            act = tf.identity, keep_prob = keep_prob)

    ########################
    # END OF YOUR CODE    #
    #######################

    return logits

  def _reg_loss(self):

      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

      if reg_losses:
          total_reg_loss = tf.add_n(reg_losses, name = 'total_reg_loss')
          tf.summary.scalar('total_reg_loss', total_reg_loss)
      else:
          total_reg_loss = None

      return total_reg_loss

  def loss(self, logits, labels):
    """
    Computes the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    In order to implement this function you should have a look at
    tf.nn.softmax_cross_entropy_with_logits.

    You can use tf.summary.scalar to save scalar summaries of
    cross-entropy loss, regularization loss, and full loss (both summed)
    for use with TensorBoard. This will be useful for compiling your report.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.
    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('mean_cross_entropy', cross_entropy)

    with tf.name_scope('reg_loss'):
        reg_loss = self._reg_loss()

    with tf.name_scope('total_loss'):
        if reg_loss is not None:
            loss = cross_entropy + reg_loss
            tf.summary.scalar('reg_loss', reg_loss)
        else:
            loss = cross_entropy
        tf.summary.scalar('total_loss', loss)

    ########################
    # END OF YOUR CODE    #
    #######################

    return loss

  def train_step(self, loss, flags):
    """
    Implements a training step using a parameters in flags.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:
      train_step: TensorFlow operation to perform one training step
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    optimizer = flags['optimizer']
    global_step = flags['global_step']

    grads = optimizer.compute_gradients(loss)
    train_step = optimizer.apply_gradients(grads_and_vars = grads, global_step = global_step)

    ########################
    # END OF YOUR CODE    #
    #######################

    return train_step

  def accuracy(self, logits, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    As in self.loss above, you can use tf.summary.scalar to save
    scalar summaries of accuracy for later use with the TensorBoard.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.
    Returns:
      accuracy: scalar float Tensor, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    pred_class = tf.argmax(input = logits, axis = 1)
    true_class = tf.argmax(input = labels, axis = 1)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_class, true_class), tf.float32), name = 'accuracy')

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('pred_class', pred_class)
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy

  def confusion_matrix(self, logits, labels):

    pred_class = tf.argmax(input = logits, axis = 1)
    true_class = tf.argmax(input = labels, axis = 1)

    confusion_matrix = tf.contrib.metrics.confusion_matrix(
            labels = true_class,
            predictions = pred_class,
            num_classes = self.n_classes,
            dtype = tf.int32,
            name = 'confusion_matrix')

    tf.summary.image('confusion_matrix',
                     tf.reshape(tf.cast(confusion_matrix, dtype=tf.float32),
                               [1, self.n_classes, self.n_classes, 1]))

    return confusion_matrix
