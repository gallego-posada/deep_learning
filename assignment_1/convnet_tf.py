"""
This module implements a convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer

class ConvNet(object):
  """
  This class implements a convolutional neural network in TensorFlow.
  It incorporates a certain graph model to be trained and to be used
  in inference.
  """

  def __init__(self, n_classes, weight_initializer = xavier_initializer(), weight_regularizer = l2_regularizer(0.001)):
    """
    Constructor for an ConvNet object. Default values should be used as hints for
    the usage of each parameter.
    Args:
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the ConvNet.
    """
    self.n_classes = n_classes
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

  def _conv_layer(self, x, layer_name):

      with tf.variable_scope(layer_name):
          conv = tf.layers.conv2d(inputs = x,
                                  filters = 64,
                                  kernel_size = (5, 5),
                                  strides = (1, 1),
                                  padding = 'same',
                                  activation = None,
                                  use_bias = True,
                                  kernel_initializer = self.weight_initializer,
                                  bias_initializer = self.bias_initializer,
                                  kernel_regularizer = self.weight_regularizer,
                                  bias_regularizer = None,
                                  data_format ='channels_last',
                                  name = layer_name + '_conv')

          activs = tf.nn.relu(conv, name = layer_name + '_act')

          pool = tf.layers.max_pooling2d(inputs = activs,
                                         pool_size = (3, 3),
                                         strides = 2,
                                         padding = 'valid',
                                         data_format ='channels_last',
                                         name = layer_name + '_pool')

          #tf.summary.histogram('preactivations', conv)
          #tf.summary.histogram('activations', activs)
          #tf.summary.histogram('pooled', pool)

        #   with tf.name_scope('weights'):
        #       w = tf.get_default_graph().get_tensor_by_name(os.path.split(conv.name)[0] + '/kernel:0')
        #       print(os.path.split(conv.name)[0] + '/kernel:0', w.shape)
        #       self._variable_summaries(w)
          #
        #   with tf.name_scope('biases'):
        #       b = tf.get_default_graph().get_tensor_by_name(os.path.split(conv.name)[0] + '/bias:0')
        #       print(os.path.split(conv.name)[0] + '/bias:0', b.shape)
        #       self._variable_summaries(b)


      return pool

  def _fc_layer(self, x, units, layer_name, keep_prob, act = tf.nn.relu):

      with tf.variable_scope(layer_name):
          fc = tf.layers.dense(inputs = x,
                                units = units,
                                activation = None,
                                use_bias = True,
                                kernel_initializer = self.weight_initializer,
                                bias_initializer = self.bias_initializer,
                                kernel_regularizer = self.weight_regularizer,
                                bias_regularizer = None,
                                trainable = True,
                                name = layer_name + '_fc')

          activs = act(fc, name = layer_name + '_act')

        #   tf.summary.histogram('preactivations', fc)
        #   tf.summary.histogram('activations', activs)
          #
        #   with tf.name_scope('weights'):
        #       w = tf.get_default_graph().get_tensor_by_name(os.path.split(fc.name)[0] + '/kernel:0')
        #       print(os.path.split(fc.name)[0] + '/kernel:0', w.shape)
        #       self._variable_summaries(w)
          #
        #   with tf.name_scope('biases'):
        #       b = tf.get_default_graph().get_tensor_by_name(os.path.split(fc.name)[0] + '/bias:0')
        #       print(os.path.split(fc.name)[0] + '/bias:0', b.shape)
        #       self._variable_summaries(b)

          with tf.name_scope('dropout'):
              output = tf.nn.dropout(activs, keep_prob)
              #tf.summary.histogram('output', output)

      return output

  def inference(self, x, keep_prob):
    """
    Performs inference given an input tensor. This is the central portion
    of the network where we describe the computation graph. Here an input
    tensor undergoes a series of convolution, pooling and nonlinear operations
    as defined in this method. For the details of the model, please
    see assignment file.

    Here we recommend you to consider using variable and name scopes in order
    to make your graph more intelligible for later references in TensorBoard
    and so on. You can define a name scope for the whole model or for each
    operator group (e.g. conv+pool+relu) individually to group them by name.
    Variable scopes are essential components in TensorFlow for parameter sharing.
    Although the model(s) which are within the scope of this class do not require
    parameter sharing it is a good practice to use variable scope to encapsulate
    model.

    Args:
      x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
              the logits outputs (before softmax transformation) of the
              network. These logits can then be used with loss and accuracy
              to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################

    pool_1 = self._conv_layer(x, "conv_1")
    pool_2 = self._conv_layer(pool_1, "conv_2")

    with tf.name_scope('flatten') as scope:
            flat = tf.contrib.layers.flatten(pool_2, scope = scope)

    fc_1 = self._fc_layer(flat, 384, "fc_1", keep_prob)
    fc_2 = self._fc_layer(fc_1, 192, "fc_2", keep_prob)
    fc_3 = self._fc_layer(fc_2, 10, "fc_3", keep_prob, act = tf.identity)

    logits = fc_3
    print(pool_1.shape, pool_2.shape, flat.shape, fc_1.shape, fc_2.shape, fc_3.shape)
    ########################
    # END OF YOUR CODE    #
    ########################
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
    Calculates the multiclass cross-entropy loss from the logits predictions and
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
    ########################
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
    ########################

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
    Calculate the prediction accuracy, i.e. the average correct predictions
    of the network.
    As in self.loss above, you can use tf.scalar_summary to save
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
    ########################
    pred_class = tf.argmax(input = logits, axis = 1)
    true_class = tf.argmax(input = labels, axis = 1)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_class, true_class), tf.float32), name = 'accuracy')

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('pred_class', pred_class)
    ########################
    # END OF YOUR CODE    #
    ########################

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
