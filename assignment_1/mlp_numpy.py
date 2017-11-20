"""
This module implements a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_hidden, n_classes, weight_decay=0.0,
               weight_scale=0.0001, input_dim=3 * 32 * 32, learning_rate= 2e-3):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter. Weights of the linear layers should be initialized
    using normal distribution with mean = 0 and std = weight_scale. Biases should be
    initialized with constant 0. All activation functions are ReLUs.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      weight_decay: L2 regularization parameter for the weights of linear layers.
      weight_scale: scale of normal distribution to initialize weights.

    """
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.weight_decay = weight_decay
    self.weight_scale = weight_scale
    self.input_dim = input_dim
    self.learning_rate = learning_rate

    # Set weight and bias shapes given specified sizes
    # W(k) has shape dim(k+1) x dim(k)
    # b(k) has shape dim(k+1) x 1
    # Convention: the input has shape input_dim x batch_size
    # h(k) = W(k) * h(k-1) + b(k)
    layer_sizes = [self.input_dim] + self.n_hidden + [self.n_classes]
    weight_shapes = [(layer_sizes[i+1], layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
    bias_shapes = [(shape[0], 1) for shape in weight_shapes]

    # Initialize weights and biases with default initializers
    weights = [self._weight_initializer(shape, self.weight_scale)
               for shape in weight_shapes]
    biases = [self._bias_initializer(shape) for shape in bias_shapes]

    # Define activation function and its derivative per layer
    activations = [self._relu] * len(self.n_hidden) + [self._linear]
    act_derivatives = [self._relu_der] * len(self.n_hidden) + [self._linear_der]

    # Use FCLayer wrappers to setup network
    self.layers = [FCLayer(W = weights[i], b = biases[i], act_fn = activations[i],
                   act_der = act_derivatives[i], name = "fc" + str(i), model = self)
                   for i in range(len(weight_shapes))]

  # Activation functions and their derivatives
  def _relu(self, x):
      return x * (x>0)

  def _linear(self, x):
      return x

  def _relu_der(self, x):
      return 1.0 * (x>0)

  def _linear_der(self, x):
      return np.ones(shape = x.shape)

  # Weight and bias initializers
  def _weight_initializer(self, weight_shape, weight_scale):
      return np.random.normal(scale = weight_scale, size = weight_shape)

  def _bias_initializer(self, bias_shape):
      return np.zeros(shape = bias_shape)

  def inference(self, x):
    """
    Performs inference given an input array. This is the central portion
    of the network. Here an input array is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    It can be useful to save some intermediate results for easier computation of
    gradients for backpropagation during training.
    Args:
      x: 2D float array of size [batch_size, input_dim]

    Returns:
      logits: 2D float array of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Make sure convention for input shape holds
    if x.shape[0] != self.input_dim:
        data_tensor = x.T
    else:
        data_tensor = x

    # Forward data tensor through the network
    for layer in self.layers:
        layer.forward(data_tensor)
        data_tensor = layer.act

    # Loss function expects logits in transposed shape
    logits = data_tensor.T

    assert np.isfinite(logits).all(), "Numerical instability in logits"

    ########################
    # END OF YOUR CODE    #
    #######################

    return logits

  def _log_sum_exp(self, logits):
      """
      Computes the log-sum-exp trick on logits. Assumes logits is a matrix of
      dimensions batch_size x n_classes.
      Reference: https://en.wikipedia.org/wiki/LogSumExp
      """

      # Extract maximum score per row
      row_max = np.max(logits, axis = 1, keepdims = True)
      logits_minus_max = logits - row_max

      # Apply LSE trick
      lse = row_max + np.expand_dims(np.log(np.einsum('ij->i',np.exp(logits_minus_max))), axis = 1)
      return lse

  def _cross_entropy_loss(self, logits, labels):
      lse = self._log_sum_exp(logits)
      log_prob = logits - lse
      return - np.einsum('ij,ij->', labels, log_prob), np.exp(log_prob)

  def _reg_loss(self):
      return np.sum([layer.complexity_penalty() for layer in self.layers])

  def loss(self, logits, labels):
    """
    Computes the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    It can be useful to compute gradients of the loss for an easier computation of
    gradients for backpropagation during training.

    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.
    Returns:
      loss: scalar float, full loss = cross_entropy + reg_loss
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = logits.shape[0]


    ce_loss, pred_probs = self._cross_entropy_loss(logits, labels)
    ce_loss /= batch_size

    reg_loss = self._reg_loss()
    loss = ce_loss + self.weight_decay * reg_loss
    assert np.isfinite(loss), "Numerical instability in logits"

    self.delta_out = (pred_probs - labels).T / batch_size

    ########################
    # END OF YOUR CODE    #
    #######################

    return loss

  def train_step(self, loss= None, flags = None):
    """
    Implements a training step using a parameters in flags.
    Use mini-batch Stochastic Gradient Descent to update the parameters of the MLP.

    Args:
      loss: scalar float.
      flags: contains necessary parameters for optimization.
    Returns:

    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    deltas = [self.delta_out]

    layer_index = list(range(1,len(self.layers)))
    layer_index.reverse()

    for k in layer_index:
        current_layer = self.layers[k-1]
        following_layer = self.layers[k]
        current_delta = following_layer.W.T.dot(deltas[0]) * current_layer.act_der(current_layer.preact)
        deltas = [current_delta] + deltas

    for k in range(len(self.layers)):
        #print(k,self.layers[k].name)
        self.layers[k].backward(deltas[k])

    self.delta_out = None

    ########################
    # END OF YOUR CODE    #
    #######################

  def accuracy(self, logits, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = logits.shape[0]

    accuracy = np.sum(np.argmax(logits, axis = 1) == np.argmax(labels, axis = 1))
    accuracy /= batch_size

    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


class FCLayer(object):
    """
    Wrapper for fully-connected layers in the MLP architecture
    """

    def __init__(self, W, b, act_fn, act_der, name, model):

        self.W = W
        self.b = b

        self.act_fn = act_fn
        self.act_der = act_der
        self.name = name

        self.model = model

        self.input = None
        self.preact = None
        self.act = None

    def forward(self, input):

        assert input.shape[0] == self.W.shape[1], "Dimension mismatch  in layer \
            {} with weight size {} and input size {}".format(self.name, self.W.shape, input.shape)

        self.input = input
        self.preact = np.dot(self.W, input) + self.b
        self.act = self.act_fn(self.preact)

    def backward(self, delta):

        grad_W = delta.dot(self.input.T) + self.model.weight_decay * self.W
        grad_b = delta.sum(axis=1, keepdims=True)

        self.W -= self.model.learning_rate * grad_W
        self.b -= self.model.learning_rate * grad_b

    def complexity_penalty(self):
        return  0.5 * np.linalg.norm(self.W)**2
