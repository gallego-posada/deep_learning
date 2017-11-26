# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-10-19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

################################################################################

class VanillaRNN(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim = input_dim
        self._num_hidden = num_hidden
        self._num_classes = num_classes
        self._batch_size = batch_size

        self._weight_initializer = tf.variance_scaling_initializer()
        self._bias_initializer = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        self._inputs = tf.placeholder(tf.float32,
                                      shape=[self._input_length, self._batch_size, self._input_dim],
                                      name='inputs')

        self._targets = tf.placeholder(tf.float32,
                                       shape=[self._batch_size, self._num_classes],
                                       name='targets')

        self._logits = self.compute_logits()
        self._loss = self.compute_loss()
        self._accuracy = self.accuracy()

    def _rnn_step(self, h_prev, x):
        """
        Single step through Vanilla RNN cell
        """

        with tf.variable_scope('rnn_step'):

            Whh = tf.get_variable(name='Whh',
                                  shape=[self._num_hidden, self._num_hidden],
                                  dtype=tf.float32,
                                  initializer=self._weight_initializer)

            Whx = tf.get_variable(name='Whx',
                                  shape=[self._input_dim, self._num_hidden],
                                  dtype=tf.float32,
                                  initializer=self._weight_initializer)

            bh = tf.get_variable(name='bh',
                                 shape=[self._num_hidden],
                                 dtype=tf.float32,
                                 initializer=self._bias_initializer)

            mmulx = tf.matmul(x, Whx)
            mmulh = tf.matmul(h_prev, Whh)
            preact = mmulx + mmulh + bh
            h = tf.tanh(preact)

        return h

    def _get_hidden_states(self):
        with tf.variable_scope('hidden_states'):
            initial_state = tf.zeros([self._batch_size, self._num_hidden], name='initial_state')
            states = tf.scan(self._rnn_step, self._inputs,
                             initializer=initial_state, name='hidden_states')
        return states

    def compute_logits(self):
        """
        Implement the logits for predicting the last digit in the palindrome
        """

        states = self._get_hidden_states()
        states = states[-1, :, :]

        with tf.variable_scope("logits"):
            Woh = tf.get_variable(name='Woh',
                                  shape=[self._num_hidden, self._num_classes],
                                  dtype=tf.float32,
                                  initializer=self._weight_initializer)

            bo = tf.get_variable(name='bo',
                                 shape=[self._num_classes],
                                 dtype=tf.float32,
                                 initializer=self._bias_initializer)

            logits = tf.add(tf.matmul(states, Woh), bo, name="logits")
        return logits

    def compute_loss(self):
        """
        Implement the cross-entropy loss for classification of the last digit
        """

        with tf.variable_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self._targets,
                                                           logits=self._logits)
            loss = tf.reduce_mean(loss, name="softmax")
            tf.summary.scalar('mean_cross_entropy', loss)

        return loss

    def accuracy(self):
        """
        Implement the accuracy of predicting the last digit over the current batch
        """
        with tf.variable_scope("accuracy"):
            pred_class = tf.argmax(input=self._logits, axis=1)
            true_class = tf.argmax(input=self._targets, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_class, true_class), tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('pred_class', pred_class)

        return accuracy
