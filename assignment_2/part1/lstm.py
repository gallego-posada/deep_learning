# MIT License
#
# Copyright (c) 2017 Tom Runia

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


class LSTM(object):
    """
    LSTM class
    """

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

        with tf.variable_scope('lstm_cell'):
            Wg = tf.get_variable(name="Wg",
                                 shape=[self._input_dim + self._num_hidden, self._num_hidden],
                                 dtype=tf.float32,
                                 initializer=self._weight_initializer)
            Wi = tf.get_variable(name="Wi",
                                 shape=[self._input_dim + self._num_hidden, self._num_hidden],
                                 dtype=tf.float32,
                                 initializer=self._weight_initializer)
            Wf = tf.get_variable(name="Wf",
                                 shape=[self._input_dim + self._num_hidden, self._num_hidden],
                                 dtype=tf.float32,
                                 initializer=self._weight_initializer)
            Wo = tf.get_variable(name="Wo",
                                 shape=[self._input_dim + self._num_hidden, self._num_hidden],
                                 dtype=tf.float32,
                                 initializer=self._weight_initializer)

            bg = tf.get_variable(name="bg", shape=[self._num_hidden],
                                 dtype=tf.float32, initializer=self._weight_initializer)
            bi = tf.get_variable(name="bi", shape=[self._num_hidden],
                                 dtype=tf.float32, initializer=self._weight_initializer)
            bf = tf.get_variable(name="bf", shape=[self._num_hidden],
                                 dtype=tf.float32, initializer=self._weight_initializer)
            bo = tf.get_variable(name="bo", shape=[self._num_hidden],
                                 dtype=tf.float32, initializer=self._weight_initializer)

            self._W = tf.concat([Wg, Wi, Wf, Wo], axis=1)
            self._b = tf.concat([bg, bi, bf, bo], axis=0)

        with tf.variable_scope("logits"):
            self._Woh = tf.get_variable(name='Woh',
                                        shape=[self._num_hidden, self._num_classes],
                                        dtype=tf.float32,
                                        initializer=self._weight_initializer)

            self._bo = tf.get_variable(name='bo',
                                       shape=[self._num_classes],
                                       dtype=tf.float32,
                                       initializer=self._bias_initializer)

        self._logits = self.compute_logits()
        self._loss = self.compute_loss()
        self._accuracy = self.accuracy()

    def _lstm_step(self, lstm_state_tuple, x):
        """
        Single step through LSTM cell
        """

        (c_prev, h_prev) = tf.unstack(lstm_state_tuple, axis=0)
        x_and_h = tf.concat([x, h_prev], axis=1)

        # Execute big matmul for efficiency. See notation in section 3.1 in
        # (Zaremba, 2015) https://arxiv.org/pdf/1409.2329.pdf
        preact = tf.add(tf.matmul(x_and_h, self._W), self._b, name='preact')

        # Split tensor into the four vector components
        g, i, f, o = tf.split(preact, num_or_size_splits=4, axis=1)

        # Calculate new cell and hidden states
        c = tf.tanh(g) * tf.sigmoid(i) + c_prev * tf.sigmoid(f)
        h = tf.tanh(c) * tf.sigmoid(o)

        return tf.stack([c, h], axis=0)

    def _get_hidden_states(self):
        with tf.variable_scope('hidden_states'):
            initial_state = tf.zeros([2, self._batch_size, self._num_hidden], name='initial_state')
            #states = tf.scan(fn=lambda lstm_state_tuple, x: self._lstm_step(lstm_state_tuple=lstm_state_tuple, x=x),
            states = tf.scan(fn=self._lstm_step,
                             elems=self._inputs,
                             initializer=initial_state,
                             name='hidden_states')
        return states

    def compute_logits(self):
        """
        Implement the logits for predicting the last digit in the palindrome
        """

        # Get hidden states
        states = self._get_hidden_states()
        # Keep only last time step
        states = states[-1, ...]
        # Separate cell and hidden states
        (_, h) = tf.unstack(states, axis=0)

        logits = tf.add(tf.matmul(h, self._Woh), self._bo, name="logits")
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
