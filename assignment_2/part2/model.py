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

class TextGenerationModel(object):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size

        self._inputs = tf.placeholder(tf.float32,
                                      shape=[self._input_length, self._batch_size, self._input_dim],
                                      name='inputs')

        self._targets = tf.placeholder(tf.float32,
                                       shape=[self._batch_size, self._num_classes],
                                       name='targets')

        init_state = tf.placeholder(tf.float32, [self._lstm_num_layers, 2, self._batch_size, self._lstm_num_hidden])
        state_per_layer_list = tf.unpack(init_state, axis=0)

        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )


    def _build_model(self):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]

        # create 2 LSTMCells
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=data,
                                           dtype=tf.float32)

        logits_per_step = None
        return logits_per_step

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

    def probabilities(self):
        # Returns the normalized per-step probabilities
        probabilities = None
        return probabilities

    def predictions(self):
        # Returns the per-step predictions
        return = tf.argmax(input=self._logits, axis=1)
