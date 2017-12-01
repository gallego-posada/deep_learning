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
                 lstm_num_hidden, lstm_num_layers, dropout_keep_prob, prediction_mode):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size
        self._prediction_mode = prediction_mode
        self._dropout_keep_prob = dropout_keep_prob

        # Initialization:
        # None = _seq_length
        self._inputs = tf.placeholder(tf.int32,
                                      shape=[self._batch_size, None],
                                      name='inputs')
        self._input_onehot = tf.one_hot(self._inputs, depth=self._vocab_size)
        self._targets = tf.placeholder(tf.int32,
                                       shape=[self._batch_size, None],
                                       name='targets')

        # Set up conditions for sampling sequences from this model
        self.sample_length = tf.placeholder(dtype=tf.int32, shape=())
        self._rand_seq_init = tf.placeholder(dtype=tf.int32, shape=(self._batch_size))
        self._rand_seq_init_onehot = tf.one_hot(self._rand_seq_init, depth=self._vocab_size)

        self._rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(
                                tf.nn.rnn_cell.LSTMCell(num_units=self._lstm_num_hidden, activation=tf.nn.relu),
                                output_keep_prob=self._dropout_keep_prob) for _ in range(self._lstm_num_layers)]

        self._multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(self._rnn_layers)

        self._Wout = tf.get_variable(name='Wout',
                                     shape=[self._lstm_num_hidden, self._vocab_size],
                                     dtype=tf.float32,
                                     initializer=tf.variance_scaling_initializer())

        self._bout = tf.get_variable(name='bout',
                                     shape=[self._vocab_size],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))

        self._logits_per_step, self.cell_state = self._build_model()
        self._loss = self._compute_loss()
        self._probabilities = self._compute_probabilities()
        self._predictions = self._compute_predictions(self._logits_per_step)

    def _build_model(self):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]

        # 'outputs' is a tensor of shape [_batch_size, _seq_length, _lstm_num_hidden]
        # 'state' is a _lstm_num_layers-tuple with one tf.contrib.rnn.LSTMStateTuple
        # for each cell one
        outputs, state = tf.nn.dynamic_rnn(cell=self._multi_rnn_cell,
                                           inputs=self._input_onehot,
                                           dtype=tf.float32)

        h = tf.reshape(outputs, [-1, self._lstm_num_hidden])
        logits = self._compute_logits(preacts=h)
        logits_per_step = tf.reshape(logits, [self._seq_length, self._batch_size, self._vocab_size])
        return logits_per_step, state

    def _compute_logits(self, preacts):
        logits = tf.add(tf.matmul(preacts, self._Wout), self._bout, name="logits")
        return logits

    def _compute_loss(self):
        # Cross-entropy loss, averaged over timestep and batch
        with tf.variable_scope("loss"):
            loss = tf.contrib.seq2seq.sequence_loss(logits=self._logits_per_step,
                                                    targets=self._targets,
                                                    weights=tf.ones(shape=(self._batch_size, self._seq_length)))
            tf.summary.scalar('mean_cross_entropy', loss)
        return loss

    def _compute_probabilities(self):
        # Returns the normalized per-step probabilities
        probabilities = tf.nn.softmax(self._logits_per_step)
        return probabilities

    def _compute_predictions(self, logits):
        # Returns the per-step predictions
        if self._prediction_mode == "sample":
            logits /= 0.8 #Temperature adjustment for diversity control
            predictions = tf.distributions.Categorical(logits=logits).sample()
        elif self._prediction_mode == "max":
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions

    def _sample(self, init_input, num_samples, sample_length=30, init_state=None):

        original_seq_length = self._seq_length
        self._seq_length = 1

        if init_state is None:
            state = self._multi_rnn_cell.zero_state(batch_size=num_samples, dtype=tf.float32)
        else:
            state = init_state

        output = init_input
        samples = [init_input]
        for _ in range(sample_length):
            input_onehot = tf.one_hot(output, depth=self._vocab_size)
            preacts, state = self._multi_rnn_cell(inputs=input_onehot, state=state)
            logits = tf.add(tf.matmul(preacts, self._Wout), self._bout)
            output = self._compute_predictions(logits)
            samples.append(output)

        self._seq_length = original_seq_length
        return tf.stack(samples)


    def _complete_sentence(self, sentence_init, sample_length=30):
        zero_state = self._multi_rnn_cell.zero_state(batch_size=sentence_init.shape[1],
                                                     dtype=tf.float32)

        input_onehot = tf.one_hot(sentence_init, depth=self._vocab_size)

        preacts, new_init_state = tf.nn.dynamic_rnn(cell=self._multi_rnn_cell,
                                                    initial_state=zero_state,
                                                    inputs=input_onehot,
                                                    time_major=True)

        h = tf.reshape(preacts, [-1, self._lstm_num_hidden])
        logits = tf.add(tf.matmul(h, self._Wout), self._bout)
        new_init_char = self._compute_predictions(logits[-1, ...])

        return self._sample(tf.expand_dims(new_init_char, 0), 1, sample_length, init_state=new_init_state)
