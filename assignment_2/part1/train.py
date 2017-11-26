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
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import os
import numpy as np
import tensorflow as tf

import utils
from vanilla_rnn import VanillaRNN
from lstm import LSTM


################################################################################
def dense_to_one_hot(labels_dense, num_classes):
    """
    Convert class labels from scalars to one-hot vectors.
    Args:
    labels_dense: Dense labels.
    num_classes: Number of classes.

    Outputs:
    labels_one_hot: One-hot encoding for labels.
    """

    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def _check_path(path):
    """
    Makes sure path for log and model saving exists
    """
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)

def train(config):
    assert config.model_type in ('RNN', 'LSTM')

    tf.reset_default_graph()

    # Setup the model that we are going to use
    if config.model_type == 'RNN':
        print("Initializing Vanilla RNN model...")
        model = VanillaRNN(
            config.input_length - 1, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )
    else:
        print("Initializing LSTM model...")
        model = LSTM(
            config.input_length - 1, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )

    ###########################################################################
    # Implement code here.
    ###########################################################################

    sess = tf.Session()

    # Setup global step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Define the optimizer
    assert config.optimizer in ('adam', 'rmsprop')
    if config.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
    elif config.optimizer == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(config.learning_rate)

    # Define summary operation
    summary_op = tf.summary.merge_all()

    ###########################################################################
    # QUESTION: what happens here and why?
    # ANSWER: we calculate the gradients and clip each of them if the magnitude
    #         is larger than config.max_norm_gradient to avoid unstable learning
    #         due to exploding gradients in cliffs in the loss surface.
    ###########################################################################
    grads_and_vars = optimizer.compute_gradients(model._loss)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables),
                                                   global_step=global_step)
    ############################################################################

    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    sess.run(fetches=[init_op, local_init_op])

    ###########################################################################
    # Implement code here.
    ###########################################################################

    train_log_path = os.path.join(config.summary_path, '{}'.format(config.name))
    _check_path(train_log_path)
    train_log_writer = tf.summary.FileWriter(train_log_path, graph=sess.graph)

    palindrome_length = config.input_length

    for train_step in range(config.train_steps):

        # Only for time measurement of step through network
        t1 = time.time()

        palindrome_batch = utils.generate_palindrome_batch(config.batch_size, palindrome_length)
        x_dense = palindrome_batch[:, :-1]
        y_dense = palindrome_batch[:, -1]

        x = np.transpose((np.arange(model._num_classes) == x_dense[..., None]).astype(int), [1, 0, 2])
        y = (np.arange(model._num_classes) == y_dense[..., None]).astype(int)

        tr_feed = {model._inputs: x, model._targets: y}
        fetches = [apply_gradients_op, model._loss, model._accuracy]

        if train_step % config.print_every == 0:
            fetches += [summary_op]
            _, train_loss, train_accuracy, train_summary = sess.run(fetches = fetches, feed_dict = tr_feed)
            train_log_writer.add_summary(train_summary, train_step)
        else:
            _, train_loss, train_accuracy, = sess.run(fetches = fetches, feed_dict = tr_feed)

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Print the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                  "Examples/Sec = {:.2f}, Accuracy = {:.3f}, Loss = {:.4f}".format(
                      datetime.now().strftime("%Y-%m-%d %H:%M"), train_step,
                      config.train_steps, config.batch_size, examples_per_second,
                      train_accuracy, train_loss))

    train_log_writer.close()
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2500, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=10.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')

    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer to use')
    parser.add_argument('--name', type=str, default="recurrent", help='Model name')

    config = parser.parse_args()

    # Train the model
    train(config)
