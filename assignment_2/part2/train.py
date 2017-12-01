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

import os
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

from dataset import TextDataset
from model import TextGenerationModel

def _check_path(path):
    """
    Makes sure path for log and model saving exists
    """
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)

def train(config):

    # Initialize the text dataset
    dataset = TextDataset(config.txt_file)

    # Initialize the model
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers,
        dropout_keep_prob=config.dropout_keep_prob,
        prediction_mode=config.prediction_mode
    )

    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Placeholders for model sampling
    init_sample_char = tf.placeholder(dtype=tf.int32, shape=(config.num_samples))
    seq_samples = model._sample(init_input=init_sample_char,
                                num_samples=config.num_samples,
                                sample_length=config.sample_length,
                                init_state=None)

    init_sentence = tf.placeholder(dtype=tf.int32, shape=(None, 1))
    completed_sentence = model._complete_sentence(init_sentence, config.sample_length)


    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem_frac, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))

    # Setup global step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Define the optimizer
    if config.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate, decay=config.learning_rate_decay)
    elif config.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(config.learning_rate)

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(model._loss)
    #train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)

    # Saver
    saver = tf.train.Saver(max_to_keep=50)
    save_path = os.path.join(config.save_path, '{}/model.ckpt'.format(config.name))
    _check_path(save_path)

    # Initialization
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    sess.run(fetches=[init_op, local_init_op])

    # Define summary operation
    summary_op = tf.summary.merge_all()

    # Logs
    train_log_path = os.path.join(config.summary_path, '{}'.format(config.name))
    _check_path(train_log_path)
    train_log_writer = tf.summary.FileWriter(train_log_path, graph=sess.graph)

    ###########################################################################
    # Implement code here.
    ###########################################################################

    print(" ******* DICTIONARY ******* ")
    print(dataset._ix_to_char)

    for train_step in range(int(config.train_steps)):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################################
        # Implement code here.
        #######################################################################
        x, y = dataset.batch(batch_size=config.batch_size, seq_length=config.seq_length)

        tr_feed = {model._inputs: x, model._targets: y}
        fetches = [apply_gradients_op, model._loss]

        if train_step % config.print_every == 0:
            fetches += [summary_op]
            _, train_loss, summary = sess.run(feed_dict=tr_feed, fetches=fetches)
            train_log_writer.add_summary(summary, train_step)
        else:
            _, train_loss = sess.run(feed_dict=tr_feed, fetches=fetches)

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Output the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {:.4f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step,
                int(config.train_steps), config.batch_size,
                examples_per_second, train_loss))

        # Sample sentences from the model
        if train_step % config.sample_every == 0:

            # Random initial character
            init_chars = np.random.choice(a=dataset.vocab_size, size=(config.num_samples))
            sampled_seq = sess.run(fetches=[seq_samples], feed_dict={init_sample_char: init_chars})[0]
            sampled_seq = np.array(sampled_seq).T
            print("\n ******* Random Initial Character *******")
            for i in range(config.num_samples):
                print('{} - {}|{}'.format(i, dataset._ix_to_char[init_chars[i]], dataset.convert_to_string(sampled_seq[i, :])))

            #Custom sentences
            custom_inits = ['To be, or not to be, that is the question: Whether ',
                            'History will be kind to me for I intend to ',
                            'Hansel and Gr',
                            'Democracy is ',
                            'Let T be a bounded linear operator in V, a vector space.',
                            'Mas vale pajaro en mano que ver un ciento v']

            print("\n ******* Sentence Completion *******")
            for init_seq in custom_inits:
                init_vec = np.array([dataset._char_to_ix[x] for x in init_seq if x in dataset._char_to_ix]).reshape((-1, 1))
                sampled_seq = sess.run(fetches=[completed_sentence], feed_dict={init_sentence: init_vec})[0]
                print('{}|{}'.format(init_seq, dataset.convert_to_string(sampled_seq.squeeze().tolist())))

            print("\n")
        # Save checkpoint
        if train_step % config.save_every == 0 and train_step > 1:
            saver.save(sess, save_path=save_path)

    train_log_writer.close()

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=10, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=250, help='How often to sample from the model')

    parser.add_argument('--save_every', type=int, default=500, help='How often to save the model')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Output path for model checkpoints')

    parser.add_argument('--optimizer', type=str, default="rmsprop", choices=['adam', 'rmsprop'], help='Optimizer to use')
    parser.add_argument('--name', type=str, default="model", help='Model name')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of randomly initialized sample sequences')
    parser.add_argument('--sample_length', type=int, default=100, help='Length of sampled sequence')

    parser.add_argument('--prediction_mode', type=str, default="sample", choices=['sample', 'max'], help='Length of sampled sequence')

    config = parser.parse_args()

    # Train the model
    train(config)
