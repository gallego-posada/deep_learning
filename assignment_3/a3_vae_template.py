import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

WEIGHT_INITIALIZATION_DICT = {'glorot_uniform': tf.contrib.layers.xavier_initializer(uniform=True),  # Xavier initialisation
                              'normal': lambda scale: tf.random_normal_initializer(stddev=scale), # Initialization from a standard normal
                              'uniform': lambda scale: tf.random_uniform_initializer(minval=-scale, maxval=scale),  # Initialization from a uniform distribution
                              }

WEIGHT_REGULARIZER_DICT = {'none': None,  # No regularization
                           'l1': lambda scale: tf.contrib.layers.l1_regularizer(scale=scale),  # L1 regularization
                           'l2': lambda scale: tf.contrib.layers.l2_regularizer(scale=scale)  # L2 regularization
                           }

ACTIVATION_DICT = {'relu': tf.nn.relu,  # ReLU
                   'elu': tf.nn.elu,  # ELU
                   'tanh': tf.nn.tanh,  # Tanh
                   'sigmoid': tf.nn.sigmoid}  # Sigmoid

OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,  # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,  # Adagrad
                  'adam': tf.train.AdamOptimizer,  # Adam
                  'rmsprop': tf.train.RMSPropOptimizer  # RMSprop
                 }

def load_mnist_images(binarize=True):
    """
    :param binarize: Turn the images into binary vectors
    :return: x_train, x_test  Where
        x_train is a (55000 x 784) tensor of training images
        x_test is a  (10000 x 784) tensor of test images
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    x_train = mnist.train.images
    x_test = mnist.test.images
    if binarize:
        x_train = (x_train>0.5).astype(x_train.dtype)
        x_test = (x_test>0.5).astype(x_test.dtype)
    return x_train, x_test

def _check_path(path):
    """
    Makes sure path for log and model saving exists
    """
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)

class VariationalAutoencoder(object):


    def __init__(self, x_dim, z_dim, encoder_hidden_sizes, decoder_hidden_sizes,
                 hidden_activation, kernel_initializer):

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.hidden_activation = hidden_activation
        self.kernel_initializer = kernel_initializer

        self.inputs = tf.placeholder(shape=[None, x_dim], dtype=tf.float32)
        _, loss = self.lower_bound(self.inputs)
        self.loss = loss

    def encoder(self, x):
        w_init = self.kernel_initializer
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            for units in self.encoder_hidden_sizes:
                h = tf.layers.dense(h, units, activation=self.hidden_activation, kernel_initializer=w_init)
            z_mean = tf.layers.dense(h, self.z_dim, activation=tf.identity, kernel_initializer=w_init)
            z_log_var = tf.layers.dense(h, self.z_dim, activation=tf.identity, kernel_initializer=w_init)
        return z_mean, z_log_var

    def decoder(self, z):
        w_init = self.kernel_initializer
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            for units in self.decoder_hidden_sizes:
                h = tf.layers.dense(h, units, activation=self.hidden_activation, kernel_initializer=w_init)
            x_mean = tf.layers.dense(h, self.x_dim, activation=tf.sigmoid, kernel_initializer=w_init)
        return x_mean

    def lower_bound(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of the lower-bound on the log-probability of each data point
        """
        z_mean, z_log_var = self.encoder(x)

        eps = tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
        z = z_mean + eps * tf.exp(z_log_var/2.)

        x_hat = self.decoder(z)
        x_hat = tf.clip_by_value(x_hat, 1e-7, 1 - 1e-7)

        rec_loss = tf.reduce_sum(x * tf.log(x_hat) + (1 - x) * tf.log(1 - x_hat), 1)
        kl_loss = 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1)

        elbo = rec_loss + kl_loss
        avg_loss = - tf.reduce_mean(elbo)

        tf.summary.scalar('lower_bound', -avg_loss)
        tf.summary.scalar('reconstruction', tf.reduce_mean(rec_loss))
        tf.summary.scalar('divergence', tf.reduce_mean(kl_loss))

        return elbo, avg_loss

    def mean_x_given_z(self, z):
        """
        :param z: A (n_samples, n_dim_z) tensor containing a set of latent data points (n_samples, n_dim_z)
        :return: A (n_samples, n_dim_x) tensor containing the mean of p(X|Z=z) for each of the given points
        """
        return self.decoder(z)

    def sample(self, n_samples, sample_or_mean = 'mean'):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        z = tf.random_normal((n_samples, self.z_dim), 0, 1, dtype=tf.float32)
        x_distro = tf.contrib.distributions.Bernoulli(probs=self.mean_x_given_z(z))

        if sample_or_mean == 'sample':
            samples = x_distro.sample()
        elif sample_or_mean == 'mean':
            samples = x_distro.mean()
        return samples

def plot_image_grid(data_tensor, im_h, im_w, hor_ims, vert_ims):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    reshaped_tensor = np.zeros((int(im_h * vert_ims), int(im_w * hor_ims)))
    for row in range(vert_ims):
        for col in range(hor_ims):
            col_inf, col_sup = (int(col*im_w), int((col+1)*im_w))
            row_inf, row_sup = (int(row*im_w), int((row+1)*im_w))
            reshaped_im = np.reshape(data_tensor[int(col + hor_ims * row), :], (im_h, im_w))
            reshaped_tensor[row_inf:row_sup, col_inf:col_sup] = reshaped_im
    plt.imshow(reshaped_tensor, cmap='gray')
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
    plt.show()

def train_vae_on_mnist(z_dim=2, kernel_initializer='glorot_uniform', optimizer = 'adam',  learning_rate=0.001, n_epochs=40,
        test_every=200, minibatch_size=100, encoder_hidden_sizes=[200, 200], decoder_hidden_sizes=[200, 200],
        hidden_activation='relu', plot_grid_size=10, plot_n_samples = 16):
    """
    Train a variational autoencoder on MNIST and plot the results.

    :param z_dim: The dimensionality of the latent space.
    :param kernel_initializer: How to initialize the weight matrices (see tf.keras.layers.Dense)
    :param optimizer: The optimizer to use
    :param learning_rate: The learning rate for the optimizer
    :param n_epochs: Number of epochs to train
    :param test_every: Test every X training iterations
    :param minibatch_size: Number of samples per minibatch
    :param encoder_hidden_sizes: Sizes of hidden layers in encoder
    :param decoder_hidden_sizes: Sizes of hidden layers in decoder
    :param hidden_activation: Activation to use for hidden layers of encoder/decoder.
    :param plot_grid_size: Number of rows, columns to use to make grid-plot of images corresponding to latent Z-points
    :param plot_n_samples: Number of samples to draw when plotting samples from model.
    """

    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)

    # White background is nicer
    x_train = 1 - x_train
    x_test = 1 - x_test

    # Only use 1k test examples for speed
    x_test = x_test[0:1000, :]

    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors

    kernel_initializer = WEIGHT_INITIALIZATION_DICT[kernel_initializer]
    hidden_activation = ACTIVATION_DICT[hidden_activation]


    # Build Model
    model = VariationalAutoencoder(784, z_dim, encoder_hidden_sizes, decoder_hidden_sizes,
                                   hidden_activation, kernel_initializer)

    in_z = tf.placeholder(shape=[None, model.z_dim], dtype=tf.float32)

    # Setup global step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Define the optimizer
    optimizer = OPTIMIZER_DICT[optimizer](learning_rate)

    # Define summary operation
    summary_op = tf.summary.merge_all()

    # Optimization step
    grads_and_vars = optimizer.compute_gradients(model.loss)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=10.)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables),
                                                   global_step=global_step)

    # Manifold specs
    mfd_samples = plot_grid_size
    x = np.linspace(-2, 2, mfd_samples)
    xv, yv = np.meshgrid(x, x)
    mfd = np.zeros((mfd_samples*mfd_samples, 784))

    with tf.Session() as sess:
        sess.run(train_iterator.initializer)  # Initialize the variables of the data-loader.
        sess.run(tf.global_variables_initializer())  # Initialize the model parameters.
        n_steps = (n_epochs * n_samples)/minibatch_size

        train_log_path = "./logs/vae_tr/"
        test_log_path = "./logs/vae_ts/"
        _check_path(train_log_path)
        _check_path(test_log_path)
        train_log_writer = tf.summary.FileWriter(train_log_path, graph=sess.graph)
        test_log_writer = tf.summary.FileWriter(test_log_path)

        for i in range(int(n_steps)):
            if i % test_every == 0:
                ts_feed = {model.inputs: x_test}
                fetches = [model.loss, summary_op]
                test_loss, test_summary = sess.run(fetches=fetches, feed_dict=ts_feed)
                test_log_writer.add_summary(test_summary, i)
                print("Step: {} \t Test ELBO: {:.3f}".format(i, -test_loss))

            tr_feed = {model.inputs: sess.run(x_minibatch)}
            fetches = [apply_gradients_op, model.loss, summary_op]
            _, train_loss, train_summary = sess.run(fetches=fetches, feed_dict=tr_feed)
            train_log_writer.add_summary(train_summary, i)

            if i % 200 == 0:
                print("Step: {} \t Train ELBO: {:.3f}".format(i, -train_loss))

            if i == 0 or i == n_steps/20 or i == n_steps-1:
                samples = sess.run(model.sample(plot_n_samples, 'sample'))
                plot_image_grid(samples, 28, 28, int(np.sqrt(plot_n_samples)), int(np.sqrt(plot_n_samples)))

        # Plot manifold of trained VAE
        for i in range(mfd_samples):
            for j in range(mfd_samples):
                my_z = np.array([[xv[i,j], yv[i,j]]])
                ts_feed = {in_z: my_z.astype('float32')}
                fetches = model.mean_x_given_z(in_z)
                mfd[j + i * mfd_samples,:] = sess.run(fetches=fetches, feed_dict=ts_feed)
        plot_image_grid(mfd, 28, 28, mfd_samples, mfd_samples)

if __name__ == '__main__':
    train_vae_on_mnist()
