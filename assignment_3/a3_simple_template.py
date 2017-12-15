import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

class NaiveBayesModel(object):

    def __init__(self, w_init, b_init = None, c_init = None):
        """
        :param w_init: An (n_categories, n_dim) array, where w[i, j] represents log p(X[j]=1 | Z[i]=1)
        :param b_init: A (n_categories, ) vector where b[i] represents log p(Z[i]=1), or None to fill with zeros
        :param c_init: A (n_dim, ) vector where b[j] represents log p(X[j]=1), or None to fill with zeros
        """

        self.w = w_init
        (self.n_categories, self.n_dim) = self.w.shape

        if b_init is None:
            self.b = tf.get_variable(name='b',
                                     shape=[self.n_categories],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
        else:
            self.b = b_init

        if c_init is None:
            self.c = tf.get_variable(name='c',
                                    shape=[self.n_dim],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        else:
            self.c = c_init

        self._inputs = tf.placeholder(tf.float32, shape=[None, self.n_dim], name='inputs')
        self._loss = self._compute_loss()

    def log_p_x_given_z(self, x):
        """
        :param x: An (n_samples, n_dims) tensor
        :return: An (n_samples, n_labels) tensor  p_x_given_z where result[n, k] indicates p(X=x[n] | Z=z[k])
        """
        # D x K
        alpha = tf.transpose(self.w + self.c)

        # N x K
        return tf.matmul(x, tf.log_sigmoid(alpha)) + tf.matmul((1 - x),  tf.log_sigmoid(- alpha))


    def log_p_x(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of log-probabilities assigned to each point
        """
        # K x 1
        log_prior = tf.nn.log_softmax(self.b)

        # N x K
        log_p_x_given_z = self.log_p_x_given_z(x)

        # N x 1
        return tf.reduce_logsumexp(tf.add(log_p_x_given_z, tf.transpose(log_prior)), axis=1)

    def _compute_loss(self):
        nll = -tf.reduce_mean(self.log_p_x(self._inputs), axis = 0)
        tf.summary.scalar('log_like', -nll)
        return nll

    def sample(self, n_samples=None, z_samples=None, sample_or_mean='sample'):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """

        if z_samples is None:
            latent_var_distro = tf.distributions.Categorical(logits=tf.squeeze(self.b))
            # N x K
            z_samples = latent_var_distro.sample(int(n_samples))

        # N x K
        z_one_hot = tf.one_hot(z_samples, self.n_categories)

        # N x D
        logits = tf.add(tf.matmul(z_one_hot, self.w), self.c, name='sample_logits')

        batch_distro = tf.contrib.distributions.BernoulliWithSigmoidProbs(logits=logits)

        if sample_or_mean == 'sample':
            samples = batch_distro.sample()
        elif sample_or_mean == 'mean':
            samples = batch_distro.mean()

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

def train_simple_generative_model_on_mnist(n_categories=20, initial_mag = 0.01, optimizer='rmsprop', learning_rate=.01, n_epochs=20, test_every=100,
                                           minibatch_size=100, plot_n_samples=16):
    """
    Train a simple Generative model on MNIST and plot the results.

    :param n_categories: Number of latent categories (K in assignment)
    :param initial_mag: Initial weight magnitude
    :param optimizer: The name of the optimizer to use
    :param learning_rate: Learning rate for the optimization
    :param n_epochs: Number of epochs to train for
    :param test_every: Test every X iterations
    :param minibatch_size: Number of samples in a minibatch
    :param plot_n_samples: Number of samples to plot
    """

    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)

    # White background is nicer
    x_train = 1 - x_train
    x_test = 1 - x_test

    # Create Frankenstein digits
    frank, orig_digits = create_frankenstein(x_test, 10)

    # Only use 1k test examples for speed
    x_test = x_test[0:1000, :]

    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors

    # Build the model
    w_init = tf.get_variable(name="w",
                             shape=[n_categories, n_dims],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=initial_mag))

    model = NaiveBayesModel(w_init)

    # Setup global step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Define the optimizer
    assert optimizer in ('adam', 'rmsprop')
    if optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

    # Define summary operation
    summary_op = tf.summary.merge_all()

    # Optimization step
    grads_and_vars = optimizer.compute_gradients(model._loss)
    grads, variables = zip(*grads_and_vars)
    apply_gradients_op = optimizer.apply_gradients(zip(grads, variables), global_step=global_step)

    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        n_steps = (n_epochs * n_samples)/minibatch_size

        train_log_path = "./logs/nb_tr/"
        test_log_path = "./logs/nb_ts/"
        _check_path(train_log_path)
        _check_path(test_log_path)
        train_log_writer = tf.summary.FileWriter(train_log_path, graph=sess.graph)
        test_log_writer = tf.summary.FileWriter(test_log_path)

        for i in range(int(n_steps)):
            if i % test_every == 0:
                ts_feed = {model._inputs: x_test}
                fetches = [model._loss, summary_op]
                test_loss, test_summary = sess.run(fetches=fetches, feed_dict=ts_feed)
                test_log_writer.add_summary(test_summary, i)
                print("Step: {} \t Test LL: {:.3f}".format(i, -test_loss))

            tr_feed = {model._inputs: sess.run(x_minibatch)}
            fetches = [apply_gradients_op, model._loss, summary_op]
            _, train_loss, train_summary = sess.run(fetches=fetches, feed_dict=tr_feed)
            train_log_writer.add_summary(train_summary, i)

            if i % 50 == 0:
                print("Step: {} \t Train LL: {:.3f}".format(i, -train_loss))

        # Problem 6: Expected pixel values given that the latent variable
        samples = sess.run(model.sample(z_samples=list(range(n_categories)),
                                        sample_or_mean="mean"))
        plot_image_grid(samples, 28, 28, 5, 4)

        #Problem 7: Show 16 images samples from your trained model
        samples = sess.run(model.sample(plot_n_samples))
        plot_image_grid(samples, 28, 28, int(np.sqrt(plot_n_samples)), int(np.sqrt(plot_n_samples)))

        #Problem 9: Frankenstein digits + statistical test
        frank_ll = sess.run(model.log_p_x(frank))
        orig_ll = sess.run(model.log_p_x(orig_digits))

        print("\nFrankenstein Digits\n")
        print(frank_ll, np.mean(frank_ll), np.std(frank_ll))
        print(orig_ll, np.mean(orig_ll), np.std(orig_ll))
        print(stats.ttest_ind(frank_ll, orig_ll, equal_var=False))
        plot_image_grid(frank, 28, 28, 5, 2)
        plot_image_grid(orig_digits, 28, 28, 5, 2)

    test_log_writer.close()
    train_log_writer.close()


def create_frankenstein(x_test, num_samples):

    (num_test_samples, x_dim) = x_test.shape

    rand_ix = np.random.randint(num_test_samples, size=int(2 * num_samples))

    orig_digits = x_test[rand_ix, :]
    frank_tensor = np.zeros((num_samples, x_dim))

    frank_tensor[:, 0: int(x_dim/2)] = orig_digits[[2*i for i in range(num_samples)], 0:int(x_dim/2)]
    frank_tensor[:, int(x_dim/2):] = orig_digits[[2*i + 1 for i in range(num_samples)], int(x_dim/2):]

    return np.array(frank_tensor, dtype = 'float32'), np.array(orig_digits[0:num_samples, :], dtype = 'float32')

if __name__ == '__main__':
    train_simple_generative_model_on_mnist()
