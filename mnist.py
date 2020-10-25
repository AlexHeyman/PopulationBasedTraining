"""
Information related to MNIST, including a convolutional neural network for it.
"""

from typing import List
import tensorflow as tf


MNIST_TRAIN_SIZE = 60000
MNIST_TEST_SIZE = 10000
MNIST_TRAIN_BATCH_SIZE = 50
MNIST_TEST_BATCH_SIZE = 100

mnist_train_data: tf.data.Dataset = None
mnist_test_data: tf.data.Dataset = None


def get_mnist_data():
    """
    Returns the on-record MNIST dataset.

    The format of the return value is a tuple of two Dataset objects
    (train_data, test_data). train_data consists of MNIST_TRAIN_SIZE elements,
    each consisting of a tf.float32 Tensor image with shape [28, 28] and a
    tf.uint8 Tensor label with shape []. test_data has the same structure, but
    with MNIST_TEST_SIZE elements.

    Before load_mnist_data() has been called for the first time, this function
    will return None.
    """
    global mnist_train_data, mnist_test_data
    return mnist_train_data, mnist_test_data


def load_mnist_data(path: str = 'mnist.npz') -> None:
    """
    Loads the MNIST dataset into memory.

    <path> is the path where the dataset should be cached locally.
    """
    global mnist_train_data, mnist_test_data
    train, test = tf.keras.datasets.mnist.load_data(path)
    mnist_train_data = tf.data.Dataset.from_tensor_slices((tf.cast(train[0]/255, tf.float32), train[1]))
    mnist_test_data = tf.data.Dataset.from_tensor_slices((tf.cast(test[0]/255, tf.float32), test[1]))


def weight_variable(shape) -> tf.Variable:
    """
    Returns a new weight Variable with shape <shape> for an MNIST convnet.
    """
    return tf.compat.v1.Variable(tf.random.truncated_normal(shape, stddev=0.1))


def bias_variable(shape) -> tf.Variable:
    """
    Returns a new bias Variable with shape <shape> for an MNIST convnet.
    """
    return tf.compat.v1.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, w):
    """
    Returns a new conv2d Operation with input <x> and filter <w> for an MNIST
    convnet.
    """
    return tf.compat.v1.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    """
    Returns a new max-pooling Operation with input <x> for an MNIST convnet.
    """
    return tf.compat.v1.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


class ConvNet:
    """
    A convolutional neural network for MNIST with two convolutional layers and
    two fully connected layers.

    A ConvNet has an associated TensorFlow Session that should be used to
    run and evaluate its graph elements.
    """

    sess: tf.compat.v1.Session
    w_conv1: tf.compat.v1.Variable
    b_conv1: tf.compat.v1.Variable
    w_conv2: tf.compat.v1.Variable
    b_conv2: tf.compat.v1.Variable
    w_fc1: tf.compat.v1.Variable
    b_fc1: tf.compat.v1.Variable
    w_fc2: tf.compat.v1.Variable
    b_fc2: tf.compat.v1.Variable
    y: tf.compat.v1.Tensor
    accuracy: tf.compat.v1.Tensor
    vars: List[tf.compat.v1.Variable]

    def __init__(self, sess: tf.compat.v1.Session, x, y_, keep_prob) -> None:
        """
        Creates a new ConvNet with associated Session <sess>.

        <x> is the input batch's images, a numerical Tensor with shape [None,
        28, 28]. <y_> is the batch's labels in one-hot vector form, a numerical
        Tensor with shape [None, 10]. keep_prob is the dropout keep
        probability, a numerical Tensor with shape [].
        """
        self.sess = sess

        self.w_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])

        x_image = tf.compat.v1.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.compat.v1.nn.relu(conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        self.w_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        h_conv2 = tf.compat.v1.nn.relu(conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        self.w_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.compat.v1.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.compat.v1.nn.relu(tf.matmul(h_pool2_flat, self.w_fc1) + self.b_fc1)
        h_fc1_drop = tf.compat.v1.nn.dropout(h_fc1, rate=(1 - keep_prob))

        self.w_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

        self.y = tf.compat.v1.matmul(h_fc1_drop, self.w_fc2) + self.b_fc2

        correct_prediction = tf.compat.v1.equal(tf.compat.v1.argmax(self.y, 1), tf.compat.v1.argmax(y_, 1))
        self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.float32))

        self.vars = [self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2,
                     self.w_fc1, self.b_fc1, self.w_fc2, self.b_fc2]

    def initialize_variables(self) -> None:
        """
        Runs the initializer Operations of all of the TensorFlow Variables that
        this ConvNet uses.
        """
        self.sess.run([var.initializer for var in self.vars])
