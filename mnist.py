"""
Information related to MNIST, including a convolutional neural network for it.
"""

from typing import List, Tuple
import tensorflow as tf


MNIST_TRAIN_SIZE = 60000
MNIST_TEST_SIZE = 10000
MNIST_TEST_BATCH_SIZE = 100

mnist_train_data: tf.data.Dataset = None
mnist_test_data: tf.data.Dataset = None


def get_mnist_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Returns the on-record MNIST training Dataset and testing Dataset as a
    tuple.

    Before set_mnist_data() has been called for the first time, both Datasets
    will be None.
    """
    return (mnist_train_data, mnist_test_data)


def set_mnist_data(train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> None:
    """
    Sets the on-record MNIST training Dataset and testing Dataset to
    <train_data> and <test_data>, respectively.
    """
    global mnist_train_data, mnist_test_data
    mnist_train_data = train_data
    mnist_test_data = test_data


def weight_variable(shape) -> tf.Variable:
    """
    Returns a new weight Variable with shape <shape> for an MNIST convnet.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape) -> tf.Variable:
    """
    Returns a new bias Variable with shape <shape> for an MNIST convnet.
    """
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, w):
    """
    Returns a new conv2d Operation with input <x> and filter <w> for an MNIST
    convnet.
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    """
    Returns a new max-pooling Operation with input <x> for an MNIST convnet.
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="SAME")


class ConvNet:
    """
    A convolutional neural network for MNIST with two convolutional layers and
    two fully connected layers.

    A ConvNet has an associated TensorFlow Session that should be used to run
    and evaluate its graph elements.
    """

    sess: tf.Session
    w_conv1: tf.Variable
    b_conv1: tf.Variable
    w_conv2: tf.Variable
    b_conv2: tf.Variable
    w_fc1: tf.Variable
    b_fc1: tf.Variable
    w_fc2: tf.Variable
    b_fc2: tf.Variable
    y: tf.Tensor
    accuracy: tf.Tensor
    vars: List[tf.Variable]

    def __init__(self, sess: tf.Session, x, y_, keep_prob) -> None:
        """
        Creates a new ConvNet with associated Session <sess>.

        <x> is the input batch's images, a tf.float32 Tensor with shape [None,
        784]. <y_> is the batch's labels in one-hot vector form, a tf.float32
        Tensor with shape [None, 10]. keep_prob is the dropout keep
        probability, a tf.float32 Tensor with shape [].
        """
        self.sess = sess

        self.w_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        self.w_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        self.w_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.w_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        self.w_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

        self.y = tf.matmul(h_fc1_drop, self.w_fc2) + self.b_fc2

        correct_prediction = tf.equal(
            tf.argmax(self.y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.vars = [self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2,
                     self.w_fc1, self.b_fc1, self.w_fc2, self.b_fc2]

    def initialize_variables(self) -> None:
        """
        Runs the initializer Operations of all of the TensorFlow Variables that
        this ConvNet created in its initializer.
        """
        self.sess.run([var.initializer for var in self.vars])
