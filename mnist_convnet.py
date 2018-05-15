"""
A convolutional neural network for MNIST.
"""

import tensorflow as tf


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


class MNISTConvNet:
    """
    A convolutional neural network for MNIST with two convolutional layers and
    two fully connected layers.
    """

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

    def __init__(self, x, y_, keep_prob):
        """
        Creates a new MNISTConvNet.

        <x> is the input batch's images, a tf.float32 Tensor with shape [None,
        784]. <y_> is the batch's labels in one-hot vector form, a tf.float32
        Tensor with shape [None, 10]. keep_prob is the dropout keep
        probability, a tf.float32 Tensor with shape [].
        """

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
