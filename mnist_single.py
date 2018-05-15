"""
Trains a single MNIST convnet to minimize cross entropy for 10,000 steps,
periodically reporting its accuracy and the time spent training it.
"""

import datetime
import tensorflow as tf
from mnist_convnet import MNISTConvNet
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    net = MNISTConvNet(x, y_, keep_prob)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net.y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    training_start = None
    training_time = datetime.timedelta()

    step_num = 0
    while step_num < 10000:
        if step_num % 100 == 0:
            print('Step', step_num)
            if step_num % 500 == 0:
                if training_start is not None:
                    training_time += datetime.datetime.now() - training_start
                print('Training time:', str(training_time))
                print('Accuracy: %a' % sess.run(net.accuracy, feed_dict={x: mnist.test.images,
                                                                         y_: mnist.test.labels,
                                                                         keep_prob: 1}))
                training_start = datetime.datetime.now()
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        step_num += 1

    print('Step', step_num)
    if training_start is not None:
        training_time += datetime.datetime.now() - training_start
    print('Training time:', training_time)
    print('Accuracy: %a' % sess.run(net.accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels,
                                                             keep_prob: 1}))
