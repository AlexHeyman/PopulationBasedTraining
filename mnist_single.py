"""
Trains a single MNIST convnet to minimize cross entropy, periodically reporting
its accuracy and the time spent training it.
"""

import datetime
import tensorflow as tf
from mnist_convnet import MNISTConvNet, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE, MNIST_NUM_CATGS
from tensorflow.models.official.mnist.dataset import train, test


if __name__ == '__main__':
    train_data = train('MNIST_data/').cache()
    test_data = test('MNIST_data/').cache()
    train_next = train_data\
        .shuffle(MNIST_TRAIN_SIZE).batch(50).repeat().make_one_shot_iterator().get_next()
    test_next = test_data\
        .apply(tf.contrib.data.batch_and_drop_remainder(MNIST_TEST_SIZE))\
        .repeat().make_one_shot_iterator().get_next()

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int32, [None])
    one_hot_y_ = tf.one_hot(y_, MNIST_NUM_CATGS)
    keep_prob = tf.placeholder(tf.float32)

    net = MNISTConvNet(x, one_hot_y_, keep_prob)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y_, logits=net.y))
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
                test_images, test_labels = sess.run(test_next)
                print('Accuracy: %a' % sess.run(net.accuracy, feed_dict={x: test_images,
                                                                         y_: test_labels,
                                                                         keep_prob: 1}))
                training_start = datetime.datetime.now()
        train_images, train_labels = sess.run(train_next)
        sess.run(train_step, feed_dict={x: train_images, y_: train_labels, keep_prob: 1})
        step_num += 1

    print('Step', step_num)
    if training_start is not None:
        training_time += datetime.datetime.now() - training_start
    print('Training time:', training_time)
    test_images, test_labels = sess.run(test_next)
    print('Accuracy: %a' % sess.run(net.accuracy, feed_dict={x: test_images,
                                                             y_: test_labels,
                                                             keep_prob: 1}))
