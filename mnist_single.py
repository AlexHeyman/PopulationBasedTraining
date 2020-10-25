"""
Trains a single MNIST convnet to minimize cross entropy, periodically reporting
its accuracy and the time spent training it.
"""

import datetime
import tensorflow as tf
from mnist import ConvNet, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE,\
    MNIST_TRAIN_BATCH_SIZE, MNIST_TEST_BATCH_SIZE, get_mnist_data, load_mnist_data


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    load_mnist_data()
    train_data, test_data = get_mnist_data()
    batch_type = (tf.float32, tf.uint8)
    batch_shape = (tf.TensorShape([None, 28, 28]), tf.TensorShape([None]))
    train_iterator = tf.compat.v1.data.Iterator.from_structure(batch_type, batch_shape)
    train_initializer = tf.compat.v1.data.Iterator.make_initializer(
        train_iterator, train_data.shuffle(MNIST_TRAIN_SIZE).batch(MNIST_TRAIN_BATCH_SIZE).repeat())
    train_next = train_iterator.get_next()
    test_iterator = tf.compat.v1.data.Iterator.from_structure(batch_type, batch_shape)
    test_initializer = tf.compat.v1.data.Iterator.make_initializer(
        test_iterator, test_data.batch(MNIST_TEST_BATCH_SIZE))
    test_next = test_iterator.get_next()

    sess = tf.compat.v1.Session()

    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28])
    y_ = tf.compat.v1.placeholder(tf.int32, [None])
    one_hot_y_ = tf.compat.v1.one_hot(y_, 10)
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    net = ConvNet(sess, x, one_hot_y_, keep_prob)
    cross_entropy = tf.compat.v1.reduce_mean(
        tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y_, logits=net.y))
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entropy, var_list=net.vars)

    net.initialize_variables()
    sess.run([var.initializer for var in optimizer.variables()])

    training_start = None
    training_time = datetime.timedelta()

    def print_stats() -> None:
        """
        Calculates and prints the net's total training time and accuracy.
        """
        global training_start, training_time
        if training_start is not None:
            training_time += datetime.datetime.now() - training_start
        print('Training time:', str(training_time))
        sess.run(test_initializer)
        size_accuracy = 0
        try:
            while True:
                test_images, test_labels = sess.run(test_next)
                batch_size = test_images.shape[0]
                batch_accuracy = sess.run(net.accuracy, feed_dict={x: test_images,
                                                                   y_: test_labels,
                                                                   keep_prob: 1})
                size_accuracy += batch_size * batch_accuracy
        except tf.errors.OutOfRangeError:
            pass
        print('Accuracy: %a' % (size_accuracy / MNIST_TEST_SIZE))
        training_start = datetime.datetime.now()

    sess.run(train_initializer)
    step_num = 0
    while step_num < 20000:
        if step_num % 100 == 0:
            print('Step', step_num)
            if step_num % 500 == 0:
                print_stats()
        train_images, train_labels = sess.run(train_next)
        sess.run(train_step, feed_dict={x: train_images, y_: train_labels, keep_prob: 1})
        step_num += 1
    print_stats()
