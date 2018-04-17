"""Trains a single MNIST convnet to minimize cross entropy for 2,000 steps,
periodically reporting its accuracy and the time spent training it.
"""

import datetime
import tensorflow as tf
from mnist_convnet import MNISTConvNet
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

net = MNISTConvNet()
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=net.y_, logits=net.y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_start = None
training_time = datetime.timedelta()

step_num = 0
while step_num < 2000:
    if step_num % 200 == 0:
        print("Step", step_num)
        if training_start is not None:
            training_time += datetime.datetime.now() - training_start
        print("Training time:", str(training_time))
        print("Accuracy: %a" % sess.run(net.accuracy, feed_dict={net.x: mnist.test.images,
                                                                 net.y_: mnist.test.labels,
                                                                 net.keep_prob: 1.0}))
        training_start = datetime.datetime.now()
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={net.x: batch[0], net.y_: batch[1], net.keep_prob: 0.5})
    step_num += 1

print("Step", step_num)
if training_start is not None:
    training_time += datetime.datetime.now() - training_start
print("Training time:", str(training_time))
print("Accuracy: %a" % sess.run(net.accuracy, feed_dict={net.x: mnist.test.images,
                                                         net.y_: mnist.test.labels,
                                                         net.keep_prob: 1.0}))
