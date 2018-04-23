"""
Uses population-based training to train MNIST convnets to minimize cross
entropy for 10,000 steps each, periodically reporting their accuracy and
reporting the time spent training them at the end.
"""

from typing import List
import math
import random
import datetime
import tensorflow as tf
from pbt import PBTAbleGraph, LocalPBTCluster
from mnist_convnet import MNISTConvNet
from tensorflow.examples.tutorials.mnist import input_data


def random_perturbation(value: float, factor: float, min_val: float = None, max_val: float = None) -> float:
    """
    Returns <value> randomly multiplied or divided by <factor>. If <min_val> is
    not None, the returned value will be no smaller than it. If <max_val> is
    not None, the returned value will be no larger than it.
    """
    if random.random() < 0.5:
        value *= factor
    else:
        value /= factor
    if min_val is not None:
        value = max(value, min_val)
    if max_val is not None:
        value = min(value, max_val)
    return value


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
num_nets = 0


class PBTAbleMNISTConvNet(PBTAbleGraph['PBTAbleMNISTConvNet']):
    """
    A PBTAbleGraph version of an MNIST convnet that trains itself to minimize
    cross entropy with a variable learning rate and dropout keep probability.
    """

    num: int
    net: MNISTConvNet
    learning_rate: tf.Variable
    keep_prob: float
    train_op: tf.Operation
    copyable_vars: List[tf.Variable]
    step_num: int
    accuracy: float
    update_accuracy: bool

    def __init__(self, learning_rate: float, keep_prob: float) -> None:
        """
        Creates a new PBTAbleMNISTConvNet with initial learning rate
        <learning_rate> and dropout keep probability <keep_prob>.
        """
        global num_nets
        super().__init__()
        self.num = num_nets
        num_nets += 1
        self.net = MNISTConvNet()
        net_vars = [self.net.w_conv1, self.net.b_conv1, self.net.w_conv2, self.net.b_conv2,
                    self.net.w_fc1, self.net.b_fc1, self.net.w_fc2, self.net.b_fc2]
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        self.keep_prob = keep_prob
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.net.y_, logits=self.net.y))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(cross_entropy)
        self.copyable_vars = list(net_vars)
        self.copyable_vars.extend(optimizer.get_slot(var, name)
                                  for name in optimizer.get_slot_names() for var in net_vars)
        self.copyable_vars.extend(optimizer._get_beta_accumulators())
        self.step_num = 0
        self.accuracy = 0
        self.update_accuracy = True

    def initialize_variables(self, sess: tf.Session) -> None:
        sess.run([var.initializer for var in self.copyable_vars])
        sess.run(self.learning_rate.initializer)

    def get_accuracy(self, sess: tf.Session) -> float:
        """
        Returns this PBTAbleMNISTConvNet's accuracy score on the MNIST test
        data set.
        """
        if self.update_accuracy:
            self.accuracy = sess.run(self.net.accuracy, feed_dict={self.net.x: mnist.test.images,
                                                                   self.net.y_: mnist.test.labels,
                                                                   self.net.keep_prob: 1})
            self.update_accuracy = False
        return self.accuracy

    def get_metric(self, sess: tf.Session) -> float:
        return self.get_accuracy(sess)

    def train_step(self, sess: tf.Session) -> None:
        if self.step_num % 100 == 0:
            print('Net', self.num, 'step', self.step_num)
        batch = mnist.train.next_batch(50)
        sess.run(self.train_op, feed_dict={self.net.x: batch[0],
                                           self.net.y_: batch[1],
                                           self.net.keep_prob: self.keep_prob})
        self.update_accuracy = True
        self.step_num += 1

    def exploit_and_or_explore(self, sess: tf.Session, population: List['PBTAbleMNISTConvNet']) -> None:
        if self.step_num % 500 == 0:
            print('Net', self.num, 'ranking nets')
            # Rank population by accuracy
            ranked_pop = sorted(population, key=lambda net: net.get_accuracy(sess))
            print('Net', self.num, 'finished ranking')
            if ranked_pop.index(self) < math.ceil(0.2*len(ranked_pop)):  # In the bottom 20%?
                # Copy a net from the top 20%
                net_to_copy = ranked_pop[random.randrange(math.floor(0.8*len(ranked_pop)), len(ranked_pop))]
                print('Net', self.num, 'copying net', net_to_copy.num)
                for i in range(len(self.copyable_vars)):
                    self.copyable_vars[i].assign(net_to_copy.copyable_vars[i])
                # Possibly perturb learning rate and/or keep probability
                new_learning_rate = net_to_copy.learning_rate.value()
                new_keep_prob = net_to_copy.keep_prob
                rand = random.randrange(3)
                if rand <= 1:
                    new_learning_rate = random_perturbation(new_learning_rate, 1.2, 0.00001, 0.001)
                if rand >= 1:
                    new_keep_prob = random_perturbation(new_keep_prob, 1.2, 0.1, 1)
                self.learning_rate.assign(new_learning_rate)
                self.keep_prob = new_keep_prob
                self.update_accuracy = True
                print('Net', self.num, 'finished copying')


def random_mnist_convnet() -> PBTAbleMNISTConvNet:
    """
    Returns a new PBTAbleMNISTConvNet with randomized initial variable values.
    """
    return PBTAbleMNISTConvNet(min(max(random.gauss(0.0001, 0.0001), 0.00001), 0.001),
                               min(max(random.gauss(0.5, 0.1), 0.1), 1))


pop_size = 10
cluster = LocalPBTCluster[PBTAbleMNISTConvNet](pop_size, random_mnist_convnet)
cluster.initialize_variables()
training_start = datetime.datetime.now()
cluster.train(lambda sess, net, population: net.step_num < 10000)
print('Training time:', datetime.datetime.now() - training_start)
print('Highest accuracy:', cluster.get_highest_metric_graph().get_accuracy())
