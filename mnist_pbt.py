"""
A convolutional neural network for MNIST that is compatible with
population-based training.
"""

from typing import Any, List
import math
import random
import tensorflow as tf
from pbt import Device, Hyperparameter, StepNumHyperparameter, HyperparamsPBTAbleGraph
from mnist_convnet import MNISTConvNet


class MNISTFloatHyperparameter(Hyperparameter):
    """
    A Hyperparameter with a single floating-point value used by
    PBTAbleMNISTConvNets.
    """

    value: tf.Variable
    factor: float
    min_value: float
    max_value: float

    def __init__(self, name: str, graph: HyperparamsPBTAbleGraph,
                 value: float, factor: float, min_value: float, max_value: float) -> None:
        """
        Creates a new MNISTFloatHyperparameter of graph <graph> with
        descriptive name <name>.

        <value> is the initial value. <factor> is the factor by which the value
        will be randomly multiplied or divided when perturbed. <min_value> is
        the minimum possible value, or None if there should be none.
        <max_value> is the maximum possible value, or None if there should be
        none.
        """
        super().__init__(name, graph)
        if min_value is not None:
            value = max(value, min_value)
        if max_value is not None:
            value = min(value, max_value)
        self.value = tf.Variable(value, trainable=False)
        self.factor = factor
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self) -> str:
        self.graph.lock.acquire()
        string = str(self._get_value())
        self.graph.lock.release()
        return string

    def _get_value(self) -> float:
        return self.graph.sess.run(self.value)

    def _set_value(self, value: float) -> None:
        with tf.device(self.graph.device):
            self.graph.sess.run(self.value.assign(value))

    def initialize_variables(self) -> None:
        self.graph.sess.run(self.value.initializer)

    def copy(self, hyperparam: 'Hyperparameter') -> None:
        self.graph.lock.acquire()
        hyperparam.graph.lock.acquire()
        new_value = hyperparam._get_value()
        self._set_value(new_value)
        hyperparam.graph.lock.release()
        self.graph.lock.release()

    def perturb(self) -> None:
        self.graph.lock.acquire()
        value = self._get_value()
        if random.random() < 0.5:
            value *= self.factor
        else:
            value /= self.factor
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        self._set_value(value)
        self.graph.lock.release()


num_nets = 0


class PBTAbleMNISTConvNet(HyperparamsPBTAbleGraph['PBTAbleMNISTConvNet']):
    """
    A PBTAbleGraph version of an MNIST convnet that trains itself to minimize
    cross entropy with a variable learning rate and dropout keep probability.

    A PBTAbleMNISTConvNet draws its training and testing data from a TensorFlow
    MNIST dataset specified in its initializer. The dataset's labels must be in
    one-hot vector format.
    """

    num: int
    vars: List[tf.Variable]
    step_num: StepNumHyperparameter
    dataset: Any
    net: MNISTConvNet
    learning_rate: MNISTFloatHyperparameter
    keep_prob: MNISTFloatHyperparameter
    train_op: tf.Operation
    accuracy: float
    update_accuracy: bool

    def __init__(self, device: Device, sess: tf.Session, dataset) -> None:
        """
        Creates a new PBTAbleMNISTConvNet with device <device>, Session <sess>,
        and dataset <dataset>.
        """
        global num_nets
        super().__init__(device, sess)
        with tf.device(self.device):
            self.num = num_nets
            num_nets += 1
            self.step_num = StepNumHyperparameter(self)
            self.dataset = dataset
            self.x = tf.placeholder(tf.float32, [None, 784])
            self.y_ = tf.placeholder(tf.float32, [None, 10])
            self.learning_rate = MNISTFloatHyperparameter('Learning rate', self,
                                                          10 ** random.gauss(-4, 0.5), 1.2, 0.00001, 0.001)
            self.keep_prob = MNISTFloatHyperparameter('Keep probability', self,
                                                      random.gauss(0.5, 0.2), 1.2, 0.1, 1)
            self.net = MNISTConvNet(self.x, self.y_, self.keep_prob.value)
            net_vars = [self.net.w_conv1, self.net.b_conv1, self.net.w_conv2, self.net.b_conv2,
                        self.net.w_fc1, self.net.b_fc1, self.net.w_fc2, self.net.b_fc2]
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.net.y))
            optimizer = tf.train.AdamOptimizer(self.learning_rate.value)
            self.train_op = optimizer.minimize(cross_entropy)
            self.vars = list(net_vars)
            self.vars.extend(optimizer.get_slot(var, name)
                             for name in optimizer.get_slot_names() for var in net_vars)
            self.vars.extend(optimizer._get_beta_accumulators())
            self.accuracy = 0
            self.update_accuracy = True

    def initialize_variables(self) -> None:
        super().initialize_variables()
        self.sess.run([var.initializer for var in self.vars])
        self.record_update()

    def get_accuracy(self) -> float:
        """
        Returns this PBTAbleMNISTConvNet's accuracy score on the MNIST test
        data set.
        """
        self.lock.acquire()
        if self.update_accuracy:
            self.accuracy = self.sess.run(self.net.accuracy,
                                          feed_dict={self.x: self.dataset.test.images,
                                                     self.y_: self.dataset.test.labels,
                                                     self.keep_prob.value: 1})
            self.update_accuracy = False
            print('Net', self.num, 'step', self.step_num, 'accuracy:', self.accuracy)
        self.lock.release()
        return self.accuracy

    def get_metric(self) -> float:
        return self.get_accuracy()

    def _train_step(self) -> None:
        self.lock.acquire()
        batch = self.dataset.train.next_batch(50)
        self.sess.run(self.train_op, feed_dict={self.x: batch[0], self.y_: batch[1]})
        self.update_accuracy = True
        self.step_num.value += 1
        self.lock.release()

    def train(self) -> None:
        print('Net', self.num, 'starting training run at step', self.step_num)
        self._train_step()
        while self.step_num.value % 500 != 0:
            self._train_step()
        print('Net', self.num, 'ending training run at step', self.step_num)

    def copy_and_explore(self, net: 'PBTAbleMNISTConvNet'):
        """
        Copies the specified PBTAbleMNISTConvNet, randomly changing the copied
        hyperparameters.
        """
        self.lock.acquire()
        net.lock.acquire()
        print('Net', self.num, 'copying net', net.num)
        for i in range(len(self.vars)):
            net_var_value = net.sess.run(net.vars[i])
            with tf.device(self.device):
                self.sess.run(self.vars[i].assign(net_var_value))
        rand = random.randrange(1, 2 ** len(self.hyperparams))
        for i in range(len(self.hyperparams)):
            self.hyperparams[i].copy(net.hyperparams[i])
            if rand & (2 ** i) != 0:
                self.hyperparams[i].perturb()
        self.update_accuracy = True
        self.last_update = net.last_update
        print('Net', self.num, 'finished copying')
        net.lock.release()
        self.record_update()
        self.lock.release()

    def exploit_and_or_explore(self, population: List['PBTAbleMNISTConvNet']) -> None:
        # Rank population by accuracy
        print('Net', self.num, 'ranking nets')
        accuracies = {}
        shuffled_pop = random.sample(population, len(population))
        for net in shuffled_pop:
            net.lock.acquire()
            accuracies[net] = net.get_accuracy()
        ranked_pop = sorted(population, key=lambda net: accuracies[net])
        print('Net', self.num, 'finished ranking')
        if (len(ranked_pop) > 1
                and ranked_pop.index(self) < math.ceil(0.2 * len(ranked_pop))):  # In the bottom 20%?
            # Copy a net from the top 20%
            net_to_copy = ranked_pop[random.randrange(math.floor(0.8 * len(ranked_pop)), len(ranked_pop))]
            for net in shuffled_pop:
                if net is not self and net is not net_to_copy:
                    net.lock.release()
            self.copy_and_explore(net_to_copy)
            net_to_copy.lock.release()
            self.lock.release()
        else:
            for net in shuffled_pop:
                net.lock.release()

    @staticmethod
    def population_exploit_explore(population: List['PBTAbleMNISTConvNet']) -> None:
        # Rank population by accuracy
        print('Ranking nets')
        accuracies = {}
        shuffled_pop = random.sample(population, len(population))
        for net in shuffled_pop:
            net.lock.acquire()
            accuracies[net] = net.get_accuracy()
        ranked_pop = sorted(population, key=lambda net: accuracies[net])
        print('Finished ranking')
        if len(ranked_pop) > 1:
            # Bottom 20% copies top 20%
            percentile20 = math.ceil(0.2 * len(ranked_pop))
            percentile80 = math.floor(0.8 * len(ranked_pop))
            for net in ranked_pop[percentile20:percentile80]:
                net.lock.release()
            worst_nets = ranked_pop[:percentile20]
            best_nets = ranked_pop[percentile80:]
            for i in range(len(worst_nets)):
                worst_nets[i].copy_and_explore(best_nets[i])
                worst_nets[i].lock.release()
                best_nets[i].lock.release()
        else:
            for net in shuffled_pop:
                net.lock.release()
