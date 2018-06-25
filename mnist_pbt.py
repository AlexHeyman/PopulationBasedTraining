"""
A convolutional neural network for MNIST that is compatible with
population-based training.
"""

from typing import Any, List, Callable
import math
import random
import tensorflow as tf
from pbt import Device, Hyperparameter, HyperparamsGraph
from mnist_convnet import ConvNet as MNISTConvNet, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE, MNIST_TEST_BATCH_SIZE


class FloatHyperparameter(Hyperparameter):
    """
    A type of Hyperparameter with a single floating-point value.
    """

    value_setter: Callable[[], float]
    value: tf.Variable
    factor: float
    min_value: float
    max_value: float

    def _limited(self, value: float) -> float:
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        return value

    def _get_value(self) -> float:
        return self.graph.sess.run(self.value)

    def _set_value(self, value: float) -> None:
        self.value.load(value, self.graph.sess)

    def __init__(self, name: str, graph: HyperparamsGraph, hidden: bool,
                 value_setter: Callable[[], float], factor: float,
                 min_value: float, max_value: float) -> None:
        """
        Creates a new FloatHyperparameter of graph <graph> with descriptive
        name <name> and initial hidden status <hidden>.

        <value_setter> is a Callable that samples and returns an initial value.
        <factor> is the factor by which the value will be randomly multiplied
        or divided when perturbed. <min_value> is the minimum possible value,
        or None if there should be none. <max_value> is the maximum possible
        value, or None if there should be none.
        """
        super().__init__(name, graph, hidden)
        with tf.device(self.graph.device):
            self.value_setter = value_setter
            self.factor = factor
            self.min_value = min_value
            self.max_value = max_value
            self.value = tf.Variable(self._limited(value_setter()), trainable=False)

    def __str__(self) -> str:
        self.graph.lock.acquire()
        string = str(self._get_value())
        self.graph.lock.release()
        return string

    def initialize_variables(self) -> None:
        self.graph.sess.run(self.value.initializer)

    def copy(self, hyperparam: 'FloatHyperparameter') -> None:
        self.graph.lock.acquire()
        hyperparam.graph.lock.acquire()
        self._set_value(hyperparam._get_value())
        hyperparam.graph.lock.release()
        self.graph.lock.release()

    def perturb(self) -> None:
        self.graph.lock.acquire()
        value = self._get_value()
        if random.random() < 0.5:
            value *= self.factor
        else:
            value /= self.factor
        self._set_value(self._limited(value))
        self.graph.lock.release()

    def resample(self) -> None:
        self.graph.lock.acquire()
        self._set_value(self._limited(self.value_setter()))
        self.graph.lock.release()


class OptimizerInfo:
    """
    Stores a TensorFlow Optimizer and information about it.
    """

    optimizer: tf.train.Optimizer
    minimizer: tf.Operation
    vars: List[tf.Variable]
    hyperparams: List[Hyperparameter]

    def __init__(self, optimizer: tf.train.Optimizer,
                 to_minimize, hyperparams: List[Hyperparameter]) -> None:
        """
        Creates a new OptimizerInfo for <optimizer>.

        <to_minimize> is a TensorFlow Tensor that <optimizer> should be used to
        minimize, and <hyperparams> is a list of all of the Hyperparameters
        that affect <optimizer>'s behavior.
        """
        self.optimizer = optimizer
        self.minimizer = optimizer.minimize(to_minimize)
        self.vars = optimizer.variables()
        self.hyperparams = hyperparams


class OptimizerHyperparameter(Hyperparameter):
    """
    A Hyperparameter whose value is one of several TensorFlow Optimizers.
    """

    opt_info: List[OptimizerInfo]
    opt_index: int

    def _set_sub_hyperparams_hidden(self, hidden: bool) -> None:
        for hyperparam in self.opt_info[self.opt_index].hyperparams:
            hyperparam.hidden = hidden

    def __init__(self, graph: HyperparamsGraph, to_minimize) -> None:
        """
        Creates a new OptimizerHyperparameter of <graph> with Optimizers that
        can be used to minimize the TensorFlow Tensor <to_minimize>.
        """
        super().__init__('Optimizer', graph, False)
        with tf.device(self.graph.device):
            self.opt_info = []
            learning_rate = FloatHyperparameter('Learning rate', self.graph, True,
                                                lambda: 10 ** random.uniform(-6, 0), 1.2, 10 ** -6, 1)
            # GradientDescentOptimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate.value)
            self.opt_info.append(OptimizerInfo(optimizer, to_minimize, [learning_rate]))
            # AdagradOptimizer
            optimizer = tf.train.AdagradOptimizer(learning_rate.value, 0.01)
            self.opt_info.append(OptimizerInfo(optimizer, to_minimize, [learning_rate]))
            # MomentumOptimizer
            momentum = FloatHyperparameter('Momentum', self.graph, True,
                                           lambda: random.uniform(0, 1), 1.2, 0, 1)
            optimizer = tf.train.MomentumOptimizer(learning_rate.value, momentum.value)
            self.opt_info.append(OptimizerInfo(optimizer, to_minimize, [learning_rate, momentum]))
            # AdamOptimizer
            optimizer = tf.train.AdamOptimizer(learning_rate.value)
            self.opt_info.append(OptimizerInfo(optimizer, to_minimize, [learning_rate]))
            self.opt_index = random.randrange(len(self.opt_info))
            self._set_sub_hyperparams_hidden(False)

    def __str__(self) -> str:
        self.graph.lock.acquire()
        string = self.opt_info[self.opt_index].optimizer.__class__.__name__
        self.graph.lock.release()
        return string

    def initialize_variables(self) -> None:
        self.graph.sess.run([var.initializer for info in self.opt_info for var in info.vars])

    def copy(self, hyperparam: 'OptimizerHyperparameter') -> None:
        self.graph.lock.acquire()
        hyperparam.graph.lock.acquire()
        self._set_sub_hyperparams_hidden(True)
        opt_index = hyperparam.opt_index
        self.opt_index = opt_index
        vars = self.opt_info[opt_index].vars
        hyperparam_vars = hyperparam.opt_info[opt_index].vars
        for i in range(len(vars)):
            vars[i].load(hyperparam.graph.sess.run(hyperparam_vars[i]), self.graph.sess)
        self._set_sub_hyperparams_hidden(False)
        hyperparam.graph.lock.release()
        self.graph.lock.release()

    def _switch_to_opt(self, opt_index: int):
        self._set_sub_hyperparams_hidden(True)
        self.opt_index = opt_index
        info = self.opt_info[self.opt_index]
        self.graph.sess.run([var.initializer for var in info.vars])
        for hyperparam in info.hyperparams:
            hyperparam.resample()
            hyperparam.hidden = False

    def perturb(self) -> None:
        self.graph.lock.acquire()
        num_opts = len(self.opt_info)
        if num_opts >= 2:
            self._switch_to_opt((self.opt_index + random.randrange(1, num_opts)) % num_opts)
        self.graph.lock.release()

    def resample(self) -> None:
        self.graph.lock.acquire()
        self._switch_to_opt(random.randrange(len(self.opt_info)))
        self.graph.lock.release()

    def get_current_minimizer(self) -> tf.Operation:
        """
        Returns a TensorFlow Operation that uses this OptimizerHyperparameter's
        current Optimizer to minimize the Tensor specified in its initializer.
        """
        return self.opt_info[self.opt_index].minimizer


num_nets = 0


class ConvNet(HyperparamsGraph):
    """
    A PBT-compatible version of an MNIST convnet that trains itself to minimize
    cross entropy with a variable learning rate and dropout keep probability.
    """

    num: int
    vars: List[tf.Variable]
    step_num: int
    train_next: Any
    test_next: Any
    net: MNISTConvNet
    optimizer: OptimizerHyperparameter
    keep_prob: FloatHyperparameter
    accuracy: float
    update_accuracy: bool

    def __init__(self, device: Device, sess: tf.Session, train_data, test_data) -> None:
        """
        Creates a new ConvNet with device <device>, Session <sess>, training
        Dataset <train_data>, and testing Dataset <test_data>.
        """
        global num_nets
        super().__init__(device, sess)
        with tf.device(self.device):
            self.num = num_nets
            num_nets += 1
            self.step_num = 0
            self.train_next = train_data\
                .shuffle(MNIST_TRAIN_SIZE).batch(50).repeat().make_one_shot_iterator().get_next()
            self.test_iterator = test_data.batch(MNIST_TEST_BATCH_SIZE).make_initializable_iterator()
            self.test_next = self.test_iterator.get_next()
            self.x = tf.placeholder(tf.float32, [None, 784])
            self.y_ = tf.placeholder(tf.int32, [None])
            one_hot_y_ = tf.one_hot(self.y_, 10)
            self.keep_prob = FloatHyperparameter('Keep probability', self, False,
                                                 lambda: random.uniform(0.1, 1), 1.2, 0.1, 1)
            self.net = MNISTConvNet(self.x, one_hot_y_, self.keep_prob.value)
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y_, logits=self.net.y))
            self.optimizer = OptimizerHyperparameter(self, cross_entropy)
            self.vars = [self.net.w_conv1, self.net.b_conv1, self.net.w_conv2, self.net.b_conv2,
                         self.net.w_fc1, self.net.b_fc1, self.net.w_fc2, self.net.b_fc2]
            self.accuracy = 0
            self.update_accuracy = True
            print('Net', self.num, 'created')

    def initialize_variables(self) -> None:
        super().initialize_variables()
        self.sess.run([var.initializer for var in self.vars])
        print('Net', self.num, 'variables initialized')

    def get_accuracy(self) -> float:
        """
        Returns this ConvNet's accuracy score on its testing Dataset.
        """
        self.lock.acquire()
        if self.update_accuracy:
            self.sess.run(self.test_iterator.initializer)
            size_accuracy = 0
            try:
                while True:
                    test_images, test_labels = self.sess.run(self.test_next)
                    batch_size = test_images.shape[0]
                    batch_accuracy = self.sess.run(self.net.accuracy,
                                                   feed_dict={self.x: test_images, self.y_: test_labels,
                                                              self.keep_prob.value: 1})
                    size_accuracy += batch_size * batch_accuracy
            except tf.errors.OutOfRangeError:
                pass
            self.accuracy = size_accuracy / MNIST_TEST_SIZE
            print('Net', self.num, 'step', self.step_num, 'accuracy:', self.accuracy)
            self.update_accuracy = False
        accuracy = self.accuracy
        self.lock.release()
        return accuracy

    def get_metric(self) -> float:
        return self.get_accuracy()

    def get_step_num(self) -> int:
        self.lock.acquire()
        step_num = self.step_num
        self.lock.release()
        return step_num

    def _train_step(self) -> None:
        self.lock.acquire()
        train_images, train_labels = self.sess.run(self.train_next)
        self.sess.run(self.optimizer.get_current_minimizer(),
                      feed_dict={self.x: train_images, self.y_: train_labels})
        self.update_accuracy = True
        self.step_num += 1
        self.lock.release()

    def train(self) -> None:
        print('Net', self.num, 'starting training run at step', self.step_num)
        self._train_step()
        while self.step_num % 500 != 0:
            self._train_step()
        print('Net', self.num, 'ending training run at step', self.step_num)

    def copy_and_explore(self, net: 'ConvNet'):
        """
        Copies the specified ConvNet, randomly changing the copied
        hyperparameters.
        """
        self.lock.acquire()
        net.lock.acquire()
        print('Net', self.num, 'copying net', net.num)
        self.step_num = net.step_num
        for i in range(len(self.vars)):
            self.vars[i].load(net.sess.run(net.vars[i]), self.sess)
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

    def exploit_and_or_explore(self, population: List['ConvNet']) -> None:
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
    def population_exploit_explore(population: List['ConvNet']) -> None:
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
