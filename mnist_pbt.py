"""
A convolutional neural network for MNIST that is compatible with
population-based training.
"""

from typing import Any, List, Callable
import math
import random
import os
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import tensorflow as tf
from pbt import Cluster, Hyperparameter, HyperparamsGraph
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

    def __init__(self, name: str, graph: HyperparamsGraph, unused: bool,
                 value_setter: Callable[[], float], factor: float,
                 min_value: float, max_value: float) -> None:
        """
        Creates a new FloatHyperparameter of graph <graph> with descriptive
        name <name> and initial unused status <unused>.

        <value_setter> is a Callable that samples and returns an initial value.
        <factor> is the factor by which the value will be randomly multiplied
        or divided when perturbed. <min_value> is the minimum possible value,
        or None if there should be none. <max_value> is the maximum possible
        value, or None if there should be none.
        """
        super().__init__(name, graph, unused)
        self.value_setter = value_setter
        self.factor = factor
        self.min_value = min_value
        self.max_value = max_value
        self.value = tf.Variable(self._limited(value_setter()), trainable=False)

    def __str__(self) -> str:
        return str(self._get_value())

    def initialize_variables(self) -> None:
        self.graph.sess.run(self.value.initializer)

    def copy(self, hyperparam: 'FloatHyperparameter') -> None:
        self._set_value(hyperparam._get_value())

    def perturb(self) -> None:
        value = self._get_value()
        if random.random() < 0.5:
            value *= self.factor
        else:
            value /= self.factor
        self._set_value(self._limited(value))

    def resample(self) -> None:
        self._set_value(self._limited(self.value_setter()))


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

    def _set_sub_hyperparams_unused(self, unused: bool) -> None:
        for hyperparam in self.opt_info[self.opt_index].hyperparams:
            hyperparam.unused = unused

    def __init__(self, graph: HyperparamsGraph, to_minimize) -> None:
        """
        Creates a new OptimizerHyperparameter of <graph> with Optimizers that
        can be used to minimize the TensorFlow Tensor <to_minimize>.
        """
        super().__init__('Optimizer', graph, False)
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
        self._set_sub_hyperparams_unused(False)

    def __str__(self) -> str:
        return self.opt_info[self.opt_index].optimizer.__class__.__name__

    def initialize_variables(self) -> None:
        self.graph.sess.run([var.initializer for info in self.opt_info for var in info.vars])

    def copy(self, hyperparam: 'OptimizerHyperparameter') -> None:
        self._set_sub_hyperparams_unused(True)
        opt_index = hyperparam.opt_index
        self.opt_index = opt_index
        vars = self.opt_info[opt_index].vars
        hyperparam_vars = hyperparam.opt_info[opt_index].vars
        for i in range(len(vars)):
            vars[i].load(hyperparam.graph.sess.run(hyperparam_vars[i]), self.graph.sess)
        self._set_sub_hyperparams_unused(False)

    def _switch_to_opt(self, opt_index: int):
        self._set_sub_hyperparams_unused(True)
        self.opt_index = opt_index
        info = self.opt_info[self.opt_index]
        self.graph.sess.run([var.initializer for var in info.vars])
        for hyperparam in info.hyperparams:
            hyperparam.resample()
            hyperparam.unused = False

    def perturb(self) -> None:
        num_opts = len(self.opt_info)
        if num_opts >= 2:
            self._switch_to_opt((self.opt_index + random.randrange(1, num_opts)) % num_opts)

    def resample(self) -> None:
        self._switch_to_opt(random.randrange(len(self.opt_info)))

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
    cross entropy with a variable optimizer, optimizer parameters, and dropout
    keep probability.
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

    def __init__(self, sess: tf.Session, train_data, test_data) -> None:
        """
        Creates a new ConvNet with Session <sess>, training Dataset
        <train_data>, and testing Dataset <test_data>.
        """
        global num_nets
        super().__init__(sess)
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
        return self.accuracy

    def get_metric(self) -> float:
        return self.get_accuracy()

    def get_step_num(self) -> int:
        return self.step_num

    def _train_step(self) -> None:
        train_images, train_labels = self.sess.run(self.train_next)
        self.sess.run(self.optimizer.get_current_minimizer(),
                      feed_dict={self.x: train_images, self.y_: train_labels})
        self.update_accuracy = True
        self.step_num += 1

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
        print('Net', self.num, 'copying net', net.num)
        self.step_num = net.step_num
        for i in range(len(self.vars)):
            self.vars[i].load(net.sess.run(net.vars[i]), self.sess)
        for i in range(len(self.hyperparams)):
            self.hyperparams[i].copy(net.hyperparams[i])
        # Ensure that at least one used hyperparameter is perturbed
        rand = random.randrange(1, 2 ** sum(1 for hyperparam in self.hyperparams if not hyperparam.unused))
        perturbed_used_hyperparam = False
        for i in range(len(self.hyperparams)):
            hyperparam = self.hyperparams[i]
            if perturbed_used_hyperparam or hyperparam.unused:
                if random.random() < 0.5:
                    hyperparam.perturb()
            elif rand & (2 ** i) != 0:
                hyperparam.perturb()
                perturbed_used_hyperparam = True
        self.update_accuracy = True
        self.last_update = net.last_update
        print('Net', self.num, 'finished copying')
        self.record_update()

    def exploit_and_or_explore(self, population: List['ConvNet']) -> None:
        # Rank population by accuracy
        print('Net', self.num, 'ranking nets')
        ranked_pop = sorted(population, key=lambda net: net.get_accuracy())
        print('Net', self.num, 'finished ranking')
        if (len(ranked_pop) > 1
                and ranked_pop.index(self) < math.ceil(0.2 * len(ranked_pop))):  # In the bottom 20%?
            # Copy a net from the top 20%
            net_to_copy = ranked_pop[random.randrange(math.floor(0.8 * len(ranked_pop)), len(ranked_pop))]
            self.copy_and_explore(net_to_copy)

    @staticmethod
    def population_exploit_explore(population: List['ConvNet']) -> None:
        # Rank population by accuracy
        print('Ranking nets')
        ranked_pop = sorted(population, key=lambda net: net.get_accuracy())
        print('Finished ranking')
        if len(ranked_pop) > 1:
            # Bottom 20% copies top 20%
            worst_nets = ranked_pop[:math.ceil(0.2 * len(ranked_pop))]
            best_nets = ranked_pop[math.floor(0.8 * len(ranked_pop)):]
            for i in range(len(worst_nets)):
                worst_nets[i].copy_and_explore(best_nets[i])


RED = '#FF0000'
ORANGE = '#FF8000'
GREEN = '#008000'
BLUE = '#0000FF'
LIGHTER = {RED: '#FFC0C0', ORANGE: '#FFE0C0', GREEN: '#C0E0C0', BLUE: '#C0C0FF'}
IDENTITY = {color: color for color in LIGHTER.keys()}
OPTS = ['AdagradOptimizer', 'AdamOptimizer', 'GradientDescentOptimizer', 'MomentumOptimizer']
OPT_COLORS = {'AdagradOptimizer': RED,
              'AdamOptimizer': ORANGE,
              'GradientDescentOptimizer': GREEN,
              'MomentumOptimizer': BLUE
              }
_NO_DATA = []
OPT_LINES = [Line2D(_NO_DATA, _NO_DATA, color=OPT_COLORS[opt]) for opt in OPTS]


def _plot_net_hyperparams(net: ConvNet, max_step_num: int,
                          kp_ax: Axes, opt_ax: Axes, mom_ax: Axes, best: bool) -> None:
    if best:
        colormap = IDENTITY
        zorder = 1
    else:
        colormap = LIGHTER
        zorder = 0
    current_opt = None
    # Keep probability data
    kp_step_nums = []
    kps = []
    # Learning rate data since the optimizer last changed
    lr_step_nums = []
    log_lrs = []
    # Momentum data since the optimizer last became MomentumOptimizer
    mom_step_nums = []
    moms = []
    for update in net.get_update_history():
        new_opt = update.hyperparams['Optimizer']
        if new_opt != current_opt:
            if current_opt is not None:
                # Finish and plot a segment of learning rate data
                last_log_lr = log_lrs[-1]
                lr_step_nums.append(update.step_num)
                log_lrs.append(last_log_lr)
                opt_ax.step(lr_step_nums, log_lrs,
                            colormap[OPT_COLORS[current_opt]], where='post', zorder=zorder)
                # Start a new one
                lr_step_nums = [update.step_num]
                log_lrs = [last_log_lr]
                if current_opt == 'MomentumOptimizer':
                    # Finish and plot a segment of momentum data
                    last_mom = moms[-1]
                    mom_step_nums.append(update.step_num)
                    moms.append(last_mom)
                    mom_ax.step(mom_step_nums, moms, colormap[BLUE], where='post', zorder=zorder)
                    # Start a new one
                    mom_step_nums = [update.step_num]
                    moms = [last_mom]
            current_opt = new_opt
        # Add the new update to the appropriate data
        kp_step_nums.append(update.step_num)
        kps.append(float(update.hyperparams['Keep probability']))
        lr_step_nums.append(update.step_num)
        log_lrs.append(math.log(float(update.hyperparams['Learning rate']), 10))
        if current_opt == 'MomentumOptimizer':
            mom_step_nums.append(update.step_num)
            moms.append(float(update.hyperparams['Momentum']))
    # Plot the keep probability data
    kp_step_nums.append(max_step_num)
    kps.append(kps[-1])
    kp_ax.step(kp_step_nums, kps, colormap[BLUE], where='post', zorder=zorder)
    # Finish and plot the last segment of learning rate data
    lr_step_nums.append(max_step_num)
    log_lrs.append(log_lrs[-1])
    opt_ax.step(lr_step_nums, log_lrs, colormap[OPT_COLORS[current_opt]], where='post', zorder=zorder)
    if current_opt == 'MomentumOptimizer':
        # Finish and plot the last segment of momentum data
        mom_step_nums.append(max_step_num)
        moms.append(moms[-1])
        mom_ax.step(mom_step_nums, moms, colormap[BLUE], where='post', zorder=zorder)


def plot_hyperparams(cluster: Cluster[ConvNet], directory: str) -> None:
    """
    Creates step plots of the hyperparameter update histories of <cluster>'s
    population and saves them as images in <directory>.

    <directory> will be created if it does not already exist.
    """
    ranked_pop = sorted(cluster.get_population(), key=lambda net: -net.get_accuracy())
    max_step_num = max(net.step_num for net in ranked_pop)
    # Keep probability plot
    kp_fig, kp_ax = plt.subplots()
    kp_ax.set(title='Dropout keep probability', xlabel='Step', ylabel='Keep probability')
    kp_ax.set_xlim(0, max_step_num)
    kp_ax.set_ylim(-0.01, 1.01)
    # Optimizer and learning rate plot
    opt_fig, opt_ax = plt.subplots()
    opt_ax.set(title='Optimizer and learning rate', xlabel='Step', ylabel='Learning rate (log)')
    opt_ax.set_xlim(0, max_step_num)
    opt_ax.set_ylim(-6.06, 0.06)
    opt_ax.legend(OPT_LINES, OPTS, loc='best')
    # Momentum plot
    mom_fig, mom_ax = plt.subplots()
    mom_ax.set(title='Momentum optimizer momentum', xlabel='Step', ylabel='Momentum')
    mom_ax.set_xlim(0, max_step_num)
    mom_ax.set_ylim(-0.01, 1.01)
    # Add data to plots
    _plot_net_hyperparams(ranked_pop[0], max_step_num, kp_ax, opt_ax, mom_ax, True)
    for i in range(1, len(ranked_pop)):
        _plot_net_hyperparams(ranked_pop[i], max_step_num, kp_ax, opt_ax, mom_ax, False)
    # Save plots
    if not os.path.exists(directory):
        os.makedirs(directory)
    kp_fig.savefig(os.path.join(directory, "keep_probability.png"))
    opt_fig.savefig(os.path.join(directory, "optimizer_and_learning_rate.png"))
    mom_fig.savefig(os.path.join(directory, "momentum.png"))
