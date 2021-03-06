"""
An implementation of population-based training of neural networks for
TensorFlow.
"""

from typing import List, Callable, TypeVar, Generic, Optional
from collections import OrderedDict
import os
import tensorflow as tf


T = TypeVar('T', bound='Graph')


class Graph:
    """
    A TensorFlow graph that a Cluster can train.

    A Graph need not have a TensorFlow Graph object all to itself.

    A Graph has an associated TensorFlow Session that is used to run and
    evaluate its graph elements. It also has a step number that records the
    number of training steps that it has performed.
    """

    num: int
    sess: tf.compat.v1.Session
    step_num: int

    def __init__(self, num: int, sess: tf.compat.v1.Session) -> None:
        """
        Creates a new Graph, numbered <num> in its population, with associated
        Session <sess>.
        """
        self.num = num
        self.sess = sess
        self.step_num = 0

    def initialize_variables(self) -> None:
        """
        Runs the initializer Operations of all of the TensorFlow Variables that
        this Graph uses.
        """
        raise NotImplementedError

    def get_train_variables(self) -> List[tf.compat.v1.Variable]:
        """
        Returns all of the TensorFlow Variables that this Graph can manipulate
        during training.
        """
        raise NotImplementedError

    def get_value(self):
        """
        Returns a picklable data structure that represents this Graph's value -
        all of the attributes relevant to its training that vary across Graphs
        of its type.
        """
        raise NotImplementedError

    def set_value(self, value) -> None:
        """
        Sets this Graph's value to that represented by <value>, a data
        structure returned by the get_value() method of a Graph of this one's
        type.
        """
        raise NotImplementedError

    def get_metric(self) -> float:
        """
        Returns a metric for this Graph, typically its accuracy, that
        represents its effectiveness at its task and allows it to be compared
        to other Graphs with the same task.
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Trains this Graph until it is ready to consider exploitation of its
        population.
        """
        raise NotImplementedError


class Cluster(Generic[T]):
    """
    A system that can perform population-based training of Graphs.

    Any TensorFlow Variables created in the initializers of a Cluster's
    Graphs should be initialized by calling the Cluster's
    initialize_variables() method.

    T is the type of Graph that this Cluster trains.
    """

    def initialize_variables(self) -> None:
        """
        Initializes all of the TensorFlow Variables that this Cluster's Graphs
        created in their initializers.
        """
        raise NotImplementedError

    def get_population(self) -> List[T]:
        """
        Returns this Cluster's population of Graphs.

        The returned list should not be modified.
        """
        raise NotImplementedError

    def get_peak_metric(self) -> float:
        """
        Returns this Cluster's all-time highest metric, chosen from all of its
        Graphs' metrics each time they finish executing their own train()
        method.

        If none of this Cluster's Graphs have finished executing train(), this
        method returns None.
        """
        raise NotImplementedError

    def get_peak_metric_value(self):
        """
        Returns the value according to get_value() of a Graph in this Cluster
        when it achieved the Cluster's all-time highest metric according to
        get_peak_metric(), or None if get_peak_metric() returns None.
        """
        raise NotImplementedError

    def train(self, until_step_num: int) -> None:
        """
        Performs population-based training on this Cluster's population until
        all of them have performed at least <until_step_num> training steps.
        """
        raise NotImplementedError


class LocalCluster(Generic[T], Cluster[T]):
    """
    A Cluster that simulates synchronous training with a single local thread.
    """

    sess: tf.compat.v1.Session
    population: List[T]
    peak_metric: Optional[float]

    def __init__(self, pop_size: int, graph_maker: Callable[[int, tf.compat.v1.Session], T]) -> None:
        """
        Creates a new LocalCluster with <pop_size> Graphs returned by
        <graph_maker> as its population.

        <pop_size> is the number of Graphs that will make up this
        LocalCluster's population. <graph_maker> is a Callable that returns a
        new T with the specified number and Session each time it is called.
        """
        self.sess = tf.compat.v1.Session()
        self.population = []
        for num in range(pop_size):
            self.population.append(graph_maker(num, self.sess))
            print('Graph', num, 'created')
        self.peak_metric = None
        self.peak_metric_value = None

    def initialize_variables(self):
        for graph in self.population:
            graph.initialize_variables()
            print('Graph', graph.num, 'variables initialized')

    def get_population(self) -> List[T]:
        return self.population

    def get_peak_metric(self) -> float:
        return self.peak_metric

    def get_peak_metric_value(self):
        return self.peak_metric_value

    def train(self, until_step_num: int) -> None:
        while True:
            keep_training = False
            for graph in self.population:
                if graph.step_num < until_step_num:
                    keep_training = True
                    if graph.step_num > 0:
                        print('Exploiting/exploring')
                        self.exploit_and_or_explore()
                        print('Finished exploiting/exploring')
                        break
            if keep_training:
                for graph in self.population:
                    if graph.step_num < until_step_num:
                        print('Graph', graph.num, 'starting training run at step', graph.step_num)
                        graph.train()
                        print('Graph', graph.num, 'ending training run at step', graph.step_num)
                best_graph = max(self.population, key=lambda graph: graph.get_metric())
                best_metric = best_graph.get_metric()
                if self.peak_metric is None or best_metric > self.peak_metric:
                    self.peak_metric = best_metric
                    self.peak_metric_value = best_graph.get_value()
            else:
                break

    def exploit_and_or_explore(self) -> None:
        """
        Causes each of the Graphs in this LocalCluster's population to exploit
        the other Graphs to improve itself and/or modify itself to explore a
        different option, if those actions are judged to be currently necessary
        for it.
        """
        raise NotImplementedError


class Hyperparameter:
    """
    A non-trained parameter of a HyperparamsGraph.

    A Hyperparameter's __str__() method should return a string representing its
    value.

    A Hyperparameter may be declared unused, in which case new
    HyperparamsUpdates will not record it.
    """

    name: str
    graph: 'HyperparamsGraph'
    unused: bool

    def __init__(self, name: str, graph: 'HyperparamsGraph', unused: bool) -> None:
        """
        Creates a new Hyperparameter of <graph> with descriptive name <name>
        and initial unused status <unused>.
        """
        self.name = name
        self.graph = graph
        self.unused = unused
        graph.hyperparams.append(self)

    def initialize_variables(self) -> None:
        """
        Runs the initializer Operations of all of the TensorFlow Variables that
        this Hyperparameter created in its initializer.
        """
        raise NotImplementedError

    def get_value(self):
        """
        Returns a picklable data structure that represents this
        Hyperparameter's value.
        """
        raise NotImplementedError

    def set_value(self, value) -> None:
        """
        Sets this Hyperparameter's value to that represented by <value>, a data
        structure returned by the get_value() method of a Hyperparameter of
        this one's type.
        """
        raise NotImplementedError

    def perturb(self) -> None:
        """
        Alters this Hyperparameter to explore a different option for it.
        """
        raise NotImplementedError

    def resample(self) -> None:
        """
        Resets this Hyperparameter's value, re-randomizing any random choices
        that determined it.
        """
        raise NotImplementedError


class HyperparamsUpdate:
    """
    Stores information about a HyperparamsGraph's update of its
    hyperparameters.
    """

    prev: 'HyperparamsUpdate'
    step_num: int
    hyperparams: OrderedDict

    def __init__(self, graph: 'HyperparamsGraph') -> None:
        """
        Creates a new HyperparamsUpdate that stores <graph>'s current
        information.
        """
        self.prev = graph.last_update
        self.step_num = graph.step_num
        self.hyperparams = OrderedDict()
        for hyperparam in graph.hyperparams:
            if not hyperparam.unused:
                self.hyperparams[hyperparam.name] = str(hyperparam)

    def __str__(self) -> str:
        string = 'Step ' + str(self.step_num) + os.linesep
        for name, value in self.hyperparams.items():
            string += name + ': ' + value + os.linesep
        return string + os.linesep


class HyperparamsGraph(Graph):
    """
    A Graph that stores its hyperparameters as a list of Hyperparameters.
    """

    hyperparams: List[Hyperparameter]
    last_update: Optional[HyperparamsUpdate]

    def __init__(self, num: int, sess: tf.compat.v1.Session) -> None:
        """
        Creates a new HyperparamsGraph, numbered <num> in its population, with
        associated Session <sess>.
        """
        super().__init__(num, sess)
        self.hyperparams = []
        self.last_update = None

    def initialize_variables(self) -> None:
        for hyperparam in self.hyperparams:
            hyperparam.initialize_variables()
        self.record_update()

    def record_update(self) -> None:
        """
        Records this HyperparamsGraph's current information as a new update to
        its hyperparameters.
        """
        self.last_update = HyperparamsUpdate(self)

    def get_update_history(self) -> List[HyperparamsUpdate]:
        """
        Returns a list of this HyperparamsGraph's HyperparamsUpdates in order
        from least to most recent.
        """
        updates = []
        update = self.last_update
        while update is not None:
            updates.append(update)
            update = update.prev
        return list(reversed(updates))
