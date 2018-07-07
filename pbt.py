"""
An implementation of population-based training of neural networks for
TensorFlow.
"""

from typing import Iterable, List, Callable, TypeVar, Generic
from collections import OrderedDict
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
    sess: tf.Session
    step_num: int

    def __init__(self, num: int, sess: tf.Session) -> None:
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
        this Graph created in its initializer.
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

    def exploit_and_or_explore(self, population: List['Graph']) -> None:
        """
        Exploits <population>, a list of Graphs of the same type as this one,
        to improve this Graph, and/or modifies this Graph to explore a
        different option, if those actions are judged to be currently
        necessary.
        """
        raise NotImplementedError

    @staticmethod
    def population_exploit_explore(population: List['Graph']) -> None:
        """
        Causes all of the Graphs in <population>, a list of Graphs of this
        type, to exploit and/or explore each other simultaneously, like a
        combined version of all of the Graphs' exploit_and_or_explore()
        methods.
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

    def get_population(self) -> List[T]:
        """
        Returns this Cluster's population of Graphs.

        The returned list should not be modified.
        """
        raise NotImplementedError

    def initialize_variables(self) -> None:
        """
        Initializes all of the TensorFlow Variables that this Cluster's Graphs
        created in their initializers.
        """
        raise NotImplementedError

    def get_highest_metric_graph(self) -> T:
        """
        Returns this Cluster's Graph with the highest metric.
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

    sess: tf.Session
    population: List[T]

    def __init__(self, pop_size: int, graph_maker: Callable[[int, tf.Session], T]) -> None:
        """
        Creates a new LocalCluster with <pop_size> graphs returned by
        <graph_maker> as its population.

        <pop_size> is the number of Graphs that will make up this
        LocalCluster's population. <graph_maker> is a Callable that returns a
        new T with the specified number and Session each time it is called.
        """
        self.sess = tf.Session()
        self.population = []
        for i in range(pop_size):
            self.population.append(graph_maker(i, self.sess))
            print('Graph', i, 'created')

    def get_population(self) -> List[T]:
        return self.population

    def initialize_variables(self):
        for graph in self.population:
            graph.initialize_variables()
            print('Graph', graph.num, 'variables initialized')

    def get_highest_metric_graph(self) -> T:
        highest_graph = None
        highest_metric = None
        for graph in self.population:
            if highest_graph is None:
                highest_graph = graph
                highest_metric = graph.get_metric()
            else:
                metric = graph.get_metric()
                if metric > highest_metric:
                    highest_graph = graph
                    highest_metric = metric
        return highest_graph

    def train(self, until_step_num: int) -> None:
        while True:
            keep_training = False
            for graph in self.population:
                if graph.step_num < until_step_num:
                    keep_training = True
                    if graph.step_num > 0:
                        print('Exploiting/exploring')
                        graph.population_exploit_explore(self.population)
                        print('Finished exploiting/exploring')
                    break
            if keep_training:
                for graph in self.population:
                    if graph.step_num < until_step_num:
                        print('Graph', graph.num, 'starting training run at step', graph.step_num)
                        graph.train()
                        print('Graph', graph.num, 'ending training run at step', graph.step_num)
            else:
                break


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


class HyperparamsGraph(Graph):
    """
    A Graph that stores its hyperparameters as a list of Hyperparameters.
    """

    hyperparams: List[Hyperparameter]
    last_update: HyperparamsUpdate

    def __init__(self, num: int, sess: tf.Session) -> None:
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

    def get_update_history(self) -> Iterable[HyperparamsUpdate]:
        """
        Returns an iterable of this HyperparamsGraph's HyperparamsUpdates in
        order from least to most recent.
        """
        updates = []
        update = self.last_update
        while update is not None:
            updates.append(update)
            update = update.prev
        return reversed(updates)

    def print_update_history(self) -> None:
        """
        Prints this HyperparamsGraph's hyperparameter update history to the
        console.
        """
        updates = []
        update = self.last_update
        while update is not None:
            updates.append(update)
            update = update.prev
        while len(updates) > 0:
            update = updates.pop()
            print('Step', update.step_num)
            for name, value in update.hyperparams.items():
                print(name + ': ' + value)
            print()
