"""
An implementation of population-based training of neural networks for
TensorFlow.
"""

from typing import Union, List, Callable, TypeVar, Generic
import random
from threading import Thread, Lock
import tensorflow as tf

T = TypeVar('T', bound='PBTAbleGraph')
Device = Union[str, Callable[[tf.Operation], str], None]


class PBTAbleGraph(Generic[T]):
    """
    A TensorFlow graph that a PBTCluster can train.

    A PBTAbleGraph need not have a TensorFlow Graph object all to itself.

    A PBTAbleGraph has an associated device on which all of its TensorFlow
    information must be placed, as well as an associated TensorFlow Session.

    T is the type of PBTAbleGraph that this PBTAbleGraph forms populations
    with.
    """

    device: Device
    sess: tf.Session
    sess_lock: Lock

    def __init__(self, device: Device, sess: tf.Session) -> None:
        """
        Creates a new PBTAbleGraph with associated device <device> and Session
        <session>.
        """
        self.device = device
        self.sess = sess
        self.sess_lock = Lock()

    def initialize_variables(self, sess: tf.Session) -> None:
        """
        Instructs <sess> to run the initializer Operations of all of the
        TensorFlow Variables that this PBTAbleGraph created in its initializer.
        """
        raise NotImplementedError

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """
        Calls this PBTAbleGraph's Session's run() method with the specified
        parameters and returns its return value, blocking beforehand until any
        other invocations of this method in other threads have finished.

        This method should always be called instead of calling the Session's
        run() method directly to prevent multiple threads from interfering with
        each other's invocations of the Session's run() method.
        """
        self.sess_lock.acquire()
        value = self.sess.run(fetches, feed_dict, options, run_metadata)
        self.sess_lock.release()
        return value

    def assign(self, var: tf.Variable, graph: T, graph_var: tf.Variable) -> None:
        """
        Assigns the value of <graph_var>, a Variable of <graph>, to <var>, a
        Variable of this PBTAbleGraph.
        """
        with tf.device(self.device):
            value = graph.run(graph_var)
            self.run(var.assign(value))

    def get_metric(self) -> float:
        """
        Returns a metric for this PBTAbleGraph, typically its accuracy, that
        represents its effectiveness at its task and allows it to be compared
        to other PBTAbleGraphs with the same task.
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Trains this PBTAbleGraph until it is ready to consider exploitation of
        its population.
        """
        raise NotImplementedError

    def exploit_and_or_explore(self, population: List[T]) -> None:
        """
        Exploits <population> to improve this PBTAbleGraph and/or modifies this
        PBTAbleGraph to explore a different option, if those actions are judged
        to be currently necessary.
        """
        raise NotImplementedError

    @staticmethod
    def population_exploit_explore(population: List[T]) -> None:
        """
        Causes all of the PBTAbleGraphs in <population> to exploit and/or
        explore each other simultaneously, like a combined version of all of
        the graphs' exploit_and_or_explore() methods.
        """
        raise NotImplementedError


class PBTCluster(Generic[T]):
    """
    A system that can perform population-based training of PBTAbleGraphs.

    Any TensorFlow Variables created in the initializers of a PBTCluster's
    PBTAbleGraphs should be initialized by calling the PBTCluster's
    initialize_variables() method. Even if such variables are global, they may
    not be on the proper device to be initialized if a Session is instructed
    to initialize all global variables.

    T is the type of PBTAbleGraph that this PBTCluster trains.
    """

    def get_population(self) -> List[T]:
        """
        Returns this PBTCluster's population of PBTAbleGraphs.

        The returned list should not be modified.
        """
        raise NotImplementedError

    def initialize_variables(self) -> None:
        """
        Initializes all of the TensorFlow Variables that this PBTCluster's
        PBTAbleGraphs created in their initializers.
        """
        raise NotImplementedError

    def get_highest_metric_graph(self) -> T:
        """
        Returns this PBTCluster's PBTAbleGraph with the highest metric.
        """
        raise NotImplementedError

    def train(self, training_cond: Callable[[T, List[T]], bool]) -> None:
        """
        Performs population-based training on this PBTCluster's population.

        <training_cond> is a Callable that, when passed a PBTAbleGraph and this
        PBTCluster's population, returns whether the training of the specified
        PBTAbleGraph should continue.
        """
        raise NotImplementedError


def async_pbt_thread(graph: T, population: List[T],
                     training_cond: Callable[[T, List[T]], bool]) -> None:
    """
    A single thread of an AsyncPBTCluster's population-based training.
    <graph> is the PBTAbleGraph to be trained. <population> is the population
    of PBTAbleGraphs to which <graph> belongs. T is the type of PBTAbleGraph of
    which <population> consists. <training_cond> is a Callable that, when
    passed <graph> and <population>, returns whether the training should
    continue.
    """
    while training_cond(graph, population):
        graph.train()
        if not training_cond(graph, population):
            break
        graph.exploit_and_or_explore(population)


class AsyncPBTCluster(Generic[T], PBTCluster[T]):
    """
    A PBTCluster that uses distributed TensorFlow to train its PBTAbleGraphs
    on different devices asynchronously.
    """

    cluster: tf.train.ClusterSpec
    population: List[T]

    def __init__(self, addresses: List[str], graph_maker: Callable[[Device, tf.Session], T]) -> None:
        """
        Creates a new AsyncPBTCluster with graphs returned by <graph_maker> as
        its population.

        <addresses> is the list of network addresses of the devices on which
        this AsyncPBTCluster will host its tasks, with each task training one
        PBTAbleGraph. <graph_maker> is a Callable that returns a new T with the
        specified device and Session each time it is called.
        """
        self.cluster = tf.train.ClusterSpec({"worker": addresses})
        self.population = []
        for task_index in range(len(addresses)):
            device = '/job:worker/task:' + str(task_index)
            server = tf.train.Server(self.cluster, job_name="worker", task_index=task_index)
            sess = tf.Session(server.target)
            self.population.append(graph_maker(device, sess))

    def get_population(self) -> List[T]:
        return self.population

    def initialize_variables(self):
        chief_sess = self.population[0].sess
        for graph in self.population:
            graph.initialize_variables(chief_sess)

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

    def train(self, training_cond: Callable[[T, List[T]], bool]) -> None:
        threads = []
        for graph in self.population:
            threads.append(Thread(target=async_pbt_thread, name='PBT thread for ' + graph.device,
                                  args=(graph, self.population, training_cond), daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


class LocalPBTCluster(Generic[T], PBTCluster[T]):
    """
    A PBTCluster that simulates synchronous training with a single local
    thread.
    """

    sess: tf.Session
    population: List[T]

    def __init__(self, pop_size: int, graph_maker: Callable[[Device, tf.Session], T]) -> None:
        """
        Creates a new LocalPBTCluster with <pop_size> graphs returned by
        <graph_maker> as its population.

        <pop_size> is the number of PBTAbleGraphs that will make up this
        LocalPBTCluster's population. <graph_maker> is a Callable that returns
        a new T with the specified device and Session each time it is called.
        """
        self.sess = tf.Session()
        self.population = [graph_maker(None, self.sess) for _ in range(pop_size)]

    def get_population(self) -> List[T]:
        return self.population

    def initialize_variables(self):
        for graph in self.population:
            graph.initialize_variables(self.sess)

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

    def train(self, training_cond: Callable[[T, List[T]], bool]) -> None:
        while True:
            keep_training = False
            for graph in self.population:
                if training_cond(graph, self.population):
                    keep_training = True
                    graph.train()
            if keep_training:
                for graph in self.population:
                    if training_cond(graph, self.population):
                        graph.population_exploit_explore(self.population)
                        break
                else:
                    break
            else:
                break
