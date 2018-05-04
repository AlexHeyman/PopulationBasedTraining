"""
An implementation of population-based training of neural networks for
TensorFlow.
"""

from typing import List, Callable, TypeVar, Generic
import random
from threading import Thread
import tensorflow as tf

T = TypeVar('T', bound='PBTAbleGraph')


class PBTAbleGraph(Generic[T]):
    """
    A TensorFlow graph that a PBTCluster can train.

    A PBTAbleGraph need not have a TensorFlow Graph object all to itself.

    A PBTAbleGraph should create all of the TensorFlow Variables that it uses
    for its training in its initializer so that if a PBTCluster is supervising
    its initialization, the PBTCluster can place its Variables on the
    appropriate device.

    T is the type of PBTAbleGraph that this PBTAbleGraph forms populations
    with.
    """

    sess: tf.Session

    def __init__(self, sess: tf.Session) -> None:
        """
        Creates a new PBTAbleGraph with the specified associated Session.
        """
        self.sess = sess

    def initialize_variables(self, sess: tf.Session) -> None:
        """
        Instructs <sess> to run the initializer Operations of all of the
        TensorFlow Variables that this PBTAbleGraph created in its initializer.
        """
        raise NotImplementedError

    def get_metric(self) -> float:
        """
        Returns a metric for this PBTAbleGraph, typically its accuracy, that
        represents its effectiveness at its task and allows it to be compared
        to other PBTAbleGraphs with the same task.
        """
        raise NotImplementedError

    def train_step(self) -> None:
        """
        Executes one step of this PBTAbleGraph's training.
        """
        raise NotImplementedError

    def is_ready(self) -> bool:
        """
        Returns whether this PBTAbleGraph is ready to use the rest of its
        population to decide whether to exploit and/or explore.
        """
        raise NotImplementedError

    def exploit_and_or_explore(self, population: List[T]) -> None:
        """
        Exploits <population> to improve this PBTAbleGraph and/or modifies this
        PBTAbleGraph to explore a different option, if those actions are judged
        to be currently necessary.
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


class DistTaskInfo(Generic[T]):
    """
    Stores information on one of a DistPBTCluster's tasks.

    T is the type of PBTAbleGraph that this DistTaskInfo's DistPBTCluster
    trains.
    """

    name: str
    server: tf.train.Server
    sess: tf.Session
    graph: T

    def __init__(self, cluster: tf.train.ClusterSpec, task_index: int,
                 graph_maker: Callable[[tf.Session], T]) -> None:
        """
        Creates a new DistTaskInfo.

        <cluster> is the ClusterSpec that describes this DistTaskInfo's
        DistPBTCluster. <task_index> is the index of the cluster's task that
        this DistTaskInfo describes. <graph_maker> is a Callable that returns a
        new T with the specified Session each time it is called.
        """
        self.name = '/job:worker/task:' + str(task_index)
        self.server = tf.train.Server(cluster, job_name="worker", task_index=task_index)
        self.sess = tf.Session(self.server.target)
        with tf.device(self.name):
            self.graph = graph_maker(self.sess)


def pbt_thread(graph: T, population: List[T],
               training_cond: Callable[[T, List[T]], bool]) -> None:
    """
    A single thread of a DistPBTCluster's population-based training.

    <graph> is the PBTAbleGraph to be trained. <population> is the population
    of PBTAbleGraphs to which <graph> belongs. T is the type of PBTAbleGraph of
    which <population> consists. <training_cond> is a Callable that, when
    passed <graph> and <population> returns whether the training should
    continue.
    """
    while training_cond(graph, population):
        graph.train_step()
        if graph.is_ready():
            graph.exploit_and_or_explore(population)


class DistPBTCluster(Generic[T], PBTCluster[T]):
    """
    A PBTCluster that uses distributed TensorFlow to train its PBTAbleGraphs
    on different devices.
    """

    cluster: tf.train.ClusterSpec
    task_info: List[DistTaskInfo[T]]

    def __init__(self, addresses: List[str], graph_maker: Callable[[tf.Session], T]) -> None:
        """
        Creates a new DistPBTCluster with graphs returned by <graph_maker> as
        its population.

        <addresses> is the list of network addresses of the devices on which
        this DistPBTCluster will host its tasks, with each task training one
        PBTAbleGraph. <graph_maker> is a Callable that returns a new T with the
        specified Session each time it is called.
        """
        self.cluster = tf.train.ClusterSpec({"worker": addresses})
        self.task_info = []
        for task_index in range(len(addresses)):
            self.task_info.append(DistTaskInfo[T](self.cluster, task_index, graph_maker))

    def get_population(self) -> List[T]:
        return [info.graph for info in self.task_info]

    def initialize_variables(self):
        chief_sess = self.task_info[0].sess
        for info in self.task_info:
            info.graph.initialize_variables(chief_sess)

    def get_highest_metric_graph(self) -> T:
        highest_graph = None
        highest_metric = None
        for info in self.task_info:
            graph = info.graph
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
        population = []
        threads = []
        for info in self.task_info:
            population.append(info.graph)
            threads.append(Thread(target=pbt_thread, name='PBT thread for ' + info.name,
                                  args=(info.graph, population, training_cond), daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


class LocalPBTCluster(Generic[T], PBTCluster[T]):
    """
    A PBTCluster that simulates parallel training with a single local thread.
    """

    sess: tf.Session
    population: List[T]

    def __init__(self, pop_size: int, graph_maker: Callable[[tf.Session], T]) -> None:
        """
        Creates a new LocalPBTCluster with <pop_size> graphs returned by
        <graph_maker> as its population.

        <pop_size> is the number of PBTAbleGraphs that will make up this
        LocalPBTCluster's population. <graph_maker> is a Callable that returns
        a new T with the specified Session each time it is called.
        """
        self.sess = tf.Session()
        self.population = [graph_maker(self.sess) for _ in range(pop_size)]

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
        unfinished_graphs = list(self.population)
        i = -1
        while len(unfinished_graphs) > 0:
            i = (i + 1) % len(unfinished_graphs)
            graph = unfinished_graphs[i]
            if training_cond(graph, self.population):
                # Small chance to skip each training step to simulate the
                # "devices" not necessarily being perfectly synchronized
                if random.random() < 0.9:
                    graph.train_step()
                    if graph.is_ready():
                        graph.exploit_and_or_explore(self.population)
            else:
                unfinished_graphs.pop(i)
                i -= 1
