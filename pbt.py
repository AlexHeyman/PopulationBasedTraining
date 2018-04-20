"""
An implementation of population-based training of neural networks for
TensorFlow.
"""

from typing import List, Callable, TypeVar, Generic
from threading import Thread
import tensorflow as tf

T = TypeVar('T', bound='PBTAbleGraph')


class PBTAbleGraph(Generic[T]):
    """
    A TensorFlow graph that a PBTCluster can train.

    A PBTAbleGraph need not have a TensorFlow Graph object all to itself. T is
    the type of PBTAbleGraph that this PBTAbleGraph forms populations with.
    """

    def get_metric(self, sess: tf.Session) -> float:
        """
        Returns a metric for this PBTAbleGraph, typically its accuracy, that
        represents its effectiveness at its task and allows it to be compared
        to other PBTAbleGraphs with the same task.

        <sess> is the Session on which this PBTAbleGraph is running.
        """
        raise NotImplementedError

    def train_step(self, sess: tf.Session) -> None:
        """
        Executes one step of this PBTAbleGraph's training.

        <sess> is the Session on which this PBTAbleGraph is running.
        """
        raise NotImplementedError

    def exploit_and_or_explore(self, sess: tf.Session, population: List[T]) -> None:
        """
        Exploits <population> to improve this PBTAbleGraph and/or modifies this
        PBTAbleGraph to explore a different option, if those actions are judged
        to be currently appropriate.

        <sess> is the Session on which this PBTAbleGraph is running."""
        raise NotImplementedError


class PBTClusterTaskInfo(Generic[T]):
    """
    Stores information on one of a PBTCluster's tasks.

    T is the type of PBTAbleGraph that this PBTClusterTaskInfo's PBTCluster
    trains.
    """

    name: str
    server: tf.train.Server
    sess: tf.Session
    graph: T

    def __init__(self, cluster: tf.train.ClusterSpec, task_index: int, graph_maker: Callable[[], T]) -> None:
        """
        Creates a new PBTClusterTaskInfo.

        <cluster> is the ClusterSpec that describes this PBTClusterTaskInfo's
        PBTCluster. <task_index> is the index of the cluster's task that this
        PBTClusterTaskInfo describes. <graph_maker> is a Callable that returns
        a new T each time it is called.
        """
        self.name = '/job:worker/task:' + str(task_index)
        self.server = tf.train.Server(cluster, job_name="worker", task_index=task_index)
        self.sess = tf.Session(self.server.target)
        with tf.device(self.name):
            self.graph = graph_maker()


def pbt_thread(sess: tf.Session, graph: T, population: List[T],
               training_cond: Callable[[tf.Session, T, List[T]], bool]) -> None:
    """
    A single thread of a PBTCluster's population-based training.

    <graph> is the PBTAbleGraph to be trained. <sess> is the Session on which
    <graph> is running. <population> is the population of PBTAbleGraphs to
    which <graph> belongs. T is the type of PBTAbleGraph of which <population>
    consists. <training_cond> is a Callable that, when passed <sess>, <graph>,
    and <population>, returns whether the training should continue.
    """
    while training_cond(sess, graph, population):
        graph.train_step(sess)
        graph.exploit_and_or_explore(sess, population)


class PBTCluster(Generic[T]):
    """
    A TensorFlow cluster that can perform population-based training of
    PBTAbleGraphs.

    T is the type of PBTAbleGraph that this PBTCluster trains.
    """

    cluster: tf.train.ClusterSpec
    task_info: List[PBTClusterTaskInfo[T]]

    def __init__(self, addresses: List[str], graph_maker: Callable[[], T]) -> None:
        """
        Creates a new PBTCluster with graphs returned by <graph_maker> as its
        population.

        <addresses> is the list of network addresses of the devices or
        processes on which this PBTCluster will host its tasks, with each task
        training one PBTAbleGraph. <graph_maker> is a Callable that returns a
        new T each time it is called.
        """
        self.cluster = tf.train.ClusterSpec({"worker": addresses})
        self.task_info = []
        for task_index in range(len(addresses)):
            self.task_info.append(PBTClusterTaskInfo[T](self.cluster, task_index, graph_maker))

    def get_highest_metric_graph(self) -> T:
        """
        Returns this PBTCluster's PBTAbleGraph with the highest metric.
        """
        highest_graph = None
        highest_metric = None
        for info in self.task_info:
            graph = info.graph
            if highest_graph is None:
                highest_graph = graph
                highest_metric = graph.get_metric(info.sess)
            else:
                metric = graph.get_metric(info.sess)
                if metric > highest_metric:
                    highest_graph = graph
                    highest_metric = metric
        return highest_graph

    def train(self, training_cond: Callable[[tf.Session, T, List[T]], bool]) -> None:
        """
        Performs population-based training on this PBTCluster's population.

        <training_cond> is a Callable that, when passed a PBTAbleGraph's
        Session, the PBTAbleGraph itself, and this PBTCluster's population,
        returns whether the training of the specified PBTAbleGraph should
        continue.
        """
        population = []
        threads = []
        for info in self.task_info:
            population.append(info.graph)
            threads.append(Thread(target=pbt_thread, name='PBT thread for ' + info.name,
                                  args=(info.sess, info.graph, population, training_cond), daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
