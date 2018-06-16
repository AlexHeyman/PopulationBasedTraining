"""
An implementation of population-based training of neural networks for
TensorFlow.
"""

from typing import Union, List, Tuple, Callable, TypeVar, Generic
from threading import Thread, RLock
from multiprocessing import Process, Queue
import tensorflow as tf

T = TypeVar('T', bound='PBTAbleGraph')
Device = Union[str, Callable[[tf.Operation], str], None]


class PBTAbleGraph(Generic[T]):
    """
    A TensorFlow graph that a PBTCluster can train.

    A PBTAbleGraph need not have a TensorFlow Graph object all to itself.

    A PBTAbleGraph has an associated device on which all of its TensorFlow
    information must be placed, an associated TensorFlow Session, and an RLock
    that should be used to prevent multiple threads from interacting with it at
    once.

    T is the type of PBTAbleGraph that this PBTAbleGraph forms populations
    with.
    """

    device: Device
    sess: tf.Session
    lock: RLock

    def __init__(self, device: Device, sess: tf.Session) -> None:
        """
        Creates a new PBTAbleGraph with associated device <device> and Session
        <session>.
        """
        self.device = device
        self.sess = sess
        self.lock = RLock()

    def initialize_variables(self) -> None:
        """
        Runs the initializer Operations of all of the TensorFlow Variables that
        this PBTAbleGraph created in its initializer.
        """
        raise NotImplementedError

    def get_metric(self) -> float:
        """
        Returns a metric for this PBTAbleGraph, typically its accuracy, that
        represents its effectiveness at its task and allows it to be compared
        to other PBTAbleGraphs with the same task.
        """
        raise NotImplementedError

    def get_step_num(self) -> int:
        """
        Returns the number of training steps that this PBTAbleGraph has
        performed.
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
            graph.initialize_variables()

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


def _pbt_localhost_process(queue: Queue, addresses: List[str], task_index: int) -> None:
    cluster = tf.train.ClusterSpec({'worker': addresses})
    server = tf.train.Server(cluster, job_name='worker', task_index=task_index)
    queue.put(server.target)
    server.join()


def _async_pbt_thread(graph: T, population: List[T],
                      training_cond: Callable[[T, List[T]], bool]) -> None:
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
        self.cluster = tf.train.ClusterSpec({'worker': addresses})
        self.population = []
        for task_index in range(len(addresses)):
            device = '/job:worker/task:' + str(task_index)
            if addresses[task_index].startswith('localhost'):
                queue = Queue()
                process = Process(target=_pbt_localhost_process, args=(queue, addresses, task_index))
                process.daemon = True
                process.start()
                target = queue.get()
            else:
                server = tf.train.Server(self.cluster, job_name='worker', task_index=task_index)
                target = server.target
            sess = tf.Session(target)
            self.population.append(graph_maker(device, sess))

    def get_population(self) -> List[T]:
        return self.population

    def initialize_variables(self):
        for graph in self.population:
            graph.initialize_variables()

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
            threads.append(Thread(target=_async_pbt_thread, name='PBT thread for ' + graph.device,
                                  args=(graph, self.population, training_cond), daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


class Hyperparameter:
    """
    A non-trained parameter of a HyperparamsPBTAbleGraph.

    A Hyperparameter's __str__() method should return a string representing its
    value.
    """

    name: str
    graph: 'HyperparamsPBTAbleGraph'

    def __init__(self, name: str, graph: 'HyperparamsPBTAbleGraph') -> None:
        """
        Creates a new Hyperparameter of <graph> with descriptive name <name>.
        """
        self.name = name
        self.graph = graph
        graph.hyperparams.append(self)

    def initialize_variables(self) -> None:
        """
        Runs the initializer Operations of all of the TensorFlow Variables that
        this Hyperparameter created in its initializer.
        """
        raise NotImplementedError

    def copy(self, hyperparam: 'Hyperparameter') -> None:
        """
        Sets this Hyperparameter's value to that of <hyperparam>, a
        Hyperparameter of the same type.
        """
        raise NotImplementedError

    def perturb(self) -> None:
        """
        Alters this Hyperparameter to explore a different option for it.
        """
        raise NotImplementedError


class HyperparamsUpdate:
    """
    Stores information about a HyperparamsPBTAbleGraph's update of its
    hyperparameters.
    """

    prev: 'HyperparamsUpdate'
    step_num: int
    hyperparams: List[Tuple[str, str]]

    def __init__(self, graph: 'HyperparamsPBTAbleGraph') -> None:
        """
        Creates a new HyperparamsUpdate that stores <graph>'s current
        information.
        """
        self.prev = graph.last_update
        self.step_num = graph.get_step_num()
        self.hyperparams = [(hyperparam.name, str(hyperparam)) for hyperparam in graph.hyperparams]


class HyperparamsPBTAbleGraph(Generic[T], PBTAbleGraph[T]):
    """
    A PBTAbleGraph that stores its hyperparameters as a list of
    Hyperparameters.
    """

    hyperparams: List[Hyperparameter]
    last_update: HyperparamsUpdate

    def __init__(self, device: Device, sess: tf.Session) -> None:
        """
        Creates a new HyperparamsPBTAbleGraph with associated device <device>
        and Session <session>.
        """
        super().__init__(device, sess)
        self.hyperparams = []
        self.last_update = None

    def initialize_variables(self) -> None:
        for hyperparam in self.hyperparams:
            hyperparam.initialize_variables()

    def record_update(self):
        """
        Records this HyperparamsPBTAbleGraph's current information as a new
        update to its hyperparameters.
        """
        self.lock.acquire()
        self.last_update = HyperparamsUpdate(self)
        self.lock.release()

    def print_update_history(self):
        """
        Prints this HyperparamsPBTAbleGraph's hyperparameter update history to
        the console.
        """
        self.lock.acquire()
        updates = []
        update = self.last_update
        while update is not None:
            updates.append(update)
            update = update.prev
        while len(updates) > 0:
            update = updates.pop()
            print('Step', update.step_num)
            for name, value in update.hyperparams:
                print(name + ': ' + value)
            print()
        self.lock.release()
