"""
Executes synchronous population-based training on MNIST ConvNets using MPI for
Python, reporting its results at the end.

This file must be run with the following command:

mpiexec -n <m> python </path/to/>mnist_pbt_sync.py

where <m>, an integer no less than 2, is the number of processes to use.

This file is intended for use with mpi4py v3.0.3.
"""

from typing import Any, List, Tuple, Dict, Optional
from enum import Enum, auto
from collections import OrderedDict
import math
from mpi4py import MPI
import datetime
import tensorflow as tf
from pbt import Cluster as PBTCluster
from mnist import load_mnist_data
from mnist_pbt import ConvNet, plot_hyperparams


Device = Optional[str]


class Attribute(Enum):
    """
    An attribute of a ConvNet that its Cluster can access remotely.
    """
    STEP_NUM = auto()
    VALUE = auto()
    UPDATE_HISTORY = auto()
    ACCURACY = auto()


GETTERS = {Attribute.STEP_NUM: (lambda graph: graph.step_num),
           Attribute.VALUE: (lambda graph: graph.get_value()),
           Attribute.UPDATE_HISTORY: (lambda graph: graph.get_update_history()),
           Attribute.ACCURACY: (lambda graph: graph.get_accuracy())
           }


class Instruction(Enum):
    """
    A type of instruction that a Cluster can send to its worker processes.
    """
    EXIT = auto()
    INIT = auto()
    GET = auto()
    COPY_TRAIN_GET = auto()


def worker(comm: MPI.Comm, cluster_rank: int) -> None:
    """
    The behavior of a Cluster's worker process.

    <comm> is the MPI Comm that the Cluster and its workers use to communicate,
    and <cluster_rank> is the rank of the Cluster's process.
    """
    sess = tf.compat.v1.Session()
    device, start_num, end_num, vary_opts = comm.recv(source=cluster_rank)
    with tf.compat.v1.device(device):
        graphs = OrderedDict()
        for num in range(start_num, end_num):
            graphs[num] = ConvNet(num, sess, vary_opts)
        while True:
            data = comm.recv(source=cluster_rank)
            instruction = data[0]
            if instruction == Instruction.EXIT:
                break
            elif instruction == Instruction.INIT:
                for graph in graphs.values():
                    graph.initialize_variables()
            else:
                if instruction == Instruction.COPY_TRAIN_GET:
                    new_values = data[3]
                    for num, new_value in new_values.items():
                        graphs[num].set_value(new_value)
                        graphs[num].explore()
                    until_step_num = data[4]
                    for graph in graphs.values():
                        if graph.step_num < until_step_num:
                            graph.train()
                nums = data[1]
                attributes = data[2]
                attribute_getters = [GETTERS[attribute] for attribute in attributes]
                comm.send({num: tuple(getter(graphs[num]) for getter in attribute_getters) for num in nums},
                          dest=cluster_rank)


class Cluster(PBTCluster[ConvNet]):
    """
    A PBT Cluster that synchronously trains ConvNets, distributed over multiple
    worker processes, using MPI for Python.

    A Cluster's get_population() method returns copies of its ConvNets on the
    Cluster's own process.
    """

    sess: tf.compat.v1.Session
    pop_size: int
    vary_opts: bool
    comm: MPI.Comm
    rank_graphs: Dict[int, List[int]]
    graph_ranks: List[int]
    peak_metric: Optional[float]

    def __init__(self, pop_size: int, vary_opts: bool, comm: MPI.Comm, rank_devices: Dict[int, Device]) -> None:
        """
        Creates a new Cluster with <pop_size> ConvNets.

        If <vary_opts> is True, the TensorFlow Optimizers used by the ConvNets
        will be sampled at random and can be perturbed. Otherwise, they will
        always be AdamOptimizers.

        <comm> is the MPI Comm that this Cluster and its worker processes use
        to communicate. <rank_devices> is a dictionary in which each key is a
        worker's process rank and its corresponding value is the TensorFlow
        device on which that worker should create its assigned ConvNets.

        worker(<comm>, <rank>), where <rank> is the rank of this Cluster's
        process, must be called independently in each worker process.
        """
        print('Varying Optimizers:', vary_opts)
        self.sess = tf.compat.v1.Session()
        self.pop_size = pop_size
        self.vary_opts = vary_opts
        self.comm = comm
        self.rank_graphs = {rank: [] for rank in rank_devices.keys()}
        self.graph_ranks = []
        self.peak_metric = None
        self.peak_metric_value = None
        graphs_per_worker = pop_size / len(rank_devices)
        graph_num = 0
        graphs_to_make = 0
        reqs = []
        for rank, device in rank_devices.items():
            graphs_to_make += graphs_per_worker
            start_num = graph_num
            graph_num = min(graph_num + math.ceil(graphs_to_make), pop_size)
            self.rank_graphs[rank].extend(range(start_num, graph_num))
            self.graph_ranks.extend(rank for _ in range(start_num, graph_num))
            reqs.append(comm.isend((device, start_num, graph_num, vary_opts), dest=rank))
            graphs_to_make -= (graph_num - start_num)
        for req in reqs:
            req.wait()

    def initialize_variables(self) -> None:
        reqs = []
        for rank in self.rank_graphs.keys():
            reqs.append(self.comm.isend((Instruction.INIT,), dest=rank))
        for req in reqs:
            req.wait()
        print('Variables initialized')

    def get_population(self) -> List[ConvNet]:
        attributes = self.get_attributes([Attribute.VALUE])
        population = []
        for num in range(self.pop_size):
            graph = ConvNet(num, self.sess, self.vary_opts)
            graph.set_value(attributes[num][0])
            population.append(graph)
        return population

    def get_peak_metric(self) -> float:
        return self.peak_metric

    def get_peak_metric_value(self):
        return self.peak_metric_value

    def _exploit_and_or_explore(self, attributes: List[Tuple[int, float]]) -> Dict[int, Any]:
        for num in range(self.pop_size):
            print('Graph', num, 'accuracy:', attributes[num][1])
        new_values = {}
        if self.pop_size > 1:
            # Rank population by accuracy
            ranked_nums = sorted(range(self.pop_size), key=lambda num: attributes[num][1])
            # Bottom 20% copies top 20%
            worst_nums = ranked_nums[:math.ceil(0.2 * len(ranked_nums))]
            best_nums = ranked_nums[math.floor(0.8 * len(ranked_nums)):]
            best_attributes = self.get_attributes([Attribute.VALUE], best_nums)
            best_metric = attributes[-1][1]
            if self.peak_metric is None or best_metric > self.peak_metric:
                self.peak_metric = best_metric
                self.peak_metric_value = best_attributes[-1][0]
            for i in range(len(worst_nums)):
                print('Graph', worst_nums[i], 'copying graph', best_nums[i])
                new_values[worst_nums[i]] = best_attributes[i][0]
        return new_values

    def train(self, until_step_num: int) -> None:
        trained = False
        attribute_ids = [Attribute.STEP_NUM, Attribute.ACCURACY]
        attributes = self.get_attributes(attribute_ids)
        while True:
            keep_training = False
            new_values = {}
            for graph_attributes in attributes:
                step_num = graph_attributes[0]
                if step_num < until_step_num:
                    keep_training = True
                    if step_num > 0:
                        print('Exploiting/exploring')
                        new_values = self._exploit_and_or_explore(attributes)
                        print('Finished exploiting/exploring')
                        break
            if keep_training:
                print('Starting training runs')
                attributes_dict = {}
                reqs = []
                for rank, graphs in self.rank_graphs.items():
                    rank_new_values = {num: new_values[num] for num in graphs if num in new_values.keys()}
                    reqs.append(self.comm.isend(
                        (Instruction.COPY_TRAIN_GET, graphs, attribute_ids, rank_new_values, until_step_num),
                        dest=rank))
                for req in reqs:
                    req.wait()
                for rank in self.rank_graphs.keys():
                    attributes_dict.update(self.comm.recv(source=rank))
                trained = True
                attributes = [attributes_dict[num] for num in range(self.pop_size)]
                print('Finished training runs')
            else:
                if trained:
                    best_num = max(range(self.pop_size), key=lambda num: attributes[num][1])
                    best_metric = attributes[best_num][1]
                    if self.peak_metric is None or best_metric > self.peak_metric:
                        self.peak_metric = best_metric
                        self.peak_metric_value = self.get_attributes([Attribute.VALUE], [best_num])[0][0]
                break

    def get_attributes(self, attribute_ids: List[Attribute], graph_nums: List[int] = None) -> List[Tuple]:
        """
        Returns the attributes specified by <attribute_ids> of this Cluster's
        ConvNets with numbers <graph_nums>.

        The return value will be a list of tuples, each containing the
        attributes of one ConvNet in the order they are listed in
        <attribute_ids>. If <graph_nums> is None or not specified, the list
        will contain a tuple for each of this Cluster's ConvNets in order of
        increasing number. Otherwise, the list will contain a tuple for each
        ConvNet in the order their numbers appear in <graph_nums>.
        """
        if graph_nums is None:
            graph_nums = list(range(self.pop_size))
            rank_graphs = self.rank_graphs
        else:
            rank_graphs = {}
            for num in graph_nums:
                rank = self.graph_ranks[num]
                if rank in rank_graphs:
                    rank_graphs[rank].append(num)
                else:
                    rank_graphs[rank] = [num]
        attributes_dict = {}
        reqs = []
        for rank, graphs in rank_graphs.items():
            reqs.append(self.comm.isend((Instruction.GET, graphs, attribute_ids), dest=rank))
        for req in reqs:
            req.wait()
        for rank in rank_graphs.keys():
            attributes_dict.update(self.comm.recv(source=rank))
        return [attributes_dict[num] for num in graph_nums]

    def exit_workers(self):
        """
        Instructs this Cluster's worker processes to exit their worker()
        functions, rendering this Cluster unable to communicate with them.

        None of this Cluster's methods should be called after this one.
        """
        reqs = []
        for rank in self.rank_graphs.keys():
            reqs.append(self.comm.isend((Instruction.EXIT,), dest=rank))
        for req in reqs:
            req.wait()


tf.compat.v1.disable_eager_execution()
load_mnist_data()
comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    cluster = Cluster(40, True, comm, {rank: '/cpu:0' for rank in range(1, comm.Get_size())})
    cluster.initialize_variables()
    training_start = datetime.datetime.now()
    cluster.train(20000)
    print('Training time:', datetime.datetime.now() - training_start)
    print()
    peak_value = cluster.get_peak_metric_value()
    print('Peak graph')
    print('Trained for', peak_value[0], 'steps')
    print('Accuracy:', cluster.get_peak_metric())
    print('Hyperparameter update history:')
    print()
    peak_updates = []
    update = peak_value[3]
    while update is not None:
        peak_updates.append(update)
        update = update.prev
    print(''.join(str(update) for update in reversed(peak_updates)))
    attributes = cluster.get_attributes(
        [Attribute.STEP_NUM, Attribute.UPDATE_HISTORY, Attribute.ACCURACY])
    for num in sorted(range(len(attributes)), key=lambda num: -attributes[num][2]):
        graph_info = attributes[num]
        print('Graph', num)
        print('Accuracy:', graph_info[2])
        print('Hyperparameter update history:')
        print()
        print(''.join(str(update) for update in graph_info[1]))
    plot_hyperparams(attributes, cluster.get_peak_metric_value(), 'plots/')
    cluster.exit_workers()
else:
    worker(comm, 0)
