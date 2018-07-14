"""
"""

from typing import Any, List, Tuple, Dict
from enum import Enum, auto
from collections import OrderedDict
import math
from mpi4py import MPI
import datetime
import tensorflow as tf
from tensorflow.models.official.mnist.dataset import train, test
from pbt import Cluster as PBTCluster
from mnist import set_mnist_data
from mnist_pbt import ConvNet, plot_hyperparams


class Instruction(Enum):
    EXIT = auto()
    INIT = auto()
    GET = auto()
    SET_TRAIN_GET = auto()


class Attribute(Enum):
    STEP_NUM = auto()
    VALUE = auto()
    UPDATE_HISTORY = auto()
    ACCURACY = auto()


GETTERS = {Attribute.STEP_NUM: (lambda graph: graph.step_num),
           Attribute.VALUE: (lambda graph: graph.get_value()),
           Attribute.UPDATE_HISTORY: (lambda graph: graph.get_update_history()),
           Attribute.ACCURACY: (lambda graph: graph.get_accuracy())
           }

comm = MPI.COMM_WORLD


def worker(cluster_rank: int) -> None:
    sess = tf.Session()
    start_num, end_num = comm.recv(source=cluster_rank)
    graphs = OrderedDict()
    for num in range(start_num, end_num):
        graphs[num] = ConvNet(num, sess)
    while True:
        data = comm.recv(source=cluster_rank)
        instruction = data[0]
        if instruction == Instruction.EXIT:
            break
        elif instruction == Instruction.INIT:
            for graph in graphs.values():
                graph.initialize_variables()
        else:
            if instruction == Instruction.SET_TRAIN_GET:
                new_values = data[3]
                for num, new_value in new_values.items():
                    graphs[num].set_value(new_value)
                    graphs[num].explore()
                for graph in graphs.values():
                    graph.train()
            nums = data[1]
            attributes = data[2]
            attribute_getters = [GETTERS[attribute] for attribute in attributes]
            comm.send({num: tuple(getter(graphs[num]) for getter in attribute_getters) for num in nums},
                      dest=cluster_rank)


class Cluster(PBTCluster[ConvNet]):

    sess: tf.Session
    pop_size: int
    rank_graphs: Dict[int, List[int]]
    graph_ranks: List[int]

    def __init__(self, pop_size: int, worker_ranks: List[int]) -> None:
        self.sess = tf.Session()
        self.pop_size = pop_size
        self.rank_graphs = {rank: [] for rank in worker_ranks}
        self.graph_ranks = []
        graphs_per_worker = pop_size / len(worker_ranks)
        graph_num = 0
        graphs_to_make = 0
        reqs = []
        for rank in worker_ranks:
            graphs_to_make += graphs_per_worker
            start_num = graph_num
            graph_num = min(graph_num + math.ceil(graphs_to_make), pop_size)
            self.rank_graphs[rank].extend(range(start_num, graph_num))
            self.graph_ranks.extend(rank for _ in range(start_num, graph_num))
            reqs.append(comm.isend((start_num, graph_num), dest=rank))
            graphs_to_make -= (graph_num - start_num)
        for req in reqs:
            req.wait()

    def get_population(self) -> List[ConvNet]:
        attributes = self.get_attributes([Attribute.VALUE])
        population = []
        for num in range(self.pop_size):
            graph = ConvNet(num, self.sess)
            graph.set_value(attributes[num][0])
            population.append(graph)
        return population

    def initialize_variables(self) -> None:
        reqs = []
        for rank in self.rank_graphs.keys():
            reqs.append(comm.isend((Instruction.INIT,), dest=rank))
        for req in reqs:
            req.wait()
        print('Variables initialized')

    def get_highest_metric_graph(self) -> ConvNet:
        attributes = self.get_attributes([Attribute.ACCURACY])
        best_num = None
        best_acc = None
        for num in range(self.pop_size):
            accuracy = attributes[num][0]
            if best_num is None or accuracy > best_acc:
                best_num = num
                best_acc = accuracy
        graph = ConvNet(best_num, self.sess)
        best_rank = self.graph_ranks[best_num]
        comm.send((Instruction.GET, [best_num], [Attribute.VALUE]), dest=best_rank)
        graph.set_value(comm.recv(source=best_rank)[best_num][0])
        return graph

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
            for i in range(len(worst_nums)):
                print('Graph', worst_nums[i], 'copying graph', best_nums[i])
                new_values[worst_nums[i]] = best_attributes[i][0]
        return new_values

    def train(self, until_step_num: int) -> None:
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
                    reqs.append(comm.isend(
                        (Instruction.SET_TRAIN_GET, graphs, attribute_ids, rank_new_values), dest=rank))
                for req in reqs:
                    req.wait()
                for rank in self.rank_graphs.keys():
                    attributes_dict.update(comm.recv(source=rank))
                attributes = [attributes_dict[num] for num in range(self.pop_size)]
                print('Finished training runs')
            else:
                break

    def get_attributes(self, attribute_ids: List[Attribute], graph_nums: List[int]=None) -> List[Tuple]:
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
            reqs.append(comm.isend((Instruction.GET, graphs, attribute_ids), dest=rank))
        for req in reqs:
            req.wait()
        for rank in rank_graphs.keys():
            attributes_dict.update(comm.recv(source=rank))
        return [attributes_dict[num] for num in graph_nums]

    def exit_workers(self):
        reqs = []
        for rank in self.rank_graphs.keys():
            reqs.append(comm.isend((Instruction.EXIT,), dest=rank))
        for req in reqs:
            req.wait()


with tf.device('/cpu:0'):
    set_mnist_data(train('MNIST_data/'), test('MNIST_data/'))
    if comm.Get_rank() == 0:
        cluster = Cluster(50, list(range(1, comm.Get_size())))
        cluster.initialize_variables()
        training_start = datetime.datetime.now()
        cluster.train(20000)
        print('Training time:', datetime.datetime.now() - training_start)
        attributes = cluster.get_attributes(
            [Attribute.STEP_NUM, Attribute.UPDATE_HISTORY, Attribute.ACCURACY])
        ranked_nums = sorted(range(len(attributes)), key=lambda num: -attributes[num][2])
        print()
        for num in ranked_nums:
            graph_info = attributes[num]
            print('Graph', num)
            print('Accuracy:', graph_info[2])
            print('Hyperparameter update history:')
            print()
            print(''.join(str(update) for update in graph_info[1]))
        plot_hyperparams(attributes, 'plots/')
        cluster.exit_workers()
    else:
        worker(0)
