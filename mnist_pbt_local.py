"""
Performs local population-based training on MNIST ConvNets.

If this file is run directly, the training will be executed and its results
reported at the end.
"""

import math
import datetime
import tensorflow as tf
from pbt import LocalCluster
from mnist import load_mnist_data
from mnist_pbt import ConvNet, plot_hyperparams


class Cluster(LocalCluster[ConvNet]):
    """
    A LocalCluster that trains ConvNets.
    """

    def __init__(self, pop_size: int, vary_opts: bool) -> None:
        """
        Creates a new Cluster with <pop_size> ConvNets.

        If <vary_opts> is True, the TensorFlow Optimizers used by the ConvNets
        will be sampled at random and can be perturbed. Otherwise, they will
        always be AdamOptimizers.
        """
        print('Varying Optimizers:', vary_opts)
        super().__init__(pop_size, lambda num, sess: ConvNet(num, sess, vary_opts))

    def exploit_and_or_explore(self) -> None:
        accuracies = {}
        for graph in self.population:
            accuracy = graph.get_accuracy()
            print('Graph', graph.num, 'accuracy:', accuracy)
            accuracies[graph] = accuracy
        if len(self.population) > 1:
            # Rank population by accuracy
            ranked_pop = sorted(self.population, key=lambda graph: accuracies[graph])
            # Bottom 20% copies top 20%
            worst_graphs = ranked_pop[:math.ceil(0.2 * len(ranked_pop))]
            best_graphs = ranked_pop[math.floor(0.8 * len(ranked_pop)):]
            for i in range(len(worst_graphs)):
                bad_graph = worst_graphs[i]
                good_graph = best_graphs[i]
                print('Graph', bad_graph.num, 'copying graph', good_graph.num)
                bad_graph.set_value(good_graph.get_value())
                bad_graph.explore()

    def plot_hyperparams(self, directory: str) -> None:
        """
        Creates step plots of the hyperparameter update histories of this
        Cluster's population and saves them as images in <directory>.

        <directory> will be created if it does not already exist.
        """
        plot_hyperparams([(graph.step_num, graph.get_update_history(), graph.accuracy) for graph in self.population],
                         self.get_peak_metric_value(), directory)


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    load_mnist_data()
    cluster = Cluster(40, True)
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
    for graph in sorted(cluster.get_population(), key=lambda graph: -graph.get_accuracy()):
        print('Graph', graph.num)
        print('Accuracy:', graph.get_accuracy())
        print('Hyperparameter update history:')
        print()
        print(''.join(str(update) for update in graph.get_update_history()))
    cluster.plot_hyperparams('plots/')
