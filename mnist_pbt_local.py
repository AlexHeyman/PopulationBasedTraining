"""
Performs local population-based training on MNIST ConvNets.

If this file is run directly, the training will be executed and its results
reported at the end.
"""

import math
import datetime
from tensorflow.models.official.mnist.dataset import train, test
from pbt import LocalCluster
from mnist import set_mnist_data
from mnist_pbt import ConvNet, plot_hyperparams


class Cluster(LocalCluster[ConvNet]):
    """
    A LocalCluster that trains ConvNets.
    """

    def __init__(self, pop_size: int) -> None:
        """
        Creates a new Cluster with <pop_size> ConvNets.
        """
        super().__init__(pop_size, lambda num, sess: ConvNet(num, sess))

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
        plot_hyperparams([(graph.step_num, graph.get_update_history(), graph.accuracy)
                          for graph in self.population], directory)


if __name__ == '__main__':
    set_mnist_data(train('MNIST_data/'), test('MNIST_data/'))
    cluster = Cluster(50)
    cluster.initialize_variables()
    training_start = datetime.datetime.now()
    cluster.train(20000)
    print('Training time:', datetime.datetime.now() - training_start)
    ranked_pop = sorted(cluster.get_population(), key=lambda graph: -graph.get_accuracy())
    print()
    for graph in ranked_pop:
        print('Graph', graph.num)
        print('Accuracy:', graph.get_accuracy())
        print('Hyperparameter update history:')
        print()
        print(''.join(str(update) for update in graph.get_update_history()))
    cluster.plot_hyperparams('plots/')
