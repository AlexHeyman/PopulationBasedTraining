"""
Performs local population-based training on MNIST convnets, reporting relevant
information at the end.
"""

import datetime
from tensorflow.models.official.mnist.dataset import train, test
from mnist import set_mnist_data
from mnist_pbt import LocalCluster


if __name__ == '__main__':
    set_mnist_data(train('MNIST_data/'), test('MNIST_data/'))
    cluster = LocalCluster(50)
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
