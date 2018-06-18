"""
Performs distributed asynchronous population-based training on MNIST convnets,
reporting relevant information at the end.
"""

import datetime
from pbt import AsyncCluster
from mnist_pbt import ConvNet
from tensorflow.models.official.mnist.dataset import train, test


if __name__ == '__main__':
    train_data = train('MNIST_data/')
    test_data = test('MNIST_data/')
    pop_size = 50
    addresses = ['localhost:' + str(2220 + i) for i in range(pop_size)]
    cluster = AsyncCluster[ConvNet](addresses,
                                    lambda device, sess: ConvNet(device, sess, train_data, test_data))
    cluster.initialize_variables()
    for net in cluster.get_population():
        net.get_accuracy()
    training_start = datetime.datetime.now()
    cluster.train(lambda net, population: net.step_num < 20000)
    print('Training time:', datetime.datetime.now() - training_start)
    ranked_pop = reversed(sorted(cluster.get_population(), key=lambda net: net.get_accuracy()))
    print()
    for net in ranked_pop:
        print('Net', net.num)
        print('Accuracy:', net.get_accuracy())
        print('Hyperparameter update history:')
        print()
        net.print_update_history()
