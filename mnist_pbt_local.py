"""
Performs local population-based training on MNIST convnets, reporting relevant
information at the end.
"""

import datetime
from pbt import LocalPBTCluster
from mnist_pbt import PBTAbleMNISTConvNet
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    pop_size = 10
    cluster = LocalPBTCluster[PBTAbleMNISTConvNet](pop_size, lambda device, sess:
                                                   PBTAbleMNISTConvNet(device, sess, mnist))
    cluster.initialize_variables()
    for net in cluster.get_population():
        net.get_accuracy()
    training_start = datetime.datetime.now()
    cluster.train(lambda net, population: net.step_num < 10000)
    print('Training time:', datetime.datetime.now() - training_start)
    ranked_pop = reversed(sorted(cluster.get_population(), key=lambda net: net.get_accuracy()))
    print()
    for net in ranked_pop:
        print('Net', net.num)
        print('Accuracy:', net.get_accuracy())
        print('Hyperparameter update history:')
        print()
        net.print_update_history()
