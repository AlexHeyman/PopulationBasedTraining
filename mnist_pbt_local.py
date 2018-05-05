"""
Performs local population-based training on MNIST convnets for 10,000 steps
each, reporting relevant information at the end.
"""

import datetime
from pbt import LocalPBTCluster
from mnist_pbt import PBTAbleMNISTConvNet, random_mnist_convnet


pop_size = 10
cluster = LocalPBTCluster[PBTAbleMNISTConvNet](pop_size, random_mnist_convnet)
cluster.initialize_variables()
training_start = datetime.datetime.now()
cluster.train(lambda net, population: net.step_num < 10000)
print('Training time:', datetime.datetime.now() - training_start)
print()
for net in reversed(sorted(cluster.get_population(), key=lambda net: net.get_accuracy())):
    print('Net', net.num, 'accuracy:', net.get_accuracy())
    net.print_update_history()
    print()
