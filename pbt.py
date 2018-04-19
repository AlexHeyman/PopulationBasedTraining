"""An implementation of population-based training of neural networks for
TensorFlow."""

from typing import List, TypeVar, Generic
import tensorflow as tf

T = TypeVar('T')


class PBTAbleGraph(Generic[T]):
    """A TensorFlow graph that the PBT implementation can train.

    A PBTAbleGraph need not have a TensorFlow Graph object all to itself. T
    should be the type of PBTAbleGraph that this PBTAbleGraph forms populations
    with."""

    x: tf.Tensor
    y_: tf.Tensor
    y: tf.Tensor

    def train_step(self, sess: tf.Session) -> None:
        """Executes one step of this PBTAbleGraph's training."""
        raise NotImplementedError

    def get_metric(self, sess: tf.Session) -> float:
        """Returns a metric for this PBTAbleGraph, typically its accuracy,
        that represents its effectiveness at its task and allows it to be
        compared to other PBTAbleGraphs with the same task."""
        raise NotImplementedError

    def exploit_and_explore(self, sess: tf.Session, population: List[T]) -> None:
        """Judges whether to exploit <population> to improve this PBTAbleGraph
        and, if it should be exploited, exploits it and possibly modifies this
        PBTAbleGraph to explore more options."""
        raise NotImplementedError
