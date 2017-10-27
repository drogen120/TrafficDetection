import tensorflow as tf
import numpy as np

def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        print static_shape
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        print dynamic_shape
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

x = tf.placeholder(tf.int32, shape=(1, None, 224, 3))
print tensor_shape(x, 4)
