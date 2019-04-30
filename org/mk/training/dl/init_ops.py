#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:24:59 2019

@author: mohit
"""

from tensorflow.python.ops import init_ops
import numpy as np


class Constant(init_ops.Initializer):
    """Initializer that generates tensors with constant values.

    The resulting tensor is populated with values of type `dtype`, as
    specified by arguments `value` following the desired `shape` of the
    new tensor (see examples below).

  The argument `value` can be a constant value, or a list of values of type
  `dtype`. If `value` is a list, then the length of the list must be less
  than or equal to the number of elements implied by the desired shape of the
  tensor. In the case where the total number of elements in `value` is less
  than the number of elements required by the tensor shape, the last element
  in `value` will be used to fill the remaining entries. If the total number of
  elements in `value` is greater than the number of elements required by the
  tensor shape, the initializer will raise a `ValueError`.

  Args:
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy
      array. All elements of the initialized variable will be set to the
      corresponding value in the `value` argument.
    dtype: The data type.
    verify_shape: Boolean that enables verification of the shape of `value`. If
      `True`, the initializer will throw an error if the shape of `value` is not
      compatible with the shape of the initialized tensor.

  Raises:
    TypeError: If the input `value` is not one of the expected types.

  Examples:
    The following example can be rewritten using a numpy.ndarray instead
    of the `value` list, even reshaped, as shown in the two commented lines
    below the `value` list initialization.

  ```python
    >>> import numpy as np
    >>> import tensorflow as tf

    >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
    >>> # value = np.array(value)
    >>> # value = value.reshape([2, 4])
    >>> init = tf.constant_initializer(value)

    >>> print('fitting shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[2, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    fitting shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]]

    >>> print('larger shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    larger shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 7.  7.  7.  7.]]

    >>> print('smaller shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[2, 3], initializer=init)

    ValueError: Too many elements provided. Needed at most 6, but received 8

    >>> print('shape verification:')
    >>> init_verify = tf.constant_initializer(value, verify_shape=True)
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init_verify)

    TypeError: Expected Tensor's shape: (3, 4), got (8,).
  ```
  """

    def __init__(self, value=0, dtype=float, verify_shape=False):
        if not (np.isscalar(value) or isinstance(value, (list, tuple, np.ndarray))):
            raise TypeError(
                      "Invalid type for initial value: %s (expected Python scalar, list or "
                      "tuple of values, or numpy.ndarray)." % type(value))
        self.value = value
        self.dtype = dtype
        self._verify_shape = verify_shape

    def __call__(self, shape, dtype=None, partition_info=None, verify_shape=None):
        if dtype is None:
            dtype = self.dtype
        if verify_shape is None:
            verify_shape = self._verify_shape

        if(isinstance(self.value,list)):
            return self.value
        if(isinstance(self.value,np.ndarray)):
            return self.value
        if(isinstance(self.value,tuple)):
            return self.value

        if(isinstance(shape,tuple)):
            kernel=np.ones((shape))*self.value;
            return kernel
        else:
            kernel=np.ones((shape))*self.value
            return kernel.reshape([1,shape])

    def get_config(self):
        # We don't include `verify_shape` for compatibility with Keras.
        # `verify_shape` should be passed as an argument to `__call__` rather
        # than as a constructor argument: conceptually it isn't a property
        # of the initializer.
        return {"value": self.value, "dtype": self.dtype.name}

class RandomUniform(init_ops.Initializer):

    def __init__(self, minval=0, maxval=None, seed=None, dtype=float):
        self.minval = minval
        if (maxval is None):
            self.maxval = 1
        self.seed = seed
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        #return random_ops.random_uniform(
         #   shape, self.minval, self.maxval, dtype, seed=self.seed)
        return np.random.uniform(self.minval,self.maxval,shape)



