import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.contrib.rnn.python.ops import rnn_cell

res = []

with tf.Session() as sess:
  with variable_scope.variable_scope(
      "other", initializer=init_ops.constant_initializer(0.5)) as vs:
    x = array_ops.zeros(
      [1, 3])  # Test BasicLSTMCell with input_size != num_units.
    c = array_ops.zeros([1, 2])
    h = array_ops.zeros([1, 2])
    state = (c, h)
    cell = rnn_cell.LayerNormBasicLSTMCell(2, layer_norm=False)

    g, out_m = cell(x, state)
    sess.run([variables.global_variables_initializer()])
    res = sess.run([g, out_m], {
      x.name: np.array([[1., 1., 1.]]),
      c.name: 0.1 * np.asarray([[0, 1]]),
      h.name: 0.1 * np.asarray([[2, 3]]),
    })

    print(res[1].c)
    print(res[1].h)

    expected_h = np.array([[ 0.64121795, 0.68166804]])
    expected_c = np.array([[ 0.88477188, 0.98103917]])