#!/usr/bin/env python3
from org.mk.training.dl.rnn_cell import LSTMCell
from org.mk.training.dl.rnn import LSTMStateTuple

from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl import init_ops
import numpy as np


n_hidden=2

with WeightsInitializer(initializer=init_ops.Constant(0.5)) as vs:
    cell = LSTMCell(n_hidden,debug=True)

c=0.1 * np.asarray([[0],
                    [1]])
h=0.1 * np.asarray([[2],
                    [ 3]])

x=np.array([[1],
            [1],
            [1]])

print(cell(x,(c,h)))


expected_h = np.array([[ 0.64121795, 0.68166804]])
expected_c = np.array([[ 0.88477188, 0.98103917]])

