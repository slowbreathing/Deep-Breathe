#!/usr/bin/env python3

import numpy as np
import sys
from org.mk.training.dl.common import softmax
from org.mk.training.dl.common import _softmax_grad

x = np.array([[1, 3, 5, 7],
      [1,-9, 4, 8]])
print("x:",x)
sm=softmax(x)
print("softmax:",sm)
jacobian=_softmax_grad(sm[0])
print(jacobian)
print(jacobian.diagonal())
jacobian=_softmax_grad(sm[1])
print(jacobian)