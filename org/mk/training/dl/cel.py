#!/usr/bin/env python3

import numpy as np
from org.mk.training.dl.common import softmax
from org.mk.training.dl.common import _softmax_grad
from org.mk.training.dl.common import cross_entropy_loss

x = np.array([[[1, 3, 5, 7],
      [1,-9, 4, 8]]])
y=np.array([[3,1]])

print(cross_entropy_loss(x,y))
