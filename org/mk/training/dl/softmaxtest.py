#!/usr/bin/env python3

import numpy as np
from org.mk.training.dl.common import softmax
from org.mk.training.dl.common import _softmax_grad
from org.mk.training.dl.common import cross_entropy_loss
from org.mk.training.dl.common import input_one_hot
from org.mk.training.dl.common import loss

"""
Example-1 Softmax and its gradient
"""
x = np.array([[1, 3, 5, 7],
      [1,-9, 4, 8]])

y = np.array([3,1])
print("x:",x.shape)
sm=softmax(x)
print("softmax:",sm)
jacobian=_softmax_grad(sm[0])
print("jacobian:",jacobian)
jacobian=_softmax_grad(sm[1])
print(jacobian)


"""
Example-2 Softmax and loss
"""
x = np.array([[1, 3, 5, 7],
      [1,-9, 4, 8]])
y = np.array([3,1])

sm=softmax(x)

#prints out 0.145
print(loss(sm[0],input_one_hot(y[0],4)))
#prints out 17.01
print(loss(sm[1],input_one_hot(y[1],4)))


"""
Example-3 Softmax and crossentropyloss
"""
x = np.array([[[1, 3, 5, 7],
      [1,-9, 4, 8]]])
y = np.array([[3,1]])

#prints array([[ 0.14507794, 17.01904505]]))
softmaxed,loss=cross_entropy_loss(x,y)
print("loss:",loss)

"""
Example-4 Combined Gradient of Loss with respect to x
"""
batch,seq,size=x.shape
target_one_hot=np.zeros((batch,seq,size))
for batnum in range(batch):
    for i in range(seq):
        target_one_hot[batnum][i]=input_one_hot(y[batnum][i],size)
dy = softmaxed.copy()
dy = dy - target_one_hot
print("gradient:",dy)