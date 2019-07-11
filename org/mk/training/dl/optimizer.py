#!/usr/bin/env python3


import numpy as np
from org.mk.training.dl.execution import ExecutionContext

class BatchGradientDescent(object):
    def __init__(self,learning_rate):
      self.learning_rate=learning_rate

    def compute_gradients(self,yhat, target):

        ec=ExecutionContext.getInstance()
        l=ec.get_layer(-1)
        l.yhat=yhat

        l.target=target
        grad=ec.compute_gradients()
        ec.clean()
        return grad


    def apply_gradients(self,gradients):
        """
        list of tuple of name, gradient
        where the gradient is a dict of cell name and its gradients as shown
        [(layer,grad)]
        ||------------|----------|----------------|
        ||            |  Y       |((dWy,y)(dBy,b))|
        ||            |----------|----------------|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||   Decoder  |     fw_cell       |--------------------|------------------------------------------------------------------|
        ||            |                   |--------------------|------------------------------------------------------------------|
        ||            |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |     bw_cell       |--------------------|------------------------------------------------------------------|
        ||            |                   |--------------------|------------------------------------------------------------------|
        ||            |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||------------|-------------------|--------------------|------------------------------------------------------------------|
        ||------------|-------------------|--------------------|------------------------------------------------------------------|
        ||            |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |     fw_cell       |--------------------|------------------------------------------------------------------|
        ||  Encoder   |                   |--------------------|------------------------------------------------------------------|
        ||            |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |     bw_cell       |--------------------|------------------------------------------------------------------|
        ||            |                   |--------------------|------------------------------------------------------------------|
        ||            |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||------------|-------------------|--------------------|------------------------------------------------------------------|
        """
        for layer in gradients:
            name,grad=layer
            if(grad is not None):
                for key in grad.keys():
                    val=grad[key]
                    for i in range(len(val)):
                        item=val[i]

                        ds,ws=zip(*item)
                        self._apply(ds,ws)
    def _apply(self,ds,ws):
        for param, dparam in zip(list(ws), list(ds)):
            param += -self.learning_rate * dparam
