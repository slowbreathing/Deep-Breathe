from operator import iadd

import numpy as np
from org.mk.training.dl import init_ops
from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl.common import input_one_hot
from org.mk.training.dl.common import checkdatadim
from org.mk.training.dl.common import checklistdim
from org.mk.training.dl.common import checktupledim
from org.mk.training.dl.execution import ExecutionContext
from org.mk.training.dl.common import retrieve_name
from org.mk.training.dl.nn import TrainableVariable

class FFLayer(object):
    def __init__(self,name=None,layer=None,yhat=None,target=None):
        self.name=name
        self.layer=layer
        self.yhat=yhat
        self.target=target
        self.grad=None
        self.prev=None
        self.next=None


    def __repr__(self):
        return "FFLayer("+str(self.name)+","+str(self.layer)+","+str(self.target)+")"
    def compute_gradient(self):
        return compute_gradient(self)

class Dense(object):

    def __init__(self,units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.Constant(0),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               debug=False):

        self.debug=debug
        self.use_bias=use_bias
        self.units=units;
        # First preference to static initializer through "WeightsInitializer"
        # but dont ise both to avoid confusion
        if(WeightsInitializer.initializer is not None):
            self.init_function=WeightsInitializer.initializer
        else:
            #If that's not then the usual "kernel_initializer"
            if kernel_initializer is None:
                self.init_function=init_ops.RandomUniform()
            else:
                self.init_function= kernel_initializer

        if (self.use_bias):
            if(bias_initializer is not None):
                self.bias_initializer=bias_initializer
        self.kernelname=None
        self.biasname=None

        self.use_act=False
        self.activation=None
        if(activation is not None):
            self.use_act=True
            self.activation=activation
        self.trainable=trainable
        if name is None:
            self.name="FeedForward"
        else:
            self.name=name
        self.ffl=FFLayer(name=self.name,layer=self)


    #def __repr__(self):
        #return "Dense("+str(self.name)+str(self.dense_kernel_shape)+")"

    @property
    def kernel_shape(self):
        return self.kernel.shape

    @property
    def kernel(self):
        return TrainableVariable.getInstance(self.kernelname).value

    @property
    def bias(self):
        return TrainableVariable.getInstance(self.biasname).value

    def __call__(self,inputs):
        """
        args:
            inputs-shape of the inputs should be batchsize,vocab_size
            or if 3D then batchsize,sequence,vocab_size
            result- will be of shape batchsize,num_units
            or batchsize,sequence,num_units
        """
        kernel=None
        bias=None
        if(self.trainable):
            ec=ExecutionContext.getInstance()
            ec.register(self.ffl)
        if self.kernelname is None:
            self.kernelname=str(retrieve_name(self)+":"+"kernel")
            if(self.use_bias):
                self.biasname=str(retrieve_name(self)+":"+"bias")
        kernel=TrainableVariable.getInstance(self.kernelname)
        bias=TrainableVariable.getInstance(self.biasname)
        if kernel is None:
            inputcolumnsize=self.get_input_columnshape(inputs)
            kernel=TrainableVariable.getInstance(self.kernelname,self.init_function((inputcolumnsize,self.units))).value
            if(self.use_bias):
                bias=TrainableVariable.getInstance(self.biasname,self.bias_initializer(self.units)).value
            if(self.debug):
                print("KernelName:",self.kernelname)
                print("self.init_function:",self.init_function)
                print("self.kernel:",kernel)
        else:
            kernel=kernel.value
            if self.use_bias:
                bias=bias.value
        pred=np.dot(inputs,kernel)
        if(self.use_bias):
            pred=(pred+bias)
            if(self.use_act):
                yhat=self.activation(pred)
                return yhat

        return pred

    def get_input_columnshape(self,inputs):
        if(self.debug):
            print("KernelName:",self.kernelname)
        shape=inputs.shape
        if (len(shape)in [2,3]):
            return shape[-1]
        else:
            raise ValueError("shape of input not understood.",shape)


def compute_gradient(fflayer):
    """
    args:
        yhat-Size of yhat should be batch,seq,size
        target-Size of target should be batch,seq
    """
    #Works for regular LSTM
    #Seq is a count of times the diff was done.
    #print("fflayer:",fflayer)
    yhat=fflayer.yhat
    target=fflayer.target
    checkdatadim(yhat,3)
    checkdatadim(target,2)
    batch,seq,size=yhat.shape
    target_one_hot=np.zeros((batch,seq,size))
    for batnum in range(batch):
        for i in range(seq):
            target_one_hot[batnum][i]=input_one_hot(target[batnum][i],size)
    dy = yhat.copy()
    dy = dy - target_one_hot
    #a convention to save to this field the grad that has to be passes back to the next layer in reverse
    fflayer.grad=dy
    # return whatever has to be returned to be applied
    return None

