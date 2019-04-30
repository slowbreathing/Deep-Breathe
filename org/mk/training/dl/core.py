from operator import iadd

import numpy as np
from org.mk.training.dl import init_ops
from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl.common import input_one_hot
from org.mk.training.dl.common import checkdatadim
from org.mk.training.dl.common import checklistdim
from org.mk.training.dl.common import checktupledim
from org.mk.training.dl.execution import ExecutionContext

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
        #return "FFLayer("+str(self.__dict__)+")"
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
               name=None):
        self.use_bias=use_bias
        self.units=units;
        if kernel_initializer is None:
            self.kernel_initializer=init_ops.RandomUniform()
        else:
            self.kernel_initializer= kernel_initializer

        if(WeightsInitializer.initializer is None):
            #WeightsInitializer.initializer=init_ops.RandomUniform()
            self.init_function=None
        else:
            self.init_function=WeightsInitializer.initializer

        self.kernel=None
        self.bias=None
        if(self.use_bias):
            self.bias=bias_initializer(units);

        self.use_act=False
        self.activation=None
        if(activation is not None):
            self.use_act=True
            self.activation=activation
        self.pred=None
        self.yhat=None
        self.trainable=trainable
        if name is None:
            self.name="FeedForward"
        else:
            self.name=name
        self.ffl=FFLayer(name=self.name,layer=self)

    def __repr__(self):
        #return "FFLayer("+str(self.__dict__)+")"
        return "Dense("+str(self.name)+str(self.dense_kernel_shape)+")"

    @property
    def dense_kernel_shape(self):
        return self.kernel.shape

    @property
    def dense_kernel(self):
        return self.kernel

    def __call__(self,inputs):
        """
        args:
            inputs-shape of the inputs should be batchsize,vocab_size
            or if 3D then batchsize,sequence,vocab_size
            result- will be of shape batchsize,num_units
            or batchsize,sequence,num_units
        """
        if(self.trainable):
            ec=ExecutionContext.getInstance()
            ec.register(self.ffl)


        # initialize the kernel,first priority to WeightsInitializer
        if self.kernel is None:
            print("inputs:::::::::::::::::::::::",inputs.shape," Kernel Shape:",self.name,":",self.get_input_columnshape(inputs),self.units)
            #r,c=inputs.shape
            inputcolumnsize=self.get_input_columnshape(inputs)
            if(self.init_function is not None):
                #self.kernel=WeightsInitializer.initializer((c,self.units))
                self.kernel=self.init_function((inputcolumnsize,self.units))
                #print("self.kernel:",self.kernel)
            else:
                self.kernel=self.kernel_initializer((inputcolumnsize,self.units));
        if(self.use_bias):
            if(self.use_act):
                self.pred=np.dot(inputs,self.kernel)+self.bias
                self.yhat=self.activation(self.pred)
                return self.yhat
            else:
                self.pred=np.dot(inputs,self.kernel)+self.bias
                return self.pred
        else:
            self.pred=np.dot(inputs,self.kernel)
            return self.pred

    def get_input_columnshape(self,inputs):
        shape=inputs.shape
        if (len(shape)in [2,3]):

            return shape[-1]

        else:
            raise ValueError("shape of input not understood.",shape)

    """
    def __call__(self,inputs):
        ec=ExecutionContext.getInstance()
        ec.register(self.ffl)


        # initialize the kernel,first priority to WeightsInitializer
        if self.kernel is None:
            r,c=inputs.shape
            print("inputs:::::::::::::::::::::::",inputs.shape)
            if(self.init_function is not None):
                #self.kernel=WeightsInitializer.initializer((c,self.units))
                self.kernel=self.init_function((c,self.units))
                #print("self.kernel:",self.kernel)
            else:
                self.kernel=self.kernel_initializer((c,self.units));
        if(self.use_bias):
            if(self.use_act):
                self.pred=self._multiply(inputs)+self.bias
                self.yhat=self.activation(self.pred)
                return self.yhat
            else:
                self.pred=self._multiply(inputs)+self.bias
                return self.pred
        else:
            self.pred=self._multiply(inputs)
            return self.pred


    def _multiply(self,inputs):

        shape=inputs.shape
        if (len(shape)==2):
            return np.dot(inputs,self.kernel)

        elif(len(shape)==3):
            bat,seq,vc=inputs.shape
            print("direct:",np.dot(inputs,self.kernel))
            result=np.zeros((bat,seq,self.units))
            for i in range(bat):
                result[i]=np.dot(inputs[i],self.kernel)
            return result
        else:
            raise ValueError("shape of input not understood.")
    """

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

