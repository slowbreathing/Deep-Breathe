#!/usr/bin/env python3

import numpy as np
from org.mk.training.dl import init_ops


def loss(pred,labels):
    """
    One at a time.
    args:
        pred-(seg=1),input_size
        labels-(seg=1),input_size
    """
    return np.multiply(labels, -np.log(pred)).sum(1)

def softmax(logits):
    """row represents num classes but they may be real numbers
    So the shape of input is important
    ([[1,2,3],
      [1,2,5]]
    softmax will be for each of the 2 rows
    [[0.09003057 0.24472847 0.66524096]
    [0.01714783 0.04661262 0.93623955]]
    respectively
    But if the input is Tranposed clearly the answer will be wrong.

    That needs to be converted to probability
    column represents the vocabulary size.
    """
    r, c = logits.shape
    predsl = []
    for row in logits:
        inputs = np.asarray(row)
        #print("inputs:",inputs)
        predsl.append(np.exp(inputs) / float(sum(np.exp(inputs))))
    return np.array(predsl)
"""
def softmax_grad(s):
    #print()
    checkdatadim(s,2)
    decseq,size=s.shape
    #smg=np.zeros((decseq,size))
    #for i in range(decseq):
        #print("row:",s[i].T,s[i].T.shape)
        #row=np.squeeze(s[i].T,axis=0)
    print("row:",s[0].T)
        #if(row.)
    return _softmax_grad(s[0].T)
"""
def softmax_grad(scoret,dalignments):
    #print()
    checkdatadim(scoret,3)
    batch,size,seq= scoret.shape
    dscore=np.zeros_like(dalignments)
    for i in range(batch):
        checkdatadim(scoret[i],2)
            #decseq,size=s.shape
            #smg=np.zeros((decseq,size))
            #for i in range(decseq):
        #print("row:",s[i].T,s[i].T.shape)
        #row=np.squeeze(s[i].T,axis=0)
        score=np.squeeze(scoret[i],axis=0)
        #print("row:",score)
        #print("dalignments:",dalignments.shape)
        dscore[i]=np.dot(dalignments[None,i,:,:],_softmax_grad(score))
        #if(row.)
    return dscore

"""
def batch_multiply(x,y):
    xshape=x.shape
    yshape=y.shape
    lenx=len(xshape)
    leny=len(yshape)
    result=None
    if (lenx>leny):
        exparams=lenx-leny
        for i in range(exparams):
"""


def _softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    #print("jacobian_m:",jacobian_m)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                #print("equal:",i,j)
                jacobian_m[i][j] = s[i] * (1-s[i])
            else:
                #print("not equal:",i,j)
                jacobian_m[i][j] = -s[i]*s[j]
    #print("jacobian_m:",jacobian_m)
    return jacobian_m

def cross_entropy_loss(pred,labels):
    """
    Does an internal softmax before loss calculation.
    args:
        pred- batch,seq,input_size
        labels-batch,seq(has to be transformed before comparision with preds(line-43).)
    """
    checkdatadim(pred,3)
    checkdatadim(labels,2)
    batch,seq,size=pred.shape
    yhat=np.zeros((batch,seq,size))

    for batnum in range(batch):
        for seqnum in range(seq):
            yhat[batnum][seqnum]=softmax(np.reshape(pred[batnum][seqnum],[-1,size]))
    lossesa=np.zeros((batch,seq))
    for batnum in range(batch):
        for seqnum in range(seq):
            lossesa[batnum][seqnum]=loss(np.reshape(yhat[batnum][seqnum],[1,-1]),input_one_hot(labels[batnum][seqnum],size))
    return yhat,lossesa

def input_one_hot(num,vocab_size):
    print(num)
    x = np.zeros(vocab_size)
    x[int(num)] = 1
    x=np.reshape(x,[1,-1])
    #print(":",x,x.shape)
    return x;

def checkdatadim(data , degree, msg=None):
    if(len(data.shape)!=degree):
        if msg is None:
            raise ValueError('Dimension must be', degree," but is ", len(data.shape),".")
        else:
            raise ValueError(msg)

def checkarrayshape(data , degree, msg=None):
    if(data.shape!=degree):
        if(data.T.shape!=degree):
            if msg is None:
                raise ValueError('Shape must be', degree," but is ", data.shape,".")
            else:
                raise ValueError(msg)
        else:
            return data.T
    return None

def checktupledim(data , degree, msg=None):
    #print("checkTupleShape:",data,len(data))
    if(len(data)!=degree):
        if msg is None:
            raise ValueError('Length must be', degree," but is ", len(data),".")
        else:
            raise ValueError(msg)

def checklistdim(dlist,degree,msg=None):
    if(len(dlist)!=degree):
        if msg is None:
            raise ValueError('Length must be', degree," but is ", len(dlist),".")
        else:
            raise ValueError(msg)



class WeightsInitializer:
    initializer=None
    def __init__(self,initializer):

        self.winit=initializer
        WeightsInitializer.initializer=initializer
    def __enter__(self):

        return self.initializer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.winit=None
        WeightsInitializer.initializer=None

def avg_gradient_over_batch_seq(gradient,batch,seq):

    for i in range(len(gradient)):
            item=gradient[i]
            #print("len(item):",len(item))
            ds,ws=zip(*item)
            ds=[d[0]/(batch*seq) for d in zip(list(ds))]
            item=tuple(zip(ds,ws))
            gradient[i]=item
    return gradient


def make_mask(data,dtype=float):
    #just a hack for now. Needs to change
    if(dtype is float):
        return (data!=-1)

def change_internal_state_type(lstmstatetuple):

    if not isinstance(lstmstatetuple,tuple):
        lstmstatetuple=(lstmstatetuple.c,lstmstatetuple.h)
    return lstmstatetuple

def change_internal_state_types(lstmstatetuples):
    listinternalstate=[]
    for i in range(len(lstmstatetuples)):
        intstate=change_internal_state_type(lstmstatetuples[i])
        checktupledim(intstate,2)
        listinternalstate.append(intstate)
    print(len(lstmstatetuples),len(listinternalstate))
    return listinternalstate

def _item_or_tuple(dat):
    if(isinstance(dat,(list,tuple))):
        if (len(dat)==1):
            return dat[0]
        else:
            return tuple(dat)

def _item_or_lastitem(dat):
    if(isinstance(dat,(list,tuple))):
        if (len(dat)==1):
            return dat[0]
        else:
            return dat[-1]