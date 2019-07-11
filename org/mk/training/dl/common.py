#!/usr/bin/env python3

import numpy as np
from org.mk.training.dl import init_ops
import inspect


def loss(pred,labels):
    """
    One at a time.
    args:
        pred-(seq=1),input_size
        labels-(seq=1),input_size
    """
    #print(np.multiply(labels, -np.log(pred)))
    return np.multiply(labels, -np.log(pred)).sum(1)

def softmax(logits):
    """row represents num classes but they may be real numbers
    So the shape of input is important
    ([[1, 3, 5, 7],
      [1,-9, 4, 8]]
    softmax will be for each of the 2 rows
    [[2.14400878e-03 1.58422012e-02 1.17058913e-01 8.64954877e-01]
    [8.94679461e-04 4.06183847e-08 1.79701173e-02 9.81135163e-01]]
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
        score=np.squeeze(scoret[i],axis=0)
        dscore[i]=np.dot(dalignments[None,i,:,:],_softmax_grad(score))
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


def _softmax_grad(sm):
    # Below is the softmax value for [1, 3, 5, 7]
    # [2.14400878e-03 1.58422012e-02 1.17058913e-01 8.64954877e-01]
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(sm)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                #print("equal:",i, sm[i],(1-sm[i]))
                #equal: 0 0.002144008783584634 0.9978559912164153
                #equal: 1 0.015842201178506925 0.9841577988214931
                #equal: 2 0.11705891323853292 0.8829410867614671
                #equal: 3 0.8649548767993754 0.13504512320062456
                jacobian_m[i][j] = sm[i] * (1-sm[i])
            else:
                #print("not equal:",i,j,sm[i],sm[j])
                #not equal: 0 1 0.002144008783584634 0.015842201178506925
                #not equal: 0 2 0.002144008783584634 0.11705891323853292
                #not equal: 0 3 0.002144008783584634 0.8649548767993754

                #not equal: 1 0 0.015842201178506925 0.002144008783584634
                #not equal: 1 2 0.015842201178506925 0.11705891323853292
                #not equal: 1 3 0.015842201178506925 0.8649548767993754

                #not equal: 2 0 0.11705891323853292 0.002144008783584634
                #not equal: 2 1 0.11705891323853292 0.015842201178506925
                #not equal: 2 3 0.11705891323853292 0.8649548767993754

                #not equal: 3 0 0.8649548767993754 0.002144008783584634
                #not equal: 3 1 0.8649548767993754 0.015842201178506925
                #not equal: 3 2 0.8649548767993754 0.11705891323853292
                jacobian_m[i][j] = -sm[i]*sm[j]

    #finally resulting in
    #[[ 2.13941201e-03 -3.39658185e-05 -2.50975338e-04 -1.85447085e-03]
    #[-3.39658185e-05  1.55912258e-02 -1.85447085e-03 -1.37027892e-02]
    #[-2.50975338e-04 -1.85447085e-03  1.03356124e-01 -1.01250678e-01]
    #[-1.85447085e-03 -1.37027892e-02 -1.01250678e-01  1.16807938e-01]]

    return jacobian_m

def cross_entropy_loss(pred,labels):
    """
    Does an internal softmax before loss calculation.
    args:
        pred- batch,seq,input_size
        labels-batch,seq(has to be transformed before comparision with preds(line-133).)
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
    #print(num)
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

def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        clname=type(var).__name__
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            #print("names",names)
            if len(names) > 0:
                #name=str(clname+":"+names[0])
                name=str(clname+":"+names[0])
                #print("Objectname:",name)
                return name
                #return str(clname+":"+names[0]+":"+str(id(var)))
