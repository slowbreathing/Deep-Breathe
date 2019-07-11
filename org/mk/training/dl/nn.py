from operator import iadd

import numpy as np
from org.mk.training.dl.execution import ExecutionContext
from org.mk.training.dl.common import avg_gradient_over_batch_seq




class EmbeddingLayer(object):
    def __init__(self,name=None, table=None):
        self.name=name
        self.table=table
        self.grad=None
        self.prev=None
        self.next=None

    def __repr__(self):
        #return "EmbeddingLayer("+str(self.__dict__)+")"
        return "EmbeddingLayer("+str(self.name)+")"
    def compute_gradient(self):
        return compute_gradient(self)

class TrainableVariable(object):
    def __init__(self,variable):
        self.value=variable

    def __repr__(self):
        return "TrainableVariable("+str(self.__dict__)+")"


    lookupdict={}

    @staticmethod
    def getInstance(name,var=None):
        if(name in TrainableVariable.lookupdict):
            return TrainableVariable.lookupdict[name]

        else:
            if var is not None:
                lt=TrainableVariable(var)
                TrainableVariable.lookupdict[name]=lt
                return lt
            else:
                return None

class EmbeddingTable(TrainableVariable):
    def __init__(self,word_embeddings):
        super(EmbeddingTable, self).__init__(word_embeddings)
        self.inputs=None
        self.vocab_size,self.dims=self.value.shape
        self.embedding_layer=EmbeddingLayer(name="EmbeddingLayer",table=self)
        self.embeddeding_input=[]
        self.ec=ExecutionContext.getInstance()
    def lookup(self,input_data):
        print("EmbeddingTable:")
        self.ec.register(self.embedding_layer)
        self.inputs=input_data
        return lookup(input_data,self.value,self.dims,self.embeddeding_input)

    def clean(self):
        """"""
        self.ec.clean_current();
        self.inputs=None
        self.embeddeding_input=[]


def lookup(input_data,word_embeddings,dims,inputs_list=None):
    num_sents,sent_len=input_data.shape
    embedding_input=np.zeros((num_sents,sent_len,dims))
    for seqnum1 in range(num_sents):
        cursent=input_data[seqnum1]
        for seqnum2 in range(sent_len):
            embedding_input[seqnum1][seqnum2]=word_embeddings[cursent[seqnum2]]
            if(inputs_list is not None):
                inputs_list.append(word_embeddings[cursent[seqnum2]])
    return embedding_input


def embedding_lookup(input_word_embeddings, input_data):
    embedding_input=None
    if (isinstance(input_word_embeddings,EmbeddingTable)):
        embedding_input=input_word_embeddings.lookup(input_data)
    elif (isinstance(input_word_embeddings,TrainableVariable)):
        table=EmbeddingTable(input_word_embeddings.value)
        embedding_input=table.lookup(input_data)
    else:
        vocab_size,dims=input_word_embeddings.shape
        embedding_input=lookup(input_data,input_word_embeddings,dims)
    return embedding_input

def compute_gradient(embedding_layer):
    embedded_inputs=embedding_layer.table.embeddeding_input
    xgrads=embedding_layer.prev.grad
    xgradtupdict={}

    grads_fw,bat,seq=xgrads[0]
    num_seq=len(grads_fw)
    xgradtups=[]
    """
    Per every step of X as input on the encoding side.
    (dx[0],x[0])(dx[1],x[1])......(dx[n],x[n])
    """
    for i in range(num_seq):
        tup=(grads_fw[i],embedded_inputs[i])
        xgradtups.append(tuple(tup))

    """
    This to keep the structure of grads the same as others
    [((dx[0],x[0])(dx[1],x[1])......(dx[n],x[n]))]
    """
    xgradtupstop=[]
    xgradtupstop.append(xgradtups)
    xgradtupstop=list(tuple(xgradtupstop))

    xgradtupstop=avg_gradient_over_batch_seq(xgradtupstop,bat,seq)
    xgradtupdict['fw_X']=xgradtupstop

    embedding_layer.table.clean()
    return xgradtupdict

def print_gradients(gradients):
    for layer in gradients:
        name,grad=layer
        if(name is "EmbeddingLayer"):
            """"""
            if(grad is not None):
                print("Layer-",name,":")
                xfwgrad=grad['fw_X']
                for i in range(len(xfwgrad)):
                    item=xfwgrad[i]
                    ds,ws=zip(*item)
                    print(ds)
                print("MATRIX:",TrainableVariable.getInstance("input_word_embedding").value)
