from operator import iadd

# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
import numpy as np
import collections
from org.mk.training.dl.common import input_one_hot
from org.mk.training.dl.common import checkdatadim
from org.mk.training.dl.common import checklistdim
from org.mk.training.dl.common import checktupledim
from org.mk.training.dl.common import avg_gradient_over_batch_seq

import org.mk.training.dl.core as core
from org.mk.training.dl.execution import ExecutionContext
from org.mk.training.dl.core import FFLayer
from org.mk.training.dl.nn import EmbeddingLayer
from org.mk.training.dl.nn import LookupTable
from org.mk.training.dl.common import change_internal_state_types
from org.mk.training.dl.common import _item_or_tuple
class RNNLayer(object):
    def __init__(self,name=None,bi=False,fw_cell=None,bw_cell=None,prev=None):
        self.name=name
        self.bi=bi
        self.fw_cell=fw_cell
        self.bw_cell=bw_cell
        self.prev=None
        self.next=None
        self.grad=None

    def compute_gradient(self):
        return compute_gradient(self)
        """"""
    def __repr__(self):
        #return "RNNLayer("+str(self.__dict__)+")"
        return "RNNLayer("+str(self.name)+")"

class Cell(object):
    # initialise the Recurrent Neural Network

    def __init__(self,hidden_size,debug):
        self.input_size = 0
        self.hidden_size = hidden_size
        self.debug = debug
        self.ht={}
        self.batch_size=0
        self.h=None
        self.init=False
        """"""

    def seth(self, h):
        self.h=np.copy(h)


    def __call__(self, X, state=None):
        """"""

class MultiRNNCell(Cell):
    # initialise the Recurrent Neural Network
    def __init__(self,cells,state=None):
        super().__init__(cells[0].hidden_size,cells[0].debug)
        self.feedforwardcells=cells
        self.feedforwarddepth =len(cells)
        self.seqsize = 0

    def _setinitparams(self,batch, seq, input_size,gen_X_Ds=False):
        self.seqsize=seq
        self.batch_size=batch

        for ffi in range(self.feedforwarddepth):
            cell=self.feedforwardcells[ffi]
            if not cell.init:
                if(ffi == 0):
                    cell._setinitparams(batch, seq, input_size,gen_X_Ds=gen_X_Ds)
                else:
                    cell._setinitparams(batch, seq, cell.hidden_size, Xfacing=False)
        self.init=True

    def setreverseDs(self,dh_next,dc_next):
        if(isinstance(dh_next,tuple)):
            """"""
        else:
            dh_next=(dh_next,)
        checktupledim(dh_next,self.feedforwarddepth)
        if(isinstance(dc_next,tuple)):
            """"""
        else:
            dc_next=(dc_next,)
        checktupledim(dc_next,self.feedforwarddepth)
        for ffi in range(self.feedforwarddepth):
            cell=self.feedforwardcells[ffi]
            cell.setreverseDs(dh_next[ffi],dc_next[ffi])

    def getreverseDs(self):
        dh_nexts=[]
        dc_nexts=[]
        for ffi in range(self.feedforwarddepth):
            cell=self.feedforwardcells[ffi]
            dh_nexts.append(cell.dh_next)
            dc_nexts.append(cell.dC_next)

        if(len(dh_nexts) == 1):
            dh_nexts=dh_nexts[0]
            dc_nexts=dc_nexts[0]
        else:
            dh_nexts=tuple(dh_nexts)
            dc_nexts=tuple(dc_nexts)
        return dh_nexts,dc_nexts

    def __call__(self, X, state=None):
        feedforwardstate=[]
        feedforwardoutput=[]
        if state is not None:
            if(isinstance(state,(list,tuple))):
                """"""

                """elif (isinstance(state,tuple)):
                    state=[x for x in state]
                    print("initial_state:",state,len(state))
                """
            else:
                state=[state]
            checklistdim(state,self.feedforwarddepth)

        for ffi in range(self.feedforwarddepth):
            cell=self.feedforwardcells[ffi]
            if state is not None:

                output,returnstate =cell(X,state[ffi])
            else:
                output,returnstate =cell(X)
            c,h=returnstate.c, returnstate.h
            self.ht[self.seqsize]=h.T
            X = h.T;
            feedforwardstate.append(returnstate)
            feedforwardoutput.append(output)
        self.seqsize += 1
        #return _item_or_tuple(feedforwardoutput),_item_or_tuple(feedforwardstate)
        return feedforwardoutput,feedforwardstate

    def compute_gradients(self,dhtf,dh_nextmlco,t):

        for tc in reversed(range(self.feedforwarddepth)):

            cell = self.feedforwardcells[tc]
            cell.dh_next+=dh_nextmlco
            dh_nextmlco=cell.compute_gradients(dhtf,dh_nextmlco, t)
            dhtf = np.zeros_like(dhtf)

    def get_Xgradients(self):
        return self.feedforwardcells[0].get_Xgradients();

    def get_gradients(self):
        gradients=[]
        for t in reversed(range(self.feedforwarddepth)):
            cell = self.feedforwardcells[t]
            grad=cell.get_gradients()
            gradients.append(grad);

        """
        Set of 8 pairs of ds and weights for each cell.
        1 set of gradients/cell.
        Sequence of cell gradient in list is, Y facing at position (0) and X facing at last.
                        |------------------------------------------------------------------|
        position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
                        |------------------------------------------------------------------|
                        |------------------------------------------------------------------|
        position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
                        |------------------------------------------------------------------|
        """
        return gradients;

    def clearStatePerSequence(self,seqmax):
        self.seqsize = 0
        self.feedforwardstate=[]
        for cell in self.feedforwardcells:
            cell.clearStatePerSequence(seqmax)

    def clearDs(self):
        for cell in self.feedforwardcells:
            cell.clearDs()

    def zero_state(self, batch_size, dtype=float):
        zerostate=[]
        for cell in self.feedforwardcells:
            zerostate.append(cell.zero_state(batch_size, dtype))

        return _item_or_tuple(zerostate)



class LSTMStateTuple(object):
    def __init__(self,c, h):
        self.c=c
        self.h=h
    def __repr__(self):
        return "LSTMStateTuple("+str(self.__dict__)+")"

    def clone(self):
        clonec=np.copy(self.c)
        cloneh=np.copy(self.h)
        return LSTMStateTuple(clonec,cloneh)

def zero_state_initializer(shape, batch_size):
    return np.zeros((shape, batch_size))

def dynamic_rnn(cell, X, initial_state=None):
    """
    Args:
        cell- RNN/LSTMCell/GRU
        X- input, whose shape should of dimension 3.More precisely
            X.shape must return batch, seq, input_size
        initial_state-if present should be in a of shape (hidden-size,batch_size)
                      in case of multi feed forward levels it should  be wrapped in a
                      list of same length
    """
    #static context like session to get things running and track current cells
    ec=ExecutionContext.getInstance()
    checkdatadim(X,3)
    batch, seq, input_size = X.shape

    #Wrap cell with MultiRNN because of code reuse.
    if ec.get_current_layer() is None:
        if issubclass(type(cell), MultiRNNCell):
            """"""
        else:
            cell =MultiRNNCell([cell])

        #print("ec.get_prev_layer():",isinstance(ec.get_prev_layer(),EmbeddingLayer))
        rl=RNNLayer(name="RNN",bi=False,fw_cell=cell,bw_cell=None,prev=None)
        ec.current_layer(rl)
        ec.register(ec.get_current_layer())
    else:
         cell=ec.get_current_layer().fw_cell

    """
    if initial_state is not None:
        if(isinstance(initial_state,list)):
            """"""
        else:
            initial_state=[initial_state]
        initial_state=change_internal_state_types(initial_state)
        checklistdim(initial_state,cell.feedforwarddepth)"""

    #Set all params which are available from input.
    #batch, seq, input_size. This in turn sizes the weights
    if not cell.init:
        if(isinstance(ec.get_prev_layer(),EmbeddingLayer)):
            cell._setinitparams(batch, seq, input_size,gen_X_Ds=True)
        else:
            cell._setinitparams(batch, seq, input_size)

    #Actual call to the cell
    result = {}
    cell.clearStatePerSequence(seq)
    #print("type(initial_state):",type(initial_state))
    for seqnum in range(seq):
        if(initial_state is None):
            output,newstate = cell(X[0:batch,seqnum].T)
        else:
            output,newstate = cell(X[0:batch,seqnum].T,initial_state)
            #because it is initial state
            initial_state=None
        result[seqnum] = newstate[-1].h;
    result_array=np.array(list(result.values())).reshape(seq,cell.batch_size*cell.hidden_size)
    result=np.zeros((batch,seq,cell.hidden_size))
    for item in range(batch):
        result[item]=result_array[:,item*cell.hidden_size:item*cell.hidden_size+cell.hidden_size]

    ec.clean_current();
    """
    if(len(newstate) == 1):
        newstate=newstate[0]
    else:
        newstate=tuple(newstate)
    """
    newstate=_item_or_tuple(newstate)
    return result, newstate


def bidirectional_dynamic_rnn(fw_cell,bw_cell,X,fw_initial_state=None,bw_initial_state=None):
    """
    Args:
        cell- RNN/LSTMCell/GRU
        X- input, whose shape should of dimension 3.More precisely
            X.shape must return batch, seq, input_size
        initial_state-if present should be in a of shape (hidden-size,batch_size)
                      in case of multi feed forward levels it should  be wrapped in a
                      list of same length
    """
    ec=ExecutionContext.getInstance()
    checkdatadim(X,3)
    batch, seq, input_size = X.shape
    print("batch:",batch, "seq:",seq, "input_size:",input_size)

    #forword cell sanity checks
    #Wrap cell with MultiRNN because of code reuse.
    if ec.get_current_layer() is None:
        if issubclass(type(fw_cell), MultiRNNCell):
            """"""
        else:
            fw_cell =MultiRNNCell([fw_cell])
        rl=RNNLayer(name="RNN",bi=True,fw_cell=fw_cell,bw_cell=None,prev=None)
        ec.current_layer(rl)
        ec.register(ec.get_current_layer())
    else:
         fw_cell=ec.get_current_layer().fw_cell

    """
    if fw_initial_state is not None:
        if(isinstance(fw_initial_state,list)):
            """"""
        else:
            fw_initial_state=[fw_initial_state]
        fw_initial_state=change_internal_state_types(fw_initial_state)
        checklistdim(fw_initial_state,fw_cell.feedforwarddepth)"""

    #Set all params which are available from input.
    #batch, seq, input_size. This in turn sizes the weights
    if not fw_cell.init:
        if(isinstance(ec.get_prev_layer(),EmbeddingLayer)):

            fw_cell._setinitparams(batch, seq, input_size,gen_X_Ds=True)
        else:
            fw_cell._setinitparams(batch, seq, input_size)

    #backword cell sanity checks
    #Wrap cell with MultiRNN because of code reuse.
    if rl.bw_cell is None:
        if issubclass(type(bw_cell), MultiRNNCell):
            rl.bw_cell=bw_cell
        else:
            bw_cell =MultiRNNCell([bw_cell])
            rl.bi=True
            rl.bw_cell=bw_cell
    else:
         bw_cell=rl.bw_cell
    """
    if bw_initial_state is not None:
        if(isinstance(bw_initial_state,list)):
            """"""
        else:
            bw_initial_state=[bw_initial_state]
        bw_initial_state=change_internal_state_types(bw_initial_state)
        checklistdim(bw_initial_state,bw_cell.feedforwarddepth)"""

    #Set all params which are available from input.
    #batch, seq, input_size. This in turn sizes the weights
    if not bw_cell.init:
        if(isinstance(ec.get_prev_layer(),EmbeddingLayer)):
            bw_cell._setinitparams(batch, seq, input_size,gen_X_Ds=True)
        else:
            bw_cell._setinitparams(batch, seq, input_size)

    #forward pass for forwardcell
    fw_result = {}
    fw_cell.clearStatePerSequence(seq)
    for seqnum in range(seq):
        if(fw_initial_state is None):
            output,fw_newstate = fw_cell(X[0:batch,seqnum].T)
        else:
            output,fw_newstate = fw_cell(X[0:batch,seqnum].T,fw_initial_state)
            #because it is initial state
            fw_initial_state=None
        fw_result[seqnum] = fw_newstate[-1].h;
    fw_result_array=np.array(list(fw_result.values())).reshape(seq,fw_cell.batch_size*fw_cell.hidden_size)
    fw_result=np.zeros((batch,seq,fw_cell.hidden_size))
    for item in range(batch):
        fw_result[item]=fw_result_array[:,item*fw_cell.hidden_size:item*fw_cell.hidden_size+fw_cell.hidden_size]
    print("fw_result.shape:",fw_result.shape)
    """
    if(len(fw_newstate) == 1):
        fw_newstate=fw_newstate[0]
    else:
        fw_newstate=tuple(fw_newstate)"""
    fw_newstate=_item_or_tuple(fw_newstate)
    #forward pass for backwardcell
    bw_result = {}
    bw_cell.clearStatePerSequence(seq)
    for seqnum in reversed(range(seq)):
        if(bw_initial_state is None):
            output,bw_newstate = bw_cell(X[0:batch,seqnum].T)
        else:
            output,bw_newstate = bw_cell(X[0:batch,seqnum].T,bw_initial_state)
            #because it is initial state
            bw_initial_state=None
        bw_result[seqnum] = bw_newstate[-1].h;
    bw_result_array=np.array(list(bw_result.values())).reshape(seq,bw_cell.batch_size*bw_cell.hidden_size)
    bw_result=np.zeros((batch,seq,bw_cell.hidden_size))
    for item in range(batch):
        bw_result[item]=bw_result_array[:,item*bw_cell.hidden_size:item*bw_cell.hidden_size+bw_cell.hidden_size]
    print("bw_result:",bw_result.shape)
    """
    if(len(bw_newstate) == 1):
        bw_newstate=bw_newstate[0]
    else:
        bw_newstate=tuple(bw_newstate)"""
    bw_newstate=_item_or_tuple(bw_newstate)
    result=fw_result,bw_result
    newstate=fw_newstate,bw_newstate

    ec.clean_current();
    return result, newstate

def compute_gradient(rnnlayer):
    ycomp=rnnlayer.prev.grad
    fw_cell=rnnlayer.fw_cell
    bw_cell=rnnlayer.bw_cell
    batch=0
    seq=0
    size=0

    dy=None
    dWy=None
    dBy=None
    dWyfw=None
    dWybw=None

    out_weights=None
    out_biases=None
    encoder=False
    decoder_attn=False
    dht_attention=None
    dht_attention_fw=None
    dht_attention_bw=None
    if(isinstance(rnnlayer.prev, FFLayer)):
        out_weights=rnnlayer.prev.layer.kernel
        out_biases=rnnlayer.prev.layer.bias
        batch,seq,size=ycomp.shape
        dy=(ycomp.sum(1).sum(0,keepdims=True)/(batch*seq))
        dWyfw=np.dot(fw_cell.ht[fw_cell.seqsize - 1],dy)
        dhtf_fw = np.dot(dy,out_weights[0:fw_cell.hidden_size,:].T).T

    else:
        batch=rnnlayer.prev.fw_cell.batch_size
        seq=rnnlayer.prev.fw_cell.seqsize
        encoder=True
        dhtf_fw=np.zeros((fw_cell.hidden_size,batch))
        dWy=None

        if(rnnlayer.prev.grad is not None):
            dht_attention=rnnlayer.prev.grad
            if(isinstance(dht_attention,tuple)):
                print("Encoder is BI:::::::::::;")
                """"""
                dht_attention_fw,dht_attention_bw=dht_attention
                #dht_attention_bw=dht_attention_bw.transpose(0,2,1)
                print("Encoder is BI:dht_attention_fw:",dht_attention_fw," \ndht_attention_bw:",dht_attention_bw)
            else:
                """"""
                dht_attention_fw=dht_attention
                print("Encoder is UNI:::::::::::;",dht_attention_fw.shape)
            #print("dht_attention:",dht_attention,dht_attention.shape," dhtf_fw:",dhtf_fw.shape)
            decoder_attn=True
            #dht_attention=dht_attention.sum(axis=1)#.sum(axis=2,keepdims=True)
            #print("dht_attention:",dht_attention)
            #dht_attention=dht_attention
            #dhtf_fw=dht_attention.T

    dh_nextmlco = np.zeros_like(dhtf_fw)
    fw_cell.clearDs()
    if (encoder):
        dh_nexts,dc_nexts=rnnlayer.prev.fw_cell.getreverseDs()
        print("dh_nexts,dc_nexts:",dh_nexts,dc_nexts)
        if(rnnlayer.bi):
            layers=len(dh_nexts)
            if(layers%2 ==0 ):

                dh_next_fw,dc_next_fw=[],[]
                dh_next_bw,dc_next_bw=[],[]
                for i in range(layers):
                    if(i%2==0):
                        dh_next_fw.append(dh_nexts[i])
                        dc_next_fw.append(dc_nexts[i])
                    else:
                        dh_next_bw.append(dh_nexts[i])
                        dc_next_bw.append(dc_nexts[i])

            fw_cell.setreverseDs(tuple(dh_next_fw),tuple(dc_next_fw))
        else:
            fw_cell.setreverseDs(dh_nexts,dc_nexts)

    for seqnum in reversed(range(fw_cell.seqsize )):
        if dht_attention is not None:
            """"""
            dhtf_fw=dht_attention_fw[:,seqnum,:].T
        if dh_nextmlco is None:
            dh_nextmlco = np.zeros_like(dhtf_fw)
        dh_nextmlco=fw_cell.compute_gradients(dhtf_fw,dh_nextmlco,seqnum)
        dhtf_fw=np.zeros_like(dhtf_fw)

    #For backward cell
    if not bw_cell==None:
        bw_cell.clearDs()
        if (encoder):
            bw_cell.setreverseDs(tuple(dh_next_bw),tuple(dc_next_bw))
            if(dht_attention_bw is None):
                dhtf_bw=np.zeros((bw_cell.hidden_size,batch))
            #else:
                #dhtf_bw=dht_attention_bw
            dWy=None
        else:
            dhtf_bw = np.dot(dy, out_weights[bw_cell.hidden_size:bw_cell.hidden_size*2,:].T).T
            dWybw=np.dot(bw_cell.ht[0],dy)
            dWy=np.concatenate((dWyfw,dWybw),0)
        """else:
            dhtf_bw=np.zeros((bw_cell.hidden_size,batch))
            dWy=None"""

        #dh_nextmlco = np.zeros_like(dhtf_bw)
        print("bw_cell.seqsize:",bw_cell.seqsize)
        dh_nextmlco=np.zeros((bw_cell.hidden_size,batch))
        if(encoder):
            for seqnum in reversed(range(bw_cell.seqsize )):
                if dht_attention_bw is not None:
                    """"""
                    dhtf_bw=dht_attention_bw[:,((bw_cell.seqsize-1)-seqnum),:].T
                    #dhtf_bw=dht_attention_bw[:,seqnum,:].T
                    print("dhtf_bw:",seqnum,":",dhtf_bw)
                if dh_nextmlco is None:
                    dh_nextmlco = np.zeros_like(dhtf_bw)
                dh_nextmlco=bw_cell.compute_gradients(dhtf_bw,dh_nextmlco,seqnum)
                dhtf_bw=np.zeros_like(dhtf_bw)
        else:
            if dh_nextmlco is None:
                dh_nextmlco = np.zeros_like(dhtf_bw)
            dh_nextmlco=bw_cell.compute_gradients(dhtf_bw,dh_nextmlco,0)
            dhtf_bw=np.zeros_like(dhtf_bw)
        #dWy=np.concatenate((dWyfw,dWybw),0)
    else:
        dWy=dWyfw
    dBy = dy

    #gradients for a cell in a dict
    grads={}

    #Ycomponents
    """
    Ds for Y
    |----------------|
    |((dWy,y)(dBy,b))|
    |----------------|
    """
    if (dWy is not None):
        ycomp=[]
        ycomp.append(((dWy,out_weights),(dBy,out_biases)))
        grads['Y']=ycomp

    #"get_gradients()" simply gets the gradients calculated by compute_gradient called earlier
    #fw_cell(s)
    if(encoder):
        fw_gradient=avg_gradient_over_batch_seq(fw_cell.get_gradients(),batch,seq)
    else:
        fw_gradient=fw_cell.get_gradients()
    grads['fw_cell']=fw_gradient

    #bw_cell(s)
    if bw_cell is not None:
        if(encoder):
            bw_gradient=avg_gradient_over_batch_seq(bw_cell.get_gradients(),batch,seq)
        else:
            bw_gradient=bw_cell.get_gradients()
        grads['bw_cell']=bw_gradient
        #print("bw_gradient:",bw_gradient)

    # X gradient for Embedding Layer to process.
    if(encoder):
        xgrads_fw=fw_cell.get_Xgradients()
        print("xgrads_fw:",xgrads_fw.shape)
        rnnlayer.grad=((xgrads_fw,batch,seq),)
        if bw_cell is not None:
            xgrads=np.zeros_like(xgrads_fw)
            xgrads_bw=bw_cell.get_Xgradients()
            for b in range(batch):
                si=b*bw_cell.seqsize
                ei=(b+1)*bw_cell.seqsize
                xgseq=xgrads_bw[si:ei,]
                xgrads[si:ei,]=xgrads_fw[si:ei,]+xgseq[::-1,:]
            rnnlayer.grad=((xgrads,batch,seq),)

    """
    Organized as a dictionary. Key being the NN type. The the position shown is list index.
    |----------|----------------|
    |  Y       |((dWy,y)(dBy,b))|
    |----------|----------------|
    |-------------------|--------------------|------------------------------------------------------------------|
    |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
    |     fw_cell       |--------------------|------------------------------------------------------------------|
    |                   |--------------------|------------------------------------------------------------------|
    |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
    |-------------------|--------------------|------------------------------------------------------------------|
    |-------------------|--------------------|------------------------------------------------------------------|
    |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
    |     bw_cell       |--------------------|------------------------------------------------------------------|
    |                   |--------------------|------------------------------------------------------------------|
    |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
    |-------------------|--------------------|------------------------------------------------------------------|

    """

    #last layer in reverse so nothing to save to layer.grad.
    # returned for application
    return grads


def print_gradients(gradients):
    print("type(gradients):",type(gradients))
    for layer in gradients:

        name,grad=layer
        print("type(layer):",type(layer)," type(grad):",type(grad))
        if(grad is not None):
            print("Layer-",name,":")
            if('Y' in grad):
                print_grad("Ycomp",grad['Y'])

            fcellgrad=grad['fw_cell']
            print_RNNCellgradients(fcellgrad,"fw_cell-")


            bcellgrad=grad.get('bw_cell')
            if(bcellgrad is not None):
                print_RNNCellgradients(bcellgrad,"bw_cell-")

def print_grad(name,ygrad):
    print(name,":",ygrad)
def print_RNNCellgradients(rgrads,celltype):
    for i in reversed(range(len(rgrads))):
        item=rgrads[i]
        ds,ws=zip(*item)
        print(celltype,i,":",np.concatenate((ds[0], ds[1], ds[2], ds[3]), axis=0).T)
        print(np.concatenate((ws[0], ws[1], ws[2], ws[3]), axis=0).T)

        print(np.concatenate((ds[4], ds[5], ds[6], ds[7]), axis=0).T)
        print(np.concatenate((ws[4], ws[5], ws[6], ws[7]), axis=0).T)