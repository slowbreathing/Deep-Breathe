import numpy as np
from org.mk.training.dl.common import softmax
from org.mk.training.dl.common import softmax_grad
from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl.common import _item_or_tuple
from org.mk.training.dl.common import _item_or_lastitem
from org.mk.training.dl.common import avg_gradient_over_batch_seq
from org.mk.training.dl.core import Dense
from org.mk.training.dl import init_ops
from org.mk.training.dl.rnn_cell import LSTMCell
from org.mk.training.dl.rnn_cell import LSTMStateTuple
from org.mk.training.dl.rnn import MultiRNNCell
from org.mk.training.dl.execution import ExecutionContext
import collections


class AttentionLayer(object):
    def __init__(self,name=None,bi=False,fw_cell=None,bw_cell=None,prev=None,batch=0,sequence=0):
        self.name=name
        self.bi=bi
        self.fw_cell=fw_cell
        self.bw_cell=bw_cell
        self.prev=prev
        self.grad=None
        self.batch=batch
        self.sequence=sequence

    def compute_gradient(self):
        return compute_gradient(self)
        """"""
    def __repr__(self):
        return "AttentionLayer("+str(self.name)+")"

def _prepare_memory(memory, memory_sequence_length):
    return memory

def _luong_score(query, keys, scale):
    """"""
    bat,seq,size=keys.shape
    score=np.zeros((bat,seq,1))

    for i in range(bat):
        x=np.dot(keys[i,:,:],query[None ,i,:].T)
        score[i]=x
    return score

class AttentionWrapper(object):
    """"""
    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None,
                 attention_layer=None,
                 debug=False):
        self.seqsize=0
        self.ec=ExecutionContext.getInstance()
        self.debug=debug
        if issubclass(type(cell), MultiRNNCell):
            """"""
        else:
            cell =MultiRNNCell([cell])


        al=AttentionLayer(name="AttentionLayer",bi=False,fw_cell=self,bw_cell=None,prev=None)
        self.ec.current_layer(al)
        self.ec.register(self.ec.get_current_layer())

        self._cell=cell
        self._attention_mechanism=attention_mechanism
        self._output_attention=output_attention

        if attention_layer_size is not None and attention_layer is not None:
            raise ValueError("Only one of attention_layer_size and attention_layer "
                       "should be set")

        if(attention_layer_size is not None):
            self._attention_layer=Dense(attention_layer_size,name="attention_layer",use_bias=False,trainable=False)
            self._attention_layer_size=attention_layer_size

        #state for Ds
        self.aht,self.attenzt,self.attentiont,self.alignmentst={},{},{},{}

    def zero_state(self,batch_size,dtype=float):
        cellstate=self._cell.zero_state(batch_size)
        attention=np.zeros((batch_size,self._attention_layer_size))
        attentionstate=np.zeros((batch_size,self._attention_mechanism._seq_size))
        alignments=attentionstate
        return AttentionWrapperState(cell_state=cellstate,time=0,attention=attention,alignments=alignments,attention_state=attentionstate,alignment_history="")

    @property
    def attention_layer_dense_shape(self):
        return self._attention_layer.kernel_shape

    @property
    def attention_layer_dense(self):
        return self._attention_layer

    @property
    def attention_mechanism(self):
        return self._attention_mechanism

    @property
    def hidden_size(self):
        return self._attention_layer_size

    @property
    def init(self):
        return self._cell.init

    @property
    def batch_size(self):
        return self._attention_mechanism._batch_size

    def _setinitparams(self,batch,sequence,input_size,gen_X_Ds=False):
        self._cell._setinitparams(batch, sequence, input_size+self.hidden_size,True)

    def clearStatePerSequence(self,sequence):
        self._cell.clearStatePerSequence(sequence)

    def getreverseDs(self):
        return self._cell.getreverseDs()

    def __call__(self,X, state=None):
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                      "Received type %s instead."  % type(state))
        X=np.concatenate((X,state.attention.T),0)
        new_op,new_st=self._cell(X,state.cell_state)
        if (self.debug):
            print("new_op:",new_op, " new_st:",new_st)
        new_op=_item_or_lastitem(new_op)
        attention,alignments,next_atten,attenz=_compute_attention(self._attention_mechanism,new_op,state.cell_state,self._attention_layer)
        newaatn_st=AttentionWrapperState(cell_state=new_st,time=0,attention=attention,alignments=alignments,attention_state=next_atten,alignment_history="")
        self.aht[self.seqsize] = new_op
        self.attenzt[self.seqsize] =attenz
        self.attentiont[self.seqsize] =attention
        self.alignmentst[self.seqsize] =alignments
        self.seqsize+=1
        attenlist=[]
        if(self.debug):
                print("New State:",newaatn_st)
        if self._output_attention:
            attenlist.append(attention)
            return attenlist, newaatn_st
        else:
            return new_op, newaatn_st

    def finalize(self):
         """"""
         self.ec.clean_current();

def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
      cell_output, state=attention_state)


    context=np.zeros((attention_mechanism._batch_size,1,attention_mechanism._state_size))
    for i in range(attention_mechanism._batch_size):
        context[i]=np.dot(alignments[i],attention_mechanism.values[i])
        if(attention_mechanism.debug):
            print("ContextVector:",context[i])
    context=np.squeeze(context,1)
    attenz=np.concatenate([cell_output, context], 1)
    if attention_layer is not None:
        attention = attention_layer(attenz)
        if(attention_mechanism.debug):
            print("AttentionNew:",attention)
    else:
        attention = context
    return attention,alignments,next_attention_state,attenz


class BaseAttentionMechanism(object):
    def __init__(self,
               query_layer,
               memory,
               probability_fn,
               memory_sequence_length=None,
               memory_layer=None,
               check_inner_dims_defined=True,
               score_mask_value=None,
               name=None,
               debug=False):
        self.debug=debug
        self._query_layer = query_layer
        self._memory_layer = memory_layer
        #print("memory:",memory)
        self.encoderisbi=False
        if (isinstance(memory,tuple)):
            memory=np.concatenate((memory[0],memory[1]),axis=-1)
            self.encoderisbi=True
        bat,seq,size=memory.shape
        #print("memory.shape:",memory.shape)
        self._batch_size=bat
        self._seq_size=seq
        self._state_size=size
        self._probability_fn=probability_fn
        self._values = _prepare_memory(memory, memory_sequence_length)

        self._keys=(self._memory_layer(self._values) if self._memory_layer  # pylint: disable=not-callable
          else self._values)

    @property
    def values(self):
        return self._values

    @property
    def keys(self):
        return self._keys

    @property
    def memory_layer_dense(self):
        return self._memory_layer

    @property
    def memory_layer_dense_shape(self):
        return self._memory_layer.kernel_shape


    def __repr__(self):
        return "BaseAttentionMechanism("+str(self.__dict__)+")"


class LuongAttention(BaseAttentionMechanism):
    def __init__(self, num_units, memory, memory_sequence_length=None,scale=False,probablity_fn=None,score_mask_value=None,name="LuongAttention"):
        if probablity_fn is None:
          probablity_fn = softmax
        wrapped_probability_fn = lambda score: probablity_fn(score)
        super(LuongAttention, self).__init__(
            query_layer=None,
            memory_layer=Dense(num_units, name="memory_layer", use_bias=False,trainable=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)

        if(self.debug):
            print(self)
        self._num_units = num_units
        self._scale = scale
        self._name = name

    def __repr__(self):
            #return "LuongAttention("+str(self.__dict__)+")"
            return "LuongAttention("+str("_values:")+str(self._values)+str("\n_memory_layer:")+str(self._memory_layer)+str("\n_keys:")+str(self._keys)+")"

    def __call__(self,query,state):
        """"""
        score=_luong_score(query,self._keys,False)
        if(self.debug):
            print("Score:",score)
        alignments=np.zeros((self._batch_size,1,self._seq_size))
        for i in range(self._batch_size):
            sm=self._probability_fn(score[i].T)
            if(self.debug):
                print("ScoreSoftmaxed:",sm)
            alignments[i]=sm
        next_state=alignments
        return alignments,next_state


class AttentionWrapperState(
    collections.namedtuple("AttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history", "attention_state"))):


    def clone(self, **kwargs):
        aws=self
        aws=aws._replace(**kwargs)
        #print(aws)
        return aws


def compute_gradient(attentionlayer):

    """"""
    ycomp=attentionlayer.prev.grad
    #print("ycomp.shape:",repr(ycomp))
    attention_cell=attentionlayer.fw_cell
    fw_cell=attentionlayer.fw_cell._cell
    attention_mechanism=attention_cell.attention_mechanism
    out_weights=attentionlayer.prev.layer.kernel
    #print("out_weights:",out_weights,out_weights.shape)
    out_biases=attentionlayer.prev.layer.bias
    #print("out_biases:",repr(out_biases))
    batch,seq,size=ycomp.shape
    #print("batch,seq,size:",batch,seq,size)
    dy=(ycomp.sum(1).sum(0,keepdims=True)/(batch*seq))

    #fullh=np.array(list(fw_cell.ht.values()))
    #print("fullh[seqnum,:,:]:",fullh[0,:,:].shape)
    #fullattendh=np.array(list(attention_cell.aht.values()))
    #print("fullattendh[seqnum,:,:]:",fullattendh[0,:,:].shape,fullattendh.shape)
    fullattendz=np.array(list(attention_cell.attenzt.values()))
    #print("fullattendz:",fullattendz.shape)
    fullattention=np.array(list(attention_cell.attentiont.values()))
    #print("attention_cell.attentiont.values():",fullattention.shape)
    fullalignments=np.array(list(attention_cell.alignmentst.values()))
    #print("attention_cell.fullalignments.values():",fullalignments.shape)
    fullah=np.array(list(attention_cell.aht.values()))
    #print("fullh[seqnum,:,:]:",fullah.shape)

    fw_cell.clearDs()


    dWyfw=np.zeros((fw_cell.hidden_size,size))
    dhtf_fw=np.zeros((fw_cell.hidden_size,batch))
    dh_nextmlco = np.zeros_like(dhtf_fw)

    #dattention=np.dot(dy,out_weights.T)
    #print("dattention:",dattention)
    #dattention_layer=np.dot(np.reshape(fullattendz[:,:,:],[seq,10]).T,dattention)

    #print("dattention_layer:",dattention_layer/(batch*seq))
    dattention=np.zeros((1,5),dtype=float)
    dattention_layer=np.zeros((attention_cell.attention_layer_dense_shape),dtype=float)
    dmemory_layer=np.zeros((attention_mechanism.memory_layer_dense_shape),dtype=float)

    damvalues=np.zeros_like(attention_mechanism.values)

    #print("fw_cell.seqsize:",fw_cell.seqsize)
    dattention_recur=np.zeros_like(dattention)
    for seqnum in reversed(range(fw_cell.seqsize )):
        dWyfw+=np.dot(fullattention[seqnum,:,:].T,np.reshape(ycomp[:,seqnum,:],[batch,size]))
        dattention=np.dot(np.reshape(ycomp[:,seqnum,:],[batch,size]),out_weights.T)+dattention_recur
        dattention_layer+=np.dot(fullattendz[seqnum,:,:].T,dattention)

        dattenz=np.dot(dattention,attention_cell._attention_layer.kernel.T)

        dcontext=dattenz[:,fw_cell.hidden_size:]
        dht=dattenz[:,0:fw_cell.hidden_size]
        #print("dattenz",dattenz,dattenz.shape,dcontext.shape, dht.shape,)

        dalignment=np.zeros_like(fullalignments[0])
        for i in range(fw_cell.batch_size):
            dalignment[i]=np.dot(dcontext[None,i,:],attention_mechanism.values.transpose(0,2,1)[i])

        #print("fullalignments:",fullalignments.shape)
        for i in range(fw_cell.batch_size):
            #print("fullalignments[seqnum][None,i,:,:]seqnum:",seqnum," i:",i,":",fullalignments[seqnum][None,i,:,:]," dcontext[None,i,:]:",dcontext[None,i,:].shape)
            damvalues[None,i]+=np.dot(fullalignments[seqnum][None,i,:,:].transpose(0,2,1),dcontext[None,i,:])
        #print("damvalues0:",damvalues,damvalues.shape)

        dscore=softmax_grad(fullalignments[seqnum],dalignment)
        dquery=np.zeros((fw_cell.batch_size,1,attention_cell.hidden_size))
        for i in range(fw_cell.batch_size):
            dquery[i]=np.dot(dscore[i],attention_mechanism.keys[i])

        dkeys=np.zeros_like(attention_mechanism.keys)
        #print("dkeys:",dkeys.shape)
        for i in range (fw_cell.batch_size):
            dkeys[i]=np.dot(dscore.transpose(0,2,1)[None,i,:,:],fullah[seqnum][None,i,:])

        dmeme_l=np.zeros((fw_cell.batch_size,attention_mechanism.memory_layer_dense_shape[0],attention_mechanism.memory_layer_dense_shape[1]))
        for i in range (fw_cell.batch_size):
            dmeme_l[i]=np.dot(dkeys.transpose(0,2,1)[i],attention_mechanism.values[i]).T
        dmemory_layer+=dmeme_l.sum(0)

        for i in range(fw_cell.batch_size):
            #print("dkeys[i]:",dkeys[i].shape)
            damvaluestemp=np.dot(dkeys[i],attention_mechanism.memory_layer_dense.kernel.T)
            #damvaluestemp=np.dot(dkeys[i],attention_mechanism.memory_layer_dense.dense_kernel)
            damvalues[None,i]+=damvaluestemp
        #print("damvalues1:",damvalues,damvalues.shape)#," attention_mechanism.memory_layer_dense.dense_kernel:",attention_mechanism.memory_layer_dense.dense_kernel)

        dhtf_fw =dht.T+np.squeeze(dquery,axis=1).T
        if dh_nextmlco is None:
            dh_nextmlco = np.zeros_like(dhtf_fw)
        dh_nextmlco=fw_cell.compute_gradients(dhtf_fw,dh_nextmlco,seqnum)
        xgrads=fw_cell.get_Xgradients()

        xgrada=xgrads[seqnum::seq,:]
        dattention_recur=xgrada[:,-fw_cell.hidden_size:]

    print("damvalues:",damvalues,damvalues.shape)
    #print("damvalueslist:",damvalueslist," listlen:",len(damvalues))
    """print("dmemlayer:",dmemory_layer/(batch*seq))
    print("fw_cell.get_gradients():",avg_gradient_over_batch_seq(fw_cell.get_gradients(),batch,seq))
    print("dattention_layer:::",dattention_layer/(batch*seq))
    dWy=dWyfw/(batch*seq)
    dBy=dy
    print("dWy:",dWy)
    print("dy:",dy)"""
    dWy=dWyfw/(batch*seq)
    dBy=dy
    #gradients for a cell in a dict
    grads={}

    #Ycomponents
    """
    Ds for Y
    |----------------|
    |((dWy,y)(dBy,b))|
    |----------------|
    """
    ycomp=[]
    ycomp.append(((dWy,out_weights),(dBy,out_biases)))
    grads['Y']=ycomp

    fw_gradient=avg_gradient_over_batch_seq(fw_cell.get_gradients(),batch,seq)
    grads['fw_cell']=fw_gradient

    memory_grad=[]
    avg_memory_grad=dmemory_layer/(batch*seq)
    memory_grad.append(((avg_memory_grad,attention_mechanism.memory_layer_dense.kernel),))
    grads['memory_grad']=memory_grad

    attention_grad=[]
    avg_attention_grad=dattention_layer/(batch*seq)
    attention_grad.append(((avg_attention_grad,attention_cell.attention_layer_dense.kernel),))
    #print("attention_grad:",attention_grad)
    grads['attention_grad']=attention_grad

    #print("attention_mechanism.encoderisbi:",attention_mechanism.encoderisbi)
    if (attention_mechanism.encoderisbi):
        """"""
        damvaluestup=(damvalues[:,:,0:attention_cell.hidden_size],damvalues[:,:,attention_cell.hidden_size:])
        #print("damvaluestup:",damvalues.shape,damvalues[:,:,0:attention_cell.hidden_size].shape,damvalues[:,:,attention_cell.hidden_size:].shape)
        #print("attention_cell.hidden_size",attention_cell.hidden_size," damvalues[:,:,0:attention_cell.hidden_size]:",damvalues[:,:,0:attention_cell.hidden_size].shape)
        attentionlayer.grad=damvaluestup
    else:
        attentionlayer.grad=damvalues#(batch*seq)


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

    return grads
