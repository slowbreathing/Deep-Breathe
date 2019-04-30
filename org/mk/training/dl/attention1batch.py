import numpy as np
from org.mk.training.dl.common import softmax
from org.mk.training.dl.common import softmax_grad
from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl.common import _item_or_tuple
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
        #return "DecoderLayer("+str(self.__dict__)+")"
        return "AttentionLayer("+str(self.name)+")"

def _prepare_memory(memory, memory_sequence_length):
    return memory

def _luong_score(query, keys, scale):
    """"""
    print("keys.shape:",keys.shape," query.shape:",query.T.shape)

    score=np.dot(keys,query.T)
    #score = np.squeeze(score, [0])
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
                 attention_layer=None):
        print("AttentionWrapper.init:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        self.seqsize=0
        self.ec=ExecutionContext.getInstance()
        if issubclass(type(cell), MultiRNNCell):
            """"""
        else:
            cell =MultiRNNCell([cell])


        al=AttentionLayer(name="AttentionLayer",bi=False,fw_cell=self,bw_cell=None,prev=None)
        self.ec.current_layer(al)
        self.ec.register(self.ec.get_current_layer())

        self._cell=cell
        self._attention_mechanism=attention_mechanism
        print("self._attention_mechanism:",self._attention_mechanism)
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
    def attention_layer_shape(self):
        return self._attention_layer.kernel_shape

    @property
    def attention_mechanism(self):
        return self._attention_mechanism

    @property
    def hidden_size(self):
        return self._attention_layer_size
    @property
    def init(self):
        return self._cell.init

    def _setinitparams(self,batch,sequence,input_size,gen_X_Ds=False):
        self._cell._setinitparams(batch, sequence, input_size+self.hidden_size,True)

    def clearStatePerSequence(self,sequence):
        self._cell.clearStatePerSequence(sequence)

    def __call__(self,X, state=None):
        print("AttentionWrapper.call:",state)
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                      "Received type %s instead."  % type(state))
        X=np.concatenate((X,state.attention.T),0)
        #print("X.shape:",X.shape)
        new_op,new_st=self._cell(X,state.cell_state)
        new_op=_item_or_tuple(new_op)
        print("LSTM.new_op:",repr(new_op))
        attention,alignments,next_atten,attenz=_compute_attention(self._attention_mechanism,new_op,state.cell_state,self._attention_layer)
        newaatn_st=AttentionWrapperState(cell_state=new_st,time=0,attention=attention,alignments=alignments,attention_state=next_atten,alignment_history="")
        self.aht[self.seqsize] = new_op
        self.attenzt[self.seqsize] =attenz
        self.attentiont[self.seqsize] =attention
        self.alignmentst[self.seqsize] =alignments
        self.seqsize+=1
        print("self.seqsize:",self.seqsize,self.attentiont)
        print("self.seqsize:",self.seqsize,self.aht)
        attenlist=[]

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
    print("alignments:",alignments.T,alignments.T.shape," attention_mechanism.values.shape:",attention_mechanism.values.shape)
    context=np.dot(alignments.T, attention_mechanism.values)
    #print(cell_output.shape,context,context.shape)
    """context=np.zeros((attention_mechanism._state_size,attention_mechanism._batch_size))
    for i in range(attention_mechanism._batch_size):
        #context[i] = np.squeeze(np.dot(alignments[i].T, attention_mechanism.values),0)
        print(alignments[i].shape,attention_mechanism.values.shape)
        context[i] = np.dot(alignments[i], attention_mechanism.values)
        print("context:",context,context.shape)
    #context=np.squeeze(context,1)"""
    context=np.squeeze(context,1)
    attenz=np.concatenate([cell_output, context], 1)
    print("attenz:",attenz)
    if attention_layer is not None:
        attention = attention_layer(attenz)
        #print("attention:",attention)
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
               name=None):

        self._query_layer = query_layer
        self._memory_layer = memory_layer
        bat,seq,size=memory.shape
        self._batch_size=bat
        self._seq_size=seq
        self._state_size=size
        self._probability_fn=probability_fn
        self._values = _prepare_memory(memory, memory_sequence_length)
        print("self._values:",self._values)
        self._keys=(self._memory_layer(self._values) if self._memory_layer  # pylint: disable=not-callable
          else self._values)

    @property
    def values(self):
        return self._values

    @property
    def keys(self):
        return self._keys

    @property
    def memory_layer(self):
        return self._memory_layer


    def __repr__(self):
        return "BaseAttentionMechanism("+str(self.__dict__)+")"


class LuongAttention(BaseAttentionMechanism):
    def __init__(self, num_units, memory, memory_sequence_length=None,scale=False,probablity_fn=None,score_mask_value=None,name="LuongAttention"):
        if probablity_fn is None:
          probablity_fn = softmax
        wrapped_probability_fn = lambda score: probablity_fn(score)
        #print("memory:",memory)
        super(LuongAttention, self).__init__(
            query_layer=None,
            memory_layer=Dense(num_units, name="memory_layer", use_bias=False,trainable=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._scale = scale
        self._name = name

    def __repr__(self):
            return "LuongAttention("+str(self.__dict__)+")"

    def __call__(self,query,state):
        """"""
        print("query:",query,query.shape)
        score=_luong_score(query,self._keys,False)
        print("score:",score,score.shape)
        alignments=np.zeros((self._batch_size,1,self._seq_size))
        for i in range(self._batch_size):
            print("score[i].T:",score[i].T)
            sm=self._probability_fn(score[i].T)
            print("self._probability_fn(score[i].T):",sm)
            alignments[i]=sm
        #print("alignments:",alignments)
        alignments=np.squeeze(alignments,0).T
        #print("alignments:",alignments)

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
    print("ycomp.shape:",repr(ycomp))
    attention_cell=attentionlayer.fw_cell
    fw_cell=attentionlayer.fw_cell._cell
    attention_mechanism=attention_cell.attention_mechanism
    out_weights=attentionlayer.prev.layer.kernel
    print("out_weights:",out_weights,out_weights.shape)
    out_biases=attentionlayer.prev.layer.bias
    print("out_biases:",repr(out_biases))
    batch,seq,size=ycomp.shape
    print("batch,seq,size:",batch,seq,size)
    dy=(ycomp.sum(1).sum(0,keepdims=True)/(batch*seq))
    print("dy:",dy)
    #fullh=np.array(list(fw_cell.ht.values()))
    #print("fullh[seqnum,:,:]:",fullh[0,:,:].shape)
    #fullattendh=np.array(list(attention_cell.aht.values()))
    #print("fullattendh[seqnum,:,:]:",fullattendh[0,:,:].shape,fullattendh.shape)
    fullattendz=np.array(list(attention_cell.attenzt.values()))
    print("fullattendz:",repr(fullattendz),fullattendz.shape)
    fullattention=np.array(list(attention_cell.attentiont.values()))
    print("attention_cell.attentiont.values():",repr(fullattention),fullattention.shape)
    fullalignments=np.array(list(attention_cell.alignmentst.values()))
    print("attention_cell.fullalignments.values():",repr(fullalignments),fullalignments.shape)
    fullah=np.array(list(attention_cell.aht.values()))
    print("fullh[seqnum,:,:]:",fullah,fullah.shape)

    fw_cell.clearDs()


    dWyfw=np.zeros((fw_cell.hidden_size,size))
    dhtf_fw=np.zeros((fw_cell.hidden_size,batch))
    dh_nextmlco = np.zeros_like(dhtf_fw)

    #dattention=np.dot(dy,out_weights.T)
    #print("dattention:",dattention)
    #dattention_layer=np.dot(np.reshape(fullattendz[:,:,:],[seq,10]).T,dattention)

    #print("dattention_layer:",dattention_layer/(batch*seq))
    dattention=np.zeros((1,5),dtype=float)
    dattention_layer=np.zeros((attention_cell.attention_layer_shape),dtype=float)
    dmemory_layer=np.zeros((5,5),dtype=float)

    print("fw_cell.seqsize:",fw_cell.seqsize)
    dattention_recur=np.zeros_like(dattention)
    for seqnum in reversed(range(fw_cell.seqsize )):
        print("seqnum::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::",seqnum)
        #pred=(attention*wy)+b
        #dWyfw=attention*Ycomp/seq
        dWyfw+=np.dot(fullattention[seqnum,:,:].T,np.reshape(ycomp[:,seqnum,:],[batch,size]))
        print(np.reshape(ycomp[:,seqnum,:],[batch,size]).shape,out_weights.T.shape)
        dattention=np.dot(np.reshape(ycomp[:,seqnum,:],[batch,size]),out_weights.T)+dattention_recur
        #dattention_recur=dattention
        print("dattention:",seqnum,":",dattention,dattention.shape)
        dattention_layer+=np.dot(fullattendz[seqnum,:,:].T,dattention)
        print("dattention_layer:::",dattention_layer/(batch*seq))
        dht=np.dot(dattention,attention_cell._attention_layer.kernel.T)
        print("dht:",dht)
        print("dht:",dht[:,0:fw_cell.hidden_size])

        print("attention_mechanism.values:",repr(attention_mechanism.values),attention_mechanism.values.shape," dht[:,fw_cell.hidden_size:].T:",repr(dht[:,fw_cell.hidden_size:].T),dht[:,fw_cell.hidden_size:].T.shape)
        dalignment=np.dot(dht[:,fw_cell.hidden_size:],attention_mechanism.values.transpose(0,2,1))
        print("dalignment:",dalignment,dalignment.shape)
        #dsoftmax=np.zeros_like(fullalignments[seqnum])
        print("fullalignments:",fullalignments,fullalignments.shape)

        print("fullalignments:",fullalignments[seqnum],fullalignments[seqnum].shape)
        dsoftmax=softmax_grad(fullalignments[seqnum].T)
        print("dsoftmax:",dsoftmax,dsoftmax.shape)
        #dscore=dsoftmax*dalignment
        dscore=np.dot(dalignment,dsoftmax)
        print("dscore:",dscore,dscore.shape)
        dquery= np.dot(dscore,attention_mechanism.keys)
        print("dquery:",dquery,dquery.shape)
        dquery=np.squeeze(dquery,axis=0)
        print("dquery:",dquery,dquery.shape)
        print("fullah[seqnum]:",fullah[seqnum],fullah[seqnum].shape)
        #dkeys=np.squeeze(np.dot(fullah[seqnum].T,dscore),axis=0)
        dkeys=np.dot(dscore.transpose(0,2,1),fullah[seqnum])
        print("dkeys:",dkeys,dkeys.shape)
        dmeme_l=np.dot(dkeys.transpose(0,2,1),attention_mechanism.values)
        print("dmeme_l:",dmeme_l,dmeme_l.shape)
        dmemory_layer+=np.squeeze(dmeme_l)
        print("dmemlayer:",dmemory_layer/(batch*seq))

        dhtf_fw =dht[:,0:fw_cell.hidden_size].T+np.squeeze(dquery,axis=0).T
        print("dhtf_fw:",dhtf_fw)
        if dh_nextmlco is None:
            dh_nextmlco = np.zeros_like(dhtf_fw)
        dh_nextmlco=fw_cell.compute_gradients(dhtf_fw,dh_nextmlco,seqnum)
        xgrads=fw_cell.get_Xgradients()
        print("fw_cell.get_Xgradients():",repr(xgrads)," fw_cell.batch_size:",fw_cell.batch_size," fw_cell.seqsize:",fw_cell.seqsize)
        print("fw_cell.get_Xgradients():",xgrads[seqnum],xgrads[seqnum].shape," fw_cell.hidden_size:",fw_cell.hidden_size," fw_cell.input_size:",fw_cell.input_size)
        #xgrada=xgrads[seqnum].reshape(1,55)
        xgrada=xgrads[seqnum::seq,:]
        print("xgrada:",xgrada[:,-fw_cell.hidden_size:])
        dattention_recur=xgrada[:,-fw_cell.hidden_size:]


    dWy=dWyfw/(batch*seq)
    print("dWy:",dWy)
    print("fw_cell.get_gradients():",avg_gradient_over_batch_seq(fw_cell.get_gradients(),batch,seq))
    """
    print()
    dWy=dWyfw/(batch*seq)
    dBy = dy
    print("dWy:",dWy)

    print("dattention_layer_final:",dattention_layer/(batch*seq))"""
    return grads
