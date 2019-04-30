
import collections
from org.mk.training.dl.common import cross_entropy_loss
import numpy as np
from org.mk.training.dl.execution import ExecutionContext
from org.mk.training.dl.rnn import MultiRNNCell

from org.mk.training.dl.rnn import LSTMStateTuple
from org.mk.training.dl.rnn import print_gradients
from org.mk.training.dl.common import avg_gradient_over_batch_seq
from org.mk.training.dl.nn import EmbeddingLayer
from org.mk.training.dl.common import checkdatadim
from org.mk.training.dl.common import checklistdim
from org.mk.training.dl.common import checktupledim
from org.mk.training.dl.common import change_internal_state_types
from org.mk.training.dl.attention import AttentionWrapper
class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
  pass

def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
    if softmax_loss_function is None:
        yhat,crossent = cross_entropy_loss(logits,targets)
    else:
        crossent = softmax_loss_function(labels=targets, logits=logits)
    if average_across_timesteps and average_across_batch:
        crossent = np.sum(crossent)
        total_size = np.sum(weights)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        crossent /= total_size
    return yhat, crossent;

class DecoderLayer(object):
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
        return "DecoderLayer("+str(self.name)+")"

class TrainingHelper(object):
    def __init__(self, inputs, sequence_length, time_major=False, name=None):
        self.inputs=inputs
        self.sequence_length=sequence_length
        self.batch,self.sequence,self.input_size=self.inputs.shape
        self.time=0

    def initialize(self, name=None):
        next_inputs=self.inputs[0:self.batch,self.time].T
        finished=False
        if(self.time == self.sequence):
            finished=True
        return (finished, next_inputs)

    def next_inputs(self, time, outputs, state, name=None):

        self.time=time+1
        finished=False
        if(self.time == self.sequence):
            finished=True
        if(not finished):
            next_inputs=self.inputs[0:self.batch,self.time].T
        else:
            next_inputs=None
        return (finished, next_inputs, state)



class BasicDecoder(object):

    def __init__(self, cell, helper, initial_state, output_layer=None):
        #to keep tract of the current cell which i am wrapping around MultiRNNCell for code generality
        #same as in RNN code
        self.ec=ExecutionContext.getInstance()
        self.attention=False
        #print("cell:",cell)
        """
        if self.ec.get_current_layer() is None:
            if issubclass(type(cell), MultiRNNCell):
                """"""
            elif(isinstance(cell, AttentionWrapper)):
                """"""
                self.attention=True
            else:
                cell =MultiRNNCell([cell])

            if(self.attention is False):
                rl=DecoderLayer(name="Decoder",bi=False,fw_cell=cell,bw_cell=None,prev=None)
                self.ec.current_layer(rl)
                self.ec.register(self.ec.get_current_layer())
        else:
            cell=self.ec.get_current_layer().fw_cell"""
        if issubclass(type(cell), MultiRNNCell):
            """"""
        elif(isinstance(cell, AttentionWrapper)):
            """"""
            self.attention=True
        else:
            cell =MultiRNNCell([cell])

        if(self.attention is False):
            rl=DecoderLayer(name="Decoder",bi=False,fw_cell=cell,bw_cell=None,prev=None)
            self.ec.current_layer(rl)
            self.ec.register(self.ec.get_current_layer())

        self._cell=cell
        self._helper=helper
        self.batch=self._helper.batch
        self.sequence=self._helper.sequence
        self.input_size=self._helper.input_size
        #print("BasicDecoder.initial_state:",initial_state)
        """
        if initial_state is not None:
            if(isinstance(initial_state,list)):
                """"""
            elif (isinstance(initial_state,tuple)):
                initial_state=[x for x in initial_state]
                print("initial_state:",initial_state,len(initial_state))
            else:
                initial_state=[initial_state]"""
            #initial_state=change_internal_state_types(initial_state)
        #checklistdim(initial_state,self._cell.feedforwarddepth)
        self._initial_state=initial_state
        self.output_layer=None

        self.output_size=cell.hidden_size
        if(output_layer is not None):
            self.output_layer=output_layer
            self.output_size=output_layer.units
        self.result={}

    def initialize(self, name=None):
        if not self._cell.init:
            if(isinstance(self.ec.get_prev_layer(),EmbeddingLayer)):
                self._cell._setinitparams(self.batch, self.sequence, self.input_size,gen_X_Ds=True)
            else:
                self._cell._setinitparams(self.batch, self.sequence, self.input_size)

        self._cell.clearStatePerSequence(self.sequence)
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, initial_state, name=None):
        #print("Decoder.step.initial_state:",initial_state,self._cell)
        if(initial_state is None):
            output,newstate = self._cell(inputs)
        else:
            output,newstate = self._cell(inputs,initial_state)
        #print("Decoder.Output:",output)
        if (self.output_layer is not None):
            output=self.output_layer(output[-1])
        else:
            output=output[-1]
        self.result[time]=output
        finished, next_input, next_state=self._helper.next_inputs(time, output, newstate)
        if self.attention is False:
            next_state=None
        return (output, next_state, next_input, finished)

    def finalize(self,outputs,final_state,sequence_length=None):
        self.ec.clean_current();
        if(self.attention):
            self._cell.finalize()


    def get_final_results(self):
        result_list=list(self.result.values());
        fw_result_array=np.array(result_list).reshape(self.sequence,self.batch*self.output_size)
        fw_result=np.zeros((self.batch,self.sequence,self.output_size))

        for item in range(self.batch):
            fw_result[item]=fw_result_array[:,item*self.output_size:item*self.output_size+self.output_size]
        #print("fw_result:",fw_result)
        sampleids=np.zeros((self.batch,self.sequence))
        for item in range(self.batch):
            sampleids[item]=fw_result[item].argmax(axis=1)
        return BasicDecoderOutput(fw_result,sampleids)


def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   scope=None):

    finished,next_inputs,initial_state=decoder.initialize()
    time=0
    #print("dynamic_decode.initialstate:",initial_state)
    next_state=initial_state
    while not finished:
        outputs, next_state, next_inputs, finished=decoder.step(time,next_inputs,next_state)
        time=time+1
        #initial_state=None

    decoder.finalize(outputs,next_state)

    return decoder.get_final_results(),next_state


def compute_gradient(declayer):
    """"""
    ycomp=declayer.prev.grad
    fw_cell=declayer.fw_cell
    out_weights=declayer.prev.layer.kernel
    out_biases=declayer.prev.layer.bias
    batch,seq,size=ycomp.shape
    dy=(ycomp.sum(1).sum(0,keepdims=True)/(batch*seq))

    fullh=np.array(list(fw_cell.ht.values()))
    fw_cell.clearDs()
    dWyfw=np.zeros((fw_cell.hidden_size,size))
    dhtf_fw=np.zeros((fw_cell.hidden_size,batch))
    dh_nextmlco = np.zeros_like(dhtf_fw)
    #print("fullh[0,:,:].shape:",fullh[0,:,:].shape)
    for seqnum in reversed(range(fw_cell.seqsize )):
        dWyfw+=np.dot(fullh[seqnum,:,:],np.reshape(ycomp[:,seqnum,:],[batch,size]))
        dhtf_fw =np.dot(ycomp[:,seqnum,:], out_weights.T).T
        if dh_nextmlco is None:
            dh_nextmlco = np.zeros_like(dhtf_fw)
        dh_nextmlco=fw_cell.compute_gradients(dhtf_fw,dh_nextmlco,seqnum)

    dWy=dWyfw/(batch*seq)
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
    ycomp=[]
    ycomp.append(((dWy,out_weights),(dBy,out_biases)))
    grads['Y']=ycomp

    fw_gradient=avg_gradient_over_batch_seq(fw_cell.get_gradients(),batch,seq)
    grads['fw_cell']=fw_gradient

    """
    Organized as a dictionary. Key being the NN type
    |----------|----------------|
    |  Y       |((dWy,y)(dBy,b))|
    |----------|----------------|
    |----------|------------------------------------------------------------------|------------------------------------------------------------------|
    |  fw_cell |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
    |----------|------------------------------------------------------------------|------------------------------------------------------------------|
    |----------|------------------------------------------------------------------|------------------------------------------------------------------|
    |  bw_cell |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
    |----------|------------------------------------------------------------------|------------------------------------------------------------------|
    """
    #last layer in reverse so nothing to save to layer.grad.
    # returned for application
    return grads