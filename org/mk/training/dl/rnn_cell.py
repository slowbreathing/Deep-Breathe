from operator import iadd

# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
import numpy as np
import collections
"""from rnn import zero_state_initializer
from rnn import LSTMStateTuple
from rnn import Cell"""
from org.mk.training.dl.rnn import zero_state_initializer
from org.mk.training.dl.rnn import LSTMStateTuple
from org.mk.training.dl.rnn import Cell
from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl import init_ops
from org.mk.training.dl.common import checkdatadim
from org.mk.training.dl.common import checktupledim
from org.mk.training.dl.common import checkarrayshape
from org.mk.training.dl.common import change_internal_state_type
from org.mk.training.dl.common import retrieve_name
from org.mk.training.dl.nn import TrainableVariable


class LSTMCell(Cell):
    # initialise the Recurrent Neural Network
    def __init__(self, hidden_size, forget_bias=1, debug=False,backpassdebug=False):
        super().__init__(hidden_size,debug,backpassdebug)
        self.forget_bias = forget_bias
        self.seqsize=0
        # first column is for biases
        self.init_function=None
        if(WeightsInitializer.initializer is None):
            #WeightsInitializer.initializer=init_ops.RandomUniform()
            self.init_function=init_ops.RandomUniform()
        else:
            self.init_function=WeightsInitializer.initializer


        self.shape=None

        #self.WLSTM=None
        self.lstmname=None
        self.wf = None
        self.wi = None
        self.wc = None
        self.wo = None

        self.bf = None
        self.bi = None
        self.bc = None
        self.bo = None

        self.dwi = None
        self.dwc = None
        self.dwo = None
        self.dwf = None

        self.dbi = None
        self.dbc = None
        self.dbo = None
        self.dbf = None

        self.dht = None
        # reverse
        self.dh_next = None
        self.dC_next = None
        self.dx = None
        self.Xfacing=True
        self.gen_X_Ds=False

    def _setinitparams(self,batch, seq, input_size,Xfacing=True,gen_X_Ds=False):
        self.input_size=input_size
        self.batch_size=batch
        self.seqmax=seq
        self.shape=(4 * self.hidden_size, self.input_size + self.hidden_size + 1)


        if self.lstmname is None:
            self.lstmname=str(retrieve_name(self))
        WLSTM=TrainableVariable.getInstance(self.lstmname)
        if WLSTM is None:
            WLSTM=TrainableVariable.getInstance(self.lstmname,self.init_function(self.shape)).value
        else:
            WLSTM=TrainableVariable.getInstance(self.lstmname).value
        self.wf = WLSTM[0:self.hidden_size, 1:]
        self.wi = WLSTM[self.hidden_size:self.hidden_size * 2, 1:]
        self.wc = WLSTM[self.hidden_size * 2:self.hidden_size * 3, 1:]
        self.wo = WLSTM[self.hidden_size * 3:self.hidden_size * 4, 1:]

        self.bf = WLSTM[0:self.hidden_size, 0].reshape(self.hidden_size, 1)
        self.bi = WLSTM[self.hidden_size:self.hidden_size * 2, 0].reshape(self.hidden_size, 1)
        self.bc = WLSTM[self.hidden_size * 2:self.hidden_size * 3, 0].reshape(self.hidden_size, 1)
        self.bo = WLSTM[self.hidden_size * 3:self.hidden_size * 4, 0].reshape(self.hidden_size, 1)

        """
        self.WLSTM=self.init_function(self.shape)
        self.wf = self.WLSTM[0:self.hidden_size, 1:]
        self.wi = self.WLSTM[self.hidden_size:self.hidden_size * 2, 1:]
        self.wc = self.WLSTM[self.hidden_size * 2:self.hidden_size * 3, 1:]
        self.wo = self.WLSTM[self.hidden_size * 3:self.hidden_size * 4, 1:]

        self.bf = self.WLSTM[0:self.hidden_size, 0].reshape(self.hidden_size, 1)
        self.bi = self.WLSTM[self.hidden_size:self.hidden_size * 2, 0].reshape(self.hidden_size, 1)
        self.bc = self.WLSTM[self.hidden_size * 2:self.hidden_size * 3, 0].reshape(self.hidden_size, 1)
        self.bo = self.WLSTM[self.hidden_size * 3:self.hidden_size * 4, 0].reshape(self.hidden_size, 1)
        """
        self.dwi = np.zeros_like(self.wi)
        self.dwc = np.zeros_like(self.wc)
        self.dwo = np.zeros_like(self.wo)
        self.dwf = np.zeros_like(self.wf)

        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)
        self.dbf = np.zeros_like(self.bf)

        self.dht = np.zeros([self.hidden_size,self.batch_size])
        # reverse
        self.dh_next = np.zeros_like(self.dht)  # dh from the next character
        self.dC_next = np.zeros_like(self.dht)
        self.dx = np.zeros(self.input_size+self.hidden_size).T
        self.setzerostate()
        self.ct, self.cprojt, self.coldt, self.ft, self.it, self.ot, self.zt, self.xt=  {}, {}, {}, {}, {}, {}, {}, {}

        self.dxt=np.zeros((self.seqmax*self.batch_size,self.input_size))
        self.Xfacing=Xfacing
        self.gen_X_Ds=gen_X_Ds
        self.init=True

        if self.debug:
            print(
                "_setinitparams start*******************************************************************************************************")
            print("cell:", self)
            print("self.lstmname:", self.lstmname)
            print("self.weights shape:", WLSTM.shape)
            print("self.weights:", WLSTM)
            print("self.hidden_size:", self.hidden_size)
            print("self.input_size:", self.input_size)
            print("self.init:", self.init)
            print("self.c:", self.c)
            print("self.h:", self.h)
            print("self.dh_next:", self.dh_next)
            print("self.Xfacing:", self.Xfacing)
            print("self.gen_X_Ds:", self.gen_X_Ds)
            print(
                "_setinitparams End*********************************************************************************************************")
        pass


    def setstate(self,state):
        (c, h) = state
        crctc=checkarrayshape(c,(self.hidden_size,self.batch_size))
        if(crctc is None):
            self.setc(c)
        else:
            self.setc(crctc)

        crcth=checkarrayshape(h,(self.hidden_size,self.batch_size))

        if(crcth is None):
            self.seth(h)
        else:
            self.seth(crcth)

    def setzerostate(self):
        c=zero_state_initializer(self.hidden_size, self.batch_size)
        h=zero_state_initializer(self.hidden_size, self.batch_size)
        self.setstate((c,h))

    def setc(self, c):
        self.c=np.copy(c)

    def setreverseDs(self,dh_next,dc_next):
        checkarrayshape(dh_next,(self.hidden_size,self.batch_size))
        checkarrayshape(dc_next,(self.hidden_size,self.batch_size))
        self.dh_next=dh_next
        self.dC_next=dc_next

    def __call__(self, X, state=None):

        """
        forward pass of LSTMCell
        args:
            x-input-size,batch_size
            state=hidden-size,batch_size
        """
        #sanity checks
        checkdatadim(X,2)
        if(state is not None):
            if (isinstance(state, LSTMStateTuple)):
                state=change_internal_state_type(state)
            checktupledim(state,2)

        #for single call
        if not self.init:
            input_size=X.shape[0]
            self._setinitparams(1, 1, input_size)

        if state is not None:
            self.setstate(state)


        self.ct[-1] = self.c
        if(self.Xfacing and self.gen_X_Ds):
            self.xt[self.seqsize]=X
        z = np.concatenate((X,self.h), 0)
        WLSTM=TrainableVariable.getInstance(self.lstmname).value

        fico = np.dot(WLSTM[:, 1:], z) + WLSTM[:, 0].reshape(self.hidden_size * 4, 1)
        f = self.sigmoid_array(fico[0:self.hidden_size, :] + self.forget_bias)
        i = self.sigmoid_array(fico[self.hidden_size * 1:self.hidden_size * 2, :])
        cproj = self.tanh_array(fico[self.hidden_size * 2:self.hidden_size * 3, :])
        o = self.sigmoid_array(fico[self.hidden_size * 3:self.hidden_size * 4, :])
        cnew = (self.c * f) + (cproj * i)
        hnew = o * self.tanh_array(cnew)
        self.c=cnew
        self.h=hnew
        state = cnew, hnew
        self.coldt[self.seqsize] = self.ct[self.seqsize - 1]
        self.ct[self.seqsize] = cnew
        self.ht[self.seqsize] = hnew
        self.ft[self.seqsize] = f
        self.it[self.seqsize] = i
        self.ot[self.seqsize] = o
        self.cprojt[self.seqsize] = cproj
        self.zt[self.seqsize] = z
        if(self.debug):
            print("-----------------------------------------------------------------")
            print("ForwardPass:Debug:Name:",self.lstmname)
            print("-----------------------------------------------------------------")
            print("input+state:",z)
            print("sequence:",self.seqsize)
            print("f:",f)
            print("i:",i)
            print("cproj:",cproj)
            print("o:",o)
            print("cnew:",cnew)
            print("hnew:",hnew)
            print("-----------------------------------------------------------------")
        self.seqsize+=1

        tup=LSTMStateTuple(cnew.T,hnew.T)
        return hnew.T,tup

    def zero_state(self, batch_size, dtype=float):
        czero=zero_state_initializer(self.hidden_size,batch_size)
        hzero=zero_state_initializer(self.hidden_size,batch_size)
        return LSTMStateTuple(czero.T,hzero.T)

    def sigmoid_array(self, array):
        return 1 / (1 + np.exp(-array))

    def tanh_array(self, array):
        return np.tanh(array)

    def clearStatePerSequence(self,seqmax):
        """
        Cleans the state per sequence.
        """
        self.ct, self.cprojt, self.coldt, self.ft, self.it, self.ot, self.zt, self.xt = {}, {}, {}, {}, {}, {}, {}, {}
        self.seqmax=seqmax
        self.seqsize=0



    def clearDs(self):
        self.dwi = np.zeros_like(self.wi)
        self.dwc = np.zeros_like(self.wc)
        self.dwo = np.zeros_like(self.wo)
        self.dwf = np.zeros_like(self.wf)

        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)
        self.dbf = np.zeros_like(self.bf)

        self.dht = np.zeros([self.hidden_size,self.batch_size])
        # reverse
        self.dh_next = np.zeros_like(self.dht)  # dh from the next character
        self.dC_next = np.zeros_like(self.dht)
        self.dx = np.zeros(self.input_size+self.hidden_size).T
        self.c = zero_state_initializer(self.hidden_size, self.batch_size)
        self.h = zero_state_initializer(self.hidden_size, self.batch_size)
        self.dxt=np.zeros((self.seqmax*self.batch_size,self.input_size))

    """
        *
        *
        *
        *
        *
        """

    def compute_gradients(self,dhtf,t):
    #def compute_gradients(self,dhtf,dh_nextmlco,t):
        z = self.zt[t].T
        self.dht=dhtf
        self.dht = self.dht + self.dh_next
        if(self.backpassdebug):
            print("-----------------------------------------------------------------")
            print("BackPass:Debug:Name:",self.lstmname)
            print("-----------------------------------------------------------------")
            print("timestep:",t)
            print("input+state:",z)
            print("dhtf(recurr(ext)):",dhtf)
            print("dhtf(recurr(int)):",self.dh_next)

        dct = np.multiply(self.dht, self.ot[t] * self.dtanh(self.ct[t])) + self.dC_next

        WLSTM=TrainableVariable.getInstance(self.lstmname).value

        dft = np.multiply(dct, self.coldt[t] * self.dsigmoid(self.ft[t]))
        dxf = np.dot(dft.T, WLSTM[0:self.hidden_size, 1:])
        self.dwf += np.dot(dft, z)
        self.dbf += dft.sum(1,keepdims=True)

        dit = np.multiply(dct, self.cprojt[t] * self.dsigmoid(self.it[t]))
        dxi = np.dot(dit.T, WLSTM[self.hidden_size * 1:self.hidden_size * 2, 1:])
        self.dwi += np.dot(dit, z)
        self.dbi += dit.sum(1,keepdims=True)

        dcproj = np.multiply(dct, self.it[t] * (1 - self.cprojt[t] * self.cprojt[t]))
        dxc = np.dot(dcproj.T, WLSTM[self.hidden_size * 2:self.hidden_size * 3, 1:])
        self.dwc += np.dot(dcproj, z)
        self.dbc += dcproj.sum(1,keepdims=True)

        dot = np.multiply(self.dht, self.tanh_array(self.ct[t]) * self.dsigmoid(self.ot[t]))
        dxo = np.dot(dot.T, WLSTM[self.hidden_size * 3:self.hidden_size * 4, 1:])
        self.dwo += np.dot(dot, z)
        self.dbo += dot.sum(1,keepdims=True)

        dhx = dxf + dxi + dxc + dxo
        xcomp=dhx[:, :self.input_size]

        if(self.Xfacing and self.gen_X_Ds):
            for bi in range(self.batch_size):
                self.dxt[t+(bi*self.seqsize)]=xcomp[bi,:]#.reshape((1,-1))

        self.dh_next = dhx[:, self.input_size:].T
        self.dC_next = np.multiply(dct, self.ft[t])

        dh_next_recurr=dhx[:, :self.hidden_size].T


        if(self.backpassdebug):
            """print("*****************************************************************")
            print("BackPass:Debug:Name:",self.lstmname)
            print("*****************************************************************")
            print("timestep:",t)
            print("input+state:",z)
            print("dhtf(recurr(ext)):",dhtf)
            print("dhtf(recurr(int)):",self.dh_next)"""
            print("dhtf(combined):",self.dht)
            print("dct:",dct)

            print("dft:",dft)
            print("self.dbf:",self.dbf)
            print("dxf:",dxf)

            print("dit:",dit)
            print("dxi:",dxi)

            print("dcproj:",dcproj)
            print("dxc:",dxc)

            print("dot:",dot)
            print("dxo:",dxo)

            print("dhx:",dhx)
            print("xcomp:",xcomp)
            print("self.dxt:",self.dxt)
            print("dhtf(recurr(int(after))):",self.dh_next)
            print("dh_next_recurr:",dh_next_recurr)
            print("dc_next(recurr(int(after))):",self.dC_next)
            print("-----------------------------------------------------------------")
        return np.copy(dh_next_recurr)
        #return np.copy(self.dh_next)

    def get_Xgradients(self):
        if(self.Xfacing and self.gen_X_Ds):
            return self.dxt
        else:
            return None

    def get_gradients(self):
        """
        For a single cell return a tuple of 8 tuples of ds and ws
        |------------------------------------------------------------------|
        |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        |------------------------------------------------------------------|
        """
        return (self.dwi,self.wi),(self.dwc,self.wc),(self.dwf,self.wf),(self.dwo,self.wo),(self.dbi,self.bi),(self.dbc,self.bc),(self.dbf,self.bf),(self.dbo,self.bo)


    def dsigmoid(self,f):
        return f*(1-f)

    def dtanh(self,f):
        tanhf=np.tanh(f)
        return 1 - tanhf * tanhf
