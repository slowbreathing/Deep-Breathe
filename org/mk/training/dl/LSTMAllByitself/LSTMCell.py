from operator import iadd

# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
import numpy as np
from Cell import Cell
import collections

class LSTMCell(Cell):
	# initialise the Recurrent Neural Network
	def __init__(self, input_size, hidden_size, forget_bias=1, w=None, debug=False):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.forget_bias = forget_bias
		self.debug = debug
		self.seqsize=0
		# first column is for biases
		if w is None:
			self.WLSTM = np.random.randn(4 * hidden_size, input_size + hidden_size + 1) / np.sqrt(
				input_size + hidden_size)
		else:
			self.WLSTM = w

		self.ht, self.ct, self.cprojt, self.coldt, self.ft, self.it, self.ot, self.zt = {}, {}, {}, {}, {}, {}, {}, {}
		if self.debug:
			print(
				"Contructor start*******************************************************************************************************")
			print("self.weights shape:", self.WLSTM.shape)
			print("self.weights:", self.WLSTM)
			print("self.hidden_size:", self.hidden_size)
			print(
				"Contructor End*********************************************************************************************************")
		pass

	def __call__(self, x, state=None):

		#print("self.seqsize:",self.seqsize)
		if state is None:
			(c, h) = zero_state_initializer(self.hidden_size, 1)
		else:
			(c, h) = state
		self.ct[-1] = c
		if self.debug:
			print("c:", c, " h:", h)

		row, col = x.shape
		if self.input_size != row:
			print("self.input_size:", self.input_size, " c:", row)
			raise ValueError('Input size must match. This is the typically the vocab size depending on the encoding')
		fico = np.dot(self.WLSTM[:, 1:], np.concatenate((h, x), 0)) + self.WLSTM[:, 0].reshape(self.hidden_size * 4,
																							   col)
		#print("fico:", fico)
		z = np.concatenate((x, h), 0)
		print("z:",z)
		print("self.WLSTM:", self.WLSTM)
		fico = np.dot(self.WLSTM[:, 1:], z) + self.WLSTM[:, 0].reshape(self.hidden_size * 4, col)
		f = self.sigmoid_array(fico[0:self.hidden_size, :] + self.forget_bias)
		i = self.sigmoid_array(fico[self.hidden_size * 1:self.hidden_size * 2, :])
		cproj = self.tanh_array(fico[self.hidden_size * 2:self.hidden_size * 3, :])
		o = self.sigmoid_array(fico[self.hidden_size * 3:self.hidden_size * 4, :])
		cnew = (c * f) + (cproj * i)
		hnew = o * self.tanh_array(cnew)
		print("cnew:",cnew," hnew:",hnew)
		state = cnew, hnew
		self.coldt[self.seqsize] = self.ct[self.seqsize - 1]
		self.ct[self.seqsize] = cnew
		self.ht[self.seqsize] = hnew
		self.ft[self.seqsize] = f
		self.it[self.seqsize] = i
		self.ot[self.seqsize] = o
		self.cprojt[self.seqsize] = cproj
		self.zt[self.seqsize] = z
		self.seqsize+=1
		return state


	def sigmoid_array(self, array):
		return 1 / (1 + np.exp(-array))

	def tanh_array(self, array):
		return np.tanh(array)

	def initState(self):
		"""reset state after feeding in the data"""
		if self.debug:
			print("initState:::::")
		self.state = np.zeros((self.statesize, self.batchsize))
		self.totalloss = 0
		self.numloss = 0
		self.totallosses = []

	def compute_gradients(self,yhat,symbols_out_onehot,WOUT,BOUT):
		print("self.seqsize:",self.seqsize)
		wf = self.WLSTM[0:self.hidden_size, 1:]
		wi = self.WLSTM[self.hidden_size:self.hidden_size * 2, 1:]
		wc = self.WLSTM[self.hidden_size * 2:self.hidden_size * 3, 1:]
		wo = self.WLSTM[self.hidden_size * 3:self.hidden_size * 4, 1:]
		bf = self.WLSTM[0:self.hidden_size, 0].reshape(5, 1)
		bi = self.WLSTM[self.hidden_size:self.hidden_size * 2, 0].reshape(5, 1)
		bc = self.WLSTM[self.hidden_size * 2:self.hidden_size * 3, 0].reshape(5, 1)
		bo = self.WLSTM[self.hidden_size * 3:self.hidden_size * 4, 0].reshape(5, 1)

		dwi = np.zeros_like(wi)
		dwc = np.zeros_like(wc)
		dwo = np.zeros_like(wo)
		dwf = np.zeros_like(wf)

		dbi = np.zeros_like(bi)
		dbc = np.zeros_like(bc)
		dbo = np.zeros_like(bo)
		dbf = np.zeros_like(bf)

		# reverse
		dh_next = np.zeros_like(self.ct[0])  # dh from the next character
		dC_next = np.zeros_like(self.ct[0])
		dx = np.zeros_like(self.zt[0]).T
		dy = yhat.copy()
		dy = dy - np.reshape(symbols_out_onehot, [-1, 1])
		print("lossesperoneseq:dy:", dy)

		dWy = np.dot(dy, self.ht[self.seqsize-1].T)
		dBy = dy
		dht = np.dot(dy.T, WOUT.T).T
		print("dht:",dht)
		for t in reversed(range(self.seqsize)):
			z = self.zt[t].T
			dht = dht + dh_next
			dot = np.multiply(dht, self.tanh_array(self.ct[t]) * self.dsigmoid(self.ot[t]))
			dct = np.multiply(dht, self.ot[t] * self.dtanh(self.ct[t])) + dC_next
			print("dct:", dct)
			dcproj = np.multiply(dct, self.it[t] * (1 - self.cprojt[t] * self.cprojt[t]))
			print("dcproj:", dcproj)
			dft = np.multiply(dct, self.coldt[t] * self.dsigmoid(self.ft[t]))
			print("dft:", dft)
			dit = np.multiply(dct, self.cprojt[t] * self.dsigmoid(self.it[t]))
			print("dit:", dit)

			dwf += np.dot(dft, z)
			# print("dxf:",dft.shape,":",z.shape,":",dwf.shape)
			print("dwf:", dwf)
			dbf += dft
			dxf = np.dot(dft.T, self.WLSTM[0:self.hidden_size, 1:])
			print("dxf:",dxf)

			dwi += np.dot(dit, z)
			dbi += dit
			dxi = np.dot(dit.T, self.WLSTM[self.hidden_size * 1:self.hidden_size * 2, 1:])
			# print("dxi:",dxi)

			dwc += np.dot(dcproj, z)
			dbc += dcproj
			dxc = np.dot(dcproj.T, self.WLSTM[self.hidden_size * 2:self.hidden_size * 3, 1:])
			# print("dxc:",dxc)

			dwo += np.dot(dot, z)
			dbo += dot
			dxo = np.dot(dot.T, self.WLSTM[self.hidden_size * 3:self.hidden_size * 4, 1:])
			# print("dxo:",dxo)

			dx = dxf + dxi + dxc + dxo
			dh_next = dx[:, :self.hidden_size].T
			dC_next = np.multiply(dct, self.ft[t])
			db = np.concatenate((dit, dcproj, dft, dot), axis=0)
			dht = np.zeros_like(dht)

		dw = np.concatenate((dwi, dwc, dwf, dwo), axis=0)
		db = np.concatenate((dbi, dbc, dbf, dbo), axis=0)

		#print("WOUT.T:", WOUT.T)
		#print("BOUT.T:", BOUT.T)
		#print("dBy:", dBy)
		for param, dparam in zip([wf, wi, wc, wo, WOUT.T, bf, bi, bc, bo, BOUT.T],
								 [dwf, dwi, dwc, dwo, dWy, dbf, dbi, dbc, dbo, dBy]):
			param += -.001 * dparam


		print("dw.T:", dw.T)
		print("db.T:", db.T)
		print("biases:", np.concatenate((bi, bc, bf, bo), axis=0).T)
		print("WOUT.T:",WOUT.T)
		print("BOUT.T:", BOUT.T)
	def dsigmoid(self,f):
		#print("dsigmoid:",f)
		return f*(1-f)

	def dtanh(self,f):
		#print("dtanh:",f)
		tanhf=np.tanh(f)
		return 1 - tanhf * tanhf

def zero_state_initializer(shape, batch_size):
	c = np.zeros((shape, batch_size))
	h = np.zeros((shape, batch_size))
	state = (c, h)
	return state


def dynamic_rnn(cell, X, state=None):
	#default in tensorflow is batch major,max seq,num_classes/inputsize
	batch, seq, input_size = X.shape
	if issubclass(type(cell), LSTMCell):
		if state is None:
			state = zero_state_initializer(cell.hidden_size, batch)

	result = {}
	prevstate = state
	for seqnum in range(seq):
		newstate = cell(np.reshape(X[0][seqnum], [input_size, batch]), prevstate)
		result[seqnum] = newstate[1];
		prevstate = newstate

	return result, prevstate

