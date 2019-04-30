from operator import iadd

# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
import numpy as np
import collections
from Cell import Cell

class MultiRNNCell(Cell):
	# initialise the Recurrent Neural Network
	def __init__(self,cells):
		print("MultiRNNCell.__init__")
		self.feedforwardcells=cells
		self.feedforwarddepth =len(cells)

	def __call__(self, x, state=None):

		for cell in self.feedforwardcells:
			returnstate =cell(x,state)
			state=returnstate
			c,h=returnstate
			x=h;
			#self.seqsize += 1
		#print("MultiRNNCell:self.seqsize:",self.seqsize)
		return state;

	def compute_gradients(self,yhat,symbols_out_onehot,WOUT,BOUT):
		print("MultiRNNCell.compute_gradients.self.seqsize:", self.feedforwarddepth)
		for t in reversed(range(self.feedforwarddepth)):
			cell=self.feedforwardcells[t-1]
			cell.compute_gradients(yhat,symbols_out_onehot,WOUT,BOUT)
