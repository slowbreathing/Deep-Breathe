from operator import iadd

# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
import numpy as np
import collections
class Cell(object):
	# initialise the Recurrent Neural Network
	def __init__(self):
		""""""

	def __call__(self, x, state=None):
		return state