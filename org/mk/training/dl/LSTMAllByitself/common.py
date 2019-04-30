#!/usr/bin/env python3

import numpy as np

def loss(pred,labels):
	#print("loss:",np.multiply(labels, -np.log(pred)).sum(1))
	return np.multiply(labels, -np.log(pred)).sum(1)

def softmax(logits):
	"""row represents num classes but they may be real numbers
	That needs to be converted to probability"""
	r, c = logits.shape
	predsl = []
	for row in logits:
		inputs = np.squeeze(np.asarray(row))
		predsl.append(np.exp(inputs) / float(sum(np.exp(inputs))))
	#print("predsl:",predsl)
	return np.matrix(predsl).T
