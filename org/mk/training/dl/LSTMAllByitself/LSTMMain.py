#!/usr/bin/env python3

import numpy as np
from LSTMCell import LSTMCell
from LSTMCell import dynamic_rnn
import collections
from common import loss
from common import softmax
import sys

# data I/O

train_file = sys.argv[1]
data = open(train_file, 'r').read()

out_weights = np.array([[-0.09588283, -2.2044923, -0.74828255, 0.14180686, -0.32083616,
						 -0.9444244, 0.06826905, -0.9728962, -0.18506959, 1.0618515],
						[1.156649, 3.2738173, -1.2556943, -0.9079511, -0.82127047,
						 -1.1448543, -0.60807484, -0.5885713, 1.0378786, -0.7088431],
						[1.006477, 0.28033388, -0.1804534, 0.8093307, -0.36991575,
						 0.29115433, -0.01028167, -0.7357091, 0.92254084, -0.10753923],
						[0.19266959, 0.6108299, 2.2495654, 1.5288974, 1.0172302,
						 1.1311738, 0.2666629, -0.30611828, -0.01412263, 0.44799015],
						[0.19266959, 0.6108299, 2.2495654, 1.5288974, 1.0172302,
						 1.1311738, 0.2666629, -0.30611828, -0.01412263, 0.44799015]]
					   )

out_biases = np.array([[0.1458478, -0.3660951, -2.1647317, -1.9633691, -0.24532059,
					   0.14005205, -1.0961286, -0.43737876, 0.7028531, -1.8481724]]
					  )


def input_one_hot(num):
	print(num)
	x = np.zeros(vocab_size)
	x[num] = 1
	return x.tolist()


def read_data(fname):
	with open(fname) as f:
		data = f.readlines()
	data = [x.strip() for x in data]
	data = [data[i].lower().split() for i in range(len(data))]
	data = np.array(data)
	data = np.reshape(data, [-1, ])
	print(data)
	return data


def build_dataset(train_data):
	count = collections.Counter(train_data).most_common()
	print("count:", count)

	dictionary = dict()
	print("dictionary:", dictionary)
	sortedwords = sorted(set(train_data))
	print("sortedword:", sortedwords)

	for word in sortedwords:
		print("word:", word)
		dictionary[word] = len(dictionary)
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return dictionary, reverse_dictionary


train_data = read_data(train_file)
dictionary, reverse_dictionary = build_dataset(train_data)
vocab_size = len(dictionary)
print("dictionary:", dictionary)
print("reverse_dictionary:", reverse_dictionary)
print("vocab_size", vocab_size)

learning_rate = 0.001
# training_iters = 50000
training_iters = 2
display_step = 500
n_input = 3
n_hidden = 5
rnd = np.random.RandomState(42)

step = 0
#offset = rnd.randint(0, n_input + 1)
offset=2
end_offset = n_input + 1
acc_total = 0
loss_total = 0
print("offset:", offset)
# only for testing
weights = np.ones([4 * n_hidden, vocab_size + n_hidden + 1]) * .1

while step < training_iters:
	if offset > (len(train_data) - end_offset):
		offset = rnd.randint(0, n_input + 1)
	print("offset:", offset)
	symbols_in_keys = [input_one_hot(dictionary[str(train_data[i])]) for i in range(offset, offset + n_input)]

	symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, vocab_size])
	symbols_out_onehot = np.zeros([vocab_size], dtype=float)
	symbols_out_onehot[dictionary[str(train_data[offset + n_input])]] = 1.0
	symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])
	cell = LSTMCell(vocab_size, n_hidden, w=weights)
	result, state = dynamic_rnn(cell, symbols_in_keys)
	(c, h) = state
	print("final:", state)
	pred = np.dot(np.reshape(h, [-1, n_hidden]), out_weights) + out_biases
	print("pred:", pred)
	ps = softmax(pred)
	print("softmax:", ps)
	lossesperoneseqset = loss(np.reshape(ps, [-1, vocab_size]),symbols_out_onehot)
	print("loss", lossesperoneseqset)
	# lossesperoneseq = loss(np.reshape(ps,[vocab_size,-1]),symbols_out_onehot)
	# reduce_mean(sysmbols_out_onehot,pred)
	cell.compute_gradients(ps,symbols_out_onehot,out_weights,out_biases)
	step += 1
	offset += (n_input + 1)
print("Optimization Finished!")	
