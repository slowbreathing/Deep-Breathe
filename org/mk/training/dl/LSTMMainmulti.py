#!/usr/bin/env python3

import numpy as np
import collections

from org.mk.training.dl.rnn_cell import LSTMCell
from org.mk.training.dl.rnn import dynamic_rnn
from org.mk.training.dl.common import loss
from org.mk.training.dl.common import softmax
from org.mk.training.dl.rnn import MultiRNNCell
from org.mk.training.dl.rnn import print_gradients

import sys
from org.mk.training.dl.rnn import LSTMStateTuple
from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl import init_ops
from org.mk.training.dl.optimizer import BatchGradientDescent
from org.mk.training.dl.common import cross_entropy_loss
from org.mk.training.dl.common import input_one_hot
from org.mk.training.dl.core import Dense

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
training_iters = 200
display_step = 100
n_input = 3
n_hidden = 5
rnd = np.random.RandomState(42)

step = 0
#offset = rnd.randint(0, n_input + 1)
offset =2
end_offset = n_input + 1
acc_total = 0
loss_total = 0
print("offset:", offset)
# only for testing
"""
with WeightsInitializer(initializer=init_ops.Constant(0.1)) as vs:
    cell1 = LSTMCell(n_hidden,debug=True)
    cell2 = LSTMCell(n_hidden,debug=True)
cell = MultiRNNCell([cell1, cell2])
n_layers=2
cell_list=[]
for i in range(n_layers):
    with WeightsInitializer(initializer=init_ops.Constant(0.1)) as vs:
        cell_list.append(LSTMCell(5,debug=True))
cell=MultiRNNCell(cell_list)
gdo=BatchGradientDescent(learning_rate)
out_l = Dense(10,kernel_initializer=init_ops.Constant(out_weights),bias_initializer=init_ops.Constant(out_biases))
"""
"""
with WeightsInitializer(initializer=init_ops.Constant(0.1)) as vs:
    cell1 = LSTMCell(n_hidden,debug=True)
    cell2 = LSTMCell(n_hidden,debug=True)
cell = MultiRNNCell([cell1, cell2])"""
"""
n_layers=2
cell_list=[]
for i in range(n_layers):
    with WeightsInitializer(initializer=init_ops.Constant(0.1)) as vs:
        #cellwam=LSTMCell(5,debug=True)
        #cell_list.append(cellwam)
        cell_list.append(LSTMCell(5,debug=True))
cellw=MultiRNNCell(cell_list)
"""
with WeightsInitializer(initializer=init_ops.Constant(0.1)) as vs:
    cell1 = LSTMCell(n_hidden,debug=True)
    cell2 = LSTMCell(n_hidden,debug=True)
cell = MultiRNNCell([cell1, cell2])

gdo=BatchGradientDescent(learning_rate)
out_l = Dense(10,kernel_initializer=init_ops.Constant(out_weights),bias_initializer=init_ops.Constant(out_biases))

"""
c=np.ones((n_hidden,1))
h=np.ones((n_hidden,1))
initstate=(c,h)
initstates=[]
initstates.append(initstate)
initstates.append(initstate)
"""

while step < training_iters:
    if offset > (len(train_data) - end_offset):
        offset = rnd.randint(0, n_input + 1)
    print("offset:", offset)
    symbols_in_keys = [input_one_hot(dictionary[str(train_data[i])],vocab_size) for i in range(offset, offset + n_input)]
    symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, vocab_size])
    target=dictionary[str(train_data[offset + n_input])]

    result, state = dynamic_rnn(cell, symbols_in_keys)
    #result, state = dynamic_rnn(cell, symbols_in_keys,initstates)
    (c, h) = state[-1].c,state[-1].h
    #(c, h) = state.c,state.h
    print("final:", result,state,h.shape)

    #last layer of Feed Forward to compare to transform result to the shape of target
    pred=out_l(h)
    print("pred:",pred)

    #cross_entropy_loss internally calculates the same softmax as and then the loss as above but for a batch and sequence
    #pred- batch,seq,input_size
    #labels-batch,seq(has to be transformed before comparision with preds(line-43).)
    yhat,cel=cross_entropy_loss(pred.reshape([1,1,vocab_size]),np.array([[target]]))
    print("yhat:",yhat)
    print("CEL:",cel)

    #yhat-Size of yhat should be batch,seq,size
    #target-Size of target should be batch,seq
    gradients=gdo.compute_gradients(yhat,np.array([[target]]))
    gdo.apply_gradients(gradients)
    print_gradients(gradients)
    step += 1
    offset += (n_input + 1)
print("Optimization Finished!")
