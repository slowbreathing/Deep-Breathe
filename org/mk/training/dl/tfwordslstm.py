import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn import GRUCell
from org.mk.training.dl.common import input_one_hot


import sys
# data I/O

train_file=sys.argv[1]
data = open(train_file, 'r').read()

# Parameters
learning_rate = 0.001
#training_iters = 50000
training_iters = 2
display_step = 100
n_input = 3

# number of units in RNN cell
n_hidden = 5
rnd=np.random.RandomState(42)

def read_data(fname):
    with open(fname) as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [data[i].lower().split() for i in range(len(data))]
    data = np.array(data)
    data = np.reshape(data, [-1, ])
    return data

train_data = read_data(train_file)


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    sortedwords=sorted(set(words))

    for i,word in enumerate(sortedwords):
        dictionary[word] = i
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(train_data)
vocab_size = len(dictionary)

# Place holder for Mini batch input output
x = tf.placeholder("float", [None, n_input, vocab_size])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable([[-0.09588283, -2.2044923 , -0.74828255,  0.14180686, -0.32083616,
        -0.9444244 ,  0.06826905, -0.9728962 , -0.18506959,  1.0618515 ],
       [ 1.156649  ,  3.2738173 , -1.2556943 , -0.9079511 , -0.82127047,
        -1.1448543 , -0.60807484, -0.5885713 ,  1.0378786 , -0.7088431 ],
       [ 1.006477  ,  0.28033388, -0.1804534 ,  0.8093307 , -0.36991575,
         0.29115433, -0.01028167, -0.7357091 ,  0.92254084, -0.10753923],
       [ 0.19266959,  0.6108299 ,  2.2495654 ,  1.5288974 ,  1.0172302 ,
         1.1311738 ,  0.2666629 , -0.30611828, -0.01412263,  0.44799015],
       [ 0.19266959,  0.6108299 ,  2.2495654 ,  1.5288974 ,  1.0172302 ,
         1.1311738 ,  0.2666629 , -0.30611828, -0.01412263,  0.44799015]]

        )
}
biases = {
    'out': tf.Variable([ 0.1458478 , -0.3660951 , -2.1647317 , -1.9633691 , -0.24532059,
        0.14005205, -1.0961286 , -0.43737876,  0.7028531 , -1.8481724 ]
    )
}


#works with 2 dimension hence easy but not optimal
"""
def RNN(x, weights, biases):
    with variable_scope.variable_scope(
            "other", initializer=init_ops.constant_initializer(0.1)) as vs:
        x = tf.unstack(x, n_input, 1)
        cell = rnn_cell.LayerNormBasicLSTMCell(n_hidden, layer_norm=False)
        outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights['out']) + biases['out'],outputs,states,weights['out'],biases['out']
"""

#Same as above
#works with 3 dimensions. Line 112 takes care of extracting last (h*wy=by) in 2 dimensions
def RNN(x, weights, biases):
    with variable_scope.variable_scope(
            "other", initializer=init_ops.constant_initializer(0.1)) as vs:
        cell = rnn_cell.LayerNormBasicLSTMCell(n_hidden, layer_norm=False)
        outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return tf.expand_dims(tf.matmul(outputs[-1], weights['out'])[-1],0) + biases['out'],outputs[-1],states,weights['out'],biases['out']

pred,output,state,weights_out,biases_out = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
grads_and_vars_tf_style = optimizer.compute_gradients(cost)
train_tf_style = optimizer.apply_gradients(grads_and_vars_tf_style)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

global_step = tf.Variable(0, name='global_step', trainable=False)
# Initializing the variables
init = tf.global_variables_initializer()

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"
# Launch the graph
saver = tf.train.Saver(max_to_keep=200)
with tf.Session() as session:
    session.run(init)
    step = 0
    offset =2
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    print ("offset:",offset)


    while step < training_iters:
        if offset > (len(train_data)-end_offset):
            offset = rnd.randint(0, n_input+1)
        print("offset:", offset)
        symbols_in_keys = [ input_one_hot(dictionary[ str(train_data[i])],vocab_size) for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input,vocab_size])
        symbols_out_onehot=input_one_hot(dictionary[str(train_data[offset+n_input])],vocab_size)
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
        tfgrads_and_vars_tf_style, _,acc, loss, onehot_pred,tfoutput,tfstate,tfout_weights,tfbiases_out  = session.run([grads_and_vars_tf_style,train_tf_style, accuracy, cost, pred,output,state,weights_out,biases_out], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        print("tfoutput:",tfoutput," tfstate:",tfstate)
        print("onehot_pred:",onehot_pred)
        print("loss:",loss)
        print("tfgrads_and_vars_tf_style:",tfgrads_and_vars_tf_style)
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [train_data[i] for i in range(offset, offset + n_input)]
            symbols_out = train_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            saver.save(session,
				   "resources/tmp/rnn_words/lstm/"+"model-checkpoint-" + '%04d' % (step+1), global_step=global_step)
            print("%s - Actual word:[%s] vs Predicted word:[%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
