import tensorflow as tf
import numpy as np
import time
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper, BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, sequence_loss
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib import rnn
from org.mk.training.dl.nmtdata import get_nmt_data

from org.mk.training.dl.nmt import parse_arguments

#np.set_printoptions(threshold=np.nan)
init=np.array([[-2.0387042,  -0.7570444,  -1.549724,   -0.55742437, -0.10309707, -0.2645374,
   0.5542126,  -0.9948135,  -1.4004004,  -0.2027762,   1.8161317,   0.02489787,
   0.04653463,  0.30181375, -1.0206957,  -0.4414572,  -0.08976762,  0.86643434,
   0.06023955, 0.50390786, -1.1679714,  -0.31363872, -0.87805235, -3.808063,
  -1.2836251,   0.1762668,  -0.4557327,   1.1585172,  -0.6317208,  -0.7690312,
  -1.1819371,  -1.0957835,  -1.0487816,   0.38563657,  0.7846264,  -0.16195902,
   2.9963484,  -1.1604083,   2.127244,    1.0451506,   2.3449166,  -1.11046   ],
 [-1.3579369,   1.6391242,   0.51722956, -1.1566479,   0.5217864,   0.6738795,
   1.4439716,  -1.5845695,  -0.5321513,  -0.45986208,  0.95982075, -2.7541134,
   0.04544061, -0.24134564,  0.01985956, -0.01174978,  0.21752118, -0.96756375,
  -1.1478109,  -1.4283063,   0.33229867, -2.06857,    -1.0454241,  -0.60130537,
   1.1755886,   0.8240156,  -1.4274453,   1.1680154,  -1.4401436,   0.16765368,
   1.2770568,  -0.15272069, -0.70132256, -0.39191842,  0.14498521,  0.52371395,
  -1.0711092,   0.7994564,  -0.86202085, -0.08277576,  0.6717222,  -0.30217007],
 [ 1.1651239,   0.8676004,  -0.7326845,   1.1329368,   0.33225664,  0.42479947,
   2.442528,  -0.24212709, -0.31981337,  0.7518857,   0.09940664,  0.733886,
   0.16233322, -3.180123,   -0.05459447, -1.0913122,   0.6260485,   1.3436326,
   0.3324367,  -0.4807606,   0.80269957,  0.80319524, -1.0442443,   0.78740156,
  -0.40594986,  2.0595453,   0.95093924,  0.05337913,  0.70918155,  1.553336,
   0.91851705, -0.79104966,  0.4592584,  -1.0875456,   1.0102607,  -1.0877079,
  -0.61451066, -0.8137477,   0.19382478, -0.7905477,   1.157019,   -0.21588814],
 [-0.02875052,  1.3946419,   1.3715329,   0.03029069,  0.72896576,  1.556146,
   0.62381554,  0.28502566,  0.42641425, -0.9238658,  -1.3146611,   0.97760606,
  -0.5422947,  -0.66413164, -0.57151276, -0.52428764, -0.44747844, -0.07234555,
   1.5007111,   0.6677294,  -0.7949865,  -1.1016922,   0.00385522,  0.2087736,
   0.02533335, -0.15060721,  0.41100115,  0.04573904,  1.5402086,  -0.5570146,
   0.8980145,  -1.0776126,  0.25556734, -1.0891188,  -0.25838724,  0.28069794,
   0.25003937,  0.47946456, -0.36741912,  0.8140413,   0.5821169,  -1.8102683 ],
 [ 1.4668883,  -0.27569455,  0.19961897,  1.0866551,  0.10519085,  1.0896195,
  -0.88432556, -0.45068273,  0.37042075, -0.10234109, -0.6915803,  -1.1545025,
  -0.4954256,  -0.10934342, -0.2797343,   0.42959297, -0.6256306,  -0.04518669,
  -1.5740314,  -0.7988373, -0.5571486,  -1.4605384,   0.85387,    -1.6822307,
   0.72871834,  0.47308877, -1.3507669,  -1.4545231,   1.1324743,  -0.03236655,
   0.6779119,   0.9597622,  -1.3243811,  -0.92739224, -0.18055117,  0.71914613,
   0.5413713,  -0.3229486,  -1.7227241,  -1.2969391,   0.27593264,  0.32810318]]
)

def model_inputs():
    inputs_data = tf.placeholder(tf.int32, [None, None], name='input_data')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    dropout_probs = tf.placeholder(tf.float32, name='dropout_probs')
    en_len = tf.placeholder(tf.int32, (None,), name='en_len')
    max_en_len = tf.reduce_max(en_len, name='max_en_len')
    fr_len = tf.placeholder(tf.int32, (None,), name='fr_len')
    return inputs_data, targets, learning_rate, dropout_probs, en_len, max_en_len, fr_len


def process_encoding_input(target_data, word2int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    decoding_input = tf.concat([tf.fill([batch_size, 1], word2int['TOKEN_GO']), ending], 1)
    return decoding_input


def get_rnn_cell(rnn_cell_size, dropout_prob,n_layers):
    rnn_c=None
    if(n_layers ==1):
        with tf.variable_scope('cells_0'):
            rnn_c = rnn_cell.LayerNormBasicLSTMCell(rnn_cell_size, layer_norm=False)
    else:
        cell_list=[]
        for i in range(n_layers):
            with tf.variable_scope('cells_{}'.format(i)):
                cell_list.append(rnn_cell.LayerNormBasicLSTMCell(rnn_cell_size, layer_norm=False))
        rnn_c=rnn.MultiRNNCell(cell_list)
    return rnn_c


def encoding_layer(rnn_cell_size, sequence_len, n_layers, rnn_inputs, dropout_prob,encoder_type):

    #for l in range(n_layers):
        #with tf.variable_scope('encoding_l_{}'.format(l)):
    with variable_scope.variable_scope(
            "encoding_layer", initializer=init_ops.constant_initializer(0.1)) as vs:
        if(encoder_type is "bi"):
            n_bi_layers=int(n_layers/2)
            rnn_fw = get_rnn_cell(rnn_cell_size, dropout_prob,n_bi_layers)
            rnn_bw = get_rnn_cell(rnn_cell_size, dropout_prob,n_bi_layers)
            encoding_output, encoding_state = tf.nn.bidirectional_dynamic_rnn(rnn_fw, rnn_bw,
                                                                rnn_inputs,
                                                                sequence_len,
                                                                dtype=tf.float32)
            encoding_output = tf.concat(encoding_output,-1)
            if(n_bi_layers > 1):
                """
                Forward
                ((LSTMStateTuple({'c': array([[0.30450274, 0.30450274, 0.30450274, 0.30450274, 0.30450274]]), 'h': array([[0.16661529, 0.16661529, 0.16661529, 0.16661529, 0.16661529]])}),
                      LSTMStateTuple({'c': array([[0.27710986, 0.07844026, 0.18714019, 0.28426586, 0.28426586]]), 'h': array([[0.15019765, 0.04329417, 0.10251247, 0.1539225 , 0.1539225 ]])})),
                Backward
                (LSTMStateTuple({'c': array([[0.30499766, 0.30499766, 0.30499766, 0.30499766, 0.30499766]]), 'h': array([[0.16688152, 0.16688152, 0.16688152, 0.16688152, 0.16688152]])}),
                      LSTMStateTuple({'c': array([[0.25328871, 0.17537864, 0.21700339, 0.25627687, 0.25627687]]), 'h': array([[0.13779658, 0.09631104, 0.11861721, 0.1393639 , 0.1393639 ]])})))
                """
                encoder_state = []
                for layer_id in range(n_bi_layers):
                    encoder_state.append(encoding_state[0][layer_id])  # forward
                    encoder_state.append(encoding_state[1][layer_id])  # backward
                encoding_state = tuple(encoder_state)
                """
                First
                ((LSTMStateTuple({'c': array([[0.30450274, 0.30450274, 0.30450274, 0.30450274, 0.30450274]]), 'h': array([[0.16661529, 0.16661529, 0.16661529, 0.16661529, 0.16661529]])}),
                      LSTMStateTuple({'c': array([[0.30499766, 0.30499766, 0.30499766, 0.30499766, 0.30499766]]), 'h': array([[0.16688152, 0.16688152, 0.16688152, 0.16688152, 0.16688152]])})),
                Second
                (LSTMStateTuple({'c': array([[0.27710986, 0.07844026, 0.18714019, 0.28426586, 0.28426586]]), 'h': array([[0.15019765, 0.04329417, 0.10251247, 0.1539225 , 0.1539225 ]])}),
                      LSTMStateTuple({'c': array([[0.25328871, 0.17537864, 0.21700339, 0.25627687, 0.25627687]]), 'h': array([[0.13779658, 0.09631104, 0.11861721, 0.1393639 , 0.1393639 ]])})))
                """
        else:
            print("uni:",n_layers)
            rnn_fw = get_rnn_cell(rnn_cell_size, dropout_prob,n_layers)
            encoding_output, encoding_state = tf.nn.dynamic_rnn(rnn_fw, rnn_inputs,
                                                                          sequence_len,
                                                                          dtype=tf.float32)
        return encoding_output, encoding_state, rnn_inputs

def training_decoding_layer(decoding_embed_input, en_len, decoding_cell, initial_state, op_layer,
                            v_size, max_en_len):
    with variable_scope.variable_scope(
            "decoder", initializer=init_ops.constant_initializer(0.1)) as vs:
        helper = TrainingHelper(inputs=decoding_embed_input, sequence_length=en_len, time_major=False)
        dec = BasicDecoder(decoding_cell, helper, initial_state,op_layer )
        logits, _, _ = dynamic_decode(dec, output_time_major=False, impute_finished=True,
                                      maximum_iterations=max_en_len)
        return logits


def decoding_layer(decoding_embed_inp, embeddings, encoding_op, encoding_st, v_size, fr_len,
                   en_len, max_en_len, rnn_cell_size, word2int, dropout_prob, batch_size, n_layers):
    with tf.variable_scope("decoding_layer"):
        decoding_cell = get_rnn_cell(rnn_cell_size, dropout_prob,n_layers)
        out_l = Dense(v_size,kernel_initializer=tf.constant_initializer(init))

        logits_tr = training_decoding_layer(decoding_embed_inp,
                                            en_len,
                                            decoding_cell,
                                            encoding_st,
                                            out_l,
                                            v_size,
                                            max_en_len)
    return logits_tr


def seq2seq_model(input_data, target_en_data, dropout_prob, fr_len, en_len, max_en_len,
                  v_size, rnn_cell_size, n_layers, word2int_en, batch_size,encoder_type,fr_embeddings_matrix,en_embeddings_matrix):
    input_word_embeddings = tf.Variable(fr_embeddings_matrix, name="input_word_embeddings")
    encoding_embed_input = tf.nn.embedding_lookup(input_word_embeddings, input_data)
    encoding_op, encoding_st, rnn_inputs = encoding_layer(rnn_cell_size, fr_len, n_layers, encoding_embed_input,
                                                          dropout_prob,encoder_type)

    decoding_input = process_encoding_input(target_en_data, word2int_en, batch_size)

    decoding_embed_input = tf.nn.embedding_lookup(en_embeddings_matrix, decoding_input)

    tr_logits= decoding_layer(decoding_embed_input,
                                           en_embeddings_matrix,
                                           encoding_op,
                                           encoding_st,
                                           v_size,
                                           fr_len,
                                           en_len,
                                           max_en_len,
                                           rnn_cell_size,
                                           word2int_en,
                                           dropout_prob,
                                           batch_size,
                                           n_layers)

    return encoding_embed_input, encoding_op, encoding_st, rnn_inputs, decoding_input, decoding_embed_input,tr_logits

def pad_sentences(sentences_batch, word2int):
    max_sentence = max([len(sentence) for sentence in sentences_batch])
    return [sentence + [word2int['TOKEN_PAD']] * (max_sentence - len(sentence)) for sentence in sentences_batch]


def get_batches(en_text, fr_text, batch_size,fr_word2int,en_word2int):
    for batch_idx in range(0, len(fr_text) // batch_size):
        start_idx = batch_idx * batch_size
        en_batch = en_text[start_idx:start_idx + batch_size]
        fr_batch = fr_text[start_idx:start_idx + batch_size]
        pad_en_batch = np.array(pad_sentences(en_batch, en_word2int))
        pad_fr_batch = np.array(pad_sentences(fr_batch, fr_word2int))

        pad_en_lens = []
        for en_b in pad_en_batch:
            pad_en_lens.append(len(en_b))

        pad_fr_lens = []
        for fr_b in pad_fr_batch:
            pad_fr_lens.append(len(fr_b))

        print("pad_en_batch:", pad_en_batch)
        print("pad_en_lens:", pad_en_lens)
        print("pad_fr_batch:", pad_fr_batch)
        print("pad_fr_lens:", pad_fr_lens)

        yield pad_en_batch, pad_fr_batch, pad_en_lens, pad_fr_lens


epochs = 0
batch_size = 0
hidden_size = 0
n_layers = 0
n_bi_layers=0
lr = 0.0
dr_prob = 0.75
encoder_type=None
per_epoch=0
logs_path = '/tmp/models/'

def set_modelparams(args):
    global epochs,n_layers,encoder_type,hidden_size,batch_size,lr,rnn_fw,rnn_bw,decoding_cell,gdo,n_bi_layer,debug,per_epoch,logs_path
    epochs=args.epochs
    n_layers=args.num_layers
    encoder_type=args.encoder_type
    hidden_size=args.num_units
    batch_size = args.batch_size
    lr = args.learning_rate
    debug=args.debug
    per_epoch=args.per_epoch
    logs_path=args.out_dir

fr_embeddings_matrix,en_embeddings_matrix,fr_word2int,en_word2int,fr_filtered,en_filtered,args=get_nmt_data()
set_modelparams(args)
train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, learning_rate, dropout_probs, en_len, max_en_len, fr_len = model_inputs()
    encoding_embed_input, encoding_op, encoding_st, rnn_inputs, decoding_input,decoding_embed_input, logits_tr = seq2seq_model(
        tf.reverse(input_data, [-1]),
        targets,
        dropout_probs,
        fr_len,
        en_len,
        max_en_len,
        len(en_word2int) + 1,
        hidden_size,
        n_layers,
        en_word2int,
        batch_size,
        encoder_type,fr_embeddings_matrix,en_embeddings_matrix)

    logits_tr = tf.identity(logits_tr.rnn_output, 'logits_tr')
    seq_masks = tf.sequence_mask(en_len, max_en_len, dtype=tf.float32, name='masks')

    with tf.name_scope("optimizer"):
        tr_cost = sequence_loss(logits_tr,targets,seq_masks)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(tr_cost)
        train_op = optimizer.apply_gradients(gradients)
        tf.summary.scalar("cost", tr_cost)
print("Graph created.")

en_train = en_filtered[0:30000]
fr_train = fr_filtered[0:30000]
update_check = (len(fr_train) // batch_size // per_epoch) - 1
print("update_check:", update_check)
checkpoint = logs_path + 'best_so_far_model.ckpt'
with tf.Session(graph=train_graph) as sess:
    tf_summary_writer = tf.summary.FileWriter(logs_path, graph=train_graph)
    merged_summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for epoch_i in range(1, epochs + 1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (en_batch, fr_batch, en_text_len, fr_text_len) in enumerate(
                get_batches(en_train, fr_train, batch_size,fr_word2int,en_word2int)):
            before = time.time()

            encoding_embed_inputtf, encoding_optf, encoding_sttf, rnn_inputstf, decoding_inputtf,decoding_embed_inputtf, logits_trtf,_,gradientstf,loss,summary = sess.run(
                [encoding_embed_input, encoding_op, encoding_st, rnn_inputs, decoding_input, decoding_embed_input,logits_tr, train_op, gradients,tr_cost,merged_summary_op],
                {input_data: fr_batch,
                 targets: en_batch,
                 learning_rate: lr,
                 en_len: en_text_len,
                 fr_len: fr_text_len,
                 dropout_probs: dr_prob})

            print("batch:", batch_i, "encoding_embed_inputtf:", encoding_embed_inputtf)
            print("batch:", batch_i, ":lastch:", encoding_sttf)
            print("batch:", batch_i, ":decoding_inputtf:", decoding_inputtf)
            print("batch:", batch_i, ":decoding_embed_inputtf:", decoding_embed_inputtf)
            print("batch:", batch_i, ":decoding_embed_inputtf:", decoding_embed_inputtf.shape)
            print("batch:", batch_i, ":tr_logits:", logits_trtf)
            print("batch:", batch_i, ":tr_logits:", logits_trtf.shape)
            print("batch:", batch_i, ":loss:", loss)
            print("batch:", batch_i, ":merged_summary_op:", merged_summary_op)
            print("batch:", batch_i, ":gradients:", gradientstf)
            #saver = tf.train.Saver()
            #saver.save(sess, checkpoint)