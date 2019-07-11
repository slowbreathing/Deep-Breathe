import pandas as pd
import numpy as np
import codecs
import time
from org.mk.training.dl.rnn import bidirectional_dynamic_rnn
from org.mk.training.dl.rnn import dynamic_rnn
from org.mk.training.dl.rnn import MultiRNNCell
from org.mk.training.dl.nn import embedding_lookup
from org.mk.training.dl.nn import TrainableVariable
from org.mk.training.dl.rnn_cell import LSTMCell
from org.mk.training.dl.core import Dense
from org.mk.training.dl.seq2seq import sequence_loss
from org.mk.training.dl.seq2seq import TrainingHelper, BasicDecoder, dynamic_decode
from org.mk.training.dl.attention import LuongAttention,AttentionWrapper

from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl import init_ops
from org.mk.training.dl.optimizer import BatchGradientDescent
from org.mk.training.dl.nmt import print_gradients
from org.mk.training.dl.common import make_mask

from org.mk.training.dl.nmt import parse_arguments
from org.mk.training.dl.nmtdata import get_nmt_data
import org

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

def process_encoding_input(target_data, word2int, batch_size):
    print("target_data:", target_data)
    print("batch_size:", batch_size)
    decoding_input = np.concatenate((np.full((batch_size, 1), word2int['TOKEN_GO']), target_data[:, :-1]), 1)
    print("decoding_input:", decoding_input)
    return decoding_input


def get_rnn_cell(rnn_cell_size, dropout_prob,n_layers,debug):
    rnn_cell=None
    print("n_layers:",n_layers)
    if(n_layers==1):

        with WeightsInitializer(initializer=init_ops.Constant(0.1)) as vs:
            rnn_cell = LSTMCell(rnn_cell_size,debug=debug)
    else:
        cell_list=[]
        for i in range(n_layers):
            with WeightsInitializer(initializer=init_ops.Constant(0.1)) as vs:
                cell_list.append(LSTMCell(rnn_cell_size,debug=debug))
        rnn_cell=MultiRNNCell(cell_list)
    return rnn_cell


def encoding_layer(rnn_cell_size, sequence_len, n_layers, rnn_inputs, dropout_prob):
    if(encoder_type=="bi" and n_layers%2 == 0):
        n_bi_layer=int(n_layers/2)
        encoding_output, encoding_state=bidirectional_dynamic_rnn(get_rnn_cell(rnn_cell_size, dr_prob,n_bi_layer,debug),get_rnn_cell(rnn_cell_size, dr_prob,n_bi_layer,debug), rnn_inputs)
        print("encoding_state:",encoding_state)
        if(n_bi_layer > 1):
            #layers/2
            """
            Forward-First(0)
            ((LSTMStateTuple({'c': array([[0.30450274, 0.30450274, 0.30450274, 0.30450274, 0.30450274]]),
              'h': array([[0.16661529, 0.16661529, 0.16661529, 0.16661529, 0.16661529]])}),
            Forward-Second(1)
             LSTMStateTuple({'c': array([[0.27710986, 0.07844026, 0.18714019, 0.28426586, 0.28426586]]),
              'h': array([[0.15019765, 0.04329417, 0.10251247, 0.1539225 , 0.1539225 ]])})),
            Backward-First(0)
            (LSTMStateTuple({'c': array([[0.30499766, 0.30499766, 0.30499766, 0.30499766, 0.30499766]]),
              'h': array([[0.16688152, 0.16688152, 0.16688152, 0.16688152, 0.16688152]])}),
            Backward-Second(1)
            LSTMStateTuple({'c': array([[0.25328871, 0.17537864, 0.21700339, 0.25627687, 0.25627687]]),
              'h': array([[0.13779658, 0.09631104, 0.11861721, 0.1393639 , 0.1393639 ]])})))
            """
            encoder_state = []
            for layer_id in range(n_bi_layer):
                encoder_state.append(encoding_state[0][layer_id])  # forward
                encoder_state.append(encoding_state[1][layer_id])  # backward
            encoding_state = tuple(encoder_state)
            """
            First(0)
            ((LSTMStateTuple({'c': array([[0.30450274, 0.30450274, 0.30450274, 0.30450274, 0.30450274]]),
               'h': array([[0.16661529, 0.16661529, 0.16661529, 0.16661529, 0.16661529]])}),
            Second(1)
            LSTMStateTuple({'c': array([[0.30499766, 0.30499766, 0.30499766, 0.30499766, 0.30499766]]),
               'h': array([[0.16688152, 0.16688152, 0.16688152, 0.16688152, 0.16688152]])})),
            Third(2)
            (LSTMStateTuple({'c': array([[0.27710986, 0.07844026, 0.18714019, 0.28426586, 0.28426586]]),
               'h': array([[0.15019765, 0.04329417, 0.10251247, 0.1539225 , 0.1539225 ]])}),
            Fourth(3)
            LSTMStateTuple({'c': array([[0.25328871, 0.17537864, 0.21700339, 0.25627687, 0.25627687]]),
               'h': array([[0.13779658, 0.09631104, 0.11861721, 0.1393639 , 0.1393639 ]])})))
            """
    else:
        encoding_output, encoding_state=dynamic_rnn(get_rnn_cell(rnn_cell_size, dr_prob,n_layers,debug), rnn_inputs)
    return encoding_output, encoding_state

def create_attention(decoding_cell,encoding_op,encoding_st,fr_len):

    if(args.attention_option is "Luong"):
        with WeightsInitializer(initializer=init_ops.Constant(0.1)) as vs:
            attention_mechanism = LuongAttention(hidden_size, encoding_op, fr_len)
            decoding_cell =  AttentionWrapper(decoding_cell,attention_mechanism,hidden_size)
        attention_zero_state = decoding_cell.zero_state(batch_size)
        attention_zero_state = attention_zero_state.clone(cell_state = encoding_st)
        print("attentionstate0:",attention_zero_state)
        return decoding_cell,attention_zero_state


def training_decoding_layer(decoding_embed_input, en_len, decoding_cell, encoding_op, encoding_st, op_layer,
                            v_size, fr_len, max_en_len):

    if (args.attention_architecture is not None):

        decoding_cell,encoding_st=create_attention(decoding_cell,encoding_op,encoding_st,fr_len)
    helper = TrainingHelper(inputs=decoding_embed_input, sequence_length=en_len, time_major=False)
    dec = BasicDecoder(decoding_cell, helper, encoding_st, op_layer)
    logits, _= dynamic_decode(dec, output_time_major=False, impute_finished=True,
                                      maximum_iterations=max_en_len)
    return logits

def decoding_layer(decoding_embed_inp, embeddings, encoding_op, encoding_st, v_size, fr_len,
                   en_len, max_en_len, rnn_cell_size, word2int, dropout_prob, batch_size, n_layers):

    out_l = Dense(len(en_word2int) + 1,kernel_initializer=init_ops.Constant(init))
    logits_tr = training_decoding_layer(decoding_embed_inp,
                                            en_len,
                                            get_rnn_cell(rnn_cell_size, dr_prob,n_layers,debug),
                                            encoding_op,
                                            encoding_st,
                                            out_l,
                                            v_size,
                                            fr_len,
                                            max_en_len)

    return logits_tr


def seq2seq_model(input_data, target_en_data, dropout_prob, fr_len, en_len, max_en_len,
                  v_size, rnn_cell_size, n_layers, word2int_en, batch_size):

    #print("LookupTable.getInstance:")
    lt_input=TrainableVariable.getInstance("input_word_embedding",fr_embeddings_matrix)
    encoding_embed_input = embedding_lookup(lt_input, input_data)
    #encoding_embed_input = embedding_lookup(fr_embeddings_matrix, input_data)
    #print("encoding_embed_input:",encoding_embed_input,encoding_embed_input.shape)
    encoding_op, encoding_st = encoding_layer(rnn_cell_size, fr_len, n_layers, encoding_embed_input,
                                              dropout_prob)
    print("encoding_st:",encoding_st,type(encoding_st))
    decoding_input = process_encoding_input(target_en_data, word2int_en, batch_size)
    decoding_embed_input = embedding_lookup(en_embeddings_matrix, decoding_input)
    #print("decoding_embed_input:",decoding_embed_input)
    #print("decoding_embed_input:",decoding_embed_input)
    tr_logits = decoding_layer(decoding_embed_input,
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


    return encoding_op, encoding_st, tr_logits


def pad_sentences(sentences_batch, word2int):
    max_sentence = max([len(sentence) for sentence in sentences_batch])
    return [sentence + [word2int['TOKEN_PAD']] * (max_sentence - len(sentence)) for sentence in sentences_batch]


def get_batches(en_text, fr_text, batch_size):
    #for batch_idx in range(0, 1):
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
n_bi_layer=0
lr = 0.0
dr_prob = 0.75
encoder_type=None
display_steps=0
projectdir="nmt_custom"


min_learning_rate = 0.0006
display_step = 20
stop_early_count = 0
stop_early_max_count = 3
per_epoch = 1
debug=False
display_steps=0
update_loss = 0
batch_loss = 0
summary_update_loss = []


rnn_fw=None
rnn_bw = None
decoding_cell = None

def set_modelparams(args):
    global epochs,n_layers,encoder_type,hidden_size,batch_size,lr,rnn_fw,rnn_bw,decoding_cell,gdo,n_bi_layer,debug,per_epoch,logs_path,display_steps
    epochs=args.epochs
    n_layers=args.num_layers
    encoder_type=args.encoder_type
    hidden_size=args.num_units
    batch_size = args.batch_size
    lr = args.learning_rate
    debug=args.debug
    per_epoch=args.per_epoch
    logs_path=args.out_dir
    display_steps=args.display_steps

fr_embeddings_matrix,en_embeddings_matrix,fr_word2int,en_word2int,fr_filtered,en_filtered,args=get_nmt_data()
set_modelparams(args)

en_train = en_filtered[0:30000]
fr_train = fr_filtered[0:30000]
update_check = (len(fr_train) // batch_size // per_epoch) - 1
#out_l = Dense(len(en_word2int) + 1,kernel_initializer=init_ops.Constant(init))
for epoch_i in range(1, epochs + 1):
    update_loss = 0
    batch_loss = 0
    for batch_i, (en_batch, fr_batch, en_text_len, fr_text_len) in enumerate(
            get_batches(en_train, fr_train, batch_size)):
        before = time.time()
        encoding_optf, encoding_sttf ,logits_tr= seq2seq_model(fr_batch[:, ::-1], en_batch, dr_prob, fr_text_len, en_text_len,
                                                     np.amax(en_text_len),
                                                     len(en_word2int) + 1
                                                     , hidden_size, n_layers, en_word2int, batch_size);

        #print("batch:", batch_i, "decoding:logits:", logits_tr)
        yhat,loss=sequence_loss(logits_tr.rnn_output,en_batch,make_mask(en_batch))
        print("loss:",loss)
        gdo=BatchGradientDescent(lr)
        gradients=gdo.compute_gradients(yhat,en_batch)
        gdo.apply_gradients(gradients)
        #print_gradients(gradients)
        batch_loss += loss
        update_loss += loss
        after = time.time()
        batch_time = after - before
        if batch_i % display_steps == 0 and batch_i > 0:
            print('** Epoch {:>3}/{} Batch {:>4}/{} - Batch Loss: {:>6.3f}, seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs,
                          batch_i,
                          len(fr_filtered) // batch_size,
                          batch_loss / display_steps,
                          batch_time*display_steps))
            batch_loss = 0

        if batch_i % update_check == 0 and batch_i > 0:
            print("Average loss:", round(update_loss/update_check,3))
            summary_update_loss.append(update_loss)

            if update_loss <= min(summary_update_loss):
                print('Saving model')
                stop_early_count = 0

            else:
                print("No Improvement.")
                stop_early_count += 1
                if stop_early_count == stop_early_max_count:
                    break
            update_loss = 0
