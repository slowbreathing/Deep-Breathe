import pandas as pd
import tensorflow as tf
import numpy as np
import codecs
import time
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper, BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, sequence_loss
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
#np.set_printoptions(threshold=np.nan)

new_embedding_constants_dict={}
new_embedding_constants_dict["TOKEN_UNK"]=np.array([ 0.89184064, -0.29138342, -0.03583159, -0.42613918,  0.6836069,   0.47367492,
  0.53413385, -0.71677613,  0.43601456,  0.18015899,  0.8467093,  -0.7277722,
 -0.45845369,  0.17520632, -0.37654692, -0.34964642, -0.27047262, -0.7207279,
  0.34537598,  0.90624493,  0.90031606, -0.7217335,  -0.43304563,  0.61162627,
 -0.54297,    -0.83039856,  0.77826506,  0.47082764,  0.779062,    0.27859527,
  0.63007116,  0.06225961,  0.3334209,  -0.9162839,  -0.93299466, -0.17645347,
 -0.69076085,  0.8482602,   0.1875441,  -0.31012556, -0.00968188, -0.9798851,
 -0.9912039,  -0.28325993,  0.40350452,  0.64749193,  0.9345305,  -0.97786355,
 0.8711786,   0.16799776]
)
new_embedding_constants_dict["TOKEN_PAD"]=np.array([-0.81655216,  0.5625992,  -0.8553194,  -0.02212036,  0.4446796,   0.3162192,
  0.5735441,  -0.5139087,  -0.76085997,  0.849614,   -0.00583495,  0.5932988,
 -0.854923,   -0.76460993,  0.6792134,   0.6887459,  -0.18353513,  0.5156813,
 -0.07207575,  0.9257539,  -0.792035,    0.77723855,  0.25142267,  0.03241107,
  0.52482784,  0.52972853, -0.286012,    0.09252205, -0.31863344,  0.92613214,
  0.5293582,   0.02199265, -0.09801475, -0.75760937,  0.58405465,  0.23611522,
  0.6127986,   0.94654065, -0.24149975, -0.00815829, -0.28616875, -0.637963,
 -0.6477495,  -0.8772441,   0.07292482,  0.28938434, -0.9516554,   0.8114216,
  0.5765251,   0.8885126 ]

)
new_embedding_constants_dict["TOKEN_EOS"]=np.array([ 0.01430098, -0.10383016, -0.235747,   -0.02978121,  0.53175306, -0.12189005,
  0.8974192,  -0.6379926,   0.8266778,  -0.36756635,  0.9337619,   0.61115456,
  0.1842079,  -0.70881706,  0.40914172, -0.10583848, -0.01877687, -0.75412905,
 -0.04979828,  0.5283455,  -0.80578804,  0.9387867,   0.61484504, -0.40299845,
 -0.9809426,  -0.25743622,  0.09101433,  0.5243984,  -0.5380408,   0.76422733,
  0.9941627,  -0.6876849,  -0.7852932,  0.61294085, -0.28005806,  0.04399994,
 -0.22404692,  0.92541665, -0.6105466,   0.47965088,  0.5159493,   0.14322965,
 -0.40323815,  0.03752193,  0.95293653, -0.389435,    0.8182654,  -0.6117154,
  0.0060643,  -0.6624445 ]

)
new_embedding_constants_dict["TOKEN_GO"]=np.array([ 0.5840509,   0.33953574,  0.5874818,  -0.83531624,  0.75538135, -0.39617494,
 -0.6250137,  -0.07262408,  0.10313866,  0.40474573, -0.94559,     0.4659892,
 -0.9553411,  -0.42595065,  0.72498983,  0.06950572, -0.06518898,  0.8611347,
 -0.69269425, -0.05353237, -0.64062035,  0.90910137,  0.5812094,   0.67093456,
 -0.33927578, -0.72800404,  0.5102056,  -0.9633267,  -0.34817594,  0.46746257,
  0.18932728, -0.66804963,  0.58211786,  0.47953087, -0.631945,   -0.67186844,
  0.74663925, -0.3959075,   0.70035964, -0.7595935,   0.29630524,  0.54017925,
 -0.05252694,  0.22665581,  0.53666824, -0.8942621,  -0.8285967,  -0.0499638,
 -0.76515394, -0.13115136]
)


endata=[]
frdata=[]
with open('input/NMT/train_en_lines.txt') as enfile:
    for li in enfile:
        endata.append(li)
with open('input/NMT/train_fr_lines.txt') as frfile:
    for li in frfile:
        frdata.append(li)
mtdata = pd.DataFrame({'FR':frdata,'EN':endata})
mtdata['EN_len'] = mtdata['EN'].apply(lambda x: len(x.split(' ')))
mtdata['FR_len'] = mtdata['FR'].apply(lambda x: len(x.split(' ')))

print(mtdata['EN'].head(2).values)
print(mtdata['FR'].head(2).values)

mtdata_en = []
for en in mtdata.EN:
    mtdata_en.append(en)
mtdata_fr = []
for fr in mtdata.FR:
    mtdata_fr.append(fr)

def count_words(words_dict, text):
    for sentence in text:
        for word in sentence.split():
            #print("word:",word)
            if word not in words_dict:
                words_dict[word] = 1
            else:
                words_dict[word] += 1

word_counts_dict_en = {}
word_counts_dict_fr = {}

"""
word_counts_dict_en: {'This': 1, 'story': 1, 'exciting': 1, 'some': 2, 'sea.': 1, 'records': 1, 'to': 2,
'tell': 1, 'box': 1, 'from': 2, 'incredible': 1, "it's": 2, 'and': 1, 'though': 1, 'all': 1, 'got': 1,
 'you': 2, 'sea': 1, "We've": 1, 'Bill': 1, 'not': 2, 'stories': 1, "that's": 1, 'is': 2, 'been': 1,
 'going': 2, 'ever': 1, 'that': 1, 'Titanic': 2, 'most': 2, 'office': 1, 'matter': 1, 'Lange.': 1,
 'in': 1, 'any': 1, 'seen,': 1, 'Dave': 1, 'video': 1, '--': 2, 'video.': 1, 'it.': 1, 'Gallo.': 1,
  "I'm": 1, "we're": 2, 'The': 1, 'And': 1, 'here': 1, 'even': 1, 'show': 1, 'of': 5, 'the': 6, 'sorts': 1,
  'truth': 1, 'breaking': 1}
Total English words in Vocabulary 54
Total French words in Vocabulary: 52

"""
count_words(word_counts_dict_en, mtdata_en)
count_words(word_counts_dict_fr, mtdata_fr)
#print("word_counts_dict_en:",word_counts_dict_en)
#print("Total English words in Vocabulary", len(word_counts_dict_en))
#print("Total French words in Vocabulary:", len(word_counts_dict_fr))


def build_word_vector_matrix(vector_file):
    embedding_index = {}
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            sr = line.split()
            if(len(sr)<26):
                print("sr:",sr,":",len(sr))
                continue
            word = sr[0]
            #print("word:",word)
            embedding = np.asarray(sr[1:], dtype='float32')
            embedding_index[word] = embedding
    return embedding_index
embeddings_index = build_word_vector_matrix('resources/tmp/embeddings/glove.6B.50d.txt')

def build_word2id_mapping(word_counts_dict):
    word2int = {}
    count_threshold = 20
    value = 0
    sortedwords=list(word_counts_dict.items())
    sortedwords.sort()
    for word, count in sortedwords:
        #print("word, count:",word,":", count)
        if count >= count_threshold or word in embeddings_index:
            word2int[word] = value
            value += 1

    special_codes = ["TOKEN_UNK","TOKEN_PAD","TOKEN_EOS","TOKEN_GO"]
    for code in special_codes:
        word2int[code] = len(word2int)

    int2word = {}
    for word, value in word2int.items():
        int2word[value] = word
    return word2int,int2word

"""
Giving each word a unique int ID for represenation. This will be used to index into
embedding matrix.
En Index: {'to': 33, 'story': 28, 'exciting': 9, 'some': 25, 'been': 4, 'records': 22, 'ever': 8,
'that': 30, 'TOKEN_PAD': 38, 'most': 18, 'breaking': 6, 'incredible': 15, 'though': 32, 'TOKEN_UNK': 37,
'truth': 34, 'video': 35, 'in': 14, 'from': 10, 'not': 19, 'office': 21, '--': 0, 'box': 5, 'TOKEN_GO': 40,
'sea': 23, 'and': 2, 'sorts': 26, 'all': 1, 'tell': 29, 'got': 12, 'any': 3, 'you': 36, 'here': 13, 'even': 7,
 'is': 16, 'of': 20, 'the': 31, 'going': 11, 'show': 24, 'TOKEN_EOS': 39, 'matter': 17, 'stories': 27}
En Index(Reverse):{0: '--', 1: 'all', 2: 'and', 3: 'any', 4: 'been', 5: 'box', 6: 'breaking', 7: 'even', 8:
'ever', 9: 'exciting', 10: 'from', 11: 'going', 12: 'got', 13: 'here', 14: 'in', 15: 'incredible', 16:
'is', 17: 'matter', 18: 'most', 19: 'not', 20: 'of', 21: 'office', 22: 'records', 23: 'sea', 24: 'show', 25:
'some', 26: 'sorts', 27: 'stories', 28: 'story', 29: 'tell', 30: 'that', 31: 'the', 32: 'though', 33: 'to', 34:
'truth', 35: 'video', 36: 'you', 37: 'TOKEN_UNK', 38: 'TOKEN_PAD', 39: 'TOKEN_EOS', 40: 'TOKEN_GO'}
Fr Index: {'du': 6, 'de': 4, 'TOKEN_PAD': 33, 'vous': 30, 'suis': 27, 'records': 25, 'avons': 2, 'quelques': 24,
 'le': 14, 'et': 9, 'une': 29, 'vérité': 31, 'les': 15, 'la': 13, 'TOKEN_UNK': 32, 'TOKEN_EOS': 34, 'parmi': 20,
  'plus': 22, "l'histoire": 12, 'mer': 16, '--': 0, "s'il": 26, 'allons': 1, 'est': 8, 'que': 23, 'histoires': 10,
  'des': 5, 'toutes': 28, 'nous': 19, 'jamais': 11, 'TOKEN_GO': 35, 'pas': 21, 'continue': 3, "n'est": 18,
  'en': 7, 'même': 17}
Fr Index(Reverse):  : {0: '--', 1: 'allons', 2: 'avons', 3: 'continue', 4: 'de', 5: 'des', 6: 'du', 7: 'en',
8: 'est', 9: 'et', 10: 'histoires', 11: 'jamais', 12: "l'histoire", 13: 'la', 14: 'le', 15: 'les', 16: 'mer',
 17: 'même', 18: "n'est", 19: 'nous', 20: 'parmi', 21: 'pas', 22: 'plus', 23: 'que', 24: 'quelques',
 25: 'records', 26: "s'il", 27: 'suis', 28: 'toutes', 29: 'une', 30: 'vous', 31: 'vérité', 32: 'TOKEN_UNK',
  33: 'TOKEN_PAD', 34: 'TOKEN_EOS', 35: 'TOKEN_GO'}
Length of english word embeddings:  41
Length of french word embeddings:  36
"""
en_word2int,en_int2word = build_word2id_mapping(word_counts_dict_en)
fr_word2int,fr_int2word = build_word2id_mapping(word_counts_dict_fr)
#print("En Index:",en_word2int,":",en_int2word)
#print("Fr Index:",fr_word2int,":",fr_int2word)

def build_embeddings(word2int):
    embedding_dim = 50
    nwords = len(word2int)

    word_emb_matrix = np.zeros((nwords, embedding_dim), dtype=np.float32)
    for word, i in word2int.items():
        if word in embeddings_index:
            #print("EmbeddingWord:",word,":",i)
            word_emb_matrix[i] = embeddings_index[word]
        else:
            #print("NonEmbeddingWord:",word,":",i)
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            #new_embedding=new_embedding_constants_dict[word]

            word_emb_matrix[i] = new_embedding
    return word_emb_matrix

en_embeddings_matrix = build_embeddings(en_word2int)
#print("Length of english word embeddings: ", len(en_embeddings_matrix))
fr_embeddings_matrix = build_embeddings(fr_word2int)
#print("Length of french word embeddings: ", len(fr_embeddings_matrix))
"""Embedding for word 'some' [ 9.2871e-01 -1.0834e-01  2.1497e-01 -5.0237e-01  1.0379e-01  2.2728e-01
 -5.4198e-01 -2.9008e-01 -6.4607e-01  1.2664e-01 -4.1487e-01 -2.9343e-01
  3.6855e-01 -4.1733e-01  6.9116e-01  6.7341e-02  1.9715e-01 -3.0465e-02
 -2.1723e-01 -1.2238e+00  9.5469e-03  1.9594e-01  5.6595e-01 -6.7473e-02
  5.9208e-02 -1.3909e+00 -8.9275e-01 -1.3546e-01  1.6200e-01 -4.0210e-01
  4.1644e+00  3.7816e-01  1.5797e-01 -4.8892e-01  2.3131e-01  2.3258e-01
 -2.5314e-01 -1.9977e-01 -1.2258e-01  1.5620e-01 -3.1995e-01  3.8314e-01
  4.7266e-01  8.7700e-01  3.2223e-01  1.3292e-03 -4.9860e-01  5.5580e-01
 -7.0359e-01 -5.2693e-01]"""
print("Embedding for word 'some'",en_embeddings_matrix[en_word2int['some']])
print("Embedding for word 'Unknown'",fr_embeddings_matrix[fr_word2int['TOKEN_UNK']])
print("Embedding for word 'Go'",fr_embeddings_matrix[fr_word2int['TOKEN_GO']])
print("Embedding for word 'EOS'",fr_embeddings_matrix[fr_word2int['TOKEN_EOS']])
print("Embedding for word 'PAD'",fr_embeddings_matrix[fr_word2int['TOKEN_PAD']])
print("fr_embeddings_matrix:",fr_embeddings_matrix)

def convert_sentence_to_ids(text, word2int, eos=False):
    wordints = []
    word_count = 0
    for sentence in text:
        sentence2ints = []
        for word in sentence.split():
            word_count += 1
            if word in word2int:
                sentence2ints.append(word2int[word])
            else:
                sentence2ints.append(word2int["TOKEN_UNK"])
        if eos:
            sentence2ints.append(word2int["TOKEN_EOS"])
        wordints.append(sentence2ints)
    return wordints, word_count
id_en, word_count_en = convert_sentence_to_ids(mtdata_en, en_word2int, eos=True)
id_fr, word_count_fr = convert_sentence_to_ids(mtdata_fr, fr_word2int)
"""Sentence-to-ids(English): [[37, 16, 37, 37, 37, 37, 37, 39], [37, 37, 11, 33, 29, 36, 25, 27, 10, 31, 23, 13, 14, 37, 39],
[37, 12, 25, 20, 31, 18, 15, 35, 20, 37, 37, 8, 4, 37, 2, 37, 19, 11, 33, 24, 36, 3, 20, 37, 39],
[37, 34, 20, 31, 17, 16, 30, 31, 37, 0, 7, 32, 37, 6, 1, 26, 20, 5, 21, 22, 0, 37, 19, 31, 18, 9, 28, 10, 31, 37, 39]]

Sentence-to-ids(French): [[32, 32, 32, 32, 27, 32, 32], [32, 1, 30, 32, 24, 10, 4, 13, 16, 7, 32],
[32, 2, 5, 32, 6, 32, 20, 15, 22, 32, 11, 32, 9, 19, 32, 21, 30, 7, 32, 29, 32],
[32, 31, 8, 23, 14, 32, 0, 17, 26, 3, 4, 32, 28, 15, 25, 4, 32, 0, 18, 21, 12, 13, 22, 32]]
max_en_length(Sentence): 30  max_fr_length(Sentence): 24
"""
#print("Sentence-to-ids:",id_en)
#print("Sentence-to-ids:",id_fr)



def unknown_tokens(sentence, word2int):
    unk_token_count = 0
    for word in sentence:
        if word == word2int["TOKEN_UNK"]:
            unk_token_count += 1
    return unk_token_count

en_filtered = []
fr_filtered = []
max_en_length = int(mtdata.EN_len.max())
max_fr_length = int(mtdata.FR_len.max())
print("max_en_length:",max_en_length," max_fr_length:",max_fr_length)
min_length = 4
unknown_token_en_limit = 10
unknown_token_fr_limit = 10

for count,text in enumerate(id_en):
    unknown_token_en = unknown_tokens(id_en[count],en_word2int)
    unknown_token_fr = unknown_tokens(id_fr[count],fr_word2int)
    en_len = len(id_en[count])
    fr_len = len(id_fr[count])
    if( (unknown_token_en>unknown_token_en_limit) or (unknown_token_fr>unknown_token_fr_limit) or
       (en_len<min_length) or (fr_len<min_length) ):
        continue
    fr_filtered.append(id_fr[count])
    en_filtered.append(id_en[count])
#print("filtered french/english sentences: ", fr_filtered, en_filtered )
#print("Length of filtered french/english sentences: ", len(fr_filtered), len(en_filtered) )

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

def get_rnn_cell(rnn_cell_size,dropout_prob):
		rnn_c=rnn_cell.LayerNormBasicLSTMCell(rnn_cell_size, layer_norm=False)
		#rnn_c = GRUCell(rnn_cell_size)
		#rnn_c = DropoutWrapper(rnn_c, input_keep_prob = dropout_prob)
		return rnn_c

def encoding_layer(rnn_cell_size, sequence_len, n_layers, rnn_inputs, dropout_prob):
    for l in range(n_layers):
        with tf.variable_scope('encodings_l_{}'.format(l)):
            with variable_scope.variable_scope(
                    "other", initializer=init_ops.constant_initializer(0.1)) as vs:

                rnn_fw = rnn_cell.LayerNormBasicLSTMCell(rnn_cell_size, layer_norm=False)
                rnn_bw = rnn_cell.LayerNormBasicLSTMCell(rnn_cell_size, layer_norm=False)
                encoding_output, encoding_state = tf.nn.bidirectional_dynamic_rnn(rnn_fw, rnn_bw,
                                                                        rnn_inputs,
                                                                        sequence_len,
                                                                        dtype=tf.float32)
                encoding_output = tf.concat(encoding_output,2)
                return encoding_output, encoding_state,rnn_inputs

def training_decoding_layer(decoding_embed_input, en_len, decoding_cell, initial_state, op_layer,
                            v_size, max_en_len):
    helper = TrainingHelper(inputs=decoding_embed_input,sequence_length=en_len, time_major=False)
    dec = BasicDecoder(decoding_cell,helper,initial_state,op_layer)
    logits, _, _ = dynamic_decode(dec,output_time_major=False,impute_finished=True,
                                  maximum_iterations=max_en_len)
    return logits

def inference_decoding_layer(embeddings, start_token, end_token, decoding_cell, initial_state, op_layer,
                             max_en_len, batch_size):

    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    inf_helper = GreedyEmbeddingHelper(embeddings,start_tokens,end_token)
    inf_decoder = BasicDecoder(decoding_cell,inf_helper,initial_state,op_layer)
    inf_logits, _, _ = dynamic_decode(inf_decoder,output_time_major=False,impute_finished=True,
                                                            maximum_iterations=max_en_len)
    return inf_logits

def decoding_layer(decoding_embed_inp, embeddings, encoding_op, encoding_st, v_size, fr_len,
                   en_len,max_en_len, rnn_cell_size, word2int, dropout_prob, batch_size, n_layers):

    for l in range(n_layers):
        with tf.variable_scope('decs_rnn_layer_{}'.format(l)):
            #gru = tf.contrib.rnn.GRUCell(rnn_len)
            gru=get_rnn_cell(rnn_cell_size, dropout_prob)
            decoding_cell = tf.contrib.rnn.DropoutWrapper(gru,input_keep_prob = dropout_prob)
    out_l = Dense(v_size, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

    attention = BahdanauAttention(rnn_cell_size, encoding_op,fr_len,
                                                  normalize=False,
                                                  name='BahdanauAttention')
    decoding_cell =  AttentionWrapper(decoding_cell,attention,rnn_len)
    attention_zero_state = decoding_cell.zero_state(batch_size , tf.float32 )
    attention_zero_state = attention_zero_state.clone(cell_state = encoding_st[0])
    with tf.variable_scope("decoding_layer"):
        logits_tr = training_decoding_layer(decoding_embed_inp,
                                                  en_len,
                                                  decoding_cell,
                                                  attention_zero_state,
                                                  out_l,
                                                  v_size,
                                                  max_en_len)
    with tf.variable_scope("decoding_layer", reuse=True):
        logits_inf = inference_decoding_layer(embeddings,
                                                    word2int["TOKEN_GO"],
                                                    word2int["TOKEN_EOS"],
                                                    decoding_cell,
                                                    attention_zero_state,
                                                    out_l,
                                                    max_en_len,
                                                    batch_size)

    return logits_tr, logits_inf

def seq2seq_model(input_data, target_en_data, dropout_prob, fr_len, en_len, max_en_len,
                  v_size, rnn_cell_size, n_layers, word2int_en, batch_size):

    input_word_embeddings = tf.Variable(fr_embeddings_matrix, name="input_word_embeddings")
    encoding_embed_input = tf.nn.embedding_lookup(input_word_embeddings, input_data)
    encoding_op, encoding_st,rnn_inputs = encoding_layer(rnn_cell_size, fr_len, n_layers, encoding_embed_input, dropout_prob)

    decoding_input = process_encoding_input(target_en_data, word2int_en, batch_size)
    decoding_embed_input = tf.nn.embedding_lookup(en_embeddings_matrix, decoding_input)

    tr_logits, inf_logits  = decoding_layer(decoding_embed_input,
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

    return tr_logits, inf_logits, encoding_embed_input,encoding_op,encoding_st,rnn_inputs

def pad_sentences(sentences_batch,word2int):
    max_sentence = max([len(sentence) for sentence in sentences_batch])
    return [sentence + [word2int['TOKEN_PAD']] * (max_sentence - len(sentence)) for sentence in sentences_batch]

def get_batches(en_text, fr_text, batch_size):
    for batch_idx in range(0, len(fr_text)//batch_size):
        start_idx = batch_idx * batch_size
        en_batch = en_text[start_idx:start_idx + batch_size]
        fr_batch = fr_text[start_idx:start_idx + batch_size]
        pad_en_batch = np.array(pad_sentences(en_batch, en_word2int))
        pad_fr_batch = np.array(pad_sentences(fr_batch,fr_word2int))

        pad_en_lens = []
        for en_b in pad_en_batch:
            pad_en_lens.append(len(en_b))

        pad_fr_lens = []
        for fr_b in pad_fr_batch:
            pad_fr_lens.append(len(fr_b))
        """
        print("pad_en_batch:",pad_en_batch)
        print("pad_en_lens:",pad_en_lens)
        print("pad_fr_batch:",pad_fr_batch)
        print("pad_fr_lens:",pad_fr_lens)
        """
        yield pad_en_batch, pad_fr_batch, pad_en_lens, pad_fr_lens

epochs = 1
batch_size = 64
#batch_size = 2
rnn_len = 5
n_layers = 2
lr = 0.005
dr_prob = 0.75
logs_path='resources/tmp/nmt-models/org/'

train_graph = tf.Graph()
with train_graph.as_default():

    input_data, targets, learning_rate, dropout_probs, en_len, max_en_len, fr_len = model_inputs()

    logits_tr, logits_inf,encoding_embed_input,encoding_op,encoding_st,rnn_inputs = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets,
                                                      dropout_probs,
                                                      fr_len,
                                                      en_len,
                                                      max_en_len,
                                                      len(en_word2int)+1,
                                                      rnn_len,
                                                      n_layers,
                                                      en_word2int,
                                                      batch_size)

    logits_tr = tf.identity(logits_tr.rnn_output, 'logits_tr')
    logits_inf = tf.identity(logits_inf.sample_id, name='predictions')

    seq_masks = tf.sequence_mask(en_len, max_en_len, dtype=tf.float32, name='masks')

    with tf.name_scope("optimizer"):
        tr_cost = sequence_loss(logits_tr,targets,seq_masks)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(tr_cost)
        capped_gradients = [(tf.clip_by_value(gradient, -5., 5.), var) for gradient, var in gradients
                        if gradient is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
    tf.summary.scalar("cost", tr_cost)
print("Graph created.")


min_learning_rate = 0.0006
display_step = 20
stop_early_count = 0
stop_early_max_count = 3
per_epoch = 1


update_loss = 0
batch_loss = 0
summary_update_loss = []

en_train = en_filtered[0:30000]
fr_train = fr_filtered[0:30000]
update_check = (len(fr_train)//batch_size//per_epoch)-1
print("update_check:",update_check)
checkpoint = logs_path + 'best_so_far_model.ckpt'
with tf.Session(graph=train_graph) as sess:
    tf_summary_writer = tf.summary.FileWriter(logs_path, graph=train_graph)
    merged_summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (en_batch, fr_batch, en_text_len, fr_text_len) in enumerate(
                get_batches(en_train, fr_train, batch_size)):
            before = time.time()
            encoding_embed_inputtf,encoding_optf,encoding_sttf,rnn_inputstf,_,loss,summary = sess.run(
                [encoding_embed_input,encoding_op,encoding_st,rnn_inputs,train_op, tr_cost,merged_summary_op],
                {input_data: fr_batch,
                 targets: en_batch,
                 learning_rate: lr,
                 en_len: en_text_len,
                 fr_len: fr_text_len,
                 dropout_probs: dr_prob})
            """
            print("batch:",batch_i,"encoding_embed_inputtf:",encoding_embed_inputtf)
            print("batch:",batch_i,":h:",encoding_optf)
            print("batch:",batch_i,":lastch:",encoding_sttf)
            print("batch:",batch_i,"rnn_inputstf:",rnn_inputstf)
            print("batch:",batch_i,"rnn_inputstfshape:",np.shape(rnn_inputstf))
            #print("batch:",batch_i,":",tf.shape(encoding_optf),tf.shape(encoding_optf))"""
            batch_loss += loss
            update_loss += loss
            after = time.time()
            batch_time = after - before
            tf_summary_writer.add_summary(summary, epoch_i * batch_size + batch_i)
            if batch_i % display_step == 0 and batch_i > 0:
                print('** Epoch {:>3}/{} Batch {:>4}/{} - Batch Loss: {:>6.3f}, seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(fr_filtered) // batch_size,
                              batch_loss / display_step,
                              batch_time*display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss:", round(update_loss/update_check,3))
                summary_update_loss.append(update_loss)

                if update_loss <= min(summary_update_loss):
                    print('Saving model')
                    stop_early_count = 0
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early_count += 1
                    if stop_early_count == stop_early_max_count:
                        break
                update_loss = 0

        if stop_early_count == stop_early_max_count:
            print("Stopping Training.")
            break
