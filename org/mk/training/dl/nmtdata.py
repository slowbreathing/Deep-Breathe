from org.mk.training.dl.nmt import parse_arguments
import pandas as pd
import numpy as np
import codecs


special_codes = ["TOKEN_UNK", "TOKEN_PAD", "TOKEN_EOS", "TOKEN_GO"]
new_embedding_constants_dict = {}
new_embedding_constants_dict["TOKEN_UNK"] = np.array(
    [0.89184064, -0.29138342, -0.03583159, -0.42613918, 0.6836069, 0.47367492,
     0.53413385, -0.71677613, 0.43601456, 0.18015899, 0.8467093, -0.7277722,
     -0.45845369, 0.17520632, -0.37654692, -0.34964642, -0.27047262, -0.7207279,
     0.34537598, 0.90624493, 0.90031606, -0.7217335, -0.43304563, 0.61162627,
     -0.54297, -0.83039856, 0.77826506, 0.47082764, 0.779062, 0.27859527,
     0.63007116, 0.06225961, 0.3334209, -0.9162839, -0.93299466, -0.17645347,
     -0.69076085, 0.8482602, 0.1875441, -0.31012556, -0.00968188, -0.9798851,
     -0.9912039, -0.28325993, 0.40350452, 0.64749193, 0.9345305, -0.97786355,
     0.8711786, 0.16799776]
    )
new_embedding_constants_dict["TOKEN_PAD"] = np.array(
    [-0.81655216, 0.5625992, -0.8553194, -0.02212036, 0.4446796, 0.3162192,
     0.5735441, -0.5139087, -0.76085997, 0.849614, -0.00583495, 0.5932988,
     -0.854923, -0.76460993, 0.6792134, 0.6887459, -0.18353513, 0.5156813,
     -0.07207575, 0.9257539, -0.792035, 0.77723855, 0.25142267, 0.03241107,
     0.52482784, 0.52972853, -0.286012, 0.09252205, -0.31863344, 0.92613214,
     0.5293582, 0.02199265, -0.09801475, -0.75760937, 0.58405465, 0.23611522,
     0.6127986, 0.94654065, -0.24149975, -0.00815829, -0.28616875, -0.637963,
     -0.6477495, -0.8772441, 0.07292482, 0.28938434, -0.9516554, 0.8114216,
     0.5765251, 0.8885126]

    )
new_embedding_constants_dict["TOKEN_EOS"] = np.array(
    [0.01430098, -0.10383016, -0.235747, -0.02978121, 0.53175306, -0.12189005,
     0.8974192, -0.6379926, 0.8266778, -0.36756635, 0.9337619, 0.61115456,
     0.1842079, -0.70881706, 0.40914172, -0.10583848, -0.01877687, -0.75412905,
     -0.04979828, 0.5283455, -0.80578804, 0.9387867, 0.61484504, -0.40299845,
     -0.9809426, -0.25743622, 0.09101433, 0.5243984, -0.5380408, 0.76422733,
     0.9941627, -0.6876849, -0.7852932, 0.61294085, -0.28005806, 0.04399994,
     -0.22404692, 0.92541665, -0.6105466, 0.47965088, 0.5159493, 0.14322965,
     -0.40323815, 0.03752193, 0.95293653, -0.389435, 0.8182654, -0.6117154,
     0.0060643, -0.6624445]

    )
new_embedding_constants_dict["TOKEN_GO"] = np.array(
    [0.5840509, 0.33953574, 0.5874818, -0.83531624, 0.75538135, -0.39617494,
     -0.6250137, -0.07262408, 0.10313866, 0.40474573, -0.94559, 0.4659892,
     -0.9553411, -0.42595065, 0.72498983, 0.06950572, -0.06518898, 0.8611347,
     -0.69269425, -0.05353237, -0.64062035, 0.90910137, 0.5812094, 0.67093456,
     -0.33927578, -0.72800404, 0.5102056, -0.9633267, -0.34817594, 0.46746257,
     0.18932728, -0.66804963, 0.58211786, 0.47953087, -0.631945, -0.67186844,
     0.74663925, -0.3959075, 0.70035964, -0.7595935, 0.29630524, 0.54017925,
     -0.05252694, 0.22665581, 0.53666824, -0.8942621, -0.8285967, -0.0499638,
     -0.76515394, -0.13115136]
    )

def open_file(file):
    data=[]
    with open(file) as datafile:
        for sent in datafile:
            data.append(sent)
        return data

def count_words(words_dict, text):
    for sentence in text:
        for word in sentence.split():
            # print("word:",word)
            if word not in words_dict:
                words_dict[word] = 1
            else:
                words_dict[word] += 1

def frame_to_list(fr):
    sentlist=[]
    for en in fr:
        sentlist.append(en)
    return sentlist

def build_word2id_mapping(word_counts_dict,embeddings_index):
    word2int = {}
    count_threshold = 20
    value = 0
    sortedwords = list(word_counts_dict.items())
    sortedwords.sort()
    for word, count in sortedwords:
        # print("word, count:",word,":", count)
        if count >= count_threshold or word in embeddings_index:
            word2int[word] = value
            value += 1

    for code in special_codes:
        word2int[code] = len(word2int)

    int2word = {}
    for word, value in word2int.items():
        int2word[value] = word
    return word2int, int2word

def build_word_vector_matrix(vector_file):
    embedding_index = {}
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            sr = line.split()
            if (len(sr) < 26):
                print("sr:", sr, ":", len(sr))
                continue
            word = sr[0]
            # print("word:",word)
            embedding = np.asarray(sr[1:], dtype='float32')
            embedding_index[word] = embedding
    return embedding_index

def build_embeddings(word2int,embeddings_index):
    embedding_dim = 50
    nwords = len(word2int)

    word_emb_matrix = np.zeros((nwords, embedding_dim), dtype=np.float32)
    for word, i in word2int.items():
        if word in embeddings_index:
            # print("EmbeddingWord:",word,":",i)
            word_emb_matrix[i] = embeddings_index[word]
        else:
            print("NonEmbeddingWord:", word, ":", i)
            # new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            new_embedding = new_embedding_constants_dict[word]
            word_emb_matrix[i] = new_embedding
    return word_emb_matrix

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

def unknown_tokens(sentence, word2int):
    unk_token_count = 0
    for word in sentence:
        if word == word2int["TOKEN_UNK"]:
            unk_token_count += 1
    return unk_token_count


def get_nmt_data():
    args=parse_arguments()
    frdata=open_file(args.src)
    endata=open_file(args.tgt)
    mtdata = pd.DataFrame({'FR': frdata, 'EN': endata})
    mtdata['FR_len'] = mtdata['FR'].apply(lambda x: len(x.split(' ')))
    mtdata['EN_len'] = mtdata['EN'].apply(lambda x: len(x.split(' ')))

    print("English Sentence:",mtdata['EN'].head(2).values)
    print("French Sentence:",mtdata['FR'].head(2).values)

    #list of sentences
    sents_fr=frame_to_list(mtdata.FR)
    sents_en=frame_to_list(mtdata.EN)

    #dict of words and count
    word_counts_dict_fr = {}
    word_counts_dict_en = {}

    """
    Count by word. {'is': 1, 'And': 1, 'Dave': 1, 'stories': 1, 'here': 1, 'going': 1, 'Gallo.': 1, "I'm": 1,
    'sea': 1, 'in': 1, 'the': 1, 'video.': 1, 'Lange.': 1, 'to': 1, 'Bill': 1, 'tell': 1, 'from': 1, 'you': 1,
    'some': 1, 'This': 1, "we're": 1}
    """
    count_words(word_counts_dict_fr, sents_fr)
    count_words(word_counts_dict_en, sents_en)

    print("Total French words in Vocabulary:", len(word_counts_dict_fr))
    print("Total English words in Vocabulary", len(word_counts_dict_en))

    embeddings_index = build_word_vector_matrix(args.vocab)
    """
    Giving each word a unique int ID for represenation. This will be used to index into
    embedding matrix.
    En Index: {'is': 4, 'TOKEN_UNK': 12, 'in': 3, 'here': 2, 'going': 1,
    'TOKEN_EOS': 14, 'sea': 5, 'the': 9, 'TOKEN_PAD': 13, 'to': 10, 'TOKEN_GO': 15, 'tell': 8,
    'from': 0, 'you': 11, 'some': 6, 'stories': 7}
    En Reverse Index: {0: 'from', 1: 'going', 2: 'here', 3: 'in', 4: 'is', 5: 'sea', 6: 'some'
    , 7: 'stories', 8: 'tell', 9: 'the', 10: 'to', 11: 'you', 12: 'TOKEN_UNK', 13: 'TOKEN_PAD',
    14: 'TOKEN_EOS', 15: 'TOKEN_GO'}
    """
    fr_word2int, fr_int2word = build_word2id_mapping(word_counts_dict_fr,embeddings_index)
    en_word2int, en_int2word = build_word2id_mapping(word_counts_dict_en,embeddings_index)
    print("Fr INDEX:", fr_word2int, ":", fr_int2word)
    print("En INDEX:", en_word2int, ":", en_int2word)

    fr_embeddings_matrix = build_embeddings(fr_word2int,embeddings_index)
    print("Length of french word embeddings: ", len(fr_embeddings_matrix))
    en_embeddings_matrix = build_embeddings(en_word2int,embeddings_index)
    print("Length of english word embeddings: ", len(en_embeddings_matrix))
    """Embedding for word 'some' [ 9.2871e-01 -1.0834e-01  2.1497e-01 -5.0237e-01  1.0379e-01  2.2728e-01
     -5.4198e-01 -2.9008e-01 -6.4607e-01  1.2664e-01 -4.1487e-01 -2.9343e-01
      3.6855e-01 -4.1733e-01  6.9116e-01  6.7341e-02  1.9715e-01 -3.0465e-02
     -2.1723e-01 -1.2238e+00  9.5469e-03  1.9594e-01  5.6595e-01 -6.7473e-02
      5.9208e-02 -1.3909e+00 -8.9275e-01 -1.3546e-01  1.6200e-01 -4.0210e-01
      4.1644e+00  3.7816e-01  1.5797e-01 -4.8892e-01  2.3131e-01  2.3258e-01
     -2.5314e-01 -1.9977e-01 -1.2258e-01  1.5620e-01 -3.1995e-01  3.8314e-01
      4.7266e-01  8.7700e-01  3.2223e-01  1.3292e-03 -4.9860e-01  5.5580e-01
     -7.0359e-01 -5.2693e-01]"""
    print("Embedding for word 'some'", en_embeddings_matrix[en_word2int['some']])
    print("Embedding for word 'Unknown'", fr_embeddings_matrix[fr_word2int['TOKEN_UNK']])
    print("Embedding for word 'Go'", fr_embeddings_matrix[fr_word2int['TOKEN_GO']])
    print("Embedding for word 'EOS'", fr_embeddings_matrix[fr_word2int['TOKEN_EOS']])
    print("Embedding for word 'PAD'", fr_embeddings_matrix[fr_word2int['TOKEN_PAD']])

    id_fr, word_count_fr = convert_sentence_to_ids(sents_fr, fr_word2int)
    id_en, word_count_en = convert_sentence_to_ids(sents_en, en_word2int, eos=True)
    """English Sentence:And we're going to tell you some stories from the sea here in video.
       English Sentence in Ids:[12, 12, 1, 10, 8, 11, 6, 7, 0, 9, 5, 2, 3, 12, 14]
    """
    print("Sentence-to-ids(FR):", id_fr, " word_count_fr:",word_count_fr)
    print("Sentence-to-ids(EN):", id_en, " word_count_en:",word_count_en)

    en_filtered = []
    fr_filtered = []
    max_en_length = int(mtdata.EN_len.max())
    max_fr_length = int(mtdata.FR_len.max())
    print("max_en_length:", max_en_length, " max_fr_length:", max_fr_length)
    min_length = 0
    unknown_token_en_limit = 10
    unknown_token_fr_limit = 10

    for count, text in enumerate(id_en):
        unknown_token_en = unknown_tokens(id_en[count], en_word2int)
        unknown_token_fr = unknown_tokens(id_fr[count], fr_word2int)
        en_len = len(id_en[count])
        fr_len = len(id_fr[count])
        if ((unknown_token_en > unknown_token_en_limit) or (unknown_token_fr > unknown_token_fr_limit) or
                (en_len < min_length) or (fr_len < min_length)):
            continue
        fr_filtered.append(id_fr[count])
        en_filtered.append(id_en[count])
    print("filtered french/english sentences: ", fr_filtered, en_filtered)
    print("Length of filtered french/english sentences: ", len(fr_filtered), len(en_filtered))
    return fr_embeddings_matrix,en_embeddings_matrix,fr_word2int,en_word2int,fr_filtered,en_filtered,args

if __name__== "__main__":
    get_nmt_data()