# DEEP-Breathe

# Introduction
I feel that the barrier to entry for Deep Learning is very steep. Especially the models which are required for <strong>Natural Language Processing</strong>. <strong>Neural Machine Translation</strong> for example, uses concepts like <strong><a href="https://github.com/slowbreathing/Deep-Breathe/blob/master/org/mk/training/dl/rnn_cell.py">LSTM</a></strong>, <strong><a href="https://github.com/slowbreathing/Deep-Breathe/blob/master/org/mk/training/dl/rnn_cell.py">Bidirectional LSTM</a></strong>, <strong><a href="https://github.com/slowbreathing/Deep-Breathe/blob/master/org/mk/training/dl/rnn.py">Multi Layered LSTMs</a></strong>, <strong><a href="https://github.com/slowbreathing/Deep-Breathe/blob/master/org/mk/training/dl/attention.py">Attention</a></strong> etc.  Neither one of them is easy to understand by itself, imagine the plight of a student when these concepts are strung together for a <strong><a href="https://github.com/tensorflow/nmt">Neural Machine Translation</a></strong> or <strong><a href="https://github.com/google-research/bert">Google's BERT</a></strong> based systems. I have seen Neural Machine Translation Based systems grossly underperform and, it was simply because most of the hyperparameters were not understood at all. <strong><a href="https://github.com/slowbreathing/Deep-Breathe">Deep-Breathe</a></strong> is a complete and pure python implementation of these models, specially but not limited to <strong><a href="https://github.com/slowbreathing/Deep-Breathe/blob/master/org/mk/training/dl/rnn_cell.py">Neural Machine Translator</a></strong>. In Fact the scripts have been written to compare weights after a certain number of iterations between TF implementation and the Deep-Breathe one. Hopefully this will go a long way into lowering the barrier to entry. <strong><a href="https://github.com/slowbreathing/Deep-Breathe">Deep-Breathe</a></strong> is already being used internally by many organizations.

## Installation Requirements
1. tensorflow(only for comparisons)
2. python 3.5 or above
3. numpy.



mkdir resources/tmp/rnn_words
python org/mk/training/dl/LSTMAllByitself/LSTMMain.py input/belling_the_cat10.txt

python org/mk/training/dl/tfwordslstmsingle.py
python -m org.mk.training.dl.LSTMMainsingle


python -m org.mk.training.dl.tfwordslstm input/belling_the_cat10.txt
python -m org.mk.training.dl.LSTMMain input/belling_the_cat10.txt


python -m org.mk.training.dl.tfwordslstminitstate input/belling_the_cat10.txt
python -m org.mk.training.dl.LSTMMaininitstate input/belling_the_cat10.txt


python -m org.mk.training.dl.tfwordslstmmulti input/belling_the_cat10.txt
python -m org.mk.training.dl.LSTMMainmulti input/belling_the_cat10.txt


python -m org.mk.training.dl.tfwordslstmbi input/belling_the_cat10.txt
python -m org.mk.training.dl.LSTMMainbi input/belling_the_cat10.txt


python -m org.mk.training.dl.tfwordslstmbimulti input/belling_the_cat10.txt
python -m org.mk.training.dl.LSTMMainbimulti input/belling_the_cat10.txt


python -m org.mk.training.dl.tfmachinetranslation --encoder_type=uni --num_layers=1
python -m org.mk.training.dl.manualmachinetranslation --encoder_type=uni --num_layer=1


python -m org.mk.training.dl.tfmachinetranslation --encoder_type=uni --num_layers=2
python -m org.mk.training.dl.manualmachinetranslation --encoder_type=uni --num_layer=2


python -m org.mk.training.dl.tfmachinetranslation --num_layers=2
python -m org.mk.training.dl.manualmachinetranslation --num_layer=2


python -m org.mk.training.dl.tfmachinetranslation --num_layers=4
python -m org.mk.training.dl.manualmachinetranslation --num_layer=4




python -m org.mk.training.dl.tfmachinetranslationcurr --attention_architecture=standard --encoder_type=uni --num_layers=1 --batch_size=1 --src=input/NMT/train_fr_lines1.txt --tgt=input/NMT/train_en_lines1.txt
python -m org.mk.training.dl.manualmachinetranslationcurr --attention_architecture=standard --encoder_type=uni --num_layers=1 --batch_size=1 --src=input/NMT/train_fr_lines1.txt --tgt=input/NMT/train_en_lines1.txt

python -m org.mk.training.dl.tfmachinetranslationcurr
python -m org.mk.training.dl.manualmachinetranslationcurr
