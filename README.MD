# DEEP-Breathe

# Introduction
I feel that the barrier to entry for Deep Learning is very steep. Especially the models which are required for Natural Language Processing. Neural Machine Translation for example, uses concepts like LSTM, Bidirectional LSTM, Multi Layered LSTMs, Attention etc. Neither one of them is easy to understand by itself, imagine the plight of a student when these concepts are strung together for a Neural Machine Translation Based systems. I have seen Neural Machine Translation Based systems grossly underperform and, it was simply because most of the hyperparameters were not understood at all. Deep-Breathe is a complete and pure python implementation of these models. In Fact the scripts have been written to compare weights after a certain number of iterations between TF implementation and the Deep-Breathe one. Hopefully this will go a long way into lowering the barrier to entry


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
