#Karpathy's heavily redesigned
./DL/RNN/manualRNNkarpathy.py input/alice.txt (Not changed at all)
./DL/RNN/manualRNNreversekarpathy.py input/input.txt
./DL/RNN/RNNMain.py input/input.txt

#tensorflow implementation reverse engineered
./DL/RNN/tfmanualRNNDataGen.py
./DL/RNN/tfmanualRNNBatchDataGen.py
./DL/RNN/tfmanualRNNEpochBatchDataGen.py
./DL/RNN/tfreversemanualRNN.py
./DL/RNN/RNNMain2.py

# Change the below value 
# num_steps = 1 # number of truncated backprop steps ('n' in the discussion above)
# state_size = 4
# num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
# state_size = 4
# num_steps = 10 # number of truncated backprop steps ('n' in the discussion above)
# state_size = 16
./DL/RNN/sequenceCatchertf.py(Not changed at all)
./DL/RNN/sequenceCatcherManual.py

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


