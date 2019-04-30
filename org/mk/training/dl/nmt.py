import argparse
import org
from org.mk.training.dl.rnn import print_grad
from org.mk.training.dl.rnn import print_RNNCellgradients
def parse_arguments():
    parser = argparse.ArgumentParser()


    parser.add_argument("--src",help="Source file",default="input/NMT/train_fr_lines2.txt") 	# naming it "echo"
    parser.add_argument("--tgt", help="target file",default="input/NMT/train_en_lines2.txt")
    parser.add_argument("--vocab", help="vocab file",default="resources/tmp/embeddings/glove.6B.50d.txt")
    parser.add_argument("--num_layers",type=int, help="RNN feed forward layer",default=2) 	# naming it "echo"
    parser.add_argument("--encoder_type", help="Uni or bi directional encoder",default="bi")
    parser.add_argument("--num_units",type=int, help="hidden units",default=5)
    parser.add_argument("--learning_rate",type=float, help="learning rate",default=.005)
    parser.add_argument("--batch_size",type=int, help="parallelism",default=2)
    parser.add_argument("--epochs",type=int, help="rounds of training",default=1)
    parser.add_argument("--per_epoch",type=int, help="rounds of training",default=1)
    parser.add_argument("--debug",type=bool, help="debug output",default=False)
    parser.add_argument("--out_dir",help="output folder",default="resources/tmp/nmt-models/fresh/")

    parser.add_argument("--attention_architecture",help="attention or not",default=None)
    parser.add_argument("--attention_option",help="which attention",default="Luong")

    args = parser.parse_args()
    print("args:",args)
    return args

def print_gradients(gradients):
    org.mk.training.dl.nn.print_gradients(gradients)
    gradients.reverse()
    for layer in gradients:
        name,grad=layer
        if(name is not "EmbeddingLayer"):

            if(grad is not None):
                print("Layer-",name,":")

                if('memory_grad' in grad):
                    print_grad("memory_grad",grad['memory_grad'])


                fcellgrad=grad['fw_cell']
                print_RNNCellgradients(fcellgrad,"fw_cell-")

                if('attention_grad' in grad):
                    print_grad("attention_grad",grad['attention_grad'])


                bcellgrad=grad.get('bw_cell')
                if(bcellgrad is not None):
                    print_RNNCellgradients(bcellgrad,"bw_cell-")

                if('Y' in grad):
                    print_grad("Ycomp",grad['Y'])


