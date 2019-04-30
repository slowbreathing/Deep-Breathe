#!/usr/bin/env python3

import numpy as np
import sys
import tensorflow as tf

def softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    #print("jacobian_m:",jacobian_m)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                print("equal:",i,j)
                jacobian_m[i][j] = s[i] * (1-s[i])
            else:
                print("not equal:",i,j)
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m


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
   0.5413713,  -0.3229486,  -1.7227241,  -1.2969391,   0.27593264,  0.32810318]])
ycomp= np.array([[[ 0.02310003,  0.0278993 ,  0.02382485,  0.02473825,
          0.02611315,  0.0287924 ,  0.02985075,  0.02063007,
          0.02232488,  0.02292969,  0.02516695,  0.02151516,
          0.0231119 ,  0.01968879,  0.02181169,  0.02211449,
          0.02367149,  0.02550107,  0.0230583 ,  0.02223216,
          0.02240607,  0.01943858,  0.02158429,  0.01850561,
          0.02436397,  0.02864791,  0.02185071,  0.02529964,
          0.02574545,  0.02451832,  0.02750084,  0.0215316 ,
          0.02130829,  0.0204995 ,  0.0260001 ,  0.02440649,
          0.02681847, -0.97716611,  0.02329408,  0.02368154,
          0.03119017,  0.02049913]]])

fullattendz=np.array([[[0.03651258, 0.03651258, 0.03651258, 0.03651258, 0.03651258,
         0.06658257, 0.06658257, 0.06658257, 0.06658257, 0.06658257]]])

attl_kernel=np.ones((5,10))
print("fullattendz.shape:",fullattendz.shape)

fullattention=np.array([[[0.05154758, 0.05154758, 0.05154758, 0.05154758, 0.05154758]]])
amvalues=np.array([[[0.06658257, 0.06658257, 0.06658257, 0.06658257, 0.06658257]]])


batch=1
seq=1
hidden_size=5
size=42
dy=(ycomp.sum(1).sum(0,keepdims=True)/(batch*seq))
print("dy:",dy)

dWyfw=np.zeros((5,42))
dhtf_fw=np.zeros((hidden_size,batch))
dh_nextmlco = np.zeros_like(dhtf_fw)
alignmentst=np.array( [[1.]])
ht=np.array([[0.03651258, 0.03651258, 0.03651258, 0.03651258, 0.03651258]])
#dattention=np.dot(dy,out_weights.T)
#print("dattention:",dattention)
#dattention_layer=np.dot(np.reshape(fullattendz[:,:,:],[seq,10]).T,dattention)

#print("dattention_layer:",dattention_layer/(batch*seq))
dattention=np.zeros((1,5),dtype=float)
dattention_layer=np.zeros((10,5 ),dtype=float)
for seqnum in reversed(range(1 )):
    #pred=(attention*wy)+b
    #dWyfw=attention*Ycomp/seq
    dWyfw+=np.dot(fullattention[seqnum,:,:].T,np.reshape(ycomp[:,seqnum,:],[1,size]))
    print(np.reshape(ycomp[:,seqnum,:],[batch,size]).shape,init.T.shape)
    dattention+=np.dot(np.reshape(ycomp[:,seqnum,:],[batch,size]),init.T)
    print("dattention:",seqnum,":",dattention,dattention.shape)
    dattention_layer+=np.dot(fullattendz[seqnum,:,:].T,dattention)
    print("dattention_layer:::",dattention_layer)
    dht=np.dot(dattention,attl_kernel)
    print("dht:",dht)
    print("dht:",dht[:,0:hidden_size])

    """
    dhtf_fw =dht[:,0:fw_cell.hidden_size].T
    if dh_nextmlco is None:
        dh_nextmlco = np.zeros_like(dhtf_fw)
    dh_nextmlco=fw_cell.compute_gradients(dhtf_fw,dh_nextmlco,seqnum)
    """

    dalignment=np.dot(dht[:,hidden_size:],amvalues[seqnum].T)
    print("dalignment:",dalignment)
    dsoftmax=np.zeros_like(alignmentst)
    for i in range(batch):
        sg=softmax_grad(alignmentst[i])
        #print(sg)
        #print("softgrad:",sg.diagonal().reshape((1,seq)))
        dsoftmax[i]= sg.diagonal().reshape((1,seq))
    dsoftmax=np.dot(dsoftmax,dalignment)
    print("dsoftmax:",dsoftmax)
    dkeys=np.dot(dsoftmax,ht)
    print("dkeys:",dkeys)
    dmemlayer=np.dot(dkeys.T,amvalues)
    print("dmemlayer:",dmemlayer)

dWy=dWyfw/(batch*seq)
print("dWy:",dWy)
#print("fw_cell.get_gradients():",fw_cell.get_gradients())
