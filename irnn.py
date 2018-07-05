

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import time
import sys

irnn_enable = True
model = 'LSTM'
dry_run = 50
num_iter = 100
count = 20     # number of iterations
cuda = False   # whether GPU is used or not
train = False  # True: test training performance; False: test forward performance only
daily = False


if 'train' in sys.argv:
    train = True
if 'daily' in sys.argv:
    daily = True
if 'gru' in sys.argv:
    model = 'GRU'
if 'nn' in sys.argv:
    irnn_enable = False
if 'cuda' in sys.argv:
    cuda = True
    irnn_enable = False
if irnn_enable:
    import irnn_pytorch as irnn

print("### irnn = %s, model = %s, train = %s, cuda = %s" % (irnn_enable, model, train, cuda))

if daily:
    sizes = [[64,50,500,500],
         [128,25,4096,4096]
        ]
    print("daily test")
else:
    sizes = [
         [20,1,   800,800],
         [20,50,  800,800],
         [20,100, 800,800],
         [20,200, 800,800],
         [20,300, 800,800],
         [20,400, 800,800],
         [12,1,   1760,1760],
         [12,50,  1760,1760],
         [12,100, 1760,1760],
         [12,200, 1760,1760],
         [12,300, 1760,1760],
         [12,400, 1760,1760],
         [32,1,   1760,1760],
         [32,50,  1760,1760],
         [32,100, 1760,1760],
         [32,200, 1760,1760],
         [32,300, 1760,1760],
         [32,400, 1760,1760]
        ]




for idx in range(len(sizes)):
    size = sizes[idx]
    N = size[0]    # batch size
    T = size[1]    # sentence length
    D = size[2]    # embedding size
    H = size[3]    # hidden size

    if irnn_enable:
        if model == 'LSTM':
            rnn = irnn.LSTM(D,H,1)
        elif model == 'GRU':
            rnn = irnn.GRU(D,H,1)
    else:
        if model == 'LSTM':
            rnn = nn.LSTM(D,H,1)
        elif model == 'GRU':
            rnn = nn.GRU(D,H,1)

    input = Variable(torch.randn(T, N, D))
    if cuda:
        rnn = rnn.cuda()
        input = input.cuda()
    
    #print("rnn type = ",type(rnn))
    if train:
        rnn.train()
        targets = Variable(torch.randn(T,N,D))
        if cuda:
            targets = targets.cuda()
    else:
        rnn.eval()

    for j in range(dry_run+num_iter):
        if j == dry_run:
            start = time.time()
        output, hn = rnn(input)
        if train:
            output.sum().backward()
        if cuda:
            torch.cuda.synchronize()
    dura = (time.time() - start)/num_iter     # time of ONE iteration
    gflops = T*4*(N*H*D*2 + N*H*H*2)/1e9
    GFLOPS = gflops/dura                   # giga floating-point operations per second
    SPS = N/dura                           # number of processed sentences per second
    print("size = %s, duration = %.4f, gflops = %.4f, GFLOPS = %.4f, SPS = %.4f" %(size,dura,gflops,GFLOPS,SPS))
