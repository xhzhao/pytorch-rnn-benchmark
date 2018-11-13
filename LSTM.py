import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys

dry_run = 3
num_iter = 30
cuda = False   # whether GPU is used or not
train = False  # True: test training performance; False: test forward performance only
daily = False

if 'cuda' in sys.argv:
    cuda = True
if 'train' in sys.argv:
    train = True
if 'daily' in sys.argv:
    daily = True

if daily:
    sizes = [[64,50,500,500],
         [128,50,1024,1024]
        ]
    print("daily test")
else:
    sizes = [[64,15,500,500],
         [64,20,500,500],
         [64,25,500,500],
         [64,30,500,500],
         [64,35,500,500],
         [64,40,500,500],
         [64,45,500,500],
         [64,50,500,500],
         [16,25,512,512],
         [32,25,512,512],
         [64,25,512,512],
         [128,25,512,512],
         [16,25,1024,1024],
         [32,25,1024,1024],
         [64,25,1024,1024],
         [128,25,1024,1024],
         [16,25,2048,2048],
         [32,25,2048,2048],
         [64,25,2048,2048],
         [128,25,2048,2048],
         [16,25,4096,4096],
         [32,25,4096,4096],
         [64,25,4096,4096],
         [128,25,4096,4096]
        ]


for idx in range(len(sizes)):
    size = sizes[idx]
    N = size[0]    # batch size
    T = size[1]    # sentence length
    D = size[2]    # embedding size
    H = size[3]    # hidden size

    #torch._C._set_mkldnn_enabled(False)  
    if cuda:
        rnn = nn.LSTM(D,H,num_layers=1).float().cuda()
        input = torch.randn(T, N, D, dtype=torch.float, requires_grad=True).cuda()
        h0 = torch.randn(1, N, H, dtype=torch.float, requires_grad=True).cuda()
        c0 = torch.randn(1, N, H, dtype=torch.float, requires_grad=True).cuda()
    else:
        rnn = nn.LSTM(D,H,num_layers=1).float()
        input = torch.randn(T, N, D, dtype=torch.float, requires_grad=True)
        h0 = torch.randn(1, N, H, dtype=torch.float, requires_grad=True)
        c0 = torch.randn(1, N, H, dtype=torch.float, requires_grad=True)

    output, hn = rnn(input, (h0, c0))
    if train:
        loss_fn = torch.nn.L1Loss()

        if cuda:
            targets = Variable(torch.randn(T,N,D).cuda())
            loss_fn = loss_fn.cuda()
        else:
            targets = Variable(torch.randn(T,N,D))

    for j in range(dry_run+num_iter):
        if j == dry_run:
            start = time.time()
        output, (hy , cy) = rnn(input, (h0, c0))
        if train:
            #loss = loss_fn(output,targets)
            loss = output.sum() #+ hy.sum() + cy.sum()
            loss.backward()
        if cuda:
            torch.cuda.synchronize()
    dura = (time.time() - start)/num_iter     # time of ONE iteration
    gflops = T*4*(N*H*D*2 + N*H*H*2)/1e9
    if train:
        gflops = gflops * 3
    GFLOPS = gflops/dura                   # giga floating-point operations per second
    SPS = N/dura                           # number of processed sentences per second
    print("size = %s, duration = %.4f, gflops = %.4f, GFLOPS = %.4f, SPS = %.4f" %(size,dura,gflops,GFLOPS,SPS))
