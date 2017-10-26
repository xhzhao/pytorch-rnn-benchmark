import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

count = 50
OpenNMT = 1

sizes = [[64,10,150,1024]
        ]




for idx in range(len(sizes)):
  size = sizes[idx]
  N = size[0]
  T = size[1]
  D = size[2]
  H = size[3]
  
  rnn = nn.GRU(D,H,1).cuda()
  input = Variable(torch.randn( N, T, D).cuda())
  h0 = Variable(torch.randn(1, N, H).cuda())
  c0 = Variable(torch.randn(1, N, H).cuda())
  output, hn = rnn(input, h0)
  
  start = time.time()
  for j in range(count):
    iter_start = time.time()
    rnn(input, h0)
    #print("iter time = ",time.time()-iter_start)
  torch.cuda.synchronize()
  dura = (time.time() - start)/count
  gflops = T*4*(N*H*D*2 + N*H*H*2)/1e9
  GFLOPS = gflops/dura
  print("size = %s, duration = %.8f, gflops = %.4f, GFLOPS = %.4f" %(size,dura,gflops,GFLOPS))

