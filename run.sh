#!/bin/sh

t=$1
lscpu
which python
export KMP_AFFINITY=compact,1,0,granularity=fine

if [ $t == 'bdw' ]; then
  export OMP_NUM_THREADS=44
  python LSTM.py $2 $3
fi
if [ $t == 'knl' ]; then
  export OMP_NUM_THREADS=68
  python LSTM.py $2 $3
fi
if [ $t == 'knm' ]; then
  export OMP_NUM_THREADS=72
  python LSTM.py $2 $3
fi
if [ $t == 'skx' ]; then
  export OMP_NUM_THREADS=56
  python LSTM.py $2 $3
fi
if [ $t == 'gpu' ]; then
  python LSTM.py $2 cuda $3
fi
