#!/bin/sh

which python

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=compact,1,0,granularity=fine"
#KMP_BLOCKTIME=1
#export MKL_DYNAMIC=TRUE

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
#export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
#echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

python LSTM.py $1 $2 $3

#cpu example: ./run.sh train daily
#cuda example: ./run.sh train daily cuda
