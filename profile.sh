echo "profiling $1"
file=$1
cat $file | grep "mkldnn_rnn_lstm," | awk '{sum+=$4} END {print "iters =" NR ", forward average = " sum/NR}'
cat $file | grep "mkldnn_rnn_lstm_backward" | awk '{sum+=$4} END {print "iters =" NR ", backward average = " sum/NR}'
cat $file | grep "mkldnn_verbose,create" | sed "s/,/\t/g" | awk '{sum+=$NF} END {print "iters =" NR ", create primitive average = " 6*sum/NR}'
cat $file | grep "mkldnn_verbose,exec,reorder" | sed "s/,/\t/g" | awk '{sum+=$NF} END {print "iters =" NR ", reorder exec average = " 6*sum/NR}'
cat $file | grep "mkldnn_verbose,exec,rnn,ref:any,forward_training" | sed "s/,/\t/g" | awk '{sum+=$NF} END {print "iters =" NR ", forward exec average = " sum/NR}'
cat $file | grep "mkldnn_verbose,exec,rnn,ref:any,backward" | sed "s/,/\t/g" | awk '{sum+=$NF} END {print "iters =" NR ", backward exec average = " sum/NR}'

tail -n 2 $file


