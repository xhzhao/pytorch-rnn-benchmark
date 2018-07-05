log=$1
grep SPS $log | awk '{print $NF}'
