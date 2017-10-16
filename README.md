# pytorch-rnn-benchmark
****
A program to test performance of several RNNs(realized by Pytorch) on different hardware platforms, providing benchmark for further optimization.  
### Requirements
****
  * Pytorch


### Usage
****
	bash run.sh platform-name [train] [daily]
platform-name can be 'bdw', 'skx', 'knm', 'knl', 'gpu'.   
If you want to test performance of training, please add parameter train.  
If you want to run daily test(just test 2 size), please add parameter daily.  


