
## Introduction
### This is the implementation of ["TAAT: Learning a Robust Feature Space via Topology Aligning with Knowledge Guided"]

## Usage
### Installation
The training environment (PyTorch and dependencies) can be installed as follows:
```
cd TAAT
pip install -r requirements.txt
```
### If you want to evaluate the pre-trained model on CIFAR-10, 
```
sh ./eval_trained.sh
```
### You can also find the evalation log at ./checkpoint/cifar10/cifar-TAAT.log, The log are as follows:
```
TAAT best checkpoint on CIFAR-10
==> acc:82.29, adv_acc:55.60, epoch:167, best_epoch:167
==> Module:AT
==> Model Name :cifar10-TAAT
==> natural eval:82.29, using time:0:00:01 
==> fgsm eval:59.77, using time:0:00:04 
==> pgd eval:54.70, using time:0:05:09 
==> cw eval:51.97, using time:0:05:27 
==> auto eval:50.80, using time:0:34:36 
==> cifar10-TAAT test finished!
```
### Train TAAT fron scratch on CIFAR-10
```
sh ./train.sh
```
### Evaluate TAAT
```
sh ./eval.sh
```
