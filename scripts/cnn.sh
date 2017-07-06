#!/bin/bash
. task-util.sh

function eval_cnn {
model=$1
start minsky eval "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-evaluation"
}

function eval2 {
model=$1
start minsky eval "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-evaluation2"
}

function clf {
model=$1
start minsky eval "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} /home/juliomb/dbs/jsrt-npy/JPCLN001.npy"
}

if [ "$1" = "eval" ]; then
    eval_cnn $2
elif [ "$1" = "eval2" ]; then
    eval2 $2
elif [ "$1" = "clf" ]; then
    clf $2 $3
else
    echo "Undefined command"
fi
