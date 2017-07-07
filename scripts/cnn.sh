#!/bin/bash
. task-util.sh

function meval {
model=$1
start prim meval "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-evaluation --blob-rad 64"
start minsky meval2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-evaluation2 --blob-rad 64"
}

function clf {
model=$1
start minsky clf "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} /home/juliomb/dbs/jsrt-npy/JPCLN001.npy"
}

function msel {
model=$1
start minsky msel "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-selection --blob-rad 64"
}



if [ "$1" = "msel" ]; then
    msel $2
elif [ "$1" = "meval" ]; then
    meval $2
elif [ "$1" = "meval-jsrt" ]; then
    meval_jsrt $2
elif [ "$1" = "clf" ]; then
    clf $2 $3
else
    echo "Undefined command"
fi
