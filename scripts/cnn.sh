#!/bin/bash
. task-util.sh

function meval {
model=$1
start prim meval "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-evaluation --roi-size 64"
start minsky meval2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-evaluation2 --roi-size 64"
}

function msel {
model=$1
start minsky msel-${model} "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-selection --roi-size 64"
}

function msel-det {
model=$1
start minsky msel-det-${model} "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-selection-detailed --roi-size 64"
}

function clf {
model=$1
start minsky clf "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} /home/juliomb/dbs/jsrt-npy/JPCLN001.npy"
}

function base {
start minsky msel "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64"
}

function exp1 {
start voronoi exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.01"
start bezier exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003"
start prim exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.001"
start phong exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.0003"
start babbage exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.0001"
}

function exp1b {
start voronoi exp1b "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.01"
start bezier exp1b "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003"
start prim exp1b "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.001"
start phong exp1b "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.0003"
start babbage exp1b "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.0001"
}

function exp2 {
start minsky exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 16"
start bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 24"
start prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 40"
start babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --blob-rad -1"
}

# space [3, 6, 9, 12, 15]
function exp3rot {
br=32
start voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003 --blob-rad ${br}"
}

# space [0.03, 0.05, 0.08, 0.1]
function exp3tr {
br=32
start voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003 --blob-rad ${br}"
}

# zoom [1.05, 1.1, 1.15, 1.2, 1.25]
function exp3zoom {
br=32
start voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003 --blob-rad ${br}"
}

# zoom [1.05, 1.1, 1.15, 1.2, 1.25]
function exp3zoom {
br=32
start voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003 --blob-rad ${br}"
}

# intensity shift [0.2, 0.4, 0.5, 0.6, 0.8]
function exp3int {
br=32
start voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
start babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003 --blob-rad ${br}"
}

# schemes

function exp3sch {
br=32
start voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-med-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"

start voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-high-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad ${br}"
}

function sel-ep {
echo "sel ep"
model=$1
epoch=$2
for cpfold in data/${model}*${epoch}.hdf5; do
    echo ${cpfold} ${cpfold:0:${#cpfold}-7}h5
done
#python util.py --single-froc --model ${model} --legend ${model}
python util.py --single-froc --model ${model} 
}

function test {
start minsky test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20"
}

if [ "$1" = "msel" ]; then
    msel $2
elif [ "$1" = "msel-det" ]; then
    msel-det $2
elif [ "$1" = "meval" ]; then
    meval $2
elif [ "$1" = "meval-jsrt" ]; then
    meval_jsrt $2
elif [ "$1" = "clf" ]; then
    clf $2 $3
elif [ "$1" = "sel-ep" ]; then
    sel-ep $2 $3
# Generic call
else
    $1
fi
