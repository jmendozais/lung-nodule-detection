#!/bin/bash
. task-util.sh
. lnd-util.sh

function meval {
model=$1
start-liv prim meval "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-evaluation --roi-size 64"
start-liv minsky meval2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-evaluation2 --roi-size 64"
}

function msel {
model=$1
start-liv minsky msel-${model} "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-selection --roi-size 64"
}

function msel-det {
model=$1
start-liv minsky msel-det-${model} "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} --model-selection-detailed --roi-size 64"
}
function clf {
model=$1
start-liv minsky clf "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model ${model} /home/juliomb/dbs/jsrt-npy/JPCLN001.npy"
}

function base {
start-liv minsky msel "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64"
}

function exp1 {
#start-liv minsky exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.01"
start-liv minsky exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003"
#start-liv prim exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.001"
#start-liv phong exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.0003"
start-liv prim exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.0001"
}

function sel-epoch {
echo "sel ep"
model=$1
epoch=$2
for cpfold in data/${model}*${epoch}.hdf5; do
    echo ${cpfold} ${cpfold:0:${#cpfold}-7}h5
done
#python util.py --single-froc --model ${model} --legend ${model}
python util.py --single-froc --model ${model} 
}


function exp1lrcmp {
methods=(
'5P-no-da-lr0.01-br3-sbf-aam-val' 
'5P-no-da-lr0.003-br3-sbf-aam-val' 
'5P-no-da-lr0.001-br3-sbf-aam-val' 
'5P-no-da-lr0.0003-br3-sbf-aam-val' 
'5P-no-da-lr0.0001-br3-sbf-aam-val')
labels=('lr=0.01' 'lr=0.003' 'lr=0.001' 'lr=0.0003' 'lr=0.0001')
froc 'lr-cmp' ${methods} ${labels} 
}

function exp1det {
start-liv minsky exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.01"
start-liv prim exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003"
start-liv babbage exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.001"
start-liv phong exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.0003"
start-liv minsky exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.0001"
}

function exp1det {
start-liv minsky exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.01"
start-liv prim exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003"
start-liv babbage exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.001"
start-liv phong exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.0003"
start-liv minsky exp1det "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.0001"
}

function exp2 {
#start-liv minsky exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 16"
#start-liv bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 24"
start-liv prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
#start-liv phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 40"
#start-liv babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad -1"
}

function exp2cmp {
methods=('5P-no-da-lr0.003-br16-sbf-0.7-aam-val' '5P-no-da-lr0.003-br24-sbf-0.7-aam-val' '5P-no-da-lr0.003-br32-sbf-0.7-aam-val' '5P-no-da-lr0.003-br40-sbf-0.7-aam-val' '5P-no-da-lr0.003-br-1-sbf-0.7-aam-val')
labels=('fixed blob rad 16' 'fixed blob rad 24' 'fixed blob rad 32' 'fixed blob rad 40' 'dynamic blbo rad')
froc 'receptive-field-cmp' ${methods} ${labels} 
}

# space [0.03, 0.05, 0.08, 0.1]
function exp3tr {
da_params='--da-tr 0 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv phong exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.01 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv minsky exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.02 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv bezier exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.03 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv prim exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.04 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv babbage exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.05 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3tr1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.1 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3tr2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

# space [3, 6, 9, 12, 15]
function exp3rot {
da_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0 --da-flip 1'
start-aqp salle exp3rot1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 4 --da-zoom 1 --da-is 0 --da-flip 1'
start-aqp salle exp3rot2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 6 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot3 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 8 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot4 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 10 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot5 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

function exp3rotdet {
br=32
start-liv minsky exp3rotdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.003 --epochs 30 --blob-rad 32 --da-rot 3"
start-liv bezier exp3rotdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.003 --epochs 30 --blob-rad 32 --da-rot 6"
start-liv prim exp3rotdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.003 --epochs 30 --blob-rad 32 --da-rot 9"
start-liv phong exp3rotdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.003 --epochs 30 --blob-rad 32 --da-rot 12"
start-liv babbage exp3rotdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.003 --epochs 30 --blob-rad 32 --da-rot 15"
}

function exp3rotcmp {
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr3-fl1-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr6-fl1-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr9-fl1-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr12-fl1-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr15-fl1-sbf-0.7-aam-val'
)
labels=(
'Rotation (-3, 3)'
'Rotation (-6, 6)'
'Rotation (-9, 9)'
'Rotation (-12, 12)'
'Rotation (-15, 15)'
)

froc '3-rot-cmp' ${methods} ${labels} 
}

# zoom [1.05, 1.1, 1.15, 1.2, 1.25]
function exp3zoom {
br=32
start-liv voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003 --blob-rad 32"
}

# zoom [1.05, 1.1, 1.15, 1.2, 1.25]
function exp3zoom {
br=32
start-liv voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003 --blob-rad 32"
}

# intensity shift [0.2, 0.4, 0.5, 0.6, 0.8]
function exp3int {
br=32
start-liv voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv bezier exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv prim exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv phong exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
start-liv babbage exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection-detailed --roi-size 64 --lr 0.003 --blob-rad 32"
}

# schemes

function exp3sch {
br=32
start-liv voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-med-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"

start-liv voronoi exp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-high-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --blob-rad 32"
}

function test {
start-liv minsky test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20"
}

# Dropout
function exp4fdp {
start-liv minsky test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.1"
start-liv babbage test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.15"
start-liv bezier test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.2"
start-liv prim test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.25"
start-liv phong test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.3"
}

function exp4ldp {
start-liv minsky test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.1"
start-liv babbage test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.15"
start-liv bezier test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.2"
start-liv prim test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.25"
start-liv phong test "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003 --epochs 20 --dp 0.3"
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
