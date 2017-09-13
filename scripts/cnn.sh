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
start-liv minsky exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.01"
start-liv minsky exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.003"
start-liv prim exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.001"
start-liv phong exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.0003"
start-liv prim exp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-no-da --model-selection --roi-size 64 --lr 0.0001"
}

function sel-epoch {
model=$1
epoch=$2
echo "Select epoch ..."
for cpfold in data/${model}*${epoch}.hdf5; do
    echo ${cpfold} ${cpfold:0:${#cpfold}-16}_weights.h5
    cp ${cpfold} ${cpfold:0:${#cpfold}-16}_weights.h5
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
methods=(
'5P-no-da-lr0.003-br16-sbf-0.7-aam-val'
'5P-no-da-lr0.003-br24-sbf-0.7-aam-val'
'5P-no-da-lr0.003-br32-sbf-0.7-aam-val'
'5P-no-da-lr0.003-br40-sbf-0.7-aam-val'
'5P-no-da-lr0.003-br-1-sbf-0.7-aam-val'
)
labels=(
'Blob radius 16'
'Blob radius 24'
'Blob radius 32'
'Blob radius 40'
'SBF approximated blob radius'
)
froc 'receptive-field-cmp' ${methods} ${labels} 
}

# space [0.03, 0.05, 0.08, 0.1]
function exp3tr {
#da_params='--da-tr 0.15 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-liv phong exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"

da_params='--da-tr 0.18 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv minsky exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"

#da_params='--da-tr 0.21 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-liv prim exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"

#da_params='--da-tr 0.12 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-liv babbage exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"

#da_params='--da-tr 0.01 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3tr1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.1 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3tr2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

function exp3trcmp {
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.03-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.06-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.12-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.15-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.18-rr0-fl1-lr0.001-sbf-0.7-aam-val'
)
labels=(
'No translation'
'Translation factor in [-0.03, 0.03)'
'Translation factor in [-0.06, 0.06)'
'Translation factor in [-0.12, 0.12)'
'Translation factor in [-0.15, 0.15)'
'Translation factor in [-0.18, 0.18)'
)
froc '3-tr-cmp' ${methods} ${labels} 
}

function exp3trdet {
da_params='--da-tr 0 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv phong exp3trdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.01 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv minsky exp3trdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.02 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv bezier exp3trdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.03 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv prim exp3trdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.04 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv babbage exp3trdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

# space [3, 6, 9, 12, 15]
function exp3rot {

da_params='--da-tr 0 --da-rot 3 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 6 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 9 --da-zoom 1 --da-is 0 --da-flip 1'
start-aqp salle exp3rot3 "sTHEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"

da_params='--da-tr 0 --da-rot 12 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot4 "sleep 14000; THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 15 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot5 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

function exp3rotcmp {
#methods=(
#'5P-da-is0.0-zm1.0-tr0.0-rr3-fl1-sbf-0.7-aam-val'
#'5P-da-is0.0-zm1.0-tr0.0-rr6-fl1-sbf-0.7-aam-val'
#'5P-da-is0.0-zm1.0-tr0.0-rr9-fl1-sbf-0.7-aam-val'
#'5P-da-is0.0-zm1.0-tr0.0-rr12-fl1-sbf-0.7-aam-val'
#'5P-da-is0.0-zm1.0-tr0.0-rr15-fl1-sbf-0.7-aam-val'
#)
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr6-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr12-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr18-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr24-fl1-lr0.001-sbf-0.7-aam-val'
)
labels=(
'No rotation'
'Rotation in [-6,6)'
'Rotation in [-12,12)'
'Rotation in [-18,18)'
'Rotation in [-24,24)'
)

froc '3-rot-cmp' ${methods} ${labels} 
}

# zoom [1.05, 1.1, 1.15, 1.2, 1.25]
function exp3zoom {
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1.05 --da-is 0 --da-flip 1'
#start-liv phong exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1.1 --da-is 0 --da-flip 1'
#start-liv minsky exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1.15 --da-is 0 --da-flip 1'
#start-liv bezier exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1.2 --da-is 0 --da-flip 1'
#start-liv prim exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1.25 --da-is 0 --da-flip 1'
#start-liv babbage exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.3 --da-is 0 --da-flip 1'
start-liv phong exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.4 --da-is 0 --da-flip 1'
start-liv prim exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

function exp3zoomdet {
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.05 --da-is 0 --da-flip 1'
start-liv phong exp3zoomdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.1 --da-is 0 --da-flip 1'
start-liv minsky exp3zoomdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.15 --da-is 0 --da-flip 1'
start-liv bezier exp3zoomdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.2 --da-is 0 --da-flip 1'
start-liv prim exp3zoomdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.25 --da-is 0 --da-flip 1'
start-liv babbage exp3zoomdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

function exp3zoomcmp {
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.05-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.1-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.15-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.2-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.25-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.4-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
)

labels=(
'No scaling'
'Scale factor in [1.0, 1.05)'
'Scale factor in [1.0, 1.1)'
'Scale factor in [1.0, 1.15)'
'Scale factor in [1.0, 1.2)'
'Scale factor in [1.0, 1.25)'
'Scale factor in [1.0, 1.3)'
)

froc '3-zoom-cmp' ${methods} ${labels} 
}

# intensity shift [0.2, 0.4, 0.5, 0.6, 0.8]
function exp3int {
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.1 --da-flip 1'
start-liv phong exp3int "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.2 --da-flip 1'
start-liv minsky exp3int "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.3 --da-flip 1'
start-liv bezier exp3int "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.4 --da-flip 1'
start-liv prim exp3int "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.5 --da-flip 1'
start-liv babbage exp3int "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

function exp3intdet {
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.1 --da-flip 1'
start-liv phong exp3intdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.2 --da-flip 1'
start-liv minsky exp3intdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.3 --da-flip 1'
start-liv bezier exp3intdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.4 --da-flip 1'
start-liv prim exp3intdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.5 --da-flip 1'
start-liv babbage exp3intdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

# TODO: repeat the experiment reducing the padding on ROI extraction (current pad=1.5)
function exp3intcmp {
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.1-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.2-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.3-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.4-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.5-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
)
labels=("No intensity shift"
"Intentity shift in $\sigma$ * [0.0, 0.1)"
"Intensity shift in $\sigma$ * [0.0, 0.2)"
"Intensity shift in $\sigma$ * [0.0, 0.3)"
"Intensity shift in $\sigma$ * [0.0, 0.4)"
"Intensity shift in $\sigma$ * [0.0, 0.5)"
)
froc '3-int-cmp' ${methods} ${labels} 
}

# TODO: repeat the experiment reducing the padding on ROI extraction (current pad=1.5)
function exp3flcmp {
methods=(
"5P-da-is0.0-zm1.0-tr0.0-rr0-fl0-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val"
)
labels=(
"No flip"
"Flip"
)
froc '3-flip-cmp' ${methods} ${labels} 
}

# schemes
# moderate data aug
# full data aug
# full data aug without zoom

function exp3sch {
#da_params='--da-tr 0.01 --da-rot 1 --da-zoom 1.0 --da-is 0.1 --da-flip 1'
#start-liv babbage exp3sch "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
#start-liv bezier exp3sch "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.0 --da-is 0.2 --da-flip 1'
#start-liv prim exp3sch "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

# FI
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1 --da-is 0.2 --da-flip 1'
#start-liv minsky exp3sch "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

# FIR
da_params='--da-tr 0 --da-rot 18 --da-zoom 1 --da-is 0.2 --da-flip 1'
start-liv prim exp3sch "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

# FIRT
da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1 --da-is 0.2 --da-flip 1'
start-liv phong exp3sch "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 120 --blob-rad 32 ${da_params}"

# FIRTS
da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
start-liv minsky exp3sch "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

}

function exp3schdet {
#da_params='--da-tr 0.01 --da-rot 1 --da-zoom 1.0 --da-is 0.1 --da-flip 1'
#start-liv babbage exp3schdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
#start-liv bezier exp3schdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.0 --da-is 0.2 --da-flip 1'
#start-liv prim exp3schdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

#da_params='--da-tr 0 --da-rot 0 --da-zoom 1 --da-is 0.2 --da-flip 1'
#start-liv minsky exp3schdet "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1'
#start-aqp salle exp3sch1det "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
start-aqp salle exp3sch1det "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

#da_params='--da-tr 0.02 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1'
#start-aqp salle exp3sch2det "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
}

function exp3schcmp {
methods=(
"5P-da-is0.0-zm1.0-tr0.0-rr0-fl0-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.0-tr0.0-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.0-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
)
labels=(
"No Data Augmentation"
"Data Augmentation F"
"Data Augmentation F+I"
"Data Augmentation F+I+R"
"Data Augmentation F+I+R+T"
"Data Augmentation F+I+R+T+S"
)
froc '3-sch-cmp' ${methods} ${labels} 
}

function test-lr-congruence {
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3lrc "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.003 --epochs 30 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-aqp salle exp3lrc2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}
function exp3fl {
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.0 --da-flip 0'
#start-liv minsky exp3fl "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.0 --da-flip 1'
start-liv phong exp3fl "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params} --load-model"

}

function exp3fldet {
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.0 --da-flip 0'
start-liv minsky exp3fl "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 0 --da-zoom 1.0 --da-is 0.0 --da-flip 1'
#start-liv minsky exp3fl "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

# Dropout
function exp4fdp {
fixed_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 100'
start-liv minsky exp4fdp "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.1"
start-liv phong exp4fdp "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.15"
start-liv prim exp4fdp "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.2"
start-liv bezier exp4fdp "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.25"
start-liv babbage exp4fdp "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.3"
}

function exp4fdpdet {
fixed_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 100'
start-liv minsky exp4fdpdet1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.1"
start-liv phong exp4fdpdet2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.15"
start-aqp salle exp4fdpdet3 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.2"
start-aqp salle exp4fdpdet4 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.25"
start-aqp salle exp4fdpdet5 "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.3"
}

function exp4fdpcmp {
methods=(
"5P-da-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001dp-0.1-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001dp-0.15-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001dp-0.2-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001dp-0.25-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001dp-0.3-sbf-0.7-aam-val"
)
labels=(
"No dropout"
"Dropout 0.1"
"Dropout 0.15"
"Dropout 0.2"
"Dropout 0.25"
"Dropout 0.3"
)
froc '4-fdp-cmp' ${methods} ${labels} 
}

THEANO_OPTS="THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn"
function exp4ldp {
fixed_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 100'
#start-aqp salle exp4ldp1 "${THEANO_OPTS},device=gpu0 python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.025 --lidp"
start-liv phong exp4ldp "${THEANO_OPTS},device=cuda0 python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.05 --lidp"
#start-aqp salle exp4ldp2 "${THEANO_OPTS},device=gpu1 python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.075 --lidp"
#start-aqp salle exp4ldp3 "${THEANO_OPTS},device=gpu2 python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.1 --lidp"
}

function exp4ldpcmp {
methods=(
"5P-da-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.025-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.05-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.075-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.1-sbf-0.7-aam-val"
)
labels=(
"No dropout"
"Linear Inc. DP 0.025"
"Linear Inc. DP 0.05"
"Linear Inc. DP 0.075"
"Linear Inc. DP 0.1"
)
froc '4-ldp-cmp' ${methods} ${labels} 
}
THEANO_OPTS="THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn"
da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
fixed_params="${da_params} --lr 0.001 --epochs 120 --dropout 0.05 --lidp --fc 1"

# TODO: sel epoch 140 for 6P
function exp5conv {
fixed_params="${da_params} --lr 0.001 --epochs 120 --dropout 0.05 --lidp --fc 1"
#start-liv prim exp5conv "${THEANO_OPTS},device=cuda0 python lnd.py --model convnet --model-selection ${fixed_params} --dp-intercept 0.1 --conv 4 --roi-size 32"
#start-liv phong exp5conv "${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --dp-intercept 0.0 --conv 5 --roi-size 64"
#start-liv minsky exp5conv "${THEANO_OPTS},device=cuda0 python lnd.py --model convnet --model-selection ${fixed_params} --dp-intercept -0.1 --conv 6 --roi-size 128"
start-liv minsky exp5conv "${THEANO_OPTS},device=cuda0 python lnd.py --model convnet --model-selection ${fixed_params} --dp-intercept -0.2 --conv 7 --roi-size 256"
}

function exp5convcmp {
methods=(
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-4-fil-64-fc-1-lidp-0.05-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-5-fil-64-fc-1-lidp-0.05-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-64-fc-1-lidp-0.05-sbf-0.7-aam-val"
)
#"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.05-sbf-0.7-aam-val"
labels=(
"4 conv layer"
"5 conv layer"
"6 conv layer"
)
froc '5-conv-cmp' ${methods} ${labels} 
}
PRIMC="base_compiledir=~/.theanoprim"
MINSKYC="base_compiledir=~/.theanominksy"
PHONGC="base_compiledir=~/.theanophong"
BEZIERC="base_compiledir=~/.theanobezier"

function exp5fi {
fixed_params="${da_params} --lr 0.001 --epochs 150 --dropout 0.05 --lidp --fc 1 --dp-intercept -0.1 --conv 6 --roi-size 128"
#fixed_params="${da_params} --lr 0.001 --epochs 100 --dropout 0.1 --lidp --fc 1 --conv 5 --roi-size 64"

#start-liv phong exp5fi "${PHONGC},${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --filters 32"
#start-liv phong exp5fi "${THEANO_OPTS},${PRIMC},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --filters 64"
#start-liv prim exp5fi "${THEANO_OPTS},${PRIMC},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --filters 96"
#start-liv minsky exp5fi "${THEANO_OPTS},${MINSKYC},device=cuda0 python lnd.py --model convnet --model-selection ${fixed_params} --filters 128"
start-liv bezier exp5fi "${THEANO_OPTS},${BEZIERC},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --filters 48"
start-liv minsky exp5fi "${THEANO_OPTS},${MINSKYC},device=cuda0 python lnd.py --model convnet --model-selection ${fixed_params} --filters 16"
#start-liv exp5fi "${THEANO_OPTS},${MINSKYC},device=cuda0 python lnd.py --model convnet --model-selection ${fixed_params} --filters 24"
}

function exp5ficmp {
methods=(
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-16-fc-1-lidp-0.05-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-24-fc-1-lidp-0.05-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-1-lidp-0.05-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-64-fc-1-lidp-0.05-sbf-0.7-aam-val"
)
labels=(
"Convnet(6, 16)"
"Convnet(6, 24)"
"Convnet(6, 32)"
"Convnet(6, 64)"
)
froc '5-fil-cmp' ${methods} ${labels} 
}

function exp5fc {
fixed_params="${da_params} --lr 0.001 --epochs 150 --dropout 0.05 --lidp --dp-intercept -0.1 --conv 6 --roi-size 128 --filters 32"
#start-liv phong exp5fc "${PHONGC},${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --fc 0"
#start-liv bezier exp5fc "${BEZIERC},${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --fc 2"
echo "${PHONGC},${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --fc 0"
echo "${BEZIERC},${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --fc 2"
#start-liv prim exp5fc "${PRIMC},${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --fc 3"
#start-liv prim exp5fc "${PRIMC},${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --fc 3"
}

function exp5fccmp {
methods=(
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-0-lidp-0.05-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-1-lidp-0.05-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-3-lidp-0.05-sbf-0.7-aam-val"
)

labels=(
"No FC layers"
"1 FC layer"
"3 FC layer"
)
froc '5-fc-cmp' ${methods} ${labels} 
}

function sleval {
fixed_params="${da_params} --lr 0.001 --epochs 1 --dropout 0.05 --lidp --dp-intercept -0.1 --conv 6 --roi-size 128 --filters 32"
echo "${PRIMC},${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-evaluation ${fixed_params} --fc 1"

#start-liv minsky sleval "${THEANO_OPTS},device=cuda0 python lnd.py --model cnn --model-evaluation2 ${fixed_params}"
#start-liv prim sleval "${THEANO_OPTS},device=cuda1 python lnd.py --model cnn --model-evaluation ${fixed_params}"
#fixed_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1 --lr 0.001 --epochs 100 --dropout 0.05 --lidp --fc 2 --conv 5 --roi-size 64 --filter 32'
#start-liv babbage sleval "${THEANO_OPTS},device=cuda0 python lnd.py --model cnn --model-evaluation2 ${fixed_params}"
#start-liv phong sleval "${THEANO_OPTS},device=cuda1 python lnd.py --model cnn --model-evaluation ${fixed_params}"
}


function fixsel {
old_da='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1'
new_da='--da-tr 0.5 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1'
fixed_params='--lr 0.001 --epochs 100 --dropout 0.05 --lidp --fc 2 --conv 5 --roi-size 64 --filters 64'
# Best model with sbf 0.5
start-aqp salle fixsel0 "${THEANO_OPTS},device=gpu0 python lnd.py --model fix --model-selection ${fixed_params} ${new_da} --detector sbf-0.5-aam"

# Best model with old data aug
#start-aqp salle fixsel1 "${THEANO_OPTS},device=gpu1 python lnd.py --model fix --model-selection ${fixed_params} ${old_da}"

# Best model with sbf 0.5 + old data aug
#start-aqp salle fixsel2 "${THEANO_OPTS},device=gpu2 python lnd.py --model fix --model-selection ${fixed_params} ${new_da} --detector sbf-0.5-aam"
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
 Generic call
else
    $1 $2 $3
fi

