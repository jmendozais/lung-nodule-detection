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
model=$1
epoch=$2
echo "sel ep"
for cpfold in data/${model}*${epoch}.hdf5; do
    #echo ${cpfold} ${cpfold:0:${#cpfold}-16}_weights.h5
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
da_params='--da-tr 0.01 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv phong exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.02 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv bezier exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.03 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv prim exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0.04 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv babbage exp3tr "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"

#da_params='--da-tr 0.01 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3tr1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.1 --da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3tr2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

function exp3trcmp {
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.01-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.02-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.03-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.04-rr0-fl1-lr0.001-sbf-0.7-aam-val'
)
labels=(
'No translation'
'Translation factor in [0.0, 0.01)'
'Translation factor in [0.0, 0.02)'
'Translation factor in [0.0, 0.03)'
'Translation factor in [0.0, 0.04)'
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
da_params='--da-tr 0 --da-rot 1 --da-zoom 1 --da-is 0 --da-flip 1'
start-liv minsky exp3rot1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"

#da_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 4 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 6 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot3 "sleep 14000; THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 8 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot4 "sleep 14000; THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 10 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rot5 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
}

function exp3rotdet {
#da_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rotdet1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 4 --da-zoom 1 --da-is 0 --da-flip 1'
#start-aqp salle exp3rotdet2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"

da_params='--da-tr 0 --da-rot 6 --da-zoom 1 --da-is 0 --da-flip 1'
start-aqp salle exp3rotdet3 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 8 --da-zoom 1 --da-is 0 --da-flip 1'
start-aqp salle exp3rotdet4 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
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
'5P-da-is0.0-zm1.0-tr0.0-rr2-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr4-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr6-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.0-rr8-fl1-lr0.001-sbf-0.7-aam-val'
)
labels=(
'No rotation'
'Rotation in [-2, 2)'
'Rotation in [-4, 4)'
'Rotation in [-6, 6)'
'Rotation in [-8, 8)'
)

froc '3-rot-cmp' ${methods} ${labels} 
}

# zoom [1.05, 1.1, 1.15, 1.2, 1.25]
function exp3zoom {
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.05 --da-is 0 --da-flip 1'
start-liv phong exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.1 --da-is 0 --da-flip 1'
start-liv minsky exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.15 --da-is 0 --da-flip 1'
start-liv bezier exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.2 --da-is 0 --da-flip 1'
start-liv prim exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
da_params='--da-tr 0 --da-rot 0 --da-zoom 1.25 --da-is 0 --da-flip 1'
start-liv babbage exp3zoom "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 70 --blob-rad 32 ${da_params}"
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
)

labels=(
'No scaling'
'Scale factor in [1.0, 1.05)'
'Scale factor in [1.0, 1.1)'
'Scale factor in [1.0, 1.15)'
'Scale factor in [1.0, 1.2)'
'Scale factor in [1.0, 1.25)'
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
"Intentity shift in $\sigma$*[0.0, 0.1)"
"Intensity shift in $\sigma$*[0.0, 0.2)"
"Intensity shift in $\sigma$*[0.0, 0.3)"
"Intensity shift in $\sigma$*[0.0, 0.4)"
"Intensity shift in $\sigma$*[0.0, 0.5)"
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

#da_params='--da-tr 0 --da-rot 0 --da-zoom 1 --da-is 0.2 --da-flip 1'
#start-liv minsky exp3sch "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1'
#start-aqp salle exp3sch1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
#da_params='--da-tr 0.02 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1'
#start-aqp salle exp3sch2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

da_params='--da-tr 0 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
start-aqp salle exp3sch1 "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
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
#da_params='--da-tr 0.02 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1'
#start-aqp salle exp3sch2det "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

da_params='--da-tr 0 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
start-aqp salle exp3sch1det "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-da --model-selection-detailed --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
}

function exp3schcmp {
methods=(
"5P-da-is0.0-zm1.0-tr0.0-rr0-fl0-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.0-tr0.0-rr2-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.0-tr0.02-rr2-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001-sbf-0.7-aam-val"
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

function exp4fdpcmp {
methods=(
"5P-da-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.1-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.15-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.2-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.25-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.3-sbf-0.7-aam-val"
)
labels=(
"No dropout"
"Dropout 0.1"
"Dropout 0.15"
"Dropout 0.2"
"Dropout 0.25"
"Dropout 0.3"
)
froc '3-dp-cmp' ${methods} ${labels} 
}

function exp4fdpcmp {
methods=(
"5P-da-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.1-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.15-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.2-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.25-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.02-rr2-fl1-lr0.001dp-0.3-sbf-0.7-aam-val"
)
labels=(
"No dropout"
"Dropout 0.1"
"Dropout 0.15"
"Dropout 0.2"
"Dropout 0.25"
"Dropout 0.3"
)
froc '3-dp-cmp' ${methods} ${labels} 
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
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-liv minsky exp4fdp1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.1"
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-liv phong exp4fdp2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.15"
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-aqp salle exp4fdp3 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.2"
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-aqp salle exp4fdp4 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.25"
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-aqp salle exp4fdp5 "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.3"
}

function exp4fdpdet {
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-liv minsky exp4fdpdet1 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.1"
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-liv phong exp4fdpdet2 "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.15"
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-aqp salle exp4fdpdet3 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.2"
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-aqp salle exp4fdpdet4 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.25"
fixed_params='--da-tr 0.02 --da-rot 2 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-aqp salle exp4fdpdet5 "THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection-detailed ${fixed_params} --dropout 0.3"
}

function exp4ldp {
fixed_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-liv minsky exp4ldp "THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.5 --ldp"
fixed_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-aqp salle exp4ldp2 "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.75 --ldp"
fixed_params='--da-tr 0 --da-rot 2 --da-zoom 1 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 90'
start-aqp salle exp4ldp3 "THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 1.0 --ldp"
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
    $1 $2 $3
fi
