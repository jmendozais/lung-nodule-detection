#!/bin/bash

. task-util.sh
. lnd-util.sh
. pbs.sh

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
#python util.py --single-froc --model ${model} 
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
da_params='--da-rot 0 --da-zoom 1 --da-is 0 --da-flip 1'
fixed_params="--model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 150 --blob-rad 32 ${da_params}"
send_job manati gpu tr0.06 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-tr 0.06"
send_job manati gpu tr0.09 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-tr 0.09"
send_job manati gpu tr0.12 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-tr 0.12"
send_job manati gpu tr0.15 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-tr 0.15"
send_job manati gpu tr0.18 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-tr 0.18"
}

function exp3trcmp {
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.06-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.09-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.12-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.15-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.0-zm1.0-tr0.18-rr0-fl1-lr0.001-sbf-0.7-aam-val'
)

labels=(
'No translation'
'$t \in [-0.06, 0.06)$'
'$t \in [-0.09, 0.09)$'
'$t \in [-0.12, 0.12)$'
'$t \in [-0.15, 0.15)$'
'$t \in [-0.18, 0.18)$'
)
froc '3-tr-cmp' ${methods} ${labels} 
}

# space [3, 6, 9, 12, 15]
function exp3rot {
da_params='--da-tr 0 --da-rot 12 --da-zoom 1 --da-is 0 --da-flip 1'
send_job manati gpu rot12 "cd ~/lung-nodule-detection; python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

da_params='--da-tr 0 --da-rot 18 --da-zoom 1 --da-is 0 --da-flip 1'
send_job manati gpu rot18 "cd ~/lung-nodule-detection; python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"

da_params='--da-tr 0 --da-rot 24 --da-zoom 1 --da-is 0 --da-flip 1'
send_job manati gpu rot24 "cd ~/lung-nodule-detection; python lnd.py --model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 90 --blob-rad 32 ${da_params}"
}

function exp3rotcmp {
#methods=(
#'5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
#'5P-da-is0.0-zm1.0-tr0.0-rr6-fl1-lr0.001-sbf-0.7-aam-val'
#'5P-da-is0.0-zm1.0-tr0.0-rr12-fl1-lr0.001-sbf-0.7-aam-val'
#'5P-da-is0.0-zm1.0-tr0.0-rr18-fl1-lr0.001-sbf-0.7-aam-val'
#'5P-da-is0.0-zm1.0-tr0.0-rr24-fl1-lr0.001-sbf-0.7-aam-val'
#)
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr18-fl1-lr0.001-sbf-0.7-aam-val'
)
#labels=(
#'No rotation'
#'$\theta \in [-6,6)$'
#'$\theta \in [-12,12)$'
#'$\theta \in [-18,18)$'
#'$\theta \in [-24,24)$'
#)
labels=(
'$\theta \in [-18,18)$'
)

froc '3-rot-cmp' ${methods} ${labels} 
}

# zoom [1.05, 1.1, 1.15, 1.2, 1.25]
function exp3zoom {
da_params="--da-tr 0 --da-rot 0 --da-is 0 --da-flip 1"
fixed_params="--model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 180 --blob-rad 32 ${da_params}"
send_job manati gpu zoom1.05 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-zoom 1.05"
send_job manati gpu zoom1.1 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-zoom 1.1"
send_job manati gpu zoom1.15 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-zoom 1.15"
send_job manati gpu zoom1.2 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-zoom 1.2"
send_job manati gpu zoom1.25 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-zoom 1.25"
send_job manati gpu zoom1.3 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-zoom 1.3"
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
'$s \in [1.0, 1.05)$'
'$s \in [1.0, 1.1)$'
'$s \in [1.0, 1.15)$'
'$s \in [1.0, 1.2)$'
'$s \in [1.0, 1.25)$'
'$s \in [1.0, 1.3)$'
)

froc '3-zoom-cmp' ${methods} ${labels} 
}

# intensity shift [0.2, 0.4, 0.5, 0.6, 0.8]
function exp3int {
da_params="--da-tr 0 --da-rot 0 --da-flip 1 --da-zoom 1"
fixed_params="--model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 150 --blob-rad 32 ${da_params}"
send_job manati gpu is0.1 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-is 0.1"
send_job manati gpu is0.2 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-is 0.2"
send_job manati gpu is0.3 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-is 0.3"
send_job manati gpu is0.4 "cd lung-nodule-detection; python lnd.py ${fixed_params} --da-is 0.4"
}

# TODO: repeat the experiment reducing the padding on ROI extraction (current pad=1.5)
function exp3intcmp {
methods=(
'5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.1-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.2-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.3-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
'5P-da-is0.4-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
)
#'5P-da-is0.5-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val'
labels=('No intensity shift'
'$v \in [-0.1*\sigma, 0.1*\sigma)$'
'$v \in [-0.2*\sigma, 0.2*\sigma)$'
'$v \in [-0.3*\sigma, 0.3*\sigma)$'
'$v \in [-0.4*\sigma, 0.4*\sigma)$'
)
#'$v \in [-0.5*\sigma, 0.5*\sigma)$'
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
# F +0.35
# ROT 5.49
# TR 5.26
# IS 5.05 * Why to few 
# ZOOM/S 4.81 * TO CHECK

# F, FR, FRT, FRTI, FRTIS

function exp3sch {
# FRT
da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1 --da-is 0.0 --da-flip 1'
fixed_params="--model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 200 --blob-rad 32 ${da_params}"
send_job manati gpu FRT "cd lung-nodule-detection; python lnd.py ${fixed_params} --load-model"

# FRTI
#da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1 --da-is 0.2 --da-flip 1'
#fixed_params="--model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 200 --blob-rad 32 ${da_params}"
#send_job manati gpu FRTI "cd lung-nodule-detection; python lnd.py ${fixed_params}"

# FRTIS
#da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
#fixed_params="--model 5P-da --model-selection --roi-size 64 --lr 0.001 --epochs 200 --blob-rad 32 ${da_params}"
#send_job manati gpu FRTIS "cd lung-nodule-detection; python lnd.py ${fixed_params}"
}


function exp3schcmp {
methods=(
"5P-da-is0.0-zm1.0-tr0.0-rr0-fl0-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.0-zm1.0-tr0.0-rr0-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.0-zm1.0-tr0.0-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.0-zm1.0-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.0-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-da-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
)
#methods=(
#"5P-da-is0.0-zm1.0-tr0.0-rr18-fl1-lr0.001-sbf-0.7-aam-val"
#)

labels=(
"No D.A."
"D.A. F"
"D.A. F+R"
"D.A. F+R+T"
"D.A. F+R+T+I"
"D.A. F+R+T+I+S"
)
#labels=(
#"D.A. F+R"
#)

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
fixed_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 250'
#send_job manati gpu dp0.1 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.1"
#send_job manati gpu dp0.15 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.15"
#send_job manati gpu dp0.2 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.2"
send_job manati gpu dp0.25 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.25"
send_job manati gpu dp0.3 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.3"
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
fixed_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1 --roi-size 64 --lr 0.001 --epochs 250'
#send_job manati gpu ldp1 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.025 --lidp"
#send_job manati gpu ldp2 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.05 --lidp"
#send_job manati gpu ldp3 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.075 --lidp"
#send_job manati gpu ldp4 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.1 --lidp"
#send_job manati gpu ldp5 "cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.125 --lidp"
send_job manati gpu ldp6 " cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.15 --lidp"
send_job manati gpu ldp7 " cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.175 --lidp"
send_job manati gpu ldp8 " cd lung-nodule-detection; python lnd.py --model 5P-dp --model-selection ${fixed_params} --dropout 0.2 --lidp"
}

function exp4ldpcmp {
methods=(
"5P-da-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.05-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.075-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.1-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.125-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.15-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.175-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.2-sbf-0.7-aam-val"
)
labels=(
"No dropout"
"Linear Inc. DP 0.05"
"Linear Inc. DP 0.075"
"Linear Inc. DP 0.1"
"Linear Inc. DP 0.125"
"Linear Inc. DP 0.15"
"Linear Inc. DP 0.175"
"Linear Inc. DP 0.2"
)
methods=(
"5P-da-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.075-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.1-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.125-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.15-sbf-0.7-aam-val"
"5P-dp-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001lidp-0.175-sbf-0.7-aam-val"
)
labels=(
"No dropout"
"Linear Inc. DP 0.075"
"Linear Inc. DP 0.1"
"Linear Inc. DP 0.125"
"Linear Inc. DP 0.15"
"Linear Inc. DP 0.175"
)
froc '4-ldp-cmp' ${methods} ${labels} 
}

THEANO_OPTS="THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn"
da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1'

# TODO: sel epoch 140 for 6P
function exp5conv {
fixed_params="${da_params} --lr 0.001 --epochs 300 --dropout 0.125 --lidp --fc 1 --filters 64"
send_job manati gpu 5c4 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --dp-intercept 0.125 --conv 4 --roi-size 32"
send_job manati gpu 5c5 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --dp-intercept 0.0 --conv 5 --roi-size 64"
send_job manati gpu 5c6 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --dp-intercept -0.125 --conv 6 --roi-size 128"
}

function exp5convcmp {
methods=(
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-4-fil-64-fc-1-lidp-0.125-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-5-fil-64-fc-1-lidp-0.125-sbf-0.7-aam-val"
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
fixed_params="${da_params} --lr 0.001 --epochs 300 --dropout 0.125 --lidp --fc 1 --dp-intercept -0.125 --conv 6 --roi-size 128"
send_job manati gpu fi16 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --filters 16"
send_job manati gpu fi32 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --filters 32"
send_job manati gpu fi48 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --filters 48"
send_job manati gpu fi64 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --filters 64"
}

function exp5ficmp {
methods=(
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-16-fc-1-lidp-0.125-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-1-lidp-0.125-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-48-fc-1-lidp-0.125-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-64-fc-1-lidp-0.05-sbf-0.7-aam-val"
)
labels=(
"Convnet(6, 16)"
"Convnet(6, 32)"
"Convnet(6, 48)"
"Convnet(6, 64)"
)
froc '5-fil-cmp' ${methods} ${labels} 
}

function exp5fc {
fixed_params="${da_params} --lr 0.001 --epochs 300 --dropout 0.125 --lidp --dp-intercept -0.125 --conv 6 --roi-size 128 --filters 32"
send_job manati gpu fc0 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --fc 0"
send_job manati gpu fc2 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --fc 2"
send_job manati gpu fc3 "cd lung-nodule-detection; python lnd.py --model convnet --model-selection ${fixed_params} --fc 3"
}

function exp5fccmp {
methods=(
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-0-lidp-0.125-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-1-lidp-0.125-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-2-lidp-0.125-sbf-0.7-aam-val"
"convnet-is0.2-zm1.25-tr0.12-rr18-fl1-lr0.001-co-6-fil-32-fc-3-lidp-0.125-sbf-0.7-aam-val"
)

labels=(
"No FC layers"
"1 FC layer"
"2 FC layer"
"3 FC layer"
)
froc '5-fc-cmp' ${methods} ${labels} 
}

function sleval {
fixed_params="${da_params} --lr 0.001 --epochs 200 --dropout 0.125 --lidp --dp-intercept -0.125 --conv 6 --roi-size 128 --filters 32"
echo send_job manati gpu eval1 "cd lung-nodule-detection; python lnd.py --model convnet --model-evaluation ${fixed_params} --fc 1"
echo send_job manati gpu eval2 "cd lung-nodule-detection; python lnd.py --model convnet --model-evaluation ${fixed_params} --fc 2"
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

