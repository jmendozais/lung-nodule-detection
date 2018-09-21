#!/bin/bash
. task-util.sh
. lnd-util.sh
. pbs.sh


function ots {
da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
fixed_params="${da_params} --lr 0.001 --epochs 2 --dropout 0.05 --lidp --dp-intercept -0.1 --conv 6 --roi-size 128 --filters 32 --fc 1"
send_job manati gpu pool2 "cd ~/lung-nodule-detection; python lnd-tl.py --mode ots-feat --model-selection --pool-layer 2 ${fixed_params}"
send_job manati gpu pool3 "cd ~/lung-nodule-detection; python lnd-tl.py --mode ots-feat --model-selection --pool-layer 3 ${fixed_params}"
send_job manati gpu pool4 "cd ~/lung-nodule-detection; python lnd-tl.py --mode ots-feat --model-selection --pool-layer 4 ${fixed_params}"
}

# Call
$1 $2 $3
