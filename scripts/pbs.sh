#!/bin/bash
/bin/bash config.sh
set -ex 
# TODO: move to config file
user_email="jmendozais@gmail.com"

function send_job {
remote=$1
task_type=$2
task_name=$3
cmd=$4

tmp=/tmp/${RANDOM}.pbs
touch ${tmp}

echo "#PBS -N ${task_name}" >> ${tmp}
if [ ${task_type} = "cpu" ]; 
then 
    echo "#PBS -l nodes=1:ppn=14:gpus=0" >> ${tmp}
else 
    echo "#PBS -l nodes=1:ppn=14:gpus=1" >> ${tmp}
fi

# TODO: enable
echo "#PBS -m ae -M ${user_email}" >> ${tmp}

echo "${cmd}" >> ${tmp}

if ssh ${remote} '[ ! -d ~/.pbs ]';
then
    ssh ${remote} "mkdir -p ~/.pbs"
fi

scp ${tmp} "${remote}:~/.pbs/${task_name}.pbs"
ssh ${remote} "cd ~/.pbs; qsub ~/.pbs/${task_name}.pbs"
}

# TEST
#send_job manati gpu test1 "echo \'Test command\'"
#send_job manati gpu segment-train "cd ~/lung-nodule-detection; python segment.py --train"
#send_job manati gpu segment-dbs "cd ~/lung-nodule-detection; python segment.py --segment"
#send_job manati gpu det1 "cd ~/lung-nodule-detection; python detect.py --save-blobs --threshold 0.5"
#send_job manati gpu det2 "cd ~/lung-nodule-detection; python detect.py --save-blobs --threshold 0.7"
#THEANO_OPTS="THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn"
#da_params='--da-tr 0.12 --da-rot 18 --da-zoom 1.25 --da-is 0.2 --da-flip 1'
#fixed_params="${da_params} --lr 0.001 --epochs 2 --dropout 0.05 --lidp --dp-intercept -0.1 --conv 6 --roi-size 128 --filters 32"
#send_job manati gpu det2 "cd ~/lung-nodule-detection; ${THEANO_OPTS},device=cuda1 python lnd.py --model convnet --model-selection ${fixed_params} --fc 3"
