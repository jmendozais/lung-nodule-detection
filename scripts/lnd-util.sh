. task-util.sh
# Variables
update_salle="source ~/juliomb/.bashrc; cd ~/juliomb/lung-nodule-detection; git pull --rebase origin master"
salle1="THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn"
salle2="THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer_including=cudnn"
salle3="THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,optimizer_including=cudnn"

# Functions
function start-liv {
    start $1 $2 "cd lung-nodule-detection; ${3}"
}
function start-aqp {
    start $1 $2 "source ~/juliomb/.bashrc"
    run $1 $2 "cd ~/juliomb/lung-nodule-detection; ${3}"
}

function froc-detect {
methods=$1
labels=$2
tmp1=/tmp/${RANDOM}.txt
tmp2=/tmp/${RANDOM}.txt
touch ${tmp1}
touch ${tmp2}

for ((i=0; i <${#methods[@]}; ++i)); do
    echo "data/${methods[i]}-froc.npy,${labels[i]}" >> ${tmp1}
    echo "data/${methods[i]}-abpi.txt,${labels[i]}" >> ${tmp2}
done
python util.py --froc --list ${tmp1} --bpif ${tmp2} --out data/cmp-froc --max 200
}

function froc {
out=$1
methods=$2
labels=$3
tmp1=/tmp/${RANDOM}.txt
touch ${tmp1}
for ((i=0; i <${#methods[@]}; ++i)); do
    echo "data/${methods[i]}-froc.npy,${labels[i]}" >> ${tmp1}
done
python util.py --froc --list ${tmp1} --out data/${out} --max 10
}


