. task-util.sh

function dog_hp_start {
start bezier dog1 "python detect.py --detector dog --threshold 0.1 --save-blobs"
start babbage dog2 "python detect.py --detector dog --threshold 0.4 --save-blobs"
start prim dog3 "python detect.py --detector dog --threshold 0.7 --save-blobs"
start phong dog4 "python detect.py --detector dog --threshold 1.0 --save-blobs"
start minsky dog5 "python detect.py --detector dog --threshold 1.3 --save-blobs"
start voronoi dog6 "python detect.py --detector dog --threshold 1.6 --save-blobs"
}

function froc {
methods=$1
labels=$2
tmp1=/tmp/${RANDOM}.txt
tmp2=/tmp/${RANDOM}.txt
touch ${tmp1}
touch ${tmp2}

for ((i=0; i <${#methods[@]}; ++i)); do
    echo "data/${methods[i]}-lidc-blobs-froc.npy,${labels[i]}" >> ${tmp1}
    echo "data/${methods[i]}-lidc-blobs-abpi.txt,${labels[i]}" >> ${tmp2}
done
python util.py --froc --list ${tmp1} --bpif ${tmp2} --out data/cmp-froc --max 100
}

function hp_froc {
method=$1
label=$2
ts=$3
tmp1=/tmp/${RANDOM}.txt
tmp2=/tmp/${RANDOM}.txt
tmp3=/tmp/${RANDOM}.txt
suffix="-frocs.npy"
dataset="-lidc-blobs"
touch ${tmp1}
touch ${tmp2}
touch ${tmp3}

for i in "${ts[@]}"; do
    echo "data/${method}-${i}-lidc-blobs-froc.npy,${label} ${i}" >> ${tmp1}
    echo "data/${method}-${i}-lidc-blobs-abpi.txt,${label} ${i}" >> ${tmp2}
    echo "data/${method}-${i}-jsrt140p-blobs-abpi.txt,${label} ${i}" >> ${tmp3}
done

python util.py --froc --list ${tmp1} --bpif ${tmp2} --out data/${method}-froc --max 100
python util.py --abpi --list ${tmp2} --out data/${method}-lidc-abpi --max 200
python util.py --abpi --list ${tmp3} --out data/${method}-jsrt-abpi --max 200
}

function dog_hp_froc {
ts=(0.1 0.4 0.7 1.0 1.3 1.6)
hp_froc dog DoG ${ts}
}

# LoG

function log_hp_start {
start bezier log1 "python detect.py --detector log --threshold 0.2 --save-blobs"
start babbage log2 "python detect.py --detector log --threshold 0.4 --save-blobs"
start prim log3 "python detect.py --detector log --threshold 0.6 --save-blobs"
start phong log4 "python detect.py --detector log --threshold 0.8 --save-blobs"
start minsky log5 "python detect.py --detector log --threshold 1.0 --save-blobs"
}

function log_hp_froc {
ts=(0.2 0.4 0.6 0.8 1.0)
hp_froc log LoG ${ts}
}

# DoH

function doh_hp_start {
start bezier doh1 "python detect.py --detector doh --threshold 0.05 --save-blobs"
start babbage doh2 "python detect.py --detector doh --threshold 0.075 --save-blobs"
start prim doh3 "python detect.py --detector doh --threshold 0.1 --save-blobs"
start phong doh4 "python detect.py --detector doh --threshold 0.125 --save-blobs"
start minsky doh5 "python detect.py --detector doh --threshold 0.15 --save-blobs"
}

function doh_hp_froc {
ts=(0.05 0.075 0.1 0.125 0.15)
hp_froc doh DoH ${ts}
}

# WMCI

function wmci_hp_start {
start bezier wmci1 "python detect.py --detector wmci --threshold 0.3 --save-blobs"
start babbage wmci2 "python detect.py --detector wmci --threshold 0.4 --save-blobs"
start prim wmci3 "python detect.py --detector wmci --threshold 0.5 --save-blobs"
start phong wmci4 "python detect.py --detector wmci --threshold 0.6 --save-blobs"
start minsky wmci5 "python detect.py --detector wmci --threshold 0.7 --save-blobs"
}

function wmci_hp_froc {
ts=(0.3 0.4 0.5 0.6 0.7)
hp_froc wmci WMCI ${ts}
}

# SBF

function sbf_hp_start {
start bezier sbf1 "python detect.py --detector sbf --threshold 0.3 --save-blobs"
start babbage sbf2 "python detect.py --detector sbf --threshold 0.4 --save-blobs"
start prim sbf3 "python detect.py --detector sbf --threshold 0.5 --save-blobs"
start phong sbf4 "python detect.py --detector sbf --threshold 0.6 --save-blobs"
start minsky sbf5 "python detect.py --detector sbf --threshold 0.7 --save-blobs"
}

function sbf_hp_froc {
ts=(0.3 0.4 0.5 0.6 0.7)
hp_froc sbf SBF ${ts}
}

# Comparing

function cmp_froc {
#methods=('dog-1.5' 'log-0.5' 'doh-0.1' 'wmci-0.5' 'sbf-0.4')
#labels=('DoG 1.5' 'LoG 0.5' 'DoH 0.1' 'WMCI 0.5' 'SBF 0.4')
methods=('dog-1.0' 'log-0.4' 'doh-0.1' 'sbf-0.4')
labels=('DoG 1.0' 'LoG 0.4' 'DoH 0.1' 'SBF 0.4')
froc ${methods} ${labels}
}

if [ "$1" = "dog-start" ]; then
    dog_hp_start
elif [ "$1" = "dog-froc" ]; then
    dog_hp_froc
elif [ "$1" = "log-start" ]; then
    log_hp_start
elif [ "$1" = "log-froc" ]; then
    log_hp_froc
elif [ "$1" = "doh-start" ]; then
    doh_hp_start
elif [ "$1" = "doh-froc" ]; then
    doh_hp_froc
elif [ "$1" = "wmci-start" ]; then
    wmci_hp_start
elif [ "$1" = "wmci-froc" ]; then
    wmci_hp_froc
elif [ "$1" = "sbf-start" ]; then
    sbf_hp_start
elif [ "$1" = "sbf-froc" ]; then
    sbf_hp_froc
elif [ "$1" = "cmp-froc" ]; then
    cmp_froc
elif [ "$1" = "stop" ]; then
    stop-task $2
elif [ "$1" = "qall" ]; then
    qall
else
    echo "Undefined command"
fi
