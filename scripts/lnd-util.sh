. task-util.sh

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
