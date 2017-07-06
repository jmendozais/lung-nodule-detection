function start {
machine=$1
session=$2 
cmd=$3
ssh $machine "/usr/bin/screen -S ${session} -dm bash;"
ssh $machine "/usr/bin/screen -S ${session} -X stuff \"cd lung-nodule-detection;\"^M"
ssh $machine "/usr/bin/screen -S ${session} -X stuff \"${cmd}\"^M"
}

function close {
ssh $1 screen -X -S $2 quit
}

function stop-task {
close bezier $1
close babbage $1
close prim $1
close phong $1
close minsky $1
}

function qall {
user=$(whoami)
ssh bezier "killall --user ${user} screen"
ssh babbage "killall --user ${user} screen"
ssh prim "killall --user ${user} screen"
ssh phong "killall --user ${user} screen"
ssh minsky "killall --user ${user} screen"
}
