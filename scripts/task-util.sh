function start {
machine=$1
session=$2 
cmd=$3
ssh $machine "/usr/bin/screen -S ${session} -dm bash;"
ssh $machine "/usr/bin/screen -S ${session} -X stuff \"${cmd}\"^M"
}

function run {
machine=$1
session=$2 
cmd=$3
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
close voronoi $1
}

function qall {
read -p "Are you sure you want to abort ALL SESSIONS? (y/n): " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
user=$(whoami)
ssh bezier "killall --user ${user} screen"
ssh babbage "killall --user ${user} screen"
ssh prim "killall --user ${user} screen"
ssh phong "killall --user ${user} screen"
ssh minsky "killall --user ${user} screen"
ssh voronoi "killall --user ${user} screen"
fi
}
