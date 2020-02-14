#!/bin/bash

# Check for proper usage
[[ $# -eq 0 ]] && cat README.txt && exit 1
[[ "$1" == "-h" ]] && cat README.txt && exit 0

# Arrange input arguments and handle variable input args
startval=$1
endval=$2
step=$3
param=$4
filename=$5
raw_args=("$@")

ARGS=""
[[ $# -gt  5 ]] && ARGS=${raw_args[@]:5}

# Set up parameter arrays and progress trackers
arr=($(seq $startval $step $endval))
count=0
len=${#arr[@]}
blocksize=5
partition=$(( $len / 5 ))
marker=$(( $blocksize < $partition ? $blocksize : $partition ))

# Create data directory and add to logfile
[ ! -d "data/$filename" ] && mkdir data/$filename
[ ! -f "data/$filename/README.txt" ] && touch data/${filename}/README.txt
echo "Parameter: $param" >> data/$filename/README.txt
echo "Range: $startval:$step:$endval" >> data/$filename/README.txt
echo "Other args: $ARGS" >> data/$filename/README.txt

# Begin main runner loop, working in batches
for e in ${arr[@]}; do
  let "++count"
  printf -v num_str "%2.3f" $e 
  ./runVisionSim.out -initPopSize 25 -$param $e ARGS > "data/$filename/$filename${num_str}.txt" &
  
  [[ $(($count % $marker)) -eq 0 ]] && wait && echo "Batch complete"
done

wait
echo "Task complete"
