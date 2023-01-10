#!/bin/bash

MAX_TIME=24:00:00
N_CPU=4
N_GPU=4

sbatch -J "$1" \
--nodes=1 \
--ntasks-per-node="$N_GPU" \
--time="$MAX_TIME" \
--mem-per-cpu=8G \
--gpus="$N_GPU" \
--gres=gpumem:10G \
--tmp=5G \
< scripts/srun_python.sh "$2"

