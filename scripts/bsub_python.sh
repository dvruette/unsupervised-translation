#!/bin/bash

MAX_TIME=24:00
N_CPU=4
N_GPU=4

bsub -J "$1" -o lsf.$1_%J -n "$N_CPU" -W "$MAX_TIME" -R "rusage[mem=4096,ngpus_excl_p=$N_GPU,scratch=5000]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.1.1 eth_proxy
# Make sure that local packages take precedence over preinstalled packages
export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:$PYTHONPATH
python $1
ENDBSUB

