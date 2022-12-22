#!/bin/bash

MAX_TIME=24:00:00
N_CPU=4
N_GPU=4

sbatch -J "$1" --nodes=1 --ntasks-per-node="$N_GPU" --time="$MAX_TIME" --mem-per-cpu=8G --gpus="$N_GPU" --gres=gpumem:10G --tmp=5G <<ENDBSUB
#!/bin/sh
module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.1.1 eth_proxy
# Make sure that local packages take precedence over preinstalled packages
export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:$PYTHONPATH
export SCRATCH=$SCRATCH/unsupervised-translation
srun python $1
ENDBSUB

