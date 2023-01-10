#!/bin/sh
module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.1.1 eth_proxy
# Make sure that local packages take precedence over preinstalled packages
export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:$PYTHONPATH
export SCRATCH=$SCRATCH/unsupervised-translation
srun python $COMMAND