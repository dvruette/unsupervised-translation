#!/bin/bash

RUN_ID=$1  # e.g. 1lvyy9r2

RUN_FOLDER=$(find "$SCRATCH" -name $RUN_ID | head -1)
RUN_PATH=$(ls $RUN_FOLDER/checkpoints/*.ckpt | tail -1)

if [ -z "$RUN_PATH" ]; then
    echo "No checkpoint found for run $RUN_ID"
    exit 1
fi
echo "Found checkpoint $RUN_PATH for run $RUN_ID"

sbatch -J "generate-$1" \
-o "logs/generate-$1-%j.out" \
--mail-type=END,FAIL \
--nodes=1 \
--ntasks-per-node=1 \
--time=24:00:00 \
--mem-per-cpu=8G \
--gpus=1 \
--gres=gpumem:10G \
--tmp=5G \
< scripts/srun_python.sh "src/unsupervised/generate.py model_path=$RUN_PATH"
