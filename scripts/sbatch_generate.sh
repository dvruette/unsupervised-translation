#!/bin/bash

RUN_ID=$1  # e.g. 1lvyy9r2

RUN_FOLDER=$(find "$SCRATCH" -name $RUN_ID | head -1)
RUN_PATH=$(ls $RUN_FOLDER/checkpoints/*.ckpt | tail -1)

if [ -z "$RUN_PATH" ]; then
    echo "No checkpoint found for run $RUN_ID"
    exit 1
fi
echo "Found checkpoint $RUN_PATH for run $RUN_ID"

export COMMAND="src/unsupervised/generate.py model_path=\"$RUN_PATH\" data.max_batches=-1"

mkdir -p logs

sbatch -J "$RUN_ID-generate" \
-o "logs/$RUN_ID-generate-%j.out" \
--mail-type=END,FAIL \
--nodes=1 \
--ntasks-per-node=1 \
--time=24:00:00 \
--mem-per-cpu=8G \
--gpus=1 \
--gres=gpumem:10G \
--tmp=5G \
< scripts/srun_python.sh 

echo "Job submitted, results will be written to ./logs/$RUN_ID-generate-{job_id}.out"
