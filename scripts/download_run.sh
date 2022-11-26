#!/bin/bash

USER=$1
RUN_ID=$2

RUN_PATH=$(ssh $USER "echo \$(ls \$(find \$SCRATCH -name $RUN_ID | head -1)/checkpoints/*.ckpt | tail -1)")
echo "Downloading $RUN_PATH as $RUN_ID.ckpt"

scp $USER:$RUN_PATH ./outputs/$RUN_ID.ckpt
