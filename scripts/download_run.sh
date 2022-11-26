#!/bin/bash

USER=$1
RUN_ID=$2

RUN_PATH=$(ssh $USER "echo \$(ls \$(find \$SCRATCH -name ums9xxkw | head -1)/checkpoints/*.ckpt | tail -1)")

scp $USER:$RUN_PATH ./outputs/$RUN_ID.ckpt
