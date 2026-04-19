#!/bin/bash

module load cuda/12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
conda activate pychannel

python testGPU.py