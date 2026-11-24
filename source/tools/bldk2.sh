#!/bin/bash
# bldk2.sh TAG KERNEL.py BUILD.py NAME... : as bldk.sh with a custom build_neffs.py
set -e
X=/home/ubuntu/es-wave3/tasks/q35-2b/runs/q35-2b-20261106-064150/work/013/scratch/exp/$1; K=$2; B=$3; shift 3
mkdir -p $X/k; cp $B $X/k/build_neffs.py; cp $K $X/k/qwen35.py
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
export PATH=$V/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_VISIBLE_CORES=18
$V/bin/python $X/k/build_neffs.py $X/neff "$@"
