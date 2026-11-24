#!/bin/bash
# bld.sh TAG NAME... : build variants of the engine's current kernels into scratch/exp/TAG/neff
set -e
E=/home/ubuntu/es-wave3/tasks/q35-2b/runs/q35-2b-20261106-064150/work/013/engine
X=/home/ubuntu/es-wave3/tasks/q35-2b/runs/q35-2b-20261106-064150/work/013/scratch/exp/$1; shift
mkdir -p $X/k; cp $E/kernels/*.py $X/k/
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
export PATH=$V/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_VISIBLE_CORES=18
$V/bin/python $X/k/build_neffs.py $X/neff "$@"
