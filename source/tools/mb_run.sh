#!/bin/bash
# run.sh NAME VARIANT [ENV=..]...  : compile + profile, print summary
cd "$(dirname "$0")"
n=$1; v=$2; shift 2
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0; export PATH=$V/bin:/opt/aws/neuron/bin:$PATH
rm -rf out_$n; env "$@" python mb.py $v out_$n > log_$n.txt 2>&1 || { tail -5 log_$n.txt; exit 1; }
cd out_$n && flock /tmp/engine-gpu-18-es-gptoss.lock env NEURON_RT_VISIBLE_CORES=18 neuron-explorer capture -n kernel.neff -s prof.ntff --num-exec 3 > cap.log 2>&1
neuron-explorer view -n kernel.neff -s prof_exec_3.ntff --output-format summary-text > sum.txt 2>&1
tt=$(grep -E "^ *total_time " sum.txt | awk '{print $2}'); th=$(grep throttle_activity_1_active_time_nc6 sum.txt | awk '{print $2}'); pe=$(grep "tensor_engine_active_time_percent" sum.txt | awk '{print $2}')
echo "$n $v $* total=$tt throttle1=$th pe=$pe"
