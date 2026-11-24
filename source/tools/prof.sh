#!/bin/bash
# prof.sh NEFF OUTDIR : capture on core 18 under our lock, json + summary in OUTDIR
set -e
export PATH=/opt/aws/neuron/bin:$PATH
neff=$(realpath $1); mkdir -p $2; cd $2
flock /tmp/engine-gpu-18-es-gptoss.lock env NEURON_RT_VISIBLE_CORES=18 neuron-explorer capture -n $neff -s prof.ntff --num-exec 3 > cap.log 2>&1
neuron-explorer view -n $neff -s prof_exec_3.ntff --output-format summary-text > sum.txt 2>&1
neuron-explorer view -n $neff -s prof_exec_3.ntff --output-format json > view.log 2>&1
rm -f prof.ntff prof_exec_2.ntff
grep -E "^ *total_time |throttle_activity_1_active_time_nc.*percent|tensor_engine_active_time_percent" sum.txt
