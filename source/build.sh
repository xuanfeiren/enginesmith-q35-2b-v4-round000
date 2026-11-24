#!/usr/bin/env bash
# Builds target/engine: NKI kernels -> NEFFs (build time only, Python/NKI/neuronx-cc), embedded into one
# native executable that links libnrt.
set -euo pipefail
cd "$(dirname "$0")"
V=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
export PATH=$V/bin:$PATH                 # nki shells out to neuronx-cc
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
# NKI's documented route runs each kernel once after compiling it: point it at the run's experiment core.
export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-$(( ${ENGINESMITH_NEURON_CORE:-0} + 2 ))}
mkdir -p target
rm -rf target/neff                       # only the current variants get embedded (the cache is target/neff-cache)
"$V/bin/python" kernels/build_neffs.py target/neff
# embed every NEFF
{
  echo '    .section .rodata'
  i=0
  for f in target/neff/*.neff; do
    n=$(basename "$f" .neff)
    echo "    .balign 64"
    echo "neff_${i}_start: .incbin \"$PWD/$f\""
    echo "neff_${i}_end:"
    echo "neff_${i}_name: .asciz \"$n\""
    i=$((i+1))
  done
  echo '    .section .data.rel.ro'
  echo '    .globl engine_neffs'
  echo '    .balign 8'
  echo 'engine_neffs:'
  for j in $(seq 0 $((i-1))); do
    echo "    .quad neff_${j}_name, neff_${j}_start, neff_${j}_end"
  done
  echo '    .globl engine_neff_count'
  echo '    .balign 4'
  echo "engine_neff_count: .long $i"
  echo '    .section .note.GNU-stack,"",@progbits'
} > target/neffs.S
NRT=/opt/aws/neuron
g++ -O2 -std=c++17 -I$NRT/include -Isrc src/engine.cpp target/neffs.S -o target/engine \
    -L$NRT/lib -lnrt -lpthread -Wl,-rpath,$NRT/lib
echo "built target/engine"
