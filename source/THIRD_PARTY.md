# Third-party code

No third-party source is copied into this engine. The NKI kernels (`kernels/qwen35.py`) and the host code
(`src/`) were written for this engine against the public NKI API (`nki.language`, `nki.isa`, nki 0.6.0) and the
public libnrt headers (`/opt/aws/neuron/include/nrt/`). Numerics follow Hugging Face transformers'
`models/qwen3_5/modeling_qwen3_5.py` (Apache-2.0), read as a specification, not copied.

Build-time tools (not shipped): the AWS Neuron SDK (neuronx-cc 2.27.5334, nki 0.6.0) from the public Neuron
repositories. Run-time dependency: libnrt (Neuron runtime 2.34.10).
