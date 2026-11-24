# q35-2b-v4-round000

A specialized inference engine for [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.4 LTS
- Reference: custom, correctness only (no speed baseline)
- Judged under evaluation version 4
- Decode: one forward pass per generated token, no speculative decoding (a rule in the session brief; the harness cannot check it)
- Source and data: https://github.com/xuanfeiren/enginesmith-q35-2b-v4-round000
- Search: 1 rounds by Claude Code 2.1.317 (Claude Code) (model not reported, reasoning effort not reported)

## Notes from the author

- No framework runs this model correctly on Trainium (the speed section lists what was tried), so the reference is a CPU implementation that only decides correctness, and the score is the fraction of the hardware bound (evaluation version 4), not a speedup.
- This run re-evaluated one engine under evaluation version 4, with no new coding sessions: round 013 of an earlier search (14 rounds by Claude Code with Claude Opus 5.5, agent cost $302.65). That search is in `search-history.csv` and in "How the search went" below, where its scores are against the CPU reference's speed (evaluation version 3), not fractions of the bound.

## Speed (held-out final evaluation; no framework baseline)

The reference decides correctness only and is no speed baseline (frameworks tried: NxD Inference 0.10: no qwen3_5 architecture. NxD Inference contrib PR #141 (SDK 2.32 and its own 2.29.1): wrong from the 2nd generated token (logit gaps 6.9-8.6). PyTorch/XLA with stock transformers: correct but its chunked prefill does not compile (13-169 s per prompt). vllm-neuron 0.24, optimum-neuron 0.4.5: no qwen3_5.); the score is the weighted geometric mean of the fractions of the hardware bound, bound time / engine time (a nominal bound where it is not computed, so then only a relative measure). Score 0.071; prefill 0.030, decode 0.170. Medians over the paired rounds, 65 tokens per request. The bound is not computed here: a hybrid model (linear_attention layers).

| prompt tokens | prefill, ms | decode, ms per token | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 10.0 | 5.8 | not computed | not computed |
| 1,024 | 31.4 | 5.8 | not computed | not computed |
| 4,096 | 121.7 | 5.9 | not computed | not computed |

Correctness: passed; 3 of 2080 judged steps mismatch (at most 208 allowed).

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.03x as long as one request for the same tokens at 256 prompt tokens (386.1 ms against 373.9 ms; each request 5.95 to 6.33 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went (the earlier search)

- 14 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 2226 min.
- First correct engine: round 001, 46 min after the start.
- First engine as fast as the reference (score 1.0): round 001, 46 min after the start.
- Agent usage: 4,032 input and 2,611k output tokens, 541,910k cache reads, $302.65 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 45.7 | 3.37x | 3.39x | 3.35x | 120 / 128k | 1,281 |
| 002 | optimize | accepted (score_improved) | 297.1 | 17.33x | 29.11x | 10.31x | 346 / 323k | 1,913 |
| 003 | optimize | accepted (score_improved) | 628.4 | 22.11x | 40.99x | 11.93x | 388 / 325k | 2,206 |
| 004 | optimize | accepted (score_improved) | 906.6 | 23.56x | 44.72x | 12.41x | 466 / 279k | 2,385 |
| 005 | optimize | rejected (insufficient_improvement) | 1,102.5 | 23.07x | 43.48x | 12.24x | 356 / 230k | 2,524 |
| 006 | optimize | accepted (case_improved) | 1,213.3 | 23.66x | 45.95x | 12.18x | 250 / 102k | 2,572 |
| 007 | optimize | accepted (case_improved) | 1,355.4 | 23.74x | 47.01x | 11.98x | 324 / 191k | 2,643 |
| 008 | optimize | rejected (rejected_by_operator: fp8 copy of the LM head used to screen tokens breaks the bf16-weights rule (skill kernels-gemv-quant.md, Extra low-precision copies)) | 1,423.5 | 25.21x | 46.16x | 13.77x | 112 / 45k | 2,724 |
| 009 | optimize | accepted (case_improved) | 1,652.8 | 23.56x | 46.44x | 11.95x | 522 / 326k | 2,865 |
| 010 | optimize | accepted (score_improved) | 1,790.5 | 23.96x | 46.77x | 12.27x | 304 / 217k | 2,909 |
| 011 | optimize | rejected (insufficient_improvement) | 2,002.9 | 23.69x | 47.10x | 11.91x | 276 / 199k | 3,341 |
| 012 | optimize | rejected (insufficient_improvement) | 2,130.8 | 23.25x | 45.32x | 11.92x | 260 / 106k | 3,495 |
| 013 | optimize | accepted (case_improved) | 2,204.9 | 23.85x | 46.94x | 12.12x | 258 / 125k | 3,901 |
| 014 | optimize | rejected (stopped) | 2,226.2 | - | - | - | 50 / 13k | 3,901 |

`search-history.csv` has every round of that search with every prompt length.

## This re-evaluation

- 0 coding sessions by Claude Code 2.1.317 (Claude Code) (model not reported, reasoning effort not reported); the search took 27 min.
- First correct engine: round 000, 27 min after the start.
- Agent usage: not reported by the agent.

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 000 | seed | accepted (first_passing_candidate) | 26.7 | 0.071 | 0.030 | 0.170 | - / - | - |

`history.csv` in https://github.com/xuanfeiren/enginesmith-q35-2b-v4-round000 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local the reference):

```sh
hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir models/Qwen3.5-2B
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-q35-2b-v4-round000 --model models/Qwen3.5-2B
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
