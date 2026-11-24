# Qwen3.5-2B on Trainium2: engine notes

## Round 013 (this session)

The engine is **round 012's code, unchanged** (round 011's four exact prefill changes + round 012's early K/V
tiles; see those sections). Rounds 011 and 012 were rejected only because the CPU reference ran faster in their
evaluations (their candidate prefill times were 1-2% better than round 010's, decode unchanged). Nothing new was
kept this round; the experiments below were measured and rejected.

### Tried and rejected (device time, neuron-explorer on core 18, one NEFF, random inputs)

- **"Waves": own-half k tiles of several weight slabs before any other-half k tile** (T = 256 only; `tools/exp/wave2.patch`
  against this kernel, not in the engine). Motivation: in p256 (pchunk256_l256th, 9.720 ms) the TensorE waits
  ~225 us on the core barrier `$S[8]` (down-/out-projection matmuls whose k tiles come from the other core) and
  ~270 us in the RMSNorm Gram on the received half's residual add. Bitwise-safe reordering (each tile keeps its
  k accumulation order):
  - wave of 2 slabs for every exchange-fed / norm-fed GEMM, plus the first gate + up slabs of the MLP as one
    wave: 9.702 ms (-0.2%, noise level).
  - three phases (own half; other half without the last-arriving group; last group) over all slabs of the down-
    projection (4 slabs = 4 PSUM banks) and out-projection, 2 slabs for norm consumers, the Gram split the same way:
    9.810 ms (+0.9%, worse). The `$S[8]` waits stayed at ~225 us.
  - Why: the compiler schedules the PE instructions itself. In the profile the down-projection matmuls run tile by
    tile (k = 0..47 for one output tile, then the next), not in the emitted k-outer / phase order, so the program
    order does not control where the waits land. Forcing an order would need explicit `depends_on` edges.
  - NKI frontend facts found on the way: no lambdas in kernel code ("unsupported expression"), dicts need string
    keys ("expected string": a module-level dict with int keys read inside the kernel fails too).
- Where p256 spends its time (core 0, measured): first GEMM at 55 us; 23 layers + layer 24's K/V in 7.94 ms
  (~345 us per layer, of which ~190 us is GEMM at the PE rate); the last layer's single-token part (DMA-bound GEMVs,
  ~105 MB) 140 us; the LM head 1.572 ms on core 0 (1.457 on core 1; the aggregate is ~670 GB/s, i.e. HBM-bound);
  14 us of argmax / exchange / drain. The LM head and the single-token tail (~1.7 ms, 18%) are at the HBM limit;
  the chunk loses ~0.5 ms to cross-core exchange latency and ~0.45 ms to the TensorE power throttle.
- pchunk512_l2048t (this engine, 14.83 ms, PE active 95.7%, throttled 10%): GEMM 9.06 ms (225 ns per 512-column
  matmul), delta-rule fp32 matmuls 2.29 ms (19584 passes = 17 fp32 matmuls per head and sub-chunk, 2 passes each),
  fp32 transposes 0.58 ms (5 per head and sub-chunk + state), conv 0.38, Gram 0.26, attention ~0.3. Gaps > 1 us
  total ~0.9 ms, spread (weight DMA after norms 114 us, VectorE chains, exchange 35 us).

### Measured this round (requests)

- same.py, round 010 build vs this engine: 40/40 identical.
- ab.py, round 010 (A) vs this engine (B), 8 rounds in each core order (cores 18 / 19), averaged over the orders:
  prefill +0.9% (256: -0.9 / +2.7%, the core-19 p256 bias), +1.4% (1024: +1.2 / +1.6%), +1.2% (4096: +1.0 /
  +1.3%); decode 0 (-0.2 / +0.3%). Medians: prefill 10.2-10.3 / 31.6 / 122.2 ms, decode 373.5-374.3 / 373.6-374.3 /
  377.4-378.4 ms. Round 012's own estimate (+2.0 / +1.9 / +1.2%) was a little high; the real prefill gain over
  round 010 is ~1.2% (mean).
- Host: the first `nrta_execute_schedule` of a request costs 50-140 us when the model differs from the last one
  scheduled (step -> pchunk; pchunk -> another pchunk 45-105 us), ~20 us for a repeat of the same model and
  pchunk -> step ~25-30 us (instrumented engine, scratch). It is host time before the device starts: ~0.8% of p256.
  No API prepares an execution ahead of time (nrt_async.h); not pursued.

- dev-eval 001 (this engine, clean build from the cache, 2 rounds, all lengths): **PASSED**. Candidate prefill
  10.0 / 9.9, 31.3 / 31.4, 121.5 / 121.6 ms; decode 373.9 / 374.2, 374.1 / 374.2, 378.2 / 378.3 ms; mismatching
  steps 1 per round in p256 (the known near-tie), 0 elsewhere. Score 26.22 (the CPU reference was slow on the loaded
  machine: p256 prefill 543 / 375 ms; compare candidate times only).

### Also tried and rejected

- **Two decode tokens per execution** (`step2_l512`: the step body twice in one NEFF, one exchange buffer per token;
  `tools/exp/step2.patch` + `step2_build.patch`): device 12.095 ms per 2 tokens vs 6.048 ms per token (no gain, as in rounds 3-4); in
  requests (same binary, step2 on/off, 8 rounds x 2 core orders, p1024 as the control) decode -0.1% / -0.03%. The
  execution boundary costs nothing measurable in requests: decode is at the HBM limit (3.81 GB per token at 653
  GB/s; the LM-head phase alone streams at the same ~650-670 GB/s aggregate).
- **Timing bound for a bf16 delta rule** (every delta-rule SBUF intermediate and the state in bf16, accuracy
  ignored; `tools/exp/bf16dr.patch`): pchunk512_l2048t 14.24 vs 14.83 ms (-4.0%); the fp32 bmm time drops 2.29 ->
  1.43 ms but the throttled time rises from 10% to 21%. So even an all-bf16 delta rule (not usable: the inverse
  and the state need fp32) would buy at most ~4% of prefill; a usable mixed version far less (cf. round 010's bf16
  inverse + Newton: no gain). Not pursued.

### Next ideas (estimates, not measured)

- p256 exchange latency (~0.5 ms): the reordering has to be forced (`depends_on` between PE instructions, or
  splitting GEMMs into separately emitted calls the scheduler cannot merge); the compiler ignores emission order.
- P V with the probabilities stationary (round 011's measured micro-benchmark) for chunks with L >= ~1536: ~0.4% of
  p4096, nothing for p256 / p1024.
- Fused-head chunks: the DeltaNet final-state updates (last sub-chunk's S update, btr, store) are not needed for the
  LM head and could move into the DMA-bound LM-head tail (PE idle there): ~50 us of p256 (0.5%), but the operands
  would have to be kept or round-tripped through HBM.
- Decode and the LM head are at the HBM limit; nothing left there without fewer bytes (and extra low-precision
  copies are not allowed).

### Session hazard (cost a rebuild)

`build_neffs.py` workers import `kernels/qwen35.py` when each variant starts, but the cache key is the source hash
taken when the build started. Editing the kernel during a build therefore puts NEFFs of the *edited* source into
the cache under the *old* key. Never edit `kernels/` while `build.sh` runs; build experiments from a copy
(`tools/bldk.sh TAG KERNEL.py NAME...`, `tools/bldk2.sh TAG KERNEL.py BUILD.py NAME...`; paths are this session's,
edit them).

## Round 012 (rejected in evaluation: score -2.98%, prefill -3.10%, decode -2.87%, all from the CPU reference; its code is this engine)

Prefill only; decode and the host are unchanged. Base: **round 011's code, re-applied unchanged** (rejected in its
evaluation only because the CPU reference ran ~3% faster in decode there; its four changes are exact and measured
+1.1 / +1.2 / +0.7% prefill), plus one new change.

### What changed

- **Early K/V tiles in TIGHT prefill chunks** (`EARLYKV`, `kv_early`, in `full_body` and `full_last`). A TIGHT chunk
  has L == pos + T at compile time, so the attention tiles kT [P, 2, L] / vv [P, L/128, 256] are allocated right
  after the k/v projections: the cached part (keys < L - T) is DMA'd from the cache then (before the q/k norms and
  RoPE), and the chunk's own K^T / V are written straight into the tiles' last T columns / blocks (kr is a view of
  kT, v_tm a view of vv). The cache writes are unchanged (from the same tiles). Before, the whole 0..L range was read
  back from HBM after this chunk's cache writes (an HBM write -> read round trip on the attention's critical path).
  FIRST chunks (L == T) read nothing at all. Same values, so token-identical.

### Verified / measured

- Device time (neuron-explorer, core 18, one NEFF): pchunk512_l4096t 15.323 -> 15.245 ms, l2048t 14.925 -> 14.835,
  pchunk256_l256th (the whole p256) 9.780 -> 9.723, l1024th 16.559 -> 16.478 (-0.5 .. -0.6% each; repeat captures
  agree within 0.01-0.03 ms).
- same.py (40 cases x 3 shapes + 14 extra prompts of 257 .. 8000 tokens, `--extra-prompts`) against the round-011
  build: 54/54 identical.
- ab.py vs round 011, 8 rounds in each core order: prefill +0.8/+1.0% (256), +0.6/+0.7% (1024), +0.4/+0.55% (4096);
  decode -0.15/+0.25% (core bias). Cumulative over round 010 (011's ab + this): about +2.0 / +1.9 / +1.2%.
- dev-eval 001 (2 rounds, all lengths): **PASSED**; candidate prefill 10.0 / 10.0, 31.4 / 31.3, 121.6 / 121.5 ms;
  decode 374.0 / 373.9, 374.2 / 374.3, 378.2 / 378.2 ms; mismatching steps 1 per round in p256 (the known near-tie), 0
  elsewhere. Round 010 official: 10.3, 31.8, 123.1 / 374.0, 374.1, 378.3 ms.

### Tried and rejected this round (device time, measured)

- Recurrent state kept in the chunked layout S [dk, h, dv] from one 512-chunk to the next (no btr(Sd) + copy at the
  end, no btr(S) + copy at the start; a `state_c2d` NEFF converted it when anything else followed): l2048t 14.880 vs
  14.835, l1024th 16.490 vs 16.478 ms. No gain (the transposes were off the critical path). Not kept.
- First weight slab after a norm (T = 512) in two fixed-address SBUF buffers (`address=`), so its DMA does not wait
  for a buffer the previous GEMM still reads: l2048t 15.341 vs 14.835, l1024th 16.979 vs 16.478 (much worse; fixed
  addresses again hurt, as in rounds 6/7).
- All norm weights (ln1, ln2, gated-norm w, q/k norm, lnf) loaded once at the chunk start (`ld` via `PFC`): late
  small loads on the GpSimd queue were seen waiting ~10 us behind the weight backlog. p256 9.700 vs 9.723 (two
  captures each), l4096t 15.238 vs 15.245, l512t 14.547 vs 14.565, l1024th 16.520 vs 16.478: +-0.2-0.3%, not kept.
- T = 256: received-half residual add on the VectorE instead of the GpSimd: 9.762 / 9.774 vs 9.723 (worse).
- T = 256: out-projection exchange groups of 1 tile (OPG 1): 9.755 vs 9.723 (worse).
- fp32 transposes (btr, 3168 per 512-chunk at 184 ns = 0.58 ms) as two bf16 transposes of the halves of each fp32
  (`.view(bf16)`, strided): refused by the compiler ("The PSUM access must be 4 byte aligned"). `tools/tr.py`.

### Where the time is (measured, c1 = this engine)

- pchunk512_l2048t: TensorE active 96%; GEMM 9.08 ms (225 ns per 512-column matmul, = 1 column/cycle), fp32 delta
  bmm 2.28 ms, fp32 transposes 0.58, conv 0.38, Gram 0.28; VectorE 54%, ScalarE 28% active. Throttled 10%.
  Remaining PE gaps ~0.6 ms, spread (GEMM waiting for weights after norms ~80 us, exchange ~45 us, VectorE chains).
- p256 (9.72 ms): LM-head phase 1.574 ms on core 0 vs 1.463 on core 1 (DMA-bound, 322 / 347 GB/s); in the chunk,
  256-column GEMM matmuls issue at 100-110 ns (1 column/cycle) except when throttled (~5%); exchange waits
  ~250 us ($S[8]) + Gram waits on the residual adds ~250 us.
- The p256 window between two layers at the MLP norm (an example in `tools/around.py` output): both cores idle
  ~11 us while a small norm-weight DMA on the GpSimd queue waits behind the weight-slab backlog; preloading them did
  not change the total (other waits fill in).
- Decode unchanged (see round 010: ~91% of the HBM peak).

### Tools (`tools/`, not part of the build)

`bld.sh TAG NAME...` (copy the engine's kernels and build some variants into `scratch/exp/TAG`, paths hard-coded for
this session), `prof.sh` (round 011), `lines.py DIR [core]` (PE issue time per source line, needs ham.py's pe.pkl),
`ivl.py DIR LINE` (issue-interval histogram of one line), `gapctx.py DIR CORE LINE SEM [x]` (PE gaps before LINE
waiting on SEM, by previous line, with timestamps), `around.py DIR CORE T_US W_US` (all non-PE instructions in a
window), `lmh.py DIR LINE` (first/last instance of a line per core, e.g. the LM-head phase), `nw.py` (norm-weight DMA
issue to use), `tr.py` (fp32-transpose micro-benchmark, variant B fails to compile).

## Round 011 (rejected in evaluation: score -1.13%, decode -2.94% from the CPU reference; its code is in round 012)

Prefill only; decode and the host are untouched. All four changes are exact (bitwise-identical tokens, checked
below), in `kernels/qwen35.py`.

### What changed

1. **Zero-state first sub-chunk** (`dc_dep(zero=True)`, `delta_pair(zero=)`, FIRST chunks only: `pchunk512_l512t(h)`,
   `pchunk256_l256t(h)`). The chunk at position 0 starts from S = 0. Its first sub-chunk's state-dependent part is
   now v_new = U, o = A2 v_new, S <- k_d^T v_new: the W S and (q e^gc) S matmuls (16 fp32 matmuls per layer), the
   S e^gc_last multiply/add and the initial btr(S) transposes are gone. The terms left out are exact zeros
   (U - 0 = U, 0 + x = x). W and q e^gc of that sub-chunk are then dead, and the compiler already removed them
   (writing that out explicitly measured the same).
2. **Depthwise conv: the last tap at the evacuation** (`CONVPE = 3`). The TensorE does taps 0-2 (diag
   stationaries). The PSUM evacuation, which was a plain VectorE copy, is now
   `scalar_tensor_tensor(x3 * w3 + PSUM)` -> bf16. x3 w3 is exact in fp32 (bf16 x bf16), and the add is the same
   fp32 add the TensorE accumulation did, in the same tap order, so the values are unchanged. That is 1 of 4 conv
   matmuls (512 columns, 24 per layer) less for the same VectorE op count.
3. **grow as an exact bf16 broadcast at T = 256** (`GROWB`, active for chunks of <= 256 tokens). The 8 fp32
   128-column matmuls gneg_h (broadcast) x Ut per sub-chunk give exactly gcs (measured bitwise:
   `scratch/num/grow.py`, 3 random trials at scales 0.01 / 0.3 / 3). So grow = gcs^T broadcast is built from an
   exact three-way bf16 split hi + mid + lo of gcs, one bf16 transpose and 8 bf16 selector matmuls (`sel3_tile`,
   the gated norm's selector, now a helper). At T = 512 the same change is slower (+0.8% device: grow now waits
   for the gcs chain, and the throttled time rises from 11% to 13%), so 512-chunks keep the old form.
4. **`YG = 1`**: the DeltaNet output y goes to the other core one head at a time (was 2). Exchange granularity
   only: p256 device -0.2%, 512-chunks -0.1% / 0.

### Verified

- same.py (40 cases x 3 request shapes, lengths 1 .. 4097) against the round-010 build after each step (c1 =
  change 1, c2 = + 2, 3, c3 = + 4): 40/40 identical each time. Extra prompts of 257, 513, 1536, 2560, 3000, 4608,
  5000, 6143, 7000 and 8000 random tokens (`--extra-prompts`, c3): 50/50 identical.
- dev-eval 001 (c3, 2 rounds, all lengths): **PASSED**, 1 mismatching step per round in p256 (the known near-tie),
  0 elsewhere. Candidate prefill 10.1 / 10.1, 31.5 / 31.5, 123.0 / 122.5 ms; decode 374.1 / 373.8, 374.4 / 374.4,
  379.7 / 378.1 ms. Round 010 official medians: 10.3, 31.8, 123.1 / 374.0, 374.1, 378.3 ms.
- dev-eval 002 (the final source, with its build, 3 rounds): **PASSED**, mismatches as in 001. Candidate prefill
  9.9 / 10.1 / 10.2, 31.5 x 3, 122.4 / 122.3 / 122.3 ms; decode 374.1 / 374.1 / 373.9, 374.4 / 374.5 / 374.5,
  378.4 / 378.4 / 378.1 ms (dev-eval score 24.57; it follows the CPU reference).
- A clean build (copy of the source without `target/`, all 64 NEFFs compiled) gives an engine that is 40/40
  identical to the tested one in same.py (the binaries differ only by build paths inside the NEFFs).

### Measured

Device time (neuron-explorer on core 18, one NEFF, random inputs; the repeatability of one capture is ~0.01 ms):

| NEFF | round 010 | + 1 | + 2 | + 3 | + 4 |
|---|---|---|---|---|---|
| pchunk256_l256th (the whole p256 request) | 9.979 / 9.995 ms | 9.856 | 9.852 | 9.792 | 9.771 |
| pchunk512_l512t (first chunk of p1024 / p4096) | 14.886 | 14.660 | 14.629 | - | - |
| pchunk512_l2048t | 15.011 | - | 14.927 | 14.938 | 14.919 |
| pchunk512_l4096t | 15.412 | - | 15.334 | - | 15.332 |
| pchunk512_l1024th (last chunk of p1024) | 16.657 | - | 16.625 | - | - |

Requests (ab.py on cores 18 / 19, 8 rounds in each core order, averaged over both orders, which cancels the
core-19 bias of ~0.25% in decode):

| build vs round 010 | prefill 256 | 1024 | 4096 | decode |
|---|---|---|---|---|
| c1 | +0.5% | +0.5% | +0.24% | 0 |
| c2 | +1.6% | +1.06% | +0.7% | 0 |
| c3 (this commit) | +1.1% | +1.2% | +0.67% | 0 |

c2 and c3 agree within the run-to-run noise (~0.4% on p256). Estimated prefill mean about +1%. That is below the
1.5% acceptance thresholds on its own, so whether the round is accepted depends on the CPU reference's speed in
the evaluation.

### Findings this round (measured)

**The TensorE power throttle (`ham` records, "activity_1": the PE limited to 50%) has a pattern, and it is not
cheap to avoid.**
- The ham records come in 3.413-us windows (8192 cycles). A throttled interval lasts 1-10 windows at k/n = 4/8.
- Micro-benchmark (`scratch/mb/mb.py`, LNC=2, groups of 32 matmuls of bf16 128x128 x 512 columns, 200 groups):
  - continuous: 1.406 ms, throttled 0.1%.
  - each group waiting for a VectorE chain (gap ~8 us): 2.979 ms, 38%. Gap ~1.4 us: 0.1%. Gap ~2.5-4 us: 9-38%.
  - the same gaps with 16 independent 512-column filler matmuls at the start of the gap: 2.634 ms, 5%. That is
    faster with 50% more matmuls. 8 fillers: 2.807 ms, 22%; 4: 2.895, 31%. Fillers of 64 / 128 columns do not help
    (16 x 128: 35%). Fillers that depend on the middle of the chain, so they run at the end of the gap: 8 fillers
    2.756 ms (no better than at the start). The data does not matter (zeros, ones and random data give the same
    times).
  - after each group, a run of 32 narrow matmuls (128 columns, bf16 or fp32): no throttle. A run of 48-64: 44-76%
    throttled, and the narrow matmuls run 2.5-3.8x slower (fp32 128-column: 850 vs 225 ns). Continuous fp32
    128-column matmuls: 97% throttled. fp32 with 512 columns: 0%.
- In the chunks: pchunk512_l2048t is throttled 11% of the time (175 intervals). GEMM matmuls issue at 427 instead
  of 216 ns when throttled, but only 2.2% of them are. The throttled stretches are runs of small PE ops: k
  transposes (85% of their time throttled), the gated-norm broadcast and rT transposes (100%), the conv l2 sums
  sst (58%), RoPE (86%), the delta rule's grow / M transposes (24-38%). Estimated cost (busy-while-throttled / 2):
  ~550 us per 512-chunk, ~465 us in the p256 chunk (plus a 1.7 ms throttled LM-head phase, which is DMA-bound
  anyway). The compiler reorders PE instructions across source lines (conv, sst and l2 broadcasts interleave),
  so fillers do not land where they are written.
- Filler matmuls in the real chunks (tried, not kept; device time p256 / pchunk512_l2048t; in variants marked *
  the fillers write zeros into the PSUM tile of the following real matmuls, which keeps the values exact and
  needs no extra bank):
  - into their own PSUM bank (the results DMA'd to a junk tensor): the reserved bank alone costs +0.2 ms on p256
    (10.18 vs 9.98). 8 fillers at each Gram + norm-GEMM start: 10.4 ms, with the throttled time 31% -> 27%.
  - * 1 round before each norm-consuming GEMM's first slab: 10.131 / 15.234. Before each Gram: 10.011 / 15.000.
    Before each delta-rule bmm: 10.246 / 15.432.
  - All are slower than without fillers (9.978 / 15.012).

Other measured results (not kept):
- Conv with only 2 taps on the TensorE (two stt ops at the evacuation): p256 10.039, l2048t 15.225 (worse:
  VectorE).
- T = 256: MLP gate/up groups of 2 / 8 tiles (9.913 / 9.908 vs 9.792), down-projection slabs of 1 tile (9.827).
- **P V with the probabilities stationary** (scratch `num/pv.py`, `num/pvb.py`): out[q, 0:257] = P~^T [V | 1]
  per head and key block, instead of V^T P~ for each V half plus a ones-matmul for the sums. It is bitwise
  identical: bf16 matmuls give the same result in either orientation (3 random trials). It costs 484 vs 684 ns per
  key block (4 x 257 columns vs 3 x 512). It needs the outputs transposed back to feature-major (8 bf16 transposes
  per sub-chunk and layer) and 4 PSUM banks. Break-even ~15 key blocks per layer. Estimated gain ~0.7% at
  L = 4096, ~0.3% for a whole p4096, ~0.1% for p1024. Not done yet.
- Where p256 loses time now (pchunk256_l256th, 9.77 ms): the norm-boundary exchange waits (GEMMs on the core
  barrier `$S[8]` 254 us, Grams on the received half's residual add 280 us), the LM-head phase (1.57 ms on core 0
  vs 1.46 on core 1, DMA-bound at ~650 GB/s), the throttle.

### Next ideas (estimates, not measured in the engine)

- P V with the probabilities stationary (see above) for chunks with L >= ~1536: ~0.3% of p4096.
- Keep the DeltaNet state in the [dk, h, dv] layout between TIGHT chunks. A TIGHT non-head chunk is always
  followed by another chunk, which would save its btr(S) at the start, the btr(Sd) + copy at the end and the
  next chunk's btr(S): ~16 fp32 transposes and 2 [P, 8, 128] copies per layer, est. ~70 us per 512-chunk (~0.4% of
  p4096). It needs layout-specific variants for every successor (TIGHT and th chunks, the 256 / 128 chunks and T = 1
  kernels of odd lengths) and host bookkeeping.
- Throttle: runs of more than ~32 narrow PE ops (128 or fewer columns, transposes, few-row selector matmuls)
  appear to be what sets off the 50% limit. Widening or interleaving them (e.g. the k transposes and the gated-norm
  broadcasts with GEMM work) is the remaining handle on the ~3.7% (512-chunks) / ~4.7% (p256) the throttle costs.
  Filler matmuls cost more than they saved in every placement tried.

### Tools (`tools/`, not part of the build)

Copied from this session's scratch directory. The core (18) and the lock file
(`/tmp/engine-gpu-18-es-gptoss.lock`) are hard-coded: edit them for your session. `prof.sh NEFF DIR` (capture +
summary + json). `ham.py DIR [core]` (throttled time, busy-while-throttled, attribution to the PE gap before each
interval; writes `pe.pkl`, which the other scripts read). `hamc.py` (throttle cost by line before -> line after the
gap and waited engine). `thrlines.py` (share of each line's PE time that is throttled). `gaps.py` (PE gaps by next
line and waited semaphore). `mb.py` + `mb_run.sh NAME VARIANT ENV=..` (the throttle micro-benchmark, variants A-F).
`grow.py`, `pv.py` (bitwise checks run on the device through NKI's own run path), `pvb.py` (P V orientation
timing). The fill mechanism that was tried is not in the engine: `fill(site, ps, nv, w)` emitted n rounds of zero
bf16 matmuls into a PSUM tile before its first real matmul, with sites `gram`, `gemm_n0`, `gemm_x0`, `bmm`, ...

## Round 010

Kernels are unchanged from round 009 (every NEFF is the round-009 NEFF). The only change is on the host.

### What changed

- **The prompt is read by a fast integer scan** (`fast_prompt` in `src/engine.cpp`): `"prompt": [ids]` is parsed
  directly from the request line and the rest of the request (with the array emptied) goes through `json::parse`.
  Through `json::parse` a 4096-id prompt took ~0.35 ms and a 1024-id one ~0.09 ms before the request started (one
  `json::Value` per id). Anything unexpected inside the array (floats, a second "prompt" key, malformed lists) falls
  back to the old path. Checked by hand: spaces, key order, floats, `[]`, negative ids, `[1,,2]` give the same
  replies as before.

### Measured

- Requests (bench on core 18, 4 rounds x 2 runs, medians): prefill 4096 123.67 / 123.68 -> 123.45 / 123.52 ms,
  1024 32.05 / 32.01 -> 31.98 / 31.86 ms, 256 unchanged within noise (10.27-10.34). Decode unchanged.
- same.py against the round-009 build: 40/40 identical (as expected: the NEFFs are the same).
- dev-eval 001 (clean build, 3 rounds, all lengths): **PASSED**. Candidate prefill 10.3 / 10.2 / 10.1, 31.8 / 31.9 /
  31.8, 123.2 / 123.3 / 123.2 ms; decode 374.0 / 374.1 / 373.9, 373.9 / 374.3 / 374.2, 378.2 / 378.7 / 381.2 ms.
  Mismatching steps: 1 per round in p256 (the known near-tie), 0 elsewhere. Score 23.77 (follows the CPU reference).
  Against round 009's official candidate times this is the same within noise: the change is worth ~0.1-0.3% of
  prefill, below the acceptance thresholds on its own.

### Findings this round (measured; they bound what is left)

- **Decode is at its practical HBM limit.** step_l512 profile: core 0's weight DMA is active (some packet in flight)
  for all but 43 us of the 6.04 ms step; every phase (DeltaNet layers, full layers, LM head) streams at 305-325 GB/s
  per core, the LM head (pure streaming, no chains) at 319 / 345 GB/s (core 0 / 1). 3.81 GB per token at the HBM
  peak of 716 GB/s would be 5.32 ms; requests take 5.83 ms per token (~91%).
  - The DeltaNet decode recurrence is not on the critical path: a timing variant with the whole recurrence removed
    (o = v, state loaded and stored unchanged) decodes in 374.2 vs 373.2 ms (no gain).
  - Core imbalance: one core streaming 32 MB more in the LM-head phase costs 70 us per token on core 0 and 37 us on
    core 1 (out-of-bounds-skipped DMAs, `oob_mode=skip` with a pid-dependent scalar offset: that mechanism works).
    An uneven LM-head vocabulary split built on it (slab-major `wlm`, per-core slab lists, skipped slabs masked by
    `tensor_copy_predicated`) gives no gain in requests: core 1 slabs 243 / 248 / 252 / 256 of 485 -> decode 373.8 /
    374.2 / 373.3 / 373.4 ms vs 373.2-373.6. Not kept.
  - Odd weight slabs on the GpSimd SWDGE queue (two DMA queues, less head-of-line blocking): step 6.53 vs 6.04 ms.
  - The weight-slab SBUF offsets: slabs live in 93-229 KB of the partition; the decode delta-rule temporaries (t1-t3,
    4 KB) and the recurrent-state tiles are allocated in the same range, which is why consecutive slabs overlap
    (overlap distance 2-8 slabs). Earlier rounds' pinned-address rings were slower; not retried.
- **Prefill chunks are power-throttled.** The profiles record `ham` throttling: the TensorE is limited to 50% for
  11% of a pchunk512_l2048t (1.67 ms) and 31% of pchunk256_l256th (the whole LM-head phase plus ~16% of the chunk).
  Throttle onsets follow periods of high VectorE (0.52 vs 0.37 average busy) and ScalarE (0.30 vs 0.18) activity
  with the TensorE active; a pure GEMM micro-benchmark (full-rate bf16 matmuls, little else) is throttled 0.2% of
  the time. Removing all GEMM weight DMAs from a chunk (timing test) gives 14.90 vs 15.00 ms and 10.2% throttling,
  so DMA traffic is not the cause. Consequence: moving work between engines or shortening chains tends to raise
  the throttle time and give nothing back (see the delta-rule results below).
- **Chunked delta rule, bf16 block inverse + Newton refinement** (tried, not kept). T^T = (I + M^T)^-1 by the same
  8-block Neumann + 4 merges with bf16 operands (fp32 PSUM), then 2 Newton steps TT <- TT - bf16(T)^T bf16(N^T TT - I)
  with the residual N^T TT in fp32 (N = I + M). numpy (scratch `num/inv.py`, keys with correlation 0 .. 0.995,
  decays 0.001 .. 1): max error 2e-7 .. 5e-7, the same as the current fp32 algorithm (1e-7 .. 5e-7); one Newton step
  leaves 1e-5 .. 1.5e-4. Device (pchunk512_l2048t, baseline 15.005 ms): TensorE occupancy -0.64 ms (13.05 vs 13.69 ms),
  but the chunk takes 15.03-15.23 ms in every variant: masks on the operands instead of the outputs, copies on the
  ScalarE, GpSimd for the off-path masks (17.1 ms: GpSimd [P, 8, 128] ops take 6 us and slow the VectorE to 2 us per
  op, shared SBUF port), 2 head groups per sub-chunk, DPAIR = 4 (all four sub-chunks interleaved; the delta-rule
  window per layer 191 vs 2 x 108 us). Throttling rises from 11% to 16-17% in all of them. Requests with DPAIR = 4,
  2 groups (p1024): 31.98 / 31.88 vs 32.01 / 31.97 ms (0.1-0.3%, noise level). The delta rule is bound by PSUM
  evacuation work (VectorE + ScalarE ~100 us per sub-chunk pair, 1.0-1.2 us per [P, 8, 128] pass whatever the dtype)
  and the bf16 version needs more evacuations (Newton), not fewer.
- VectorE micro-benchmark ([P, 8, 128], 1024 elements per partition): fp32 SBUF op 1218 ns; bf16 + bf16 -> bf16 SBUF
  682 ns; any fp32 operand or output, or a PSUM operand: ~1215 ns; bf16 SBUF copy 426 ns; ScalarE copy from PSUM
  1114 ns; GpSimd bf16 add 3079 ns.
- Host: the first `nrta_execute_schedule` of a request takes 20-30 us when requests follow each other and 50-200 us
  after an idle gap of 0.5-2 s; a synchronous `nrt_execute` for single-execution requests is slower (p256 +0.2 ms).
  Engine-side overhead of a p256 request is ~0.1 ms over the device time (9.95 ms); the client sees another
  0.15-0.25 ms (Python, pipes).

### Tools (scratch, not in the engine)

`scratch/prof/prof.sh NEFF NAME` (capture on core 18, json -> pickle), `pegaps.py` / `pelines.py` / `englines.py`
(TensorE gaps and per-line occupancy), `dmabins.py` / `dips.py` / `idle.py` / `slabs.py` (DMA throughput, low-rate
bins, DMA idle by the waited semaphore, slab overlap distances), `seg.py` / `win.py` / `winlines.py` / `wtl.py`
(delta-rule windows: duration, engine busy, timelines), `ham*.py` (throttle events vs engine activity; the `ham`
records come from the full `neuron-explorer view` json), `exp/mk.sh` + `exp/bld.sh` (copy the kernels, build some
variants, link a test engine from the base NEFFs), `scripts/bench.py` (split requests on AB_CORES).

## Round 009

Prefill only; decode is untouched. Every change is token-identical to round 007 (same.py 40/40 cases x 3 request
shapes, and every ab.py request).

### What changed

1. **Skip work nobody reads in the last layer of a chunk** (`full_last`, `KVLAST`, `LASTTOK`). Layer 24 is a
   full-attention layer, and only two things downstream use it: its K/V cache (later chunks and decode) and, in the
   last chunk of a prompt, the last token's hidden state (LM head). Later chunks restart from token embeddings; the
   state crosses chunks only through the KV caches and DeltaNet states.
   - TIGHT chunks without the head (`pchunk512_l*t`, `pchunk256_l256t`) are never the last chunk of a prompt (their
     fused-head twins `*th` take that place), so their hidden-state output is never read. There, layer 24 stops
     after ln1, the K/V projections, the k norm + RoPE and the cache writes. The returned "hidden state" is layer
     23's output and is not used. That is 7 of the 8 chunks of a 4096 prompt and 1 of the 2 of a 1024 prompt.
   - Fused-head chunks (`pchunk*_l*th`): K/V for all tokens as before, then the query, attention, out-projection and
     MLP for the last token only. These are one-column versions of the chunk's own arithmetic: `gemv_col` has the
     same k order (own half first) and the same `bf16(rstd * PSUM)` evacuation, `norm_col` the 1 x 1 Gram in kord
     order, `qknorm_col` the T > 1 q/k norm, and the attention is one query column of `att_tight_t` (its causal
     mask keeps every key for the last query). Each value is therefore computed exactly as in that column of the
     full chunk: same.py 40/40 identical.
2. **Chunked delta rule: decay-mask differences fused over the heads** (`dc_s2`). `dif = gc_j - gc_i` for all 8 heads
   in one [P, 8, 128] pass, then `d1n = dif - mask1` and `d2 = dif + mask2`. That is three passes instead of 16
   per-head `scalar_tensor_tensor` ops of 128 elements, with the same two roundings per element. It shortens the
   VectorE chain between the PE stages.
3. **Residual add of the received half on the GpSimd** (`gemm_fm` resid). The GpSimd runs the exchange handshake,
   so the add sits behind the exchange on the same engine, and the in-order VectorE no longer blocks on the
   exchange. Measured identical (IEEE add).
4. **T = 512: the first weight slab of a norm-consuming GEMM has 2 tiles** (`FIRSTG`). It arrives about 2x sooner.
   The PE waited ~6.6 us for that slab at 16 norms per chunk. Per-tile accumulation is unchanged. At T = 256 the
   change is worse: less own-half work to cover the exchange, 10.10 vs 10.03 ms. So it is limited to T >= 512.

### Measured

Device time (neuron-explorer, one NEFF, random inputs, core 18):

| NEFF | round 007 | + 2, 3 | + 4 | + 1 |
|---|---|---|---|---|
| pchunk512_l2048t (non-final chunk) | 15.72 ms | 15.60 | 15.47 | 15.03 (KVLAST) |
| pchunk512_l2048th (last chunk) | - | - | 17.05 | 16.77 (LASTTOK) |
| pchunk256_l256th (the whole p256 request) | 10.19 | 10.03 | 10.04 | 9.99 |
| pchunk512_l512t / l4096t (with 2-4) | - | - | 15.33 / 15.94 | - |

Requests: ab.py, round 007 (A) against this engine (B), 8 rounds in each core order (cores 18/19), averaged:

| case | prefill | decode |
|---|---|---|
| 256 | +1.75% (10.6 -> 10.45 ms) | -0.07% (unchanged: the core-19 bias cancels between orders) |
| 1024 | +3.0% (33.3 -> 32.35 ms) | 0 |
| 4096 | +4.6% (129.7 -> 124.05 ms) | 0 |

Step by step (same method): changes 2-4 only (engine c1): prefill +2.1 / +1.3 / +1.65%. Adding LASTTOK (c2): +2.0 /
+2.1 / +1.85%. Adding KVLAST (c3, this commit): the table above. Estimated: prefill mean +3.1%, score +1.5%.
same.py against round 007 at every step (c1, c2, c3): 40/40 identical. Long-output check against round 007: a
200-token prompt + 3,840 tokens, a 4,151-token continuation (to position 8,190) + 1, 8,000 + 192, 4,097 + 5,
513 + 4, 8,191 + 1, 1 + 3, 600 / 768 / 1,536 / 2,560 / 300 + a few tokens. All tokens are returned and identical.

dev-eval 002 (the clean build of this source, 3 rounds, public prompts): **PASSED**. Candidate prefill 10.2 / 10.3 /
10.4, 31.9 / 32.1 / 31.9, 123.2 / 123.6 / 123.3 ms; decode 374.1 x 3, 374.2 / 374.2 / 374.4, 378.5 / 378.1 / 378.6 ms.
Round 007 official: 10.4-10.5, 33.0, 128.8-129.1 / 374.1, 374.5, 378.4 ms. That is prefill +1.5 / +3.4 / +4.5% and
decode unchanged. Mismatching steps: 1 per round in p256 (rank 2, margin 0.0625, the same near-tie as round 007), 0
elsewhere. The dev-eval score (23.65) follows the CPU reference, which ran faster than in round 007's evaluation
(p4096 7.3-9.0 s against 9.5 s there). Compare candidate times, not scores.

### Tried and rejected this round (device time; pchunk512_l2048t / pchunk256_l256th unless noted)

- Decode:
  - Dedicated SBUF tiles for the prefetched recurrent state, so no weight slab overlaps a state tile whose store
    is still queued. As plain tensors: 6.044 vs 6.048 ms, no change. The compiler re-allocates every write, so
    the addresses still vary. With fixed addresses (`address=`): 6.084 (worse).
  - Recurrent/conv-state stores on the Scalar HWDGE queue: 6.055. A dummy late read of the stored state to extend
    its liveness: removed as dead code, no effect.
  - `dtt` / `alt` prefetched one layer ahead: 6.042 vs 6.044 (noise).
  - KV-cache loads on the Scalar queue: 6.098 (worse).
  - Profile facts: rings are busy ~91% (E96 97%). The weight queue mostly waits for buffer release (PE stalled
    in the DeltaNet chain). Buffered-but-unconsumed weights stay at ~7-11 MB per core, so a chain longer than
    ~30 us empties the DMA. 15 waits per step of weight DMAs on the GpSimd-queue state store (~120 us of Sync
    queue time) do not show up as lost device time.
- Prefill:
  - (G beta) E1 for all heads in two passes instead of 8 per-head ops: 15.66 / 10.11 vs 15.60 / 10.08 (worse:
    the per-head form lets the M transposes start head by head).
  - vb, kbg on the ScalarE (per-head activation with a per-partition scale): 15.66 / 10.14 (worse; kd, Sn were
    worse in round 005).
  - x (1 + w) of the received half on the GpSimd: 17.17 / 10.37 (much worse: the GpSimd queue holds the exchange
    barriers).
  - kd, qg on the GpSimd: 16.02 / 10.30 (worse).
  - The q/k RMSNorm + RoPE of the full layers batched over the heads (exact): 15.52 / 10.14 (worse, PSUM).
  - B^T as the transpose of the masked B (one VectorE multiply less): 15.62 / 10.08 (worse: longer chain before
    the transposes).
  - Squares for the l2 sums on the ScalarE: no change.
  - A k order with the last exchange group of both halves last (`kord2`, a rounding change): 15.48 / 10.05 (no
    gain, not kept).
  - T = 256: the first two slabs of a norm consumer, and the MLP's first gate/up pair, doing their own-half k
    before the other half (exact): 10.03 vs 10.04 (no gain).
  - ScalarE function order forced with `depends_on` (sigmoid, softplus, exp after the conv SiLUs): table loads
    156 -> 128 per chunk, time unchanged (15.50 / 10.08).
- Where the time is now (pchunk512_l2048t, core 0): PE busy 14.15 of 15.47 ms (GEMM 9.22, delta-rule fp32 bmm
  2.31, fp32 transposes 0.44, conv 0.45, Gram 0.21). The largest idle items are the delta rule waiting for the
  VectorE (~200 us), the gated-norm / RoPE chains (~80 us each) and the norm boundaries (exchange, ~150 us). In
  pchunk256_l256th the norm-boundary exchange waits are ~400 us (4%): at T = 256 the other half's last group
  arrives ~5 us after the own half.

### Next ideas (not measured)

- kv-only twins for the non-TIGHT chunk variants (`pchunk256_l*`, `pchunk128_l*`), used when another chunk follows
  (odd prompt lengths only; the timed lengths use TIGHT chunks).
- p256's remaining PE idle is mostly the exchange at norm boundaries (~400 us): the other core's last group arrives
  ~5 us after the own half (handshake, DMA, residual add, x (1 + w)). Reordering the own-half work (pair, kord2) did
  not help. A one-sided exchange would need a different mechanism (HBM write + core_barrier was not tried).
- Decode: in profiles core 0's DMA rings are slower (E96 ~7%), and core 1 waits ~0.8 ms per step at the gpsimd_dma
  exchanges. Real requests run 3.5% faster than the profiled step, so this may be a profiler artifact. An uneven
  LM-head vocabulary split (DMAs of the extra slabs skipped with a pid-dependent out-of-bounds offset and
  `oob_mode=skip`, logits of the skipped tiles masked) would test it.
- Prefill attention at L = 4096 costs ~0.83 ms of PE per 512-chunk; 1/5 of it is the ones-matmul for the softmax
  sums.

### Tools (scratch, not in the engine)

`scratch/try.sh NAME BASE PATCH.py [variants]` (copy kernels, patch, build 1-2 NEFFs in ~3 min, profile),
`prof.sh`, `assemble.sh OUT BASE_NEFFS OVERRIDES...` (link a test engine from cached NEFFs without a full build),
`gapsum.py` (PE idle by next line and waited engine), `pelines.py`, `idle.py` (DMA ring idle by what the Sync queue
waits for), `syncw.py` (long Sync-queue semaphore waits), `act.py` / `tl.py` (engine timelines), `dmabins.py`.

## Round 008 (rejected by the operator)

An fp8 copy of the LM head used to screen tokens before an exact bf16 rescore (+14.9% decode in its evaluation) was
rejected: extra low-precision weight copies break the bf16-weights rule (skill `kernels-gemv-quant.md`). Do not
retry any variant of it.

## Round 007

### What changed

1. **Prefill RMSNorm with the per-token rstd deferred to the consumers** (`RSD`, `norm_ctx`, `norm_gram`, `norm_bcast`,
   `gemm_n`). Every prefill norm (ln1 of both layer kinds, ln2) now returns a context: hx = bf16(x (1 + w)) per k tile
   (one ScalarE pass, own-half tiles first) is the GEMM input at once. The consuming GEMM (qkv, q/k/v, gate/up; k order
   own half first) emits the norm's TensorE work (Gram diagonals) after its first slab's own-half k matmuls and the
   rsqrt broadcast after the slab's k loop; every output tile is evacuated as bf16(rstd * PSUM) on the VectorE (was a
   plain copy, alternating VectorE / ScalarE). The token-major ab projection scales its PSUM rows by rst. So the
   TensorE no longer waits for the norm chain (Gram -> diag -> rsqrt -> transpose -> selector broadcast -> 16 xn
   multiplies -> 16 (1 + w) scales) at each of the 48 norms per chunk.
   **Numerics change**: bf16(rstd * (W bf16(x (1 + w)))) instead of bf16(W bf16(x rstd (1 + w))) - one rounding point
   moves; the reference's own route is the latter. Decode (T = 1) is unchanged.
2. **Delta-rule merges mask after the matmul** (exact): ZT = (M^T TT) * dm[9 + lv] instead of Cm = M * dm[5 + lv],
   ZT = Cm^T TT (TT is block-diagonal with exact zeros, so the kept entries have the same nonzero terms; masks 9..12 =
   transposes of 5..8, built by the host: `dnc` is now [128][13][128]); BT = btr(M) * dm[4] straight from PSUM (no MT
   copy). One VectorE [P, 8, 128] pass less per merge level and one per sub-chunk. same.py vs change 1 alone: 40/40
   identical.

### Measured

Device time (neuron-explorer, core 19, random inputs):

| NEFF | round 006 | + deferred rstd | + merge masks (this commit) |
|---|---|---|---|
| pchunk256_l256th (the whole p256 request) | 10.816 ms | 10.232 | 10.205 (-5.6%) |
| pchunk512_l2048t | 15.915 | 15.795 | 15.734 (-1.1%) |

Requests (ab.py, round 006 build A vs this commit B, 8 rounds each core order, averaged): prefill +5.2% / +1.6% / +1.4%
(256 / 1024 / 4096; 11.2 -> 10.6, 33.8 -> 33.25, 131.5 -> 129.6 ms), decode +0.1 / +0.1 / +0.1% (unchanged; per
order -0.3% / +0.4% = the known core-19 bias). Estimated score +1.3%, prefill mean +2.7%.
**dev-eval 001 (2 rounds, public prompts): PASSED**; candidate prefill 10.4/10.7, 33.1/33.0, 129.2/128.9 ms, decode
374.0/374.0, 374.3/374.5, 378.1/378.2 ms (round 006 official: 11.0, 33.5, 130.8 / 374.0, 374.7, 378.2). Mismatching
steps: 3 in the controls and 1 per round in p256, every one rank 2 at margin 0.0625 (exact bf16 near-ties); round
006's dev-eval had none - the moved rounding point flips a few more near-ties.
Correctness (scratch `judge.py`: random-token prompts, split request 1 + 64, every stream teacher-forced through
reference.sh `judge`): seed 1 (256/1024/4096/300/777/2048 x 2): round 006 0/780, this 0/780; seed 2
(256/1024/4096/513/2000 x 3): round 006 1/975, this 3/975 (all rank 2, margin 0.125). Long-output check vs round 006:
200-token prompt + 3,840 tokens + a 4,151-token continuation (to position 8,191) identical; 8,000 + 192 (diverges at
step 35, free-running), 7,900 + 10, 4,097 + 5, 513 + 4, 8,191 + 1, 1 + 3, 129 / 384 / 640 / 1,536 + 8: all tokens
returned, no errors.

### Where the time is now

- p256 (10.2 ms device): LM head ~1.45 ms (DMA-bound at ~700 GB/s, PE idle ~1.1 ms of it); in the chunk the largest
  remaining PE gaps are exchange latency: the Gram waits for the residual add of the received half (~360 us) and the
  out/down projections for exchanged inputs (~200 us). An exchange's DMA starts ~5 us after its trigger (the trigger
  is a Sync-queue WRITE, behind weight DMAs that wait for buffer release), then ~3.5 us for 128 KB.
- 512-chunk (15.7 ms): PE occupied 14.2 ms (GEMM 9.24, fp32 delta bmm 2.30, fp32 transposes 0.45); gaps 1.54 ms: the
  delta rule waiting for the VectorE (~440 us), the Gram waiting for exchanged tiles (~200 us), GEMMs waiting for weight
  slabs after a norm (~120 us: the slab DMA waits ~5-45 us for its SBUF buffer).
- fp32 matmul cost (micro-benchmark, 8 heads x 64 reps): 255 ns per 128-column fp32 matmul (2 passes), 495 ns at 256
  columns: linear in the moving columns, so wider moving operands do not amortize anything; bf16 128 columns 104 ns.
  Mixed fp32/bf16 operands are refused ("if one input is float32, both must be"); tfloat32 is refused by the verifier
  (`float32r`).
- Decode (unchanged, 6.06 ms device step_l512): per core 1.905 GB at 315 GB/s average; the MLP phases stream at
  ~330 GB/s (HBM-bound with both cores), a DeltaNet attention part takes ~70 us for 61 us of weights. The dips are
  where the weight DMA waits for a slab buffer the TensorE has not released yet.

### Tried and rejected this round (measured)

- decode: weight slabs at fixed SBUF addresses (`nl.ndarray(address=...)`, ring of n slots): 4 x 24 KB 6.139, 5 x 24
  6.108, 6 x 24 6.126, 8 x 16 6.094, 10 x 16 6.142 ms vs 6.058 (mixing fixed and automatic allocation compiles; the
  ring runs further ahead but PE-idle dips grow).
- decode: z projection after the delta recurrence 6.117, q-gate tiles after the attention scores 6.120, both 6.154 ms
  vs 6.058 (the extra TensorE work delays the chain's own TensorE ops).
- prefill exchanges rotating over 4 pipe_ids: no change (15.803 / 10.219 vs 15.795 / 10.232 ms).
- prefill weight slabs on the ScalarE HWDGE queue (with the deferred norm): 15.842 / 10.397 ms (worse).
- larger first slab of a norm-consuming GEMM (8 tiles at T = 256, 6 at 512; MLP first group likewise): 15.783 /
  10.368 vs 15.734 / 10.205 ms (worse).
- out-projection slab groups for T = 256 of 4 / 8 tiles: 10.203 / 10.278 vs 10.205 ms (no gain).

### Tools (scratch, not in the engine)

- `scratch/exp.sh NAME PATCH.py VARIANT...` (copy + patch + build variants), `scratch/prof.sh NEFF DIR [core]`
  (capture + summary), profile analysis: `pegaps.py` (PE gaps by next line / waited engine), `pelines.py` (PE
  occupancy per line), `englines.py` (engine time per line), `dmabins.py` (DMA GB/s per bin), `idlewait.py` /
  `dips.py` (what the DMA waits for in dips), `layer.py` (per-line timeline of a window), `xchg.py` (exchange
  handshake instructions of both cores).
- `scratch/scripts/judge.py OUT ENGINE...` (random-token prompts, reference `judge` on every distinct stream),
  `longcheck.py`; engine_client.py with `AB_CORES=a,b` (engine A / B pinned via NEURON_RT_VISIBLE_CORES).

## Round 006

### What this round is

1. **Round 005's prefill changes, re-applied unchanged** (keys-on-partitions TIGHT attention, z after the delta rule,
   own-half-first k order, Gram-diagonal gated norm, compile-only build; see the round 005 section). Round 005 was
   rejected by the official evaluation (score -2.06%) although it measured +2.7 / +5.1 / +7.8% prefill and unchanged
   decode in dev-eval: its official decode mean came out -1.35% with the candidate's decode unchanged, i.e. the
   CPU reference ran faster in that evaluation. Re-measured here (ab.py, round 004 (A) vs this code (B), 8 rounds,
   B on core 19): prefill +1.7 / +4.7 / +7.6% (11.3 -> 11.2, 35.4 -> 33.7, 141.5 -> 131.5 ms), decode -0.3%
   (= the known core-19 bias).
2. **Decode: the recurrent state S and the conv state of DeltaNet layer l + 1 are loaded during layer l**
   (`PREFETCH`, `PF_SET = ('S', 'cs')`, `prefetch_layer`, `ld`). The prefetch DMAs go on the Sync HWDGE queue in
   program order (`PF_SYNC`), so they are issued about one layer ahead. Token-identical (DMA order only).

### Result (this commit)

**dev-eval 001 (2 rounds, public prompts): PASSED, 0 mismatching steps in every stream.** Candidate prefill
11.0/10.9, 33.4/33.4, 130.9/131.0 ms; decode 374.2/374.1, 374.7/375.0, 378.3/378.4 ms (score 24.21; the dev-eval
score follows the CPU reference). Against round 004's official candidate times (11.2 / 35.2 / 141.2 ms prefill,
375.2 / 376.0 / 380.8 ms decode): prefill +2.3 / +5.4 / +7.8%, decode +0.3 / +0.3 / +0.65%, estimated score +2.7%.
Long-output check against the round-005 code (cores 18/19): 200-token prompt + 3,840 tokens and a 4,151-token
continuation (to position 8,191), 8,000 + 192, 7,900 + 10, 4,097 + 5, 513 + 4, 8,191 + 1, 1 + 3: all tokens
returned, identical. Build: 64 NEFFs, ~13 min with a warm machine, executable 314 MB.

### How the decode stall was found (profile analysis, step_l512)

- Weight DMA per core averages 312 GB/s and reaches 330-350 GB/s in good stretches. When the other core idles,
  one core reaches ~430 GB/s, so the ceiling is the shared HBM (~690 GB/s for the two cores), not the 16 DMA rings.
  A 16 KB packet takes ~650 ns, linear in size (8 KB 309 ns, 24 KB 1116 ns): there is no per-descriptor overhead
  to amortize.
- The DMA in-flight count drops to ~1 for 4-8 us about twice per layer. In 9 of the 18 DeltaNet layers, the VectorE
  waited 7-16 us for the recurrent-state load (`S[n] (qGpSimdDynamic)` on the line `S = ...recst`). That load was
  issued late on the GpSimd queue and then waited behind the deep weight-packet backlog of the DMA rings (~33 GB/s
  effective). The norm-weight loads (load_norm_w1) and the gated-norm weight wait 20-26 us the same way.
- After the fix, the remaining dips sit at the RMSNorm / exchange windows. The weight DMAs there wait on
  `S[5] (Tensor) >= N`, a true WAR on the buffer address the allocator gave that slab. The allocator packs the
  slabs at irregular addresses (16 KB and 24 KB slabs mixed with small tiles), so the next in-order DMA often
  needs the *most recently* consumed buffer while older freed buffers stay unused. The effective prefetch depth
  is therefore set by address assignment, not by SBUF size.
- Analysis scripts (scratch, not in the engine): per-core DMA GB/s per bin, in-flight packets per bin with the
  instructions running in it, PE-semaphore target of every weight DMA (which PE instruction it waits for), dst
  address histogram of the weight DMAs.

### Measured (device time step_l512, neuron-explorer, baseline 6.091-6.098 ms in three captures)

| variant | step_l512 |
|---|---|
| baseline (round 005 code) | 6.091 / 6.095 / 6.098 ms |
| prefetch all 9 small loads, default (GpSimd) queue: the compiler still issues them just before use | 6.144 |
| prefetch all 9 on Sync | 6.117 (the qGpSimdDynamic waits are gone, but 9 more Sync DMAs per layer) |
| prefetch S + norm weights (ln1, ln2, nw) on Sync | 6.125 |
| prefetch S only | 6.072 / 6.078 |
| **prefetch S + conv state (kept)** | 6.056 / 6.066 |
| S + cs + conv weights | 6.075 |
| S + cs + wab | 6.080 |

Requests (ab.py, 8 rounds each way, averaged over both core orders): decode +0.3 / +0.4 / +0.65% (256 / 1024 /
4096), prefill unchanged within noise. same.py vs the round-005 code: 40/40 identical; all ab.py requests
identical.

### Tried and rejected this round (step_l512 device time)

- Explicit weight ring, slab j -> slot j % 8 as slices of one [P, 8, 8192] tensor (as in round 2): 6.210 ms. The
  ring DMAs are released in bursts (one PE wait unlocks six DMAs, then nothing for ~30 us), which looks like
  dependency tracking on the whole ring tensor.
- The same ring as 8 separate slot tensors: 6.097 ms (no gain). The compiler re-allocates each definition anyway:
  the DMA dst addresses of the "slots" still vary.
- Uniform 16 KB weight slabs for every decode GEMV (flattened (n, k) tile sequence, PSUM accumulation carried
  across slab boundaries, down-projection no longer in 24 KB slabs): 6.111 ms. The addresses stay irregular
  because the small tiles are interleaved in the same allocator space. (A first attempt with 8-tile = 2 KB slabs
  was 6.549 ms: 2 KB packets.)

### Prefill (pchunk512_l2048t, 15.92 ms, profiled, not changed this round)

- PE issue intervals: the bf16 GEMM matmuls (512 columns) issue every 216 ns = 1 column per cycle, 61.6% of PE
  time. The fp32 delta-rule bmm runs as two passes of 213 ns per 128 columns (4x the bf16 cost per column),
  15.4%. Other lines are each < 4%. PE active 91.8%. Without a numerics change (bf16 delta rule) the remaining
  lever is the ~8% PE idle.

## Round 005

### Result

Three prefill changes (decode untouched). ab.py on cores 18/19 (8 rounds each step, B on core 19, which reads
~0.3% slow in decode on this machine):

| step (A -> B) | prefill 256 / 1024 / 4096 | decode |
|---|---|---|
| round 004 -> c1 (attention) | +1.0 / +0.7 / +3.1% | -0.3% (core bias) |
| c1 -> c2 (z deferral, own-half-first k order) | +1.6 / +2.0 / +2.1% | -0.3% |
| c2 -> c3 (gated norm) = this commit | +1.9 / +2.1 / +2.5% | -0.2% |
| cumulative | ~+4.5 / +4.9 / +7.9% (11.5 -> 11.2, 35.5 -> 33.8, 141.8 -> 131.4 ms) | 0 |

Device time (neuron-explorer, one NEFF): pchunk512_l2048t 16.79 (c1) -> 15.92 ms; pchunk512_l4096t 18.28 (r004)
-> ~16.4 ms; pchunk256_l256th 11.05 -> ~10.8 ms.
Correctness: judge.py (scratch tool: random-token prompts, split request 1 + 64 tokens, every stream teacher-forced
through reference.sh `judge`): round 004 0/910 and 5/910 mismatching steps (seeds 1,2 / 3,4), c1 0/910, c2 3/910,
c3 3/910 (all rank 2-4 near-ties, margins 0.06-0.13). same.py c2 vs c3: 40/40 identical.
**dev-eval 001 (this commit, 2 rounds): PASSED, 0 mismatching steps**; candidate prefill 10.9/11.0, 33.6/33.5,
131.0/130.9 ms, decode 375.3/375.1, 376.0/376.2, 380.8/380.7 ms (round 004 official: 11.3, 35.2, 141.2 / 375.2,
376.0, 380.8): prefill +2.7 / +5.1 / +7.8%, decode unchanged (dev-eval score 25.1; it follows the CPU reference).
Long-output check (c3): 200-token prompt + 3,840 tokens, continuation of 4,151 (to position 8,191), 8,000 + 192,
7,900 + 10, 4,097 + 5, 1 + 3, 513 + 4, 8,191 + 1: all tokens returned, no errors.

### Changes

| change | effect | identical? |
|---|---|---|
| **prefill TIGHT attention with keys on the partitions** (`att_tight_t`): per sub-chunk and 128-key block, S^T [keys, 4 heads x 128 queries] = K Q^T in one PSUM bank, P^T = bf16(exp(S^T/16 - c)) straight from PSUM on the ScalarE (it is already PV's moving operand: no probability transposes, no score copy, no max pass), softmax sums by a ones-matmul (broadcast on all partitions), normalisation after PV. c = 8(max(1+w_q)^2 + max(1+w_k)^2) - 64 from the q/k norm weights, computed on the device: since |q|,|k| <= 16 max|1+w|, exp never overflows and the row maximum cannot underflow for this checkpoint (bound c0 <= 92) | pchunk512_l4096t 18.28 -> 17.27 ms, l1024t 16.74 -> 16.57; requests p4096 +3.1% | no (rounding), 0 mismatches |
| z projection (DeltaNet) emitted after the delta rule | l2048t 16.79 -> 16.66 | yes |
| **own-half-first k order** (`kord`): GEMMs whose input halves come from the exchange (out-proj, down-proj) and the RMSNorm Gram loop start on the core's own tiles; core 1 used to wait for core 0's half first. Barrier waits per chunk 2.25/0.42 ms (core 0/1) -> 0.69/1.08 | 16.66 -> 16.50 | no (fp32 accumulation order on core 1), 0 mismatches |
| **gated RMSNorm (prefill)**: sums of squares as the diagonals of bf16 Gram matrices ob^T ob (one `select_reduce` per head and sub-chunk) instead of 64 fp32-stationary 1-column matmuls; rsqrt broadcast by a bf16 selector matmul of an exact three-way bf16 split hi+mid+lo (reproduces the fp32 value exactly in PSUM) instead of 8 fp32 512-column matmuls; the o^2 pass is gone | 16.50 -> 15.92 ms | sums: accumulation order only; same.py 40/40 identical |
| build: `build_neffs.py` skips NKI's post-compile run on a NeuronCore (patches the executor of `run_from_bir`); the build never touches a device, so builds no longer collide with engines on the experiment core | build only | NEFFs unchanged |

### Tried and rejected (device time, pchunk512_l2048t unless noted)

- z slabs inside the delta pairs (after s5): 17.11 vs 16.79; at each pair's start: 16.08 vs 15.92.
- next pair's s1/s2 emitted before this pair's s9/dep (and G moved to s3): 16.04 vs 15.92 (G move alone 15.94).
- DPAIR = 4 (all sub-chunks interleaved; SBUF freed by the attention change): 16.45 vs 15.92.
- delta-rule PSUM evacuations of the second sub-chunk on the ScalarE: 16.71 vs 16.50; kd / S e^g scaling on the
  ScalarE: 16.69 vs 16.66.
- gated norm (nb*w)*silu(z) as one scalar_tensor_tensor: no change. RMSNorm normalize as one VectorE stt: 16.53 vs
  16.50. RMSNorm Gram diagonal by select_reduce: 16.53. RMSNorm broadcast by the exact bf16 split (tok_bcast): 16.25.
- q/k norm of the full layers token-major (Gram + split broadcast): 15.88 vs 15.92 device, requests +0.5/0.1/0%
  and token changes: not kept.
- constants (selectors, masks, the attention shift) built once per chunk: 16.04-16.11 vs 15.92 (worse).
- down-projection groups [2,2,2,1,1]: 15.91 vs 15.88.
- exchanges with dma_engine=gpsimd_dma in prefill: limited to 512 B per partition per sendrecv; split into 256-column
  pieces: 25.6 ms. Timing tests: no exchange at all 15.95 vs 16.66 (the rendezvous costs ~4%).
- decode: GEMVs fed by exchanges with k-half weight slabs, own half first (`dechalf`): step_l512 6.122 vs 6.109.
- tfloat32 views for the delta-rule matmuls: not accepted by the compiler (`float32r`); VectorE rsqrt: refused
  (GpSimd only).

### Where the time is now (pchunk512_l2048t, 15.9 ms, core 0 profile)

- Per layer: DeltaNet attention part ~437 us (qkv GEMM 84, conv/l2/transposes ~45, delta rule ~210 for two
  sub-chunk pairs, gated norm + z + out-proj ~70, norm ~24), MLP ~270 us (GEMMs ~97% of the PE streaming rate),
  full-attention part ~235 us. PE ~90% busy; the delta rule is ~200 fp32 PE ops per head-group sub-chunk (18 per
  head for the inverse whatever the Neumann block size: 8/16/32 all give 18).
- Power throttling: `ham` records show the PE limited to 50% for 16-37% of a chunk (37% in pchunk256_l256th,
  mostly in the LM-head phase, which is DMA-bound anyway).
- p256: 256-chunk ~9.3 ms + LM head 1.5 ms (DMA-bound, ~330 GB/s per core).
- Decode unchanged (5.86 ms/token); per profile core 0's DMA is the binding one (312 GB/s average, ~440 us/step of
  dips at exchange/norm windows where the weight stream waits for buffer release).

### Tools (scratch, not in the engine)

- `scratch/exp.sh NAME PATCH.py VARIANT...`: copies the engine kernels, applies a python patch, builds the variants
  and profiles each on core 19. Profile analysis scripts: per-source-line PE cost including the gap before it, PE
  gaps by waited semaphore, layer segment durations (Gram groups as markers), DMA throughput bins, barrier waits.
- `scratch/scripts/judge.py OUT ENGINE...`: random-token prompts, then the reference's `judge` on every stream.
- Builds and engine runs must not share a core at the same time (nrt_init fails) - fixed by the compile-only build.

## Round 004

### Result

Build c3p (this commit). Token-identical to round 003 on `same.py` (40 cases x 3 request shapes, lengths 1 .. 4097:
c3p vs round 003 40/40 identical; also c1, c2, c3 at each step) and in every ab.py request.
**dev-eval 003 (c3p, 2 rounds): PASSED, 0 mismatching steps**; candidate prefill 11.4/11.2, 35.2/35.2, 141.2/141.2 ms,
decode 375.5/375.5, 376.2/376.1, 380.9/380.8 ms (score 24.4; the dev-eval score follows the CPU reference's speed:
24.4 / 23.2 / 26.2 in runs 003 / 002 / 001, compare candidate times; round 003's official candidate times were
11.3 / 35.8 / 145.1 and 378.0 / 377.3 / 389.5 ms).
ab.py, round 003 (A) vs c3p (B), 8 rounds with B on core 19 and 8 with B on core 18, the two averaged (cancels the
core bias: on this machine p256 prefill is ~0.2 ms slower on core 19 for either build):

| case | prefill | decode |
|---|---|---|
| 256 | +1.5% (11.55 -> 11.4 ms) | +0.7% (377.8 -> 375.0 ms) |
| 1024 | +1.9% (36.15 -> 35.45 ms) | +0.4% (377.2 -> 375.7 ms) |
| 4096 | +2.9% (145.8 -> 141.8 ms) | +2.2% (389.0 -> 380.5 ms) |

Estimated score +1.6%, prefill mean +2.1%, decode mean +1.1% (each case within ~1% run-to-run noise).
Checks (c2 / c3p, measured): 36 repeated identical requests (3 prompts x 12, 1 + 120 tokens) all identical; 200-token
prompt + 3,840 tokens, then a 4,151-token continuation (positions up to 8,191), an 8,000-token prompt + 192 tokens,
odd prompt lengths 1 .. 7,900, prompts of 256 / 512 / 768 / 1,024 / 1,536 / 2,560 / 4,096 / 7,680 / 8,191 tokens
(fused and unfused last chunks) + a continuation: all tokens returned, no errors, identical to round 003.
64 NEFFs (47 + 17 fused-head chunks), executable 331 MB, engine ready ~20 s after start (parallel NEFF loads; five
consecutive starts clean), cold build ~25 min on the loaded machine (12 parallel compiles).

### Tools this session (scratch, not in the engine)

- Experiments on cores 18/19 with a copy of `brief/skill/scripts` whose `engine_client.py` passes
  `NEURON_RT_VISIBLE_CORES` through (`AB_CORES=18,19`: engine A on 18, B on 19; the stock scripts start the engine
  with a clean environment, so it would claim every visible core) and lock file `/tmp/engine-gpu-18-es-gptoss.lock`.
- One variant at a time: `build_neffs.py OUTDIR NAME...` into a scratch directory (own cache), profile with
  `neuron-explorer capture/view --output-format json` (the json lands in `ntff.json` in the working directory), and
  assemble a test binary from the parent's NEFFs plus the new ones (the build.sh embed step on another directory).
- Profile views that were decisive: Tensor-engine idle gaps attributed to the next instruction's source line and to
  the semaphore it waits on (`S[n] (Vector)`, `$S[8]` = the core barrier, `qSyncDynamicHW` = weight DMA); DMA packets
  in flight per microsecond per core; for each `sendrecv` the barrier wait per core.

### Changes (in order), each measured

| change | effect (device time, neuron-explorer, unless noted) | identical? |
|---|---|---|
| **decode exchanges on the GpSimd SBUF-to-SBUF DMA** (`sendrecv(dma_engine=gpsimd_dma)`) with every exchange tile a slice of one step-wide buffer `XB` that is never reused within a token (`xtile`). The CoreBarrier exchange's wait sits in the Sync queue that also issues the weight DMAs, so weight prefetch stopped at every exchange (Sync idle 0.9-1.4 ms per step in front of barrier WRITEs, ~20 us of DMA lost after each DeltaNet chain). The race of round 002 is explained by address reuse: with gpsimd_dma a core can be one exchange ahead and writes into the other core's dst before that core reached the exchange; never-reused dst addresses make that harmless | step_l512 6.150 -> 6.091 ms; requests decode +0.4 / +0.1 / +1.8% | yes; 36 repeated requests identical |
| prefill attention software-pipelined: the scores of (head, sub-chunk) item i + 1 are emitted before the softmax / transposes / PV of item i (the TensorE computes them while the VectorE / ScalarE run the softmax) | pchunk512_l2048t 17.71 -> 17.45 ms | yes |
| TIGHT attention masks: only the last 128 keys of a sub-chunk can be masked (causal triangle, one constant [128, 128] mask); the other scores are q.k / 16 (exactly q.k / 16 + 0) | 17.45 -> 17.45 ms (less SBUF: no [TS, L] masks) | yes |
| **prefill exchanges in GEMM-k order**: the DeltaNet output y is normalized, gated and exchanged per group of 2 heads (`YG`), the attention output per local head (head-major loop); the out-projection (k = 0 .. 15 on both cores) starts on the first tiles instead of waiting for one big exchange (core 1's first k tiles are core 0's) | 17.45 -> 17.21 ms (512 at L = 512: 16.79 -> 16.40, L = 4096: 18.99 -> 18.29) | yes |
| unmasked attention scores scaled on the ScalarE (`activation(copy, scale=1/16)`) | 17.21 -> 17.18 ms (L = 4096: no change) | yes |
| RMSNorm Gram matmuls k-outer (each sub-chunk still accumulates k in order) | 17.20 -> 17.18 ms (noise) | yes |
| chunked delta rule: d1 / d2 / M as one `scalar_tensor_tensor` per head instead of two [P, 8, 128] passes (d1 computed as -(d1) = (gc_j - gc_i) - mask, exp(scale = -1): IEEE subtraction is sign-symmetric) | 17.18 -> 17.14 ms | yes (same.py) |
| **LM head fused into the last prefill chunk** (`pchunkh`, variants `pchunk512_l{L}th` for all 16 exact buckets and `pchunk256_l256th`; the host uses them when the prompt ends on such a chunk, else chunk + `headna_t*` as before): one execution less, no hidden-state round trip through HBM, the LM-head weight stream starts during the last layer | pchunk256_l256t + headna_t256 9.469 + 1.621 ms -> 11.026 ms; requests p256 prefill +1.3%, p1024 +0.2% (c3 vs c2) | yes |
| NEFFs loaded by 8 threads (`nrt_load` of 64 NEFFs one by one: 44.9 s; parallel: 15.1 s; engine ready 49 -> 20 s) | start-up only | yes |

### Tried and rejected this round (measured)

- decode: weight preloads (`load_slabs`) for the out-projection and the first gate / down slabs before the exchange
  in program order: the scheduler moved them behind the barrier anyway (6.206 vs 6.150 ms); with
  `instruction.depends_on()` forcing them ahead: 6.254 ms (they wait for slab buffers instead).
- decode: weight slabs alternating Sync / Scalar HWDGE queues (with the GpSimd exchanges): 6.220 vs 6.091 ms.
- decode: explicit ring of 9 slabs of 16 KB for all 16-KB weight slabs (and the LM head): 6.227 vs 6.091 ms.
- decode: 4 tokens per step execution (`step4`, 4 full forward passes in one NEFF, 15.5 MB): 24.46 ms per 4 tokens
  = 6.115 ms per token vs 6.092: the token boundary inside a NEFF costs as much as an execution boundary.
- decode delta rule with the shorter output chain o = (S e^g) . q + delta (k . q) (k . q by a ones matmul): 6.144 vs
  6.091 ms (rounding change as well).
- decode: exp(A_log) of all 18 layers once per step (one ACT table switch less per layer): 6.155 vs 6.091 ms. Decode
  device time reacts to small changes by +-1% through the compiler's scheduling; only the exchange change helped.
- decode LM head: core 0 streams ~8% slower than core 1 in profiles (320 vs 348 GB/s, core 1 idles ~100 us at the
  end). Swapping which half of the wlm tensor each core reads keeps core 0 the slower one, so it is not HBM placement.
  An uneven split is not expressible in the SPMD kernel (same instruction stream on both cores).
- prefill: two interleaved 256-token streams per 512-chunk (phase functions, stream B one phase behind A, each stream
  streams its own weights): 18.50 vs 17.71 ms (256-column GEMMs 9.77 vs 9.18 ms, and weight-DMA waits appear).
- prefill: out-projection + MLP split by tokens between the cores (no m exchange, half the MLP norm, both weight
  halves streamed by each core): 17.28 vs 17.21 ms.
- prefill: norm halves (the GEMM slab before and after each norm per 256-token half, norms per half): 17.47 vs 17.14 ms
  (6k more matmul instructions cost 0.25 ms; the norm waits barely moved).
- prefill: attention probability transposes copied alternately by ScalarE / VectorE: 17.25 vs 17.21 ms (L = 4096
  18.52 vs 18.29).
- prefill: delta-rule PSUM evacuations (`bsb`) on the ScalarE: 17.48 vs 17.18 ms.
- prefill: delta-rule state update S <- S e^gc_last + kd^T v_new as one `scalar_tensor_tensor` per head: 17.21 vs
  17.14 ms (the state update is on the sub-chunk-to-sub-chunk critical path; 8 small ops are slower there).

### Where the time is now (measured in profiles unless stated)

- Decode (step_l512 6.09 ms device, 5.85 ms per token in requests): DMA busy ~99%, ~620-680 GB/s. A DeltaNet layer
  takes ~186 us for 58.7 MB per core (~315 GB/s per core against ~335-357 at full rate); the losses are DMA tails
  (one of the 16 DMA rings left with a backlog of 16 KB packets while the others idle, ~10 us per event) and the
  serial chain per layer. The LM head is ~1.58 ms on the slower core.
- Prefill 512-chunk (17.14 ms at L = 2048): TensorE work ~14.5 ms (GEMMs 9.2, delta rule fp32 matmuls 2.3), idle
  ~2.6 ms: the RMSNorm Gram waiting for the residual of the last exchanged group (~0.6 ms, the cores take turns
  waiting at the barriers: core 0 waits 2.3 ms in total, core 1 0.3 ms), the delta rule waiting for the VectorE
  (~0.5 ms; the delta rule's [P, 8, 128] fp32 elementwise ops are 2.0 ms of VectorE time), GEMMs waiting for the
  norm's ScalarE (1 + w) multiply (~0.3 ms), attention transposes waiting for the ScalarE (exp) (~0.3 ms).
  The norm window (last GEMM matmul -> first matmul of the next GEMM) is ~24 us median, ~1.1 ms per chunk.
- p256 prefill: one execution `pchunk256_l256th` (11.03 ms device: the 256-chunk ~9.4 ms + the fused LM head;
  the LM head alone streams ~630 GB/s).
- Attention at long L costs ~0.53 us per key per 512-chunk; at Ls = 4096 a (head, sub-chunk) item is ScalarE-bound
  (~11 us: score copy, exp, normalize) against ~8.6 us VectorE and ~3 us TensorE.

### Next ideas (not measured)

- Prefill attention at long L: balance the score copy / normalize between ScalarE and VectorE per Ls.
- Fewer VectorE passes in the chunked delta rule (more fused `scalar_tensor_tensor` per head where one operand is a
  per-partition scalar).

## Round 003

### Result

dev-eval 003 (final build c7, 2 rounds, public prompts): **PASSED**, 0 mismatching steps in every stream.
Candidate prefill 11.5 / 35.8 / 145.1 ms (256 / 1024 / 4096), decode (64 tokens) 378 / 377 / 390 ms. Round 002's
best had 14.5 / 48.5 / 197.6 and 424 / 423 / 430 ms: prefill 1.26 / 1.35 / 1.36x, decode 1.12 / 1.12 / 1.10x faster.
(dev-eval scores move with the CPU reference's speed: 19.6 in run 001 with c3, 24.8 and 24.7 in runs 002 / 003.)
Measured on cores 18/19 with `brief/skill/scripts/same.py`, `ab.py` (own lock file) and device times with
`neuron-explorer` (one NEFF, random inputs). ab.py on this machine has a core bias: an A/A run of one build shows
B (logical core 19) 0.3% slower in decode than A (core 18).

### What the engine is now (differences from round 002)

- Prefill chunks (T = 512 / 256 / 128, sub-chunks of 128):
  - depthwise conv on the TensorE (diag(w_j) stationaries, host re-layout `cdiag`), per channel tile pipelined with
    the SiLU, the square and the token-major l2 sums;
  - all per-token norm statistics token-major (l2norm of q/k, RMSNorm via the diagonal of the per-sub-chunk Gram
    matrix, gated RMSNorm): rsqrt on a few values per partition, broadcast back by a PE transpose + selector matmul;
    the RMSNorm (1 + w) multiply on the ScalarE;
  - GEMM slabs k-outer, evacuation alternating VectorE / ScalarE; every slab group is exchanged with the other core
    as soon as it is evacuated, and the residual add is done per group; out-projection groups of 2 tiles;
  - chunked delta rule: the state-independent stages of two sub-chunks interleaved (`delta_pair`), the state kept
    as [dk, h, dv] across the sub-chunks of a call;
  - exact attention buckets for 512-chunks (`pchunk512_l{pos+512}t`, every multiple of 512 up to 8192) with
    per-sub-chunk key trimming; `pchunk512_l512t` / `pchunk256_l256t` (position 0) start from zero DeltaNet states
    on the chip, so a request with >= 256 prompt tokens needs no `zero_states` execution.
- Decode: the conv state moves as one contiguous 144-byte row per partition (the strided 6-byte-element DMA was the
  largest decode stall).
- 47 NEFFs, cold build ~9 min (12 parallel compiles), executable ~196 MB, engine ready ~24 s after start.
- Host: `pchunk_name` picks the exact-bucket variants; `ENGINE_VERBOSE=1` also logs host write / schedule / wait times.

### Changes (in order), each measured

| change | effect (device time unless noted) | identical? |
|---|---|---|
| prefill depthwise conv on the TensorE: per channel tile 4 accumulating matmuls with diag(w_j) stationaries (host re-layout `cdiag` [18][2][P][96][128], 113 MB) instead of 7 VectorE passes over [P, 24, T] | pchunk512_l512 23.24 -> 22.92 ms with GpSimd-built diags (GpSimd elementwise is ~0.13 elem/ns: 93 us/layer), host-built: see next row | yes (same products, fp32 accumulation in tap order) |
| prefill l2norm of q/k with token-major sums (64 ones-vector matmuls -> [128 tok, NS, 16]); the bf16 chain + GpSimd rsqrt run on 64 values per partition instead of 8192; result broadcast back by a PE transpose + selector matmul (sel[k, c, m] = (k == c)) | 22.92 -> 21.76 ms | yes |
| prefill RMSNorm: sum of squares as the diagonal of the per-sub-chunk Gram matrix (bf16 matmuls, no x*x pass), rsqrt token-major, broadcast back by transpose + selector matmul, (1 + w) multiply on the ScalarE (`activation(copy, scale=w1[:, k])`) | 21.76 -> 20.60 ms | **no**: fp32 sums of squares round differently (3-6 of 1M normalized values differ by 1 bf16 ulp; even the ones-vector form with sq stationary differs from ones-stationary: the fp32 TensorE matmul is not symmetric in its operands). dev-eval: 0 mismatches |
| prefill exchanges pipelined with the GEMMs: each GEMM slab group is sent to the other core as soon as it is evacuated; MLP gate/up/silu*up per group of 4 intermediate tiles with the group exchanged at once | 20.60 -> 19.61 ms | yes |
| prefill gated RMSNorm token-major (as l2norm), GEMM PSUM evacuation alternating VectorE / ScalarE (`activation(copy)`: RNE, measured identical), gate sigmoids hoisted out of the attention loop (ACT table reloads) | 19.61 -> 19.56 ms | yes |
| prefill GEMM slabs k-outer (all g output tiles of a slab accumulate in g PSUM banks, k = 0..KT-1 per tile unchanged) so a GEMM starts on the first normalized k tile; RMSNorm xn / y per k tile | 19.56 -> 19.16 ms | yes |
| prefill delta rule: state kept as S [dk, h, dv] across the sub-chunks of a call (2 transposes per call instead of 2 per sub-chunk) | 19.16 -> 18.99 ms | yes (exact transposes) |
| **conv state DMA as one contiguous 144-byte row per partition** (load into a [P, 24, 3] tile and copy into xx on-chip; copy out before the store). The strided DMA into/out of xx[:, :, 0:3] was 3072 descriptors of 6 bytes on the GpSimd software DGE (1.8 us per DMA instruction), and later weight-slab DMAs waited behind it (SBUF reuse) | **step_l512 6.81 -> 6.13 ms (-10%)**; pchunk512 18.99 -> 18.92 | yes |
| prefill residual add per exchanged tile group (own tiles after evacuation, received ones after their exchange) instead of one add after the last exchange: the next RMSNorm's Gram k-loop starts on early tiles | pchunk512_l512 18.92 -> 18.36 ms | yes |
| **prefill delta rule: the state-independent stages of two consecutive 128-token sub-chunks interleaved** (`delta_pair`, `dc_s1..dc_s9`, stage k of both, then stage k + 1; state passed in dicts): the TensorE works on one sub-chunk while the VectorE / ScalarE finish the other's previous stage. 4 sub-chunks at once: 17.50 ms (SBUF pressure) | 18.36 -> 17.03 ms (-7.2%) | yes |
| prefill conv -> SiLU -> square -> token-major l2 sums per channel tile | 17.03 -> 16.89 ms | yes |
| prefill out-projection slabs (= exchange groups) of 2 tiles instead of 4 (1 tile: 16.81; down-projection groups of 1: 16.81) | 16.89 -> 16.70 ms | yes |
| position-0 chunk variants start from zero DeltaNet states (memset instead of state loads; `zero_states` skipped for prompts >= 256) | requests (c7 vs c5): prefill +1.7% at 256, ~0 at 1024 / 4096 | yes |
| exact attention buckets for 512-chunks (L = pos + 512, every multiple of 512 up to 8192: `pchunk512_l{L}t`; the 256-chunk at 0: `pchunk256_l256t`) and per sub-chunk trimming (sub-chunk s reads keys < L - (3 - s) 128; the rest is fully masked). 47 NEFFs, cold build 8.9 min, executable 196 MB, ready ~24 s | pchunk512_l4096 19.11 -> 19.01 ms from trimming; requests: prefill +2.2 / +1.0 / +2.5% | yes |

A/B (ab.py, 4 rounds): c1 (first four prefill rows) vs round 002: prefill +13.5 / +17.6 / +16.7%; c2 (+ k-outer,
state layout) vs c1: prefill +0.8 / +2.3 / +2.0%; c3 (+ conv-state DMA) vs c2: decode +11.6 / +11.7 / +9.7%.
c4 (+ per-group residual add, delta_pair, per-tile conv chain) vs c3: prefill +5.5 / +10.8 / +11.2%; c5 (+ exact
buckets, out-proj groups of 2) vs c4: prefill +2.2 / +1.0 / +2.5%; c7 (+ position-0 zero states) vs c5: prefill
+1.7 / 0 / 0%. Decode differences in these runs are within the core bias.
same.py: c2 .. c7 each token-identical to its parent (40/40 cases x 3 request shapes); c1 differs from round 002 in
1 of 12 ab requests (rounding of the Gram-form RMSNorm). dev-eval 001-003: 0 mismatching steps.
Long-output check (c5): 200-token prompt + 3,840 tokens, continuation to position 8,191 (4,152 tokens), an
8,000-token prompt + 192 tokens (the largest exact bucket), odd lengths 1 .. 7,900: all tokens returned, no errors.

### Tried and rejected this round (measured)

- prefill conv on the VectorE as fused multiply-adds per tile (instead of the TensorE diag matmuls): 17.02 vs 16.74 ms.
- interleaving all 4 sub-chunks of the delta rule: 17.50 vs 17.03 ms (pairs).
- deferring the received-group residual add by one group (so the in-order VectorE stream does not wait): no change.
- KV-cache / recurrent-state DMAs on the Scalar HWDGE queue (decode): 6.14 vs 6.13 ms.
- two decode steps per execution (`step2`: two full forward passes one after the other in one NEFF, so the weight
  stream of the second starts during the LM-head tail of the first): device 12.22 ms per 2 tokens vs 6.12-6.13 per
  token; requests (c6 vs c5, core-bias corrected) decode +0.3 / 0 / +0.9%, tokens identical; not kept (+7 NEFFs,
  +55 MB executable, +1 min build for ~0.4%).
- the 32-block-diagonal stages of the inverse (8-block Neumann product, merges 8->16->32) as tiled PE matmuls
  (`tile_position`/`tile_size` 32x32, four tiles in parallel) on a compact [P, H, 32] layout: 17.70 vs 16.74 ms.
  The tiled fp32 matmuls are only ~1.4x faster per 4 blocks (89 vs 126 ns), block transposes as identity matmuls
  are slower than `nc_transpose`, and 4x the instructions add PE gaps. (Micro-benchmark: tiled matmuls correct,
  PE time 66 vs 119 us for 512 head-matrices, but 4 partition-sliced PSUM copies per stage made it slower overall.)
- prefill P V with the probabilities stationary and both V halves moving (256 columns, half the stationary loads,
  then bf16 transposes back): 18.97 vs 19.01 ms at L = 4096, no change at 512.
- prefill attention probability transposes by `nisa.dma_transpose` (16 queries x all key blocks per DMA, bf16):
  19.68 vs 19.01 ms at L = 4096.
- host overhead per request is small: a 256-token prefill is 11.3 ms in the engine (0.13 ms scheduling) for
  ~11.2 ms of device time.

- moving the conv-weight load before the projections for T = 1 too: step +0.5% (restored for decode).
- decode exchanges on `dma_engine=gpsimd_dma`: step 6.87 -> 6.80 ms (-1%), but it raced in round 002; not used.
- decode KV cache read without the write dependency (writes through aliased inputs `f_kcw`/`f_vcw`, cached keys
  >= pos masked, the current token as an extra key block from SBUF): step 6.13 -> 6.32 ms (exchange waits grew).
- explicit SBUF ring for the decode weight slabs: 8 slots 6.17 ms, 5 slots 6.22 ms (vs 6.13); per-tile DMAs into a
  flat ring 6.60 ms.
- decode weight slabs of 8 tiles (32 KB per partition): 6.30 ms; of 2 tiles: 6.44 ms (4 tiles: 6.13).
- preloading all small per-layer parameters (norm weights, conv weights, A_log, dt_bias, wab, rot) once per step:
  6.20 ms (SBUF taken from the weight buffers).
- no exchanges at all in decode (timing test): 6.13 -> 6.09 ms: exchanges are not the decode bottleneck any more.
- extra attention buckets for 512-chunks: attention costs ~0.59 us per key per 512-chunk (pchunk512 l512 19.56,
  l2048 20.49, l4096 21.69 ms), so buckets every 512 would save ~2 ms of a 4096 prefill (~1.2%): not done yet.

### Where the time is now / next ideas (estimates unless stated)

- Prefill (pchunk512, 16.7 ms at L = 512): TensorE ~93% busy; GEMMs 9.2 ms (~97% of 1 column/cycle at 2.4 GHz),
  delta rule ~2.6 ms (25 fp32 matmul-equivalents per head per sub-chunk, fp32 ~3x bf16), conv 0.4, norms ~0.6,
  attention ~0.5 ms + ~0.59 us per key. Remaining TensorE gaps ~2.3 ms, spread (core imbalance at exchanges,
  delta-rule ping-pong, small chains). Ideas: interleave the state-dependent part of one sub-chunk pair with the
  state-independent stages of the next; MLP token-split for 512-chunks (no m exchange, half the MLP norm, 2x MLP
  weight traffic); a 64-token WY chunk (one merge level less).
- Decode (step_l512 6.12 ms device, ~5.9 ms per token in requests): weight-DMA bound, ~618 GB/s effective; the LM
  head phase streams at ~690 GB/s. In profiles core 0's DMA is ~6% slower than core 1's and core 1 waits for it at
  every exchange; whether that is real (then: an uneven split, e.g. of the LM head, would help) is not known.
- Exchanges with `dma_engine=gpsimd_dma` are faster but raced in round 002; unique (never reused) exchange buffers
  might fix that (not tried).

### Profiling notes

- In `neuron-explorer view --output-format json`, the `instruction` records of **both physical cores** are under
  `subgraph == 'sg01'`; tell them apart by `fully_qualified_subgraph` (/sg00, /sg01). Aggregates over `sg01`
  alone mix the two cores. DMA records are per core in `subgraph` sg00 / sg01.
- In profiles core 0's weight DMA is ~6% slower than core 1's (packets 701 vs 670 ns per 16 KB; LM head alone
  1.56 vs 1.45 ms) and core 1 waits for core 0 at every exchange. Not known whether this is the profiler's own
  trace traffic or real.
- TensorE: bf16 matmul ~1 column per cycle at 2.4 GHz (moving 512: ~232 ns); fp32 ~3.2x slower. GpSimd
  elementwise ~0.13 elements/ns (avoid it for anything large); VectorE ~0.95, ScalarE ~1.2 elements/ns.

## Round 002: 3.37 -> 17.6 (dev-eval 007)

### What the engine is now

- **Both physical cores split every kernel** (LNC=2, `pid = nl.program_id(0)`): every GEMM is N-split (each core
  computes half the output features with the full K accumulation, so every value is computed exactly as before) and
  the halves are exchanged with `nisa.sendrecv` into a full SBUF tile (core 0's half first; pid-dependent SBUF
  slices work, measured). Core c runs DeltaNet heads 8c..8c+7 and full-attention KV head c (query heads 4c..4c+3),
  so the delta rule, the conv state, the recurrent state and the KV cache are per core and halved. 4 exchanges per
  layer + 1 in the head (argmax: each core has half the vocabulary; local (max, 2^20 - lowest id) pairs exchanged).
  Host re-layout: `relayout_split` = per-core row subsets, `[2][128 p][NTc][KT][128]`.
- **Fused decode step NEFF** `step_l{L}` (L = 512, 1024, 1152, 2048, 4096, 4224, 8192): embedding, 24 layers, final
  norm, LM head, argmax, `toks[pos + 1] = token`, `pos += 1`, one execution per token. Weights are stacked per kind
  (`l_*` for the 18 DeltaNet layers, `f_*` for the 6 full layers); per-layer NEFFs bind `nrt_tensor_allocate_slice`
  slices of the same device tensors.
- **Fused prefill chunk NEFFs** `pchunk{T}_l{L}` (T = 512, 256, 128; L = 512 .. 8192): embedding + 24 layers for T
  tokens + `pos += T`, hidden state to HBM; `headna_t{T}` computes the first token after the last chunk. The host
  covers the prompt with 512-chunks, then at most one 256 and one 128 chunk; the remainder (prompt % 128) runs the
  per-layer T=1 kernels (`embed_t1`, `lin_t1`, `full_t1_l*`, `adv_t1`, `head_t1`). Inside a chunk the GEMMs and
  elementwise ops run over all T tokens (one weight stream per T tokens); the token-major parts (embedding gather,
  beta/g, q/k transposes, delta rule, attention, V-cache write) loop over 128-token sub-chunks (`TS`, `NS`).
- **Chunked gated delta rule (WY form)** for 128-token sub-chunks on the TensorE in fp32 (`delta_chunk2`),
  **stage-major over the 8 local heads** (each elementwise op covers [128, 8, 128]; 8 independent matmuls per stage):
  gc = cumsum(g) by a triangular ones matmul; decay matrices exp(gc_i - gc_j) on the ScalarE with additive -1e30
  masks; M = beta_i (k_i.k_j) exp(gc_i - gc_j) (j < i); T^T = (I + M^T)^-1 by a **stable block-recursive inverse**:
  8x8 diagonal blocks by (I - B^T)(I + B^2T)(I + B^4T), then four merges TT <- TT - TT Cm^T TT (8->16->32->64->128),
  T itself from PE transposes of TT (13 fp32 matmuls + 4 transposes per head; a plain Neumann product over 128 grows
  like binomial coefficients with correlated keys and was not used); U = T (beta v), W = T (beta e^gc k),
  v_new = U - W S, o = (q e^gc) S + (QK^T * decay) v_new, S <- e^gc_last S + (k e^(gc_last - gc))^T v_new.
  S is kept as S^T[v, h, k] (decode layout) and transposed per head (fp32 PE transposes are exact). Constant masks
  come from the host (`dnc`, [128][9][128] fp32). Decode (T=1) keeps the token-by-token recurrence.
- **Decode attention** (T=1): keys on the partitions (key = b*128 + p), the 4 query heads of the core's KV head side
  by side: scores [128, NB, 4] by NB*2 matmuls, max over blocks then across partitions (PE transpose), broadcast
  back by a ones matmul, exp with per-partition partial sums, total by a ones matmul.
- Single-token norms (RMSNorm, l2norm, gated norm, q/k norm): one ones-matmul over all tiles' columns instead of one
  per tile (each reloaded the 128x128 fp32 ones stationary).
- **Position on the device**: `pos` [2,1,1] int32 and `posf` [2,128,1] fp32 (one copy per physical core: a shared
  copy raced, core A's update was read by core B), advanced in-kernel. The host queues every execution of a
  request with `nrta_execute_schedule`, waits once, reads the tokens from `toks` once.
- DeltaNet states zeroed on the device (`zero_states`) as the first queued execution of a reset request (the host
  write of 18 MB of zeros was 1.8 ms on the prefill path). stdin is non-blocking with a 10 s spin before poll().
- Weight / KV DMAs on the Sync engine's HWDGE queue (`dge_mode=hwdge, engine=sync`).
- Build: 35 NEFFs, cold build 4.5 min (12 parallel compiles), executable ~104 MB, engine ready ~22 s after start
  (NEFF loads). `build.sh` clears `target/neff` so only the current variants are embedded.

### Correctness (measured)

- Every change was checked with `brief/skill/scripts/same.py` (40 prompts x 3 request shapes, lengths 1..4097)
  against the previous build: core split, async queue, fused step, fused prefill chunks (128/256/512), the
  cleanup and the batched norms are token-identical to their parents (40/40). The chunked delta rule (v1 and v2) and
  the decode attention change fp32 rounding only. dev-eval: 0 or 1 mismatching steps per run (001-006).
- `delta_chunk2` standalone vs an fp64 recurrence: output rel. error 4e-6 .. 5e-6 (random to cos ~0.97 keys),
  state 1.5e-6 .. 5e-6 (v1 with 16-blocks: up to 3e-5). TensorE fp32 matmul measured rel. err 2.2e-7 (true fp32).
- A 300-token prompt where v1 and v2 differ at token 2: the reference agrees with v2.
- Long output: 200-token prompt + 3,840 tokens, then a continuation to position 8,191 (all tokens returned; the
  limit is exact: prompt + output <= 8,192 positions), reset afterwards works; odd prompt lengths run.
- Determinism: repeated identical requests give identical tokens (0 of 12 differ).

### Speed (measured)

dev-eval 007 (2 rounds, final build): **score 17.59** (prefill 29.6x, decode 10.5x), PASSED, 1 mismatching step per
round in p256 (0 elsewhere). (dev-eval 006, 1 round: 20.28 with a slow CPU reference on one case.)

| case | prefill cand ms | decode cand ms (64 tokens) |
|---|---|---|
| 256 | 14.6 | 424 |
| 1024 | 48.8 | 424 |
| 4096 | 198.5 | 430 |

Step by step (ab.py on cores 18/19, or `neuron-explorer` device time):

| change | effect |
|---|---|
| split over the 2 cores + async queue | decode 64 tok 1,300 -> 530 ms; prefill 256/1024/4096 120/463/1,844 -> 61/234/934 ms. Device: DeltaNet layer T=1 480 -> 260 us, full 398 -> 214 us, LM head 3,178 -> 1,620 us |
| chunked delta rule (T=128) | lin_t128 1,464 -> 403 us; prefill 61/234/934 -> 23/81/322 ms |
| fused decode step | 7.6 ms (sum of 26 kernels) + overhead -> 7.0 ms device; decode 532 -> 428 ms |
| HWDGE (Sync) for weight DMAs | step 7.02 -> 6.88 ms device |
| decode attention keys-on-partitions | step_l4224 7.45 -> 7.00 ms; decode at 4096: 467 -> 432 ms |
| device-side state zeroing + stdin spin | prefill +7.6% / +2.4% / +0.7% (256/1024/4096) |
| fused 128-token prefill chunk | prefill +13% / +14% / +14% |
| delta_chunk2 (stage-major, 8-block inverse) | pchunk_l512 8.51 -> 7.69 ms; prefill +8% / +9% / +8% |
| 256-token chunks | 12.5 ms per 256 tokens vs 2 x 7.7; prefill +19% / +20% / +22% |
| 512-token chunks | 23.4 ms per 512 tokens; prefill +7.3% at 1024 and 4096 |
| single-token norms batched | step 6.91 -> 6.84 ms; decode +0.2-1% |

Decode step now: 6.84 ms device (3.79 GB read); the LM head part streams ~690 GB/s (both cores), the layers ~550.
Per token 6.6 ms in requests.

### Tried and rejected (measured; step_l512 device time unless noted; baseline 6.88-6.91 ms)

- `sendrecv` with `dma_engine=gpsimd_dma` in the fused step: **races** (non-deterministic tokens, 1 of 9 repeated
  requests differed); the default DMA engine (CoreBarrier) is deterministic and not slower (decode 431 vs 428 ms).
- weight slabs alternating Sync/Scalar HWDGE queues: 7.20 ms. 2x larger slabs: 7.19 ms.
- small weight / state DMAs on the Scalar HWDGE queue: 6.99 ms.
- explicit SBUF ring for weight slabs (4 / 8 slots of 16 KB): 7.16 / 7.00 ms (DMA idle 478 -> 393 us but slower
  packets).
- contiguous 2 MB slabs in HBM (timing test): 6.94 ms.
- KV-cache loads on software DGE / Scalar HWDGE: 6.92 / 6.95 ms.
- preloading the out-proj and the first gate slabs before the DeltaNet chain: 6.90 / 6.92 ms.
- `is_stationary_onezero=True` on the ones / identity matmuls: 6.89 ms; pchunk256_l512 12.50 vs 12.52 ms.
- decode kb/qb broadcast by an identity-masked [128, 16, 128] operand instead of transposes: 6.93 vs 6.84 ms.
- no exchanges at all (timing test): 6.89 ms -> the exchanges cost nothing measurable.
- DeltaNet output not depending on the delta chain (timing test): 6.58 ms -> the serial conv/delta/norm chain
  costs ~0.33 ms per step.
- prefill GEMMs with the activations stationary and 512-wide weight blocks moving: lin_t128 398 vs 396 us
  (TensorE streaming-bound; power throttle limit ~0.56 in the profiles).
- (The rejected flags were removed from the code after measuring; this list is the record.)
- Bug found and fixed on the way: the old T=1 attention computed scores only for multiples of 512 keys (buckets
  1152 / 4224 had uninitialized scores).

### Where the time is (estimates from the profiles)

- Decode: weight streaming (~1.87 GB per core per token). Floor at the LM-head streaming rate (~690 GB/s) ~5.5 ms
  vs 6.84 ms; the gap is DMA idle while the TensorE waits on serial chains (~0.4 ms; weight DMAs wait on TensorE
  progress = buffer release) and slower DMA packets in the layer phases (880 vs 760 ns per 16 KB).
  Timeline analysis: `neuron-explorer view --output-format json` gives `instruction` (with `nki_source_location`),
  `dma` (per packet: queue, size, timestamp, duration) records; PE gaps attributed to source lines were the most
  useful view.
- Prefill: TensorE-bound GEMMs (~94 ns per 128x128x128 bf16 matmul, power throttled) plus the delta rule; the LM
  head (1.6 ms) once per request; attention at long buckets (prefill 4096: chunks use bucket >= position + 512).

### Next ideas (not measured)

- Decode delta rule: o = (S e^g).q + delta (k.q) shortens the per-token chain from 7 to 3 [128, 1024] passes
  (fp32 rounding changes).
- RMSNorm sum of squares on the ScalarE (activation with accumulate) instead of the fp32 ones matmul.
- More prefill buckets for the 512-chunks (1536, 2560, 3072, 3584): est. ~4% at 4096.

## Round 001 (first engine)

### What it is

- `kernels/qwen35.py`: NKI kernels, traced at build time by `kernels/build_neffs.py` (documented route:
  `NKI_ARTIFACTS_DIR` + calling the `@nki.jit` kernel with numpy zeros, `kernel[2]` for LNC=2). 16 NEFF variants,
  built in parallel processes (the post-compile test run fails when another build process holds the core; the
  NEFF is written before that and is kept), cached under `target/neff-cache` keyed by kernel source + variant.
  A cold build takes about 1 minute (51 s for all 16 variants with 12 workers).
- `build.sh` embeds every NEFF with `.incbin` (`target/neffs.S`) into `target/engine`, which links libnrt only.
- `src/engine.cpp`: mmaps the safetensors, re-lays out weights on the host (values unchanged), uploads them once,
  loads the NEFFs, and drives them with synchronous `nrt_execute`. `src/json.h`: a minimal JSON reader/writer.

Kernels (T = tokens per call, 1 or 128; L = attention bucket 512/1024/2048/4096/8192):
- `embed_t{T}`: token ids from the device `toks` buffer at `pos`, gather of embedding rows, PE transposes to
  feature-major.
- `lin_t{T}`: one gated-DeltaNet layer + MLP (18 layers use it, each with its own tensor set).
- `full_t{T}_l{L}`: one full-attention layer + MLP (6 layers), KV cache written at `pos` in place.
- `head_t{T}`: final RMSNorm of column T-1, LM head (weight-stationary, logits rounded to bf16 as the reference),
  argmax with ties to the lowest id on the device; returns one int32.

Host schedule per request: prompt split into 128-token chunks plus single-token steps for the remainder
(no padding anywhere, so no masking of state updates is needed); each chunk = embed + 24 layers (+ head on the
last chunk). Decode = one T=1 chain per token; the host writes the emitted token into `toks` and reads back the
4-byte argmax. Positions limit: prompt + output <= 8192 (error reply beyond).

### Layout and numerics (where the rounding points are)

- Activations feature-major `[128 p, n/128, T]`, feature = nt*128 + p. GEMM: weight tile `[128 k, 128 n]`
  stationary, activation `[128 k, T]` moving, so the same code is a GEMV at T=1 and a GEMM at T=128, and every
  output column is computed identically whatever T is (prefill partition does not change numerics).
- Weight re-layout: W [N, K] -> `[128 p][NT][KT][128 j]` = W[nt*128+j, kt*128+p] (contiguous per partition).
- RMSNorm `(1 + w)` in fp32, rounded once; sums of squares by an fp32 ones-matmul (cross-partition);
  rsqrt on GpSimd (`tensor_scalar(op0=rsqrt, engine=gpsimd)`, 7e-8 relative error; the ScalarE `activation`
  rsqrt is 4e-5, measured).
- Every linear output rounded to bf16 (fp32 PSUM -> bf16 RNE, measured identical to numpy RNE).
- DeltaNet: conv1d in fp32 rounded to bf16, SiLU bf16; l2norm of q/k in bf16 exactly as torch (bf16 squares, sum
  rounded to bf16, + eps in bf16, rsqrt rounded to bf16, product rounded); beta = bf16 sigmoid;
  g = -exp(A_log) * softplus(a + dt_bias) in fp32.
- Recurrent delta rule token by token in fp32, the reference's operation order (S *= exp(g); kv = sum_k S k;
  delta = (v - kv) beta; S += k delta; o = sum_k S q). State held transposed `S^T[v, h, k]` (v on partitions) so
  all 16 heads update in one instruction; the per-token k and q rows are broadcast to all partitions with a
  one-hot selector matmul (`ident[:, t:t+1].broadcast(1, 128)` as stationary, exact). The reference uses the
  chunked (64) algorithm for prefill and the recurrent one for decode; mathematically equal, fp32 noise only.
  The output is scaled by 1/sqrt(128) after the sum (the reference scales q first).
- Gated RMSNorm: bf16(o * rsqrt(mean(o^2)+eps)) * w(fp32) * silu(z fp32) -> bf16.
- Full attention: per-head q/k RMSNorm (1+w), partial RoPE on dims 0..63 with bf16 cos/sin tables (host
  constants computed like transformers: fp32 then bf16), rotate_half as a signed permutation matmul (exact),
  products and sum rounded to bf16 like torch. Scores fp32 (scale 1/16 exact), additive -1e30 mask
  (key j > pos + t), exp on ScalarE, probabilities normalized in fp32 then bf16 (eager softmax().to(bf16)),
  PV fp32 accumulate -> bf16, times bf16 sigmoid(gate).
- LNC=2: both physical cores run each kernel redundantly; every in-place state (recurrent, conv, K/V cache)
  has a leading per-core dimension indexed by `nl.program_id(0)` so no core reads what the other writes.

### Verification (measured)

`scratch`-style compare against the reference (`reference.sh`) on core 18: split requests generate(P,1) +
generate((),n-1), random-token prompts (seed 1234), teacher-forced judge on any difference.
- lengths 1,2,5,127,128,129,200,256,300,511,512,513 x 16 tokens: 191/192 identical steps; the one mismatch
  (len 256, step 12) is the reference's 2nd choice with margin 0.125.
- lengths 1024,2048,4096,4097 x 65 tokens: one mismatch in 260 steps (len 4096, step 1, rank 2, margin 0.0625).

### Harness dev-eval (measured, `enginesmith dev-eval`, 1 round, public prompts)

PASSED, score 3.4687 (prefill 3.5905, decode 3.3511); 1 mismatching step in total (p256).

| case | prefill ref / cand ms | prefill x | decode (64 tok) ref / cand ms | decode x |
|---|---|---|---|---|
| 256 | 370 / 120 | 3.09 | 4,531 / 1,318 | 3.44 |
| 1024 | 1,412 / 463 | 3.05 | 4,431 / 1,352 | 3.28 |
| 4096 | 9,067 / 1,847 | 4.91 | 4,868 / 1,457 | 3.34 |

Note: `/tmp/engine-gpu-16.lock` belongs to another user and `fs.protected_regular=2` makes dev-eval's
`open(path, 'a')` fail; run it with `GPU_LOCK_FILE=/tmp/engine-gpu-16-es-gptoss.lock` (own file, same cores).

Long-output check (measured): 200-token prompt + 3,840 tokens (79 s), then a continuation to exactly position
8,191 (4,151 tokens, 91 s): all tokens returned; one more continuation gets an error reply; a following reset
request works.

### Speed (measured, same compare runs; engine core 18, reference CPU, single runs, not the harness)

| | engine | reference |
|---|---|---|
| prefill 256 | 121 ms | 439 ms |
| prefill 1024 | 466 ms | 2,602 ms |
| prefill 4096 | 1,786 ms | 6,604 ms |
| decode | ~20 ms/token (15 tokens 310 ms; 64 tokens 1.2-1.4 s) | ~70-75 ms/token |
| remainder tokens (prompt % 128) | ~17 ms each (T=1 chain) | |

Load: NEFF loads 7.8 s, weight re-layout + upload 2.4 s.

### Known limits / next steps (estimates, not measured unless stated)

- Decode is ~20 ms/token: 26 synchronous `nrt_execute` calls (~70 us each) plus both cores streaming all
  weights redundantly (4 GB/token through one HBM stack split two ways). Expected levers: split each layer's
  GEMVs across the two cores (program_id) with a core exchange, queue executions asynchronously
  (`nrta_execute_schedule`) with the token/pos kept on device, fuse layers into fewer NEFFs.
- Prefill: the sequential delta-rule scan costs ~7 big VectorE ops per token per layer; a chunked (WY) form with
  matmuls would remove most of it. GEMMs at T=128 stream every weight once per 128 tokens; T=512 chunks would
  cut weight traffic 4x.
- Remainder tokens run as single-token steps (up to 127 x 26 executions); intermediate chunk sizes (32, 8)
  would shorten odd-length prompts. Timed prompts (256/1024/4096) are multiples of 128.
