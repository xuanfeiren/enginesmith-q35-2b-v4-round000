"""NKI kernels for Qwen3.5-2B (text model) on Trainium2, logical NeuronCore config 2.

Build-time only: build_neffs.py traces these with numpy inputs and keeps the NEFFs.

Layout conventions
- Feature-major activations ("fm"): a [N] vector per token lives as [128 p, N/128 nt, T] with
  feature n = nt*128 + p. Hidden states between kernels are HBM tensors [128, 16, T] bf16.
- GEMM weights W [N, K] are re-laid out on the host as [128 p, NT, KT, 128 j] with
  element (p, nt, kt, j) = W[nt*128 + j, kt*128 + p]: tile (nt, kt) is the stationary operand
  [128 k, 128 n] and the activation [128 k, T] is the moving one, so y lands feature-major.
- The two physical cores (LNC=2, pid = nl.program_id(0)) split every GEMM along N: weights are
  re-laid out on the host per core as [2, P, NT_core, KT, 128] and each core computes its half
  of the output features with the full K accumulation (so every output value is computed
  exactly as by one core). Halves are exchanged with nisa.sendrecv into a full SBUF tile
  (core 0's half first). DeltaNet heads 8c..8c+7 and full-attention KV head c (query heads
  4c..4c+3) run on core c only. Norms, residual adds and the embedding run on both cores,
  which write identical bytes to shared outputs. Every tensor a kernel updates in place
  (recurrent state, conv state, KV cache, pos) has a leading per-core dimension of 2 and each
  core touches only its own copy.

Module globals T (tokens per call) and L (attention bucket) are set by the build script
before tracing.
"""
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
D = 2048
KT = D // P            # 16
DFF = 6144
FT = DFF // P          # 48
NQKV = 6144
QT = NQKV // P         # 48
NLH = 16               # linear-attention heads (k = v heads), head dim 128
NQH = 8                # full-attention query heads, head dim 256
NKVH = 2
HL = NLH // 2          # DeltaNet heads per core
QL = 3 * HL            # qkv channel tiles per core (q, k, v of its heads)
FL = FT // 2           # MLP intermediate tiles per core
VOCAB = 248320
VT = VOCAB // P        # 1940
VTL = VT // 2          # LM-head vocabulary tiles per core
MAXPOS = 8192
EPS = 1e-6
NEG = -1.0e30
NLAYER = 24
NMASK = 13

T = 1                  # tokens per call (set by the build script)
TS = 1                 # token-major sub-chunk (min(T, 128)), set by the build script
NS = 1                 # sub-chunks per call (T // TS), set by the build script
L = 256                # attention bucket (keys read), set by the build script
TIGHT = False          # prefill chunk variant with L == pos + T (set by the build script)
FIRST = False          # prefill chunk variant that only runs at position 0: DeltaNet states start at zero
HEAD_ADV = True        # the head kernel advances pos (False: the fused prefill chunk did)
HEADCOL = 0            # column of x the head kernel reads (T - 1 of the producing call)

FP32 = nl.float32
BF16 = nl.bfloat16
OPG = 2                # prefill out-projection slab (= exchange group) size
YG = 1                 # prefill: DeltaNet output heads per y exchange group (2 until round 010)
PREFETCH = True        # fused decode step: small per-layer loads issued one layer ahead
DPAIR = 2              # prefill delta rule: sub-chunks whose state-independent stages are interleaved


# ----------------------------------------------------------------------------------------------
# helpers (top-level functions called with constant arguments are inlined by the tracer)

def ones_tile(dtype):
    t = nl.ndarray((P, P), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=t, value=1.0)
    return t


def ident_tile(dtype=BF16):
    """identity [128, 128]."""
    ones = ones_tile(dtype)
    ident = nl.ndarray((P, P), dtype=dtype, buffer=nl.sbuf)
    nisa.affine_select(dst=ident, pattern=[[-1, P]], channel_multiplier=1, on_true_tile=ones,
                       on_false_value=0.0, cmp_op=nl.equal, offset=0)
    return ident


def identf_small(n):
    """fp32 identity [n, n]"""
    ones = nl.ndarray((n, n), dtype=FP32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    idn = nl.ndarray((n, n), dtype=FP32, buffer=nl.sbuf)
    nisa.affine_select(dst=idn, pattern=[[-1, n]], channel_multiplier=1, on_true_tile=ones,
                       on_false_value=0.0, cmp_op=nl.equal, offset=0)
    return idn


def rsqrt_accurate(dst, src):
    """dst = 1/sqrt(src) on GpSimd (accurate to ~1e-7; the ScalarE table is ~4e-5). src in SBUF."""
    nisa.tensor_scalar(dst=dst, data=src, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)


# fused decode step: small per-layer loads issued one layer ahead (see prefetch in step). PFC holds the tiles
# prefetched for the layer being emitted; ld() takes a tile from it or loads it on the spot.
PFC = []               # [dict of the current layer's prefetched tiles] in the fused decode step, else empty


def ld(name, shape, dtype, src):
    if PFC and name is not None and name in PFC[0]:
        return PFC[0][name]
    t = nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=src)
    return t


PF_SET = ('S', 'cs')   # which small loads the decode step prefetches one layer ahead
PF_SYNC = True         # prefetch DMAs on the Sync HWDGE queue (in program order with the weight slabs)


def pf_load(d, name, shape, dtype, src):
    t = nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
    if PF_SYNC:
        nisa.dma_copy(dst=t, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    else:
        nisa.dma_copy(dst=t, src=src)
    d[name] = t


def load_norm_w1(w_hbm, nt, name=None):
    """w_hbm [128, nt] bf16 -> SBUF fp32 (1 + w)."""
    wb = ld(name, (P, nt), BF16, w_hbm)
    w1 = nl.ndarray((P, nt), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w1, data=wb, op0=nl.add, operand0=1.0)
    return w1


def tok_bcast(dst, vt, n, selF, identf):
    """dst PSUM [P, n*TS] fp32: dst[f, s*TS + t] = vt[t, s] for vt SBUF [TS, n] fp32 (token-major values).
    PE transpose to [n, TS], then per sub-chunk s a selector matmul (stationary selF[0:n, s, :], row s = 1)."""
    tp = nl.ndarray((n, TS), dtype=FP32, buffer=nl.psum)
    nisa.nc_transpose(dst=tp, data=vt)
    vT = nl.ndarray((n, TS), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vT, src=tp)
    for s in range(n):
        nisa.nc_matmul(dst=dst[:, s * TS:(s + 1) * TS], stationary=selF[0:n, s, :], moving=vT)


def rmsnorm_fm(x, w1, nt, t, nd, ones_f, selF=None, identf=None, pid=None, defer=False):
    """Qwen3.5 RMSNorm over features: bf16((x * rsqrt(mean(x^2) + eps)) * (1 + w)).
    x [P, nt, t] bf16 SBUF; w1 [P, nt] fp32 = 1 + w. Returns bf16 [P, nt, t] (defer: a norm_ctx dict)."""
    if defer:
        return norm_ctx(x, w1, nt, t, nd, selF, identf, pid)
    if t > 1 and selF is not None:
        # sums of squares token-major as the diagonal of the Gram matrix of each 128-token sub-chunk (exact
        # products, fp32 accumulation as the ones-matmul form); rsqrt on NS values per partition
        gr = nl.ndarray((TS, NS, TS), dtype=FP32, buffer=nl.psum)
        # k-outer (each sub-chunk still accumulates k = 0 .. nt-1 in order): the Gram runs on early tiles of every
        # sub-chunk while the last (exchanged) tiles are still on their way
        for k0 in range(nt):
            k = kord(k0, nt, nt // 2 if pid is not None else 0, pid)
            for s in range(NS):
                nisa.nc_matmul(dst=gr[:, s, :], stationary=x[:, k, s * TS:(s + 1) * TS], moving=x[:, k, s * TS:(s + 1) * TS])
        dg = nl.ndarray((TS, NS, TS), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dg, data1=gr, data2=identf.expand_dim(1).broadcast(1, NS), op=nl.multiply)
        tot = nl.ndarray((TS, NS), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tot, op=nl.add, data=dg, axis=2)
        vart = nl.ndarray((TS, NS), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=vart, data=tot, op0=nl.multiply, operand0=1.0 / nd, op1=nl.add, operand1=EPS)
        rst = nl.ndarray((TS, NS), dtype=FP32, buffer=nl.sbuf)
        rsqrt_accurate(rst, vart)
        bc = nl.ndarray((P, t), dtype=FP32, buffer=nl.psum)
        tok_bcast(bc, rst, NS, selF, identf)
        y = nl.ndarray((P, nt, t), dtype=BF16, buffer=nl.sbuf)
        for k in range(nt):
            xn = nl.ndarray((P, t), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=xn, data1=x[:, k, :], data2=bc, op=nl.multiply)
            nisa.activation(dst=y[:, k, :], op=nl.copy, data=xn, scale=w1[:, k:k + 1])
        return y
    sq = nl.ndarray((P, nt, t), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
    var = nl.ndarray((P, t), dtype=FP32, buffer=nl.sbuf)
    if t == 1:
        # one token: one matmul gives the per-tile sums (one stationary load), then a sum over the tiles
        pst = nl.ndarray((P, nt), dtype=FP32, buffer=nl.psum)
        nisa.nc_matmul(dst=pst, stationary=ones_f, moving=sq[:, :, 0])
        tot = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tot, op=nl.add, data=pst, axis=1)
        nisa.tensor_scalar(dst=var, data=tot, op0=nl.multiply, operand0=1.0 / nd, op1=nl.add, operand1=EPS)
    else:
        ps = nl.ndarray((P, t), dtype=FP32, buffer=nl.psum)
        for k in range(nt):
            nisa.nc_matmul(dst=ps, stationary=ones_f, moving=sq[:, k, :])
        nisa.tensor_scalar(dst=var, data=ps, op0=nl.multiply, operand0=1.0 / nd, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((P, t), dtype=FP32, buffer=nl.sbuf)
    rsqrt_accurate(rstd, var)
    xn = nl.ndarray((P, nt, t), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xn, data1=x, data2=rstd.expand_dim(1).broadcast(1, nt), op=nl.multiply)
    y = nl.ndarray((P, nt, t), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=xn, data2=w1.expand_dim(2).broadcast(2, t), op=nl.multiply)
    return y


GROWB = True           # prefill delta rule: grow as an exact bf16 broadcast of gcs (see dc_s1)
CONVPE = 3             # prefill depthwise conv: taps on the TensorE (the rest at the evacuation on the VectorE)
FIRSTG = 2             # prefill: size of the first weight slab of a GEMM that consumes a norm (0: as the others)
RSD = True             # prefill: GEMMs on bf16(x (1 + w)), the per-token rstd applied at the PSUM evacuation (norm_ctx)


def norm_ctx(x, w1, nt, t, nd, selF, identf, pid):
    """Prefill RMSNorm with the per-token rstd deferred to the consumers: hx = bf16(x (1 + w)) per k tile (ScalarE)
    is the GEMM input at once; the sum of squares (Gram diagonals), rsqrt and the [P, t] broadcast are emitted by the
    first consuming GEMM after its own-half k tiles (norm_gram) and after its k loop (norm_bcast), so the TensorE
    does not wait for the norm chain; outputs are bf16(rstd * (W hx)) (rounding differs from bf16(W bf16(x rstd (1 + w))))."""
    c = {'x': x, 'nt': nt, 't': t, 'nd': nd, 'selF': selF, 'identf': identf, 'pid': pid, 'rst': None, 'bcs': None}
    hx = nl.ndarray((P, nt, t), dtype=BF16, buffer=nl.sbuf)
    for k0 in range(nt):
        k = kord(k0, nt, nt // 2, pid)
        nisa.activation(dst=hx[:, k, :], op=nl.copy, data=x[:, k, :], scale=w1[:, k:k + 1])
    c['hx'] = hx
    return c


def norm_gram(c):
    if c['rst'] is not None:
        return
    x = c['x']
    nt = c['nt']
    gr = nl.ndarray((TS, NS, TS), dtype=FP32, buffer=nl.psum)
    for k0 in range(nt):
        k = kord(k0, nt, nt // 2, c['pid'])
        for s in range(NS):
            nisa.nc_matmul(dst=gr[:, s, :], stationary=x[:, k, s * TS:(s + 1) * TS], moving=x[:, k, s * TS:(s + 1) * TS])
    dg = nl.ndarray((TS, NS, TS), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dg, data1=gr, data2=c['identf'].expand_dim(1).broadcast(1, NS), op=nl.multiply)
    tot = nl.ndarray((TS, NS), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=tot, op=nl.add, data=dg, axis=2)
    vart = nl.ndarray((TS, NS), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=vart, data=tot, op0=nl.multiply, operand0=1.0 / c['nd'], op1=nl.add, operand1=EPS)
    rst = nl.ndarray((TS, NS), dtype=FP32, buffer=nl.sbuf)
    rsqrt_accurate(rst, vart)
    c['rst'] = rst


def norm_bcast(c):
    norm_gram(c)
    if c['bcs'] is not None:
        return
    bc = nl.ndarray((P, c['t']), dtype=FP32, buffer=nl.psum)
    tok_bcast(bc, c['rst'], NS, c['selF'], c['identf'])
    bcs = nl.ndarray((P, c['t']), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bcs, src=bc)
    c['bcs'] = bcs


def kord(k, n, half, pid):
    """k-th index of an own-half-first order over n tiles whose halves [0, half) / [half, n) belong to core 0 / 1
    (half = 0: natural order). pid-dependent: core c starts with its own tiles (the others arrive by exchange)."""
    if half == 0:
        return k
    if k < half:
        return k + pid * half
    return k - pid * half


def gemm_fm(dst, w, pid, x, nt_total, kt_total, t, group, xfull=None, resid=None, kown=0, nctx=None):
    """dst[:, nt, :] = bf16(W_core x) feature-major. w HBM [2, P, NT, KT, 128] (this core: w[pid]);
    x SBUF [P, KT, t] bf16; dst SBUF view [P, NT, t]. Weight slabs of `group` output tiles on the Sync HWDGE queue.
    xfull (prefill): the full [P, 2*nt_total, t] tile dst is this core's half of; each group is sent to the other
    core as soon as it is evacuated."""
    sizes = []
    if nctx is not None and nctx['bcs'] is None and FIRSTG and T >= 512 and nt_total > FIRSTG:
        sizes.append(FIRSTG)          # the first slab after a norm is smaller: it arrives sooner
    while sum(sizes) < nt_total:
        sizes.append(min(group, nt_total - sum(sizes)))
    for si in range(len(sizes)):
        n0 = sum(sizes[:si])
        g = sizes[si]
        wt = nl.ndarray((P, g, kt_total, P), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w[pid, :, n0:n0 + g, :, :], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        if t > 1:
            # k-outer over the slab's g output tiles (g PSUM banks): the first matmuls need only x[:, 0], so a GEMM
            # starts while the producer (norm, exchange) is still finishing later k tiles; per-tile accumulation
            # order (k = 0 .. KT-1) is unchanged
            ps = nl.ndarray((P, g, t), dtype=FP32, buffer=nl.psum)
            for k0 in range(kt_total):
                k = kord(k0, kt_total, kown, pid)
                if nctx is not None and k0 == kown:
                    norm_gram(nctx)           # the norm's TensorE work after this slab's own-half k tiles
                for i in range(g):
                    nisa.nc_matmul(dst=ps[:, i, :], stationary=wt[:, i, k, :], moving=x[:, k, :])
            if nctx is not None:
                norm_bcast(nctx)
                for i in range(g):
                    nisa.tensor_tensor(dst=dst[:, n0 + i, :], data1=ps[:, i, :], data2=nctx['bcs'], op=nl.multiply)
            for i in range(g if nctx is None else 0):
                if i % 2 == 1:
                    nisa.activation(dst=dst[:, n0 + i, :], op=nl.copy, data=ps[:, i, :])   # ScalarE (RNE, as VectorE)
                else:
                    nisa.tensor_copy(dst=dst[:, n0 + i, :], src=ps[:, i, :])
        else:
            for i in range(g):
                ps = nl.ndarray((P, t), dtype=FP32, buffer=nl.psum)
                for k in range(kt_total):
                    nisa.nc_matmul(dst=ps, stationary=wt[:, i, k, :], moving=x[:, k, :])
                nisa.tensor_copy(dst=dst[:, n0 + i, :], src=ps)
        if xfull is not None:
            exchange_part(xfull, nt_total, n0, g, pid)
            if resid is not None:
                # residual add per group as soon as both halves of it are here (own tiles, then the received ones)
                rin, rout = resid
                for c in (pid, 1 - pid):
                    lo = c * nt_total + n0
                    # the received half on the GpSimd (it runs the exchange handshake; the in-order VectorE does not wait
                    # for the exchange)
                    nisa.tensor_tensor(dst=rout[:, lo:lo + g, :], data1=rin[:, lo:lo + g, :], data2=xfull[:, lo:lo + g, :],
                                       op=nl.add, engine=(nisa.engine.gpsimd if c != pid else nisa.engine.unknown))


def gemm_n(dst, w, pid, h, nt_total, kt_total, t, group):
    """gemm_fm on the output of rmsnorm_fm (a norm_ctx dict when deferred: own-half k tiles first)"""
    if t > 1 and RSD:
        gemm_fm(dst, w, pid, h['hx'], nt_total, kt_total, t, group, kown=kt_total // 2, nctx=h)
    else:
        gemm_fm(dst, w, pid, h, nt_total, kt_total, t, group)


def exchange_part(full, h, lo, n, pid):
    """full [P, 2h, t]: send this core's tiles pid*h + lo .. + n, receive the other core's same tiles."""
    nisa.sendrecv(src=full[:, pid * h + lo:pid * h + lo + n, :], dst=full[:, (1 - pid) * h + lo:(1 - pid) * h + lo + n, :],
                  send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)


def exchange(full, h, pid, t):
    """full [P, 2h, t]: this core filled tiles pid*h .. pid*h+h; receive the other core's half.
    In the fused decode step (XB set) the exchange tiles are slices of one buffer that lives for the whole step, so
    no address is ever reused and the GpSimd SBUF-to-SBUF DMA can be used: it does not hold the Sync queue (which
    streams the weights) at a CoreBarrier. Elsewhere the default DMA engine (CoreBarrier-synchronised)."""
    if XB:
        nisa.sendrecv(src=full[:, pid * h:pid * h + h, :], dst=full[:, (1 - pid) * h:(1 - pid) * h + h, :],
                      send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0, dma_engine=nisa.dma_engine.gpsimd_dma)
    else:
        nisa.sendrecv(src=full[:, pid * h:pid * h + h, :], dst=full[:, (1 - pid) * h:(1 - pid) * h + h, :],
                      send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)


XB = []                # fused decode step: [buffer, next free column] for the exchange tiles (see xtile)


def xtile(n, t):
    """[P, n, t] bf16 SBUF tile for an exchange: a fresh slice of the step-wide buffer XB (decode step), else new."""
    if XB:
        off = XB[1]
        XB[1] = off + n
        return XB[0][:, off:off + n, :]
    return nl.ndarray((P, n, t), dtype=BF16, buffer=nl.sbuf)


def mlp_block(x1, ln2, wg, wu, wd, pid, t, ones_f, selF=None, identf=None):
    """x2 = x1 + down(silu(gate(h)) * up(h)), h = rmsnorm(x1). All bf16 rounding points as HF.
    gate/up: this core's FL intermediate tiles; down: this core's 8 output tiles."""
    w1 = load_norm_w1(ln2, KT, 'ln2')
    h = rmsnorm_fm(x1, w1, KT, t, D, ones_f, selF, identf, pid, defer=(t > 1 and RSD))
    m = xtile(FT, t) if t == 1 else nl.ndarray((P, FT, t), dtype=BF16, buffer=nl.sbuf)
    if t == 1:
        gate = nl.ndarray((P, FL, t), dtype=BF16, buffer=nl.sbuf)
        gemm_fm(gate, wg, pid, h, FL, KT, t, 4)
        up = nl.ndarray((P, FL, t), dtype=BF16, buffer=nl.sbuf)
        gemm_fm(up, wu, pid, h, FL, KT, t, 4)
        act = nl.ndarray((P, FL, t), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=act, op=nl.silu, data=gate)
        nisa.tensor_tensor(dst=m[:, pid * FL:pid * FL + FL, :], data1=act, data2=up, op=nl.multiply)
        exchange(m, FL, pid, t)
    else:
        # per group of 4 intermediate tiles: gate, up, silu * up, and the group goes to the other core at once
        G = 4
        for n0 in range(0, FL, G):
            gate = nl.ndarray((P, G, t), dtype=BF16, buffer=nl.sbuf)
            gemm_n(gate, wg[:, :, n0:n0 + G, :, :], pid, h, G, KT, t, G)
            up = nl.ndarray((P, G, t), dtype=BF16, buffer=nl.sbuf)
            gemm_n(up, wu[:, :, n0:n0 + G, :, :], pid, h, G, KT, t, G)
            act = nl.ndarray((P, G, t), dtype=BF16, buffer=nl.sbuf)
            nisa.activation(dst=act, op=nl.silu, data=gate)
            nisa.tensor_tensor(dst=m[:, pid * FL + n0:pid * FL + n0 + G, :], data1=act, data2=up, op=nl.multiply)
            exchange_part(m, FL, n0, G, pid)
    dn = xtile(KT, t) if t == 1 else nl.ndarray((P, KT, t), dtype=BF16, buffer=nl.sbuf)
    x2 = nl.ndarray((P, KT, t), dtype=BF16, buffer=nl.sbuf)
    if t == 1:
        gemm_fm(dn[:, pid * 8:pid * 8 + 8, :], wd, pid, m, 8, FT, t, 2)
        exchange(dn, 8, pid, t)
        nisa.tensor_tensor(dst=x2, data1=x1, data2=dn, op=nl.add)
    else:
        gemm_fm(dn[:, pid * 8:pid * 8 + 8, :], wd, pid, m, 8, FT, t, 2, xfull=dn, resid=(x1, x2), kown=FL)
    return x2


def load_pos(pos, pid):
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos[pid])
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    return preg


# ----------------------------------------------------------------------------------------------
# chunked gated delta rule (prefill, T = 128): WY form on the TensorE in fp32, stage-major over the heads

def bmm(stats, movs, H, T=P):
    """psum [P, H, T]: head h = stats(h)^T @ movs(h)"""
    ps = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.psum)
    for h in range(H):
        nisa.nc_matmul(dst=ps[:, h, :], stationary=stats[:, h, :], moving=movs[:, h, :])
    return ps


def btr(x, H):
    """psum [P, H, P]: per-head transpose of x [P, H, P] fp32"""
    ps = nl.ndarray((P, H, P), dtype=FP32, buffer=nl.psum)
    for h in range(H):
        nisa.nc_transpose(dst=ps[:, h, :], data=x[:, h, :])
    return ps


def bsb(ps, H, T=P):
    s = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s, src=ps)
    return s


def bop(a, b, op, H, T=P):
    s = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s, data1=a, data2=b, op=op)
    return s


def sel3_tile():
    """bf16 [3HL, HL, P]: row j*HL + c selects head c (three stacked identities), for exact hi + mid + lo broadcasts"""
    i24 = identf_small(3 * HL)
    s3 = nl.ndarray((3 * HL, HL), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s3, data1=i24[:, 0:HL], data2=i24[:, HL:2 * HL], op=nl.add)
    nisa.tensor_tensor(dst=s3, data1=s3, data2=i24[:, 2 * HL:3 * HL], op=nl.add)
    sel3 = nl.ndarray((3 * HL, HL, P), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sel3, src=s3.expand_dim(2).broadcast(2, P))
    return sel3


def dc_s1(st, a, dm, identf, H, T):
    """chunked delta rule, state-independent part, stage 1: cumulative log decays (see delta_pair)"""
    Ut = dm[:, 0, :]
    gc_ps = nl.ndarray((T, H), dtype=FP32, buffer=nl.psum)
    nisa.nc_matmul(dst=gc_ps, stationary=Ut, moving=a['gneg'])
    gcs = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gcs, src=gc_ps)
    egc = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf)
    nisa.activation(dst=egc, op=nl.exp, data=gcs)
    bexp = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=bexp, data1=egc, data2=a['bg'][:, H:2 * H], op=nl.multiply)
    # grow[m, h, i] = gc_i of head h on every partition m
    gr_ps = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.psum)
    if GROWB and T * NS <= 256:
        # the fp32 matmul gneg_h (broadcast) x Ut gives exactly gcs (measured bitwise, scratch num/grow.py), so grow is
        # gcs^T broadcast: an exact three-way bf16 split hi + mid + lo of gcs, one bf16 transpose to [3H, T], and per
        # head a bf16 selector matmul (1 hi + 1 mid + 1 lo = gc exactly in the fp32 PSUM) instead of a 128-column
        # fp32 matmul
        r3 = nl.ndarray((T, 3 * H), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=r3[:, 0:H], src=gcs)
        d1 = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d1, data1=gcs, data2=r3[:, 0:H], op=nl.subtract)
        nisa.tensor_copy(dst=r3[:, H:2 * H], src=d1)
        d2 = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d2, data1=d1, data2=r3[:, H:2 * H], op=nl.subtract)
        nisa.tensor_copy(dst=r3[:, 2 * H:3 * H], src=d2)
        rtp = nl.ndarray((3 * H, T), dtype=BF16, buffer=nl.psum)
        nisa.nc_transpose(dst=rtp, data=r3)
        rT = nl.ndarray((3 * H, T), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rT, src=rtp)
        for h in range(H):
            nisa.nc_matmul(dst=gr_ps[:, h, :], stationary=a['sel3'][:, h, :], moving=rT)
    else:
        for h in range(H):
            nisa.nc_matmul(dst=gr_ps[:, h, :], stationary=a['gneg'][:, h:h + 1].broadcast(1, P), moving=Ut)
    st['grow'] = bsb(gr_ps, H, T)
    st['gcs'] = gcs
    st['bexp'] = bexp


def dc_s2(st, a, dm, identf, H, T):
    """decay matrices E1[i, h, j] = exp(gc_i - gc_j) (j < i), E2[j, h, i] (j <= i); G = K K^T"""
    grow = st['grow']
    gcs = st['gcs']
    # per head one fused op each (same operations in the same order): d1n = (gc_j - gc_i) - mask1 = -d1 exactly
    # (IEEE subtraction is sign-symmetric; masked entries become +BIG), E1 = exp(-d1n); d2 = (gc_i - gc_j) + mask2
    # dif = gc_j - gc_i once for all heads, then the two masks (the same two roundings per element as the per-head
    # fused form; three [P, H, T] passes instead of 2H small ones)
    dif = bop(grow, gcs.expand_dim(2).broadcast(2, T), nl.subtract, H, T)
    d1n = bop(dif, dm[:, 1, :].expand_dim(1).broadcast(1, H), nl.subtract, H, T)
    d2 = bop(dif, dm[:, 2, :].expand_dim(1).broadcast(1, H), nl.add, H, T)
    E1 = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.sbuf)
    nisa.activation(dst=E1, op=nl.exp, data=d1n, scale=-1.0)
    E2 = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.sbuf)
    nisa.activation(dst=E2, op=nl.exp, data=d2)
    egrow = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.sbuf)
    nisa.activation(dst=egrow, op=nl.exp, data=grow)
    dk = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dk, data1=grow[:, :, T - 1], data2=st['gcs'], op=nl.subtract)
    ekd = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf)          # exp(gc_last - gc_j)
    nisa.activation(dst=ekd, op=nl.exp, data=dk)
    eL = nl.ndarray((P, H), dtype=FP32, buffer=nl.sbuf)           # exp(gc_last)
    nisa.activation(dst=eL, op=nl.exp, data=grow[:, :, T - 1])
    qk = a['qk']
    g_ps = nl.ndarray((P, H, P), dtype=FP32, buffer=nl.psum)
    for h in range(H):
        nisa.nc_matmul(dst=g_ps[:, h, :], stationary=qk[:, H + h, :], moving=qk[:, H + h, :])
    st['g_ps'] = g_ps
    st['E1'] = E1
    st['E2'] = E2
    st['egrow'] = egrow
    st['ekd'] = ekd
    st['eL'] = eL


def dc_s3(st, a, dm, identf, H, T):
    """M[i, h, j] = beta_i (k_i . k_j) exp(gc_i - gc_j), j < i; its transpose; the 8-block parts"""
    M = nl.ndarray((P, H, P), dtype=FP32, buffer=nl.sbuf)
    for h in range(H):
        nisa.scalar_tensor_tensor(dst=M[:, h, :], data=st['g_ps'][:, h, :], op0=nl.multiply,
                                  operand0=a['bg'][:, H + h:H + h + 1], op1=nl.multiply, operand1=st['E1'][:, h, :])
    st['M'] = M
    st['B'] = bop(M, dm[:, 3, :].expand_dim(1).broadcast(1, H), nl.multiply, H)
    st['BT'] = bop(btr(M, H), dm[:, 4, :].expand_dim(1).broadcast(1, H), nl.multiply, H)   # masked M^T straight from PSUM


def dc_s4(st, a, dm, identf, H, T):
    st['B2'] = bsb(bmm(st['BT'], st['B'], H), H)
    st['B2T'] = bsb(bmm(st['B'], st['BT'], H), H)


def dc_s5(st, a, dm, identf, H, T):
    idb = identf.expand_dim(1).broadcast(1, H)
    st['IB2'] = bop(st['B2'], idb, nl.add, H)
    st['IB4'] = bop(bmm(st['B2T'], st['B2'], H), idb, nl.add, H)
    st['XT'] = bop(idb, st['BT'], nl.subtract, H)


def dc_s6(st, a, dm, identf, H, T):
    st['XT'] = bsb(bmm(st['IB2'], st['XT'], H), H)


def dc_s7(st, a, dm, identf, H, T):
    st['XT'] = bsb(bmm(st['IB4'], st['XT'], H), H)


def dc_s8(st, a, dm, identf, H, T):
    st['X'] = bsb(btr(st['XT'], H), H)


def dc_merge(st, a, dm, identf, H, T, lv):
    """T^T merge at level lv (blocks 8 << lv): TT <- TT - TT Cm^T TT. Cm^T TT = (M^T TT) masked by dm[9 + lv] = dm[5 + lv]^T:
    TT is block-diagonal (blocks 8 << lv, exact zeros elsewhere), so the unmasked product has the same nonzero terms
    in the kept entries (one VectorE pass less than masking M first)."""
    ZT = bop(bmm(st['M'], st['XT'], H), dm[:, 9 + lv, :].expand_dim(1).broadcast(1, H), nl.multiply, H)
    st['XT'] = bop(st['XT'], bmm(st['X'], ZT, H), nl.subtract, H)


def dc_mergeT(st, a, dm, identf, H, T):
    st['X'] = bsb(btr(st['XT'], H), H)


def dc_s9(st, a, dm, identf, H, T):
    """token-major operands, U = T (beta v), W^T = (T beta e^gc k)^T, A2 = (Q K^T) * E2"""
    TT = st['XT']                                                  # T^T [j, h, i]
    vt_ps = nl.ndarray((T, H, P), dtype=BF16, buffer=nl.psum)
    for h in range(H):
        nisa.nc_transpose(dst=vt_ps[:, h, :], data=a['vfm'][:, h, :])
    bg = a['bg']
    vb = bop(vt_ps, bg[:, H:2 * H].expand_dim(2).broadcast(2, P), nl.multiply, H)
    ktm = a['k_tm']
    kbg = bop(ktm, st['bexp'].expand_dim(2).broadcast(2, P), nl.multiply, H)
    st['kd'] = bop(ktm, st['ekd'].expand_dim(2).broadcast(2, P), nl.multiply, H)
    qk = a['qk']
    st['qg'] = bop(qk[:, 0:H, :], st['egrow'], nl.multiply, H, T)
    st['U'] = bsb(bmm(TT, vb, H), H)                               # [i, h, dv]
    st['WT'] = bsb(bmm(kbg, TT, H), H)                             # [dk, h, i]
    a2_ps = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.psum)
    for h in range(H):
        nisa.nc_matmul(dst=a2_ps[:, h, :], stationary=qk[:, H + h, :], moving=qk[:, h, :])
    st['A2'] = bop(a2_ps, st['E2'], nl.multiply, H, T)


def dc_dep(st, a, Sd, H, T, zero=False):
    """state-dependent part: v_new = U - W S, o = (q e^gc) S + A2 v_new, S <- e^gc_last S + (k e^(gc_last - gc))^T v_new.
    zero: S = 0 (the first sub-chunk of a request): v_new = U, o = A2 v_new, S <- (k e^(gc_last - gc))^T v_new. The
    terms left out are exact zeros (U - 0 = U, 0 + x = x), so every value is the same."""
    if zero:
        vnew = st['U']
    else:
        vnew = bop(st['U'], bmm(st['WT'], Sd, H), nl.subtract, H)
    o_ps = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.psum)
    for h in range(H):
        if not zero:
            nisa.nc_matmul(dst=o_ps[:, h, :], stationary=Sd[:, h, :], moving=st['qg'][:, h, :])
        nisa.nc_matmul(dst=o_ps[:, h, :], stationary=vnew[:, h, :], moving=st['A2'][:, h, :])
    nisa.tensor_copy(dst=a['o'], src=o_ps)
    if zero:
        nisa.tensor_copy(dst=Sd, src=bmm(st['kd'], vnew, H))
    else:
        Sn = bop(Sd, st['eL'].expand_dim(2).broadcast(2, P), nl.multiply, H)
        nisa.tensor_tensor(dst=Sd, data1=Sn, data2=bmm(st['kd'], vnew, H), op=nl.add)   # state stays [dk, h, dv]


def delta_pair(subs, Sd, dm, identf, H, T, zero=False):
    """Chunked gated delta rule (WY form, fp32 on the TensorE) for one or two consecutive 128-token sub-chunks.
    subs: list of dicts with qk [P, 2H, T] bf16 (q heads then k heads, feature-major, l2-normed), vfm [P, H, T] bf16,
    k_tm [T, H, P] bf16 token-major k, bg [T, 2H] fp32 [eg | beta], gneg [T, H] fp32 = g (log decay), o [P(v), H, T]
    fp32 output (unscaled q). Sd [dk, H, dv] fp32 state (in place).
    dm [P, 13, P] fp32 masks: 0 Ut (t <= i), 1 -BIG where j >= i, 2 -BIG where part > free, 3/4 strictly lower
    8-blocks (and transpose), 5..8 lower off-diagonal block of each 2s pair, s = 8, 16, 32, 64, 9..12 their transposes.
    The state-independent stages of the sub-chunks are emitted interleaved (stage k of each, then stage k + 1), so
    the TensorE works on one sub-chunk while the VectorE / ScalarE finish the other's previous stage; the arithmetic
    of every element is the same as for one sub-chunk at a time. T^T = (I + M^T)^-1: 8-blocks by
    (I - B^T)(I + B^2T)(I + B^4T), then merges TT <- TT - TT Cm^T TT."""
    nb = len(subs)
    sts = []
    for b in range(nb):
        sts.append({})
    for b in range(nb):
        dc_s1(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_s2(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_s3(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_s4(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_s5(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_s6(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_s7(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_s8(sts[b], subs[b], dm, identf, H, T)
    for lv in range(4):
        for b in range(nb):
            dc_merge(sts[b], subs[b], dm, identf, H, T, lv)
        if lv < 3:
            for b in range(nb):
                dc_mergeT(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_s9(sts[b], subs[b], dm, identf, H, T)
    for b in range(nb):
        dc_dep(sts[b], subs[b], Sd, H, T, zero=(zero and b == 0))


# ----------------------------------------------------------------------------------------------
# embedding: x[:, :, i] = embed[toks[pos + i]]

def embed_body(toks, preg, emb):
    """x[:, :, i] = emb[toks[pos + i]] feature-major [P, KT, T] bf16 (SBUF)."""
    xs = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.sbuf)
    for s in range(NS):
        idx = nl.ndarray((TS, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=toks.ap(pattern=[[1, TS], [1, 1]], offset=s * TS, scalar_offset=preg, indirect_dim=0))
        rows = nl.ndarray((TS, D), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=emb.ap(pattern=[[D, TS], [1, D]], offset=0, vector_offset=idx, indirect_dim=0))
        for k in range(KT):
            tp = nl.ndarray((P, TS), dtype=BF16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=rows[:, k * P:(k + 1) * P])
            nisa.tensor_copy(dst=xs[:, k, s * TS:(s + 1) * TS], src=tp)
    return xs


@nki.jit
def embed(toks, pos, emb):
    """toks [MAXPOS + 128] int32 HBM; pos [2, 1, 1] int32 (per core); emb [VOCAB, D] bf16 -> x [P, KT, T] bf16."""
    pid = nl.program_id(0)
    preg = load_pos(pos, pid)
    xs = embed_body(toks, preg, emb)
    out = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=xs)
    return out


# ----------------------------------------------------------------------------------------------
# gated DeltaNet layer (+ MLP) over T tokens

def lin_body(xs, ln1, wqkv, wz, wab, convw, alog, dtb, normw, wout, ln2, wg, wu, wd, convst, recst, dnc,
             pid, ones_f, ones_b, ident, identf, cdg=None, sel16=None, selF=None):
    """x [P, KT, T] bf16. Per-core weights (leading dim 2 = core; core c has heads 8c..8c+7):
    ln1/ln2 [P, KT] bf16; wqkv [2, P, 24, 16, 128] (q, k, v tiles of its heads); wz [2, P, 8, 16, 128];
    wab [2, P, KT, 16] (b of its heads then a, (p, kt, j) = W[j, kt*128 + p]); convw [2, P, 24, 4] bf16;
    alog [2, P, 8] fp32 and dtb [2, P, 8] bf16 (rows replicated); normw [P, 1] fp32;
    wout [2, P, 8, 16, 128] (output tiles 8c..); wg, wu [2, P, 24, 16, 128]; wd [2, P, 8, 48, 128].
    State (updated in place, per core): convst [2, P, 24, 3] bf16; recst [2, P(v), 8, 128(k)] fp32.
    dnc [P, NMASK, P] fp32: constant masks of the chunked delta rule (T >= 128 only)."""
    w1 = load_norm_w1(ln1, KT, 'ln1')
    h = rmsnorm_fm(xs, w1, KT, T, D, ones_f, selF, identf, pid, defer=(T > 1 and RSD))

    if T > 1:
        # diag(w) stationaries for the conv on the TensorE (host re-layout): cdiag[p, qt*4 + j, m] = (p == m) * w[p, qt, j]
        cdiag = nl.ndarray((P, QL * 4, P), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=cdiag, src=cdg[pid], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    # projections; qkv lands after the 3 conv-state columns
    xx = nl.ndarray((P, QL, T + 3), dtype=BF16, buffer=nl.sbuf)
    # the conv state moves as one contiguous 144-byte row per partition (a strided DMA into xx would be 3072
    # descriptors of 6 bytes on the software DGE, and later weight DMAs wait behind it)
    if FIRST:
        nisa.memset(dst=xx[:, :, 0:3], value=0.0)                   # first chunk of a request: zero state
    else:
        cs = ld('cs', (P, QL, 3), BF16, convst[pid])
        nisa.tensor_copy(dst=xx[:, :, 0:3], src=cs)
    gemm_n(xx[:, :, 3:3 + T], wqkv, pid, h, QL, KT, T, 4)
    z = nl.ndarray((P, HL, T), dtype=BF16, buffer=nl.sbuf)
    if T == 1:
        gemm_fm(z, wz, pid, h, HL, KT, T, 4)
    wabs = ld('wabs', (P, KT, 2 * HL), BF16, wab[pid])
    ab = nl.ndarray((TS, NS, 2 * HL), dtype=BF16, buffer=nl.sbuf)       # token-major per sub-chunk
    hm = h['hx'] if (T > 1 and RSD) else h
    for s in range(NS):
        ab_ps = nl.ndarray((TS, 2 * HL), dtype=FP32, buffer=nl.psum)
        for k in range(KT):
            nisa.nc_matmul(dst=ab_ps, stationary=hm[:, k, s * TS:(s + 1) * TS], moving=wabs[:, k, :])
        if T > 1 and RSD:
            norm_gram(h)
            nisa.tensor_scalar(dst=ab[:, s, :], data=ab_ps, op0=nl.multiply, operand0=h['rst'][:, s:s + 1])
        else:
            nisa.tensor_copy(dst=ab[:, s, :], src=ab_ps)

    # new conv state = last 3 inputs
    csn = nl.ndarray((P, QL, 3), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=csn, src=xx[:, :, T:T + 3])
    nisa.dma_copy(dst=convst[pid], src=csn)

    # causal depthwise conv (kernel 4) in fp32, rounded to bf16, then SiLU (bf16)
    convb = nl.ndarray((P, QL, T), dtype=BF16, buffer=nl.sbuf)
    if T == 1:
        cw = ld('cw', (P, QL, 4), BF16, convw[pid])
        acc = nl.ndarray((P, QL, T), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=acc, data1=xx[:, :, 0:T], data2=cw[:, :, 0:1].broadcast(2, T), op=nl.multiply)
        for j in range(1, 4):
            tj = nl.ndarray((P, QL, T), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tj, data1=xx[:, :, j:j + T], data2=cw[:, :, j:j + 1].broadcast(2, T), op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tj, op=nl.add)
        nisa.tensor_copy(dst=convb, src=acc)
    qkv = nl.ndarray((P, QL, T), dtype=BF16, buffer=nl.sbuf)
    sq = nl.ndarray((P, 2 * HL, T), dtype=BF16, buffer=nl.sbuf)
    if T > 1:
        # conv on the TensorE: per channel tile, 4 accumulating matmuls with diag(w_j) stationary (exact products,
        # fp32 accumulation in tap order, as the elementwise form). Per tile: conv, evacuation, SiLU, square and the
        # token-major l2 sums, so the TensorE convolves tile qt + 1 while the other engines finish tile qt
        sst_ps = nl.ndarray((TS, NS, 2 * HL), dtype=FP32, buffer=nl.psum)
        cwp = ld('cw', (P, QL, 4), BF16, convw[pid])
        for qt in range(QL):
            cps = nl.ndarray((P, T), dtype=FP32, buffer=nl.psum)
            for j in range(CONVPE):
                nisa.nc_matmul(dst=cps, stationary=cdiag[:, qt * 4 + j, :], moving=xx[:, qt, j:j + T])
            if CONVPE == 4:
                nisa.tensor_copy(dst=convb[:, qt, :], src=cps, engine=nisa.engine.vector)
            else:
                # the last tap on the VectorE at the evacuation: x3 w3 (exact in fp32) + PSUM, the same fp32 add the
                # TensorE accumulation does, then bf16
                nisa.scalar_tensor_tensor(dst=convb[:, qt, :], data=xx[:, qt, 3:3 + T], op0=nl.multiply,
                                          operand0=cwp[:, qt, 3:4], op1=nl.add, operand1=cps)
            nisa.activation(dst=qkv[:, qt, :], op=nl.silu, data=convb[:, qt, :])
            if qt < 2 * HL:
                nisa.tensor_tensor(dst=sq[:, qt, :], data1=qkv[:, qt, :], data2=qkv[:, qt, :], op=nl.multiply)
                for s in range(NS):
                    nisa.nc_matmul(dst=sst_ps[:, s, qt:qt + 1], stationary=sq[:, qt, s * TS:(s + 1) * TS],
                                   moving=ones_b[:, 0:1])
    else:
        nisa.activation(dst=qkv, op=nl.silu, data=convb)
        nisa.tensor_tensor(dst=sq, data1=qkv[:, 0:2 * HL, :], data2=qkv[:, 0:2 * HL, :], op=nl.multiply)

    # l2norm of q and k per head, in bf16 as torch does it: x * rsqrt(sum(x*x) + eps)
    qk = nl.ndarray((P, 2 * HL, T), dtype=BF16, buffer=nl.sbuf)
    if T > 1:
        # token-major sums (tokens on partitions): the elementwise chain runs on 2HL*NS values per partition
        ssbt = nl.ndarray((TS, NS, 2 * HL), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ssbt, src=sst_ps)
        sset = nl.ndarray((TS, NS, 2 * HL), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sset, data=ssbt, op0=nl.add, operand0=EPS)
        ssft = nl.ndarray((TS, NS, 2 * HL), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ssft, src=sset)
        invft = nl.ndarray((TS, NS, 2 * HL), dtype=FP32, buffer=nl.sbuf)
        rsqrt_accurate(invft, ssft)
        invbt = nl.ndarray((TS, NS, 2 * HL), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=invbt, src=invft)
        # back to feature-major: transpose to [2HL, T], then a selector matmul broadcasts row c to all partitions
        ivT = nl.ndarray((2 * HL, T), dtype=BF16, buffer=nl.sbuf)
        for s in range(NS):
            tp = nl.ndarray((2 * HL, TS), dtype=BF16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=invbt[:, s, :])
            nisa.tensor_copy(dst=ivT[:, s * TS:(s + 1) * TS], src=tp)
        for c in range(2 * HL):
            bc = nl.ndarray((P, T), dtype=FP32, buffer=nl.psum)
            nisa.nc_matmul(dst=bc, stationary=sel16[:, c, :], moving=ivT)
            nisa.tensor_tensor(dst=qk[:, c, :], data1=qkv[:, c, :], data2=bc, op=nl.multiply)
    ssb = nl.ndarray((P, 2 * HL, T), dtype=BF16, buffer=nl.sbuf)
    if T > 1:
        pass
    elif T == 1:
        ps1 = nl.ndarray((P, 2 * HL), dtype=FP32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps1, stationary=ones_b, moving=sq[:, :, 0])
        nisa.tensor_copy(dst=ssb[:, :, 0], src=ps1)
    if T == 1:
        sse = nl.ndarray((P, 2 * HL, T), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sse, data=ssb, op0=nl.add, operand0=EPS)
        ssf = nl.ndarray((P, 2 * HL, T), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ssf, src=sse)
        invf = nl.ndarray((P, 2 * HL, T), dtype=FP32, buffer=nl.sbuf)
        rsqrt_accurate(invf, ssf)
        invb = nl.ndarray((P, 2 * HL, T), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=invb, src=invf)
        nisa.tensor_tensor(dst=qk, data1=qkv[:, 0:2 * HL, :], data2=invb, op=nl.multiply)

    # token-major q and k per sub-chunk: [TS, NS, 8, 128] (prefill only)
    q_tm = nl.ndarray((TS, NS, HL, P), dtype=BF16, buffer=nl.sbuf)
    k_tm = nl.ndarray((TS, NS, HL, P), dtype=BF16, buffer=nl.sbuf)
    for s in range(NS):
        for hh in range(HL):
            tq = nl.ndarray((TS, P), dtype=BF16, buffer=nl.psum)
            nisa.nc_transpose(dst=tq, data=qk[:, hh, s * TS:(s + 1) * TS])
            nisa.tensor_copy(dst=q_tm[:, s, hh, :], src=tq)
            tk = nl.ndarray((TS, P), dtype=BF16, buffer=nl.psum)
            nisa.nc_transpose(dst=tk, data=qk[:, HL + hh, s * TS:(s + 1) * TS])
            nisa.tensor_copy(dst=k_tm[:, s, hh, :], src=tk)

    # beta = sigmoid(b) (bf16); g = -exp(A_log) * softplus(a + dt_bias) (fp32); eg = exp(g)
    bg = nl.ndarray((TS, NS, 2 * HL), dtype=FP32, buffer=nl.sbuf)       # [eg | beta]
    betab = nl.ndarray((TS, NS, HL), dtype=BF16, buffer=nl.sbuf)
    nisa.activation(dst=betab, op=nl.sigmoid, data=ab[:, :, 0:HL])
    nisa.tensor_copy(dst=bg[:, :, HL:2 * HL], src=betab)
    dtt = ld('dtt', (TS, HL), BF16, dtb[pid, 0:TS, :])
    alt = ld('alt', (TS, HL), FP32, alog[pid, 0:TS, :])
    negA = nl.ndarray((TS, HL), dtype=FP32, buffer=nl.sbuf)
    nisa.activation(dst=negA, op=nl.exp, data=alt)
    asum = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=asum, data1=ab[:, :, HL:2 * HL], data2=dtt.expand_dim(1).broadcast(1, NS), op=nl.add)
    sp = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
    nisa.activation(dst=sp, op=nl.softplus, data=asum)
    gg = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gg, data1=negA.expand_dim(1).broadcast(1, NS), data2=sp, op=nl.multiply)
    nisa.activation(dst=bg[:, :, 0:HL], op=nl.exp, data=gg, scale=-1.0)

    # gated delta rule. State S^T[v, h, k] (v on partitions).
    if FIRST and T >= 128:
        S = nl.ndarray((P, HL, P), dtype=FP32, buffer=nl.sbuf)   # written from Sd after the chunked rule
    elif FIRST:
        S = nl.ndarray((P, HL, P), dtype=FP32, buffer=nl.sbuf)
        nisa.memset(dst=S, value=0.0)
    else:
        S = ld('S', (P, HL, P), FP32, recst[pid])
    o_all = nl.ndarray((P, HL, T), dtype=FP32, buffer=nl.sbuf)
    if T >= 128:
        # chunked form per 128-token sub-chunk (mathematically the recurrence below; fp32 rounding differs)
        dms = nl.ndarray((P, NMASK, P), dtype=FP32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dms, src=dnc)
        gneg = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gneg, data=gg, op0=nl.multiply, operand0=-1.0)
        if FIRST:
            # S = 0: the first sub-chunk's state-dependent part runs in its zero-state form and writes Sd
            Sd = nl.ndarray((P, HL, P), dtype=FP32, buffer=nl.sbuf)
        else:
            Sd = bsb(btr(S, HL), HL)                               # S [dk, h, dv] for the whole call (exact transposes)
        sel3 = sel3_tile()
        for s0 in range(0, NS, DPAIR):
            subs = []
            for s in range(s0, min(NS, s0 + DPAIR)):
                subs.append({'qk': qk[:, :, s * TS:(s + 1) * TS], 'vfm': qkv[:, 2 * HL:3 * HL, s * TS:(s + 1) * TS],
                             'k_tm': k_tm[:, s, :, :], 'bg': bg[:, s, :], 'gneg': gneg[:, s, :],
                             'o': o_all[:, :, s * TS:(s + 1) * TS], 'sel3': sel3})
            delta_pair(subs, Sd, dms, identf, HL, TS, zero=(FIRST and s0 == 0))
        nisa.tensor_copy(dst=S, src=btr(Sd, HL))
        gemm_n(z, wz, pid, h, HL, KT, T, 4)
    else:
        for t in range(T):
            sel = ident[0:T, t:t + 1].broadcast(1, P)               # [T, 128]: broadcasts row t
            kb = nl.ndarray((P, HL * P), dtype=FP32, buffer=nl.psum)
            for c in range(HL * P // 512):
                nisa.nc_matmul(dst=kb[:, c * 512:(c + 1) * 512], stationary=sel, moving=k_tm[:, 0, 4 * c:4 * c + 4, :])
            bgp = nl.ndarray((P, 2 * HL), dtype=FP32, buffer=nl.psum)
            nisa.nc_matmul(dst=bgp, stationary=identf[0:T, t:t + 1].broadcast(1, P), moving=bg[:, 0, :])
            bgs = nl.ndarray((P, 2 * HL), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bgs, src=bgp)
            # S = S * exp(g)
            nisa.tensor_tensor(dst=S, data1=S, data2=bgs[:, 0:HL].expand_dim(2).broadcast(2, P), op=nl.multiply)
            # kv_mem = sum_k S * k
            t1 = nl.ndarray((P, HL, P), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t1, data1=S, data2=kb.reshape((P, HL, P)), op=nl.multiply)
            kvm = nl.ndarray((P, HL), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=kvm, op=nl.add, data=t1, axis=2)
            # delta = (v - kv_mem) * beta
            dl = nl.ndarray((P, HL), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dl, data1=qkv[:, 2 * HL:3 * HL, t], data2=kvm, op=nl.subtract)
            nisa.tensor_tensor(dst=dl, data1=dl, data2=bgs[:, HL:2 * HL], op=nl.multiply)
            # S = S + k * delta
            t2 = nl.ndarray((P, HL, P), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t2, data1=kb.reshape((P, HL, P)), data2=dl.expand_dim(2).broadcast(2, P), op=nl.multiply)
            nisa.tensor_tensor(dst=S, data1=S, data2=t2, op=nl.add)
            # o = sum_k S * q
            qb = nl.ndarray((P, HL * P), dtype=FP32, buffer=nl.psum)
            for c in range(HL * P // 512):
                nisa.nc_matmul(dst=qb[:, c * 512:(c + 1) * 512], stationary=sel, moving=q_tm[:, 0, 4 * c:4 * c + 4, :])
            t3 = nl.ndarray((P, HL, P), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t3, data1=S, data2=qb.reshape((P, HL, P)), op=nl.multiply)
            nisa.tensor_reduce(dst=o_all[:, :, t], op=nl.add, data=t3, axis=2)
    nisa.dma_copy(dst=recst[pid], src=S)

    # core_attn_out = bf16(o * scale)
    ob = nl.ndarray((P, HL, T), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ob, data=o_all, op0=nl.multiply, operand0=1.0 / 11.313708498984761)

    # gated RMSNorm over v (partitions) per head: bf16(w * bf16(o * rsqrt(mean(o^2)+eps)) * silu(z))
    if T == 1:
        o2 = nl.ndarray((P, HL, T), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o2, data1=ob, data2=ob, op=nl.multiply)
    var = nl.ndarray((P, HL, T), dtype=FP32, buffer=nl.sbuf)
    nb = nl.ndarray((P, HL, T), dtype=BF16, buffer=nl.sbuf)
    if T > 1:
        # token-major sums of squares as the diagonal of each head's bf16 Gram matrix ob^T ob per sub-chunk (exact
        # products, fp32 accumulation), picked by one select_reduce each; rsqrt on NS*HL values per partition
        idu = nl.ndarray((TS, TS), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idu, src=identf)
        sst = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
        for s in range(NS):
            gp = nl.ndarray((TS, HL, TS), dtype=FP32, buffer=nl.psum)
            for hh in range(HL):
                nisa.nc_matmul(dst=gp[:, hh, :], stationary=ob[:, hh, s * TS:(s + 1) * TS], moving=ob[:, hh, s * TS:(s + 1) * TS])
            for hh in range(HL):
                dgs = nl.ndarray((TS, TS), dtype=FP32, buffer=nl.sbuf)
                nisa.select_reduce(dst=dgs, predicate=idu, on_true=gp[:, hh, :], on_false=0.0,
                                   reduce_res=sst[:, s, hh:hh + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce,
                                   reduce_op=nl.maximum)
        vt = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=vt, data=sst, op0=nl.multiply, operand0=1.0 / P, op1=nl.add, operand1=EPS)
        rst = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
        rsqrt_accurate(rst, vt)
        # exact three-way bf16 split r = hi + mid + lo (8 + 8 + 8 significand bits), so the broadcast below is a bf16
        # matmul that reproduces r exactly in the fp32 PSUM (1*hi + 1*mid + 1*lo)
        r3 = nl.ndarray((TS, NS, 3 * HL), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=r3[:, :, 0:HL], src=rst)
        d1 = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d1, data1=rst, data2=r3[:, :, 0:HL], op=nl.subtract)
        nisa.tensor_copy(dst=r3[:, :, HL:2 * HL], src=d1)
        d2 = nl.ndarray((TS, NS, HL), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d2, data1=d1, data2=r3[:, :, HL:2 * HL], op=nl.subtract)
        nisa.tensor_copy(dst=r3[:, :, 2 * HL:3 * HL], src=d2)
        rT = nl.ndarray((3 * HL, T), dtype=BF16, buffer=nl.sbuf)
        for s in range(NS):
            tp = nl.ndarray((3 * HL, TS), dtype=BF16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=r3[:, s, :])
            nisa.tensor_copy(dst=rT[:, s * TS:(s + 1) * TS], src=tp)
        # selector [3HL, HL, P]: row j*HL + c selects head c (three stacked identities)
        sel3 = sel3_tile()
        # per group of YG heads: normalize, weight, gate and send the group to the other core at once, so the
        # out-projection (k = 0 .. 15 in order on both cores) starts on the first received / own tiles
        nw = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
        nisa.dma_copy(dst=nw, src=normw)
        sz = nl.ndarray((P, HL, T), dtype=FP32, buffer=nl.sbuf)
        nisa.activation(dst=sz, op=nl.silu, data=z)
        y = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.sbuf)
        for h0 in range(0, HL, YG):
            for hh in range(h0, h0 + YG):
                bc = nl.ndarray((P, T), dtype=FP32, buffer=nl.psum)
                nisa.nc_matmul(dst=bc, stationary=sel3[:, hh, :], moving=rT)
                nisa.tensor_tensor(dst=nb[:, hh, :], data1=ob[:, hh, :], data2=bc, op=nl.multiply)
            nwn = nl.ndarray((P, YG, T), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nwn, data=nb[:, h0:h0 + YG, :], op0=nl.multiply, operand0=nw)
            nisa.tensor_tensor(dst=y[:, pid * HL + h0:pid * HL + h0 + YG, :], data1=nwn, data2=sz[:, h0:h0 + YG, :],
                               op=nl.multiply)
            exchange_part(y, HL, h0, YG, pid)
    elif T == 1:
        ps1 = nl.ndarray((P, HL), dtype=FP32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps1, stationary=ones_f, moving=o2[:, :, 0])
        nisa.tensor_scalar(dst=var[:, :, 0], data=ps1, op0=nl.multiply, operand0=1.0 / P, op1=nl.add, operand1=EPS)
    if T == 1:
        rs = nl.ndarray((P, HL, T), dtype=FP32, buffer=nl.sbuf)
        rsqrt_accurate(rs, var)
        nisa.tensor_tensor(dst=nb, data1=ob, data2=rs, op=nl.multiply)
    if T == 1:
        nw = ld('nw', (P, 1), FP32, normw)
        nwn = nl.ndarray((P, HL, T), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nwn, data=nb, op0=nl.multiply, operand0=nw)
        sz = nl.ndarray((P, HL, T), dtype=FP32, buffer=nl.sbuf)
        nisa.activation(dst=sz, op=nl.silu, data=z)
        y = xtile(KT, T)
        nisa.tensor_tensor(dst=y[:, pid * HL:pid * HL + HL, :], data1=nwn, data2=sz, op=nl.multiply)
        exchange(y, HL, pid, T)

    att = xtile(KT, T) if T == 1 else nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.sbuf)
    x1 = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.sbuf)
    if T == 1:
        gemm_fm(att[:, pid * 8:pid * 8 + 8, :], wout, pid, y, 8, KT, T, 4)
        exchange(att, 8, pid, T)
        nisa.tensor_tensor(dst=x1, data1=xs, data2=att, op=nl.add)
    else:
        gemm_fm(att[:, pid * 8:pid * 8 + 8, :], wout, pid, y, 8, KT, T, OPG, xfull=att, resid=(xs, x1), kown=HL)
    x2 = mlp_block(x1, ln2, wg, wu, wd, pid, T, ones_f, selF, identf)
    return x2


@nki.jit
def lin_layer(x, ln1, wqkv, wz, wab, convw, alog, dtb, normw, wout, ln2, wg, wu, wd, convst, recst, dnc):
    """One DeltaNet layer + MLP; x [P, KT, T] bf16 in HBM (see lin_body)."""
    pid = nl.program_id(0)
    ones_f = ones_tile(FP32)
    ones_b = ones_tile(BF16)
    ident = ident_tile()
    identf = ident_tile(FP32)
    xs = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    x2 = lin_body(xs, ln1, wqkv, wz, wab, convw, alog, dtb, normw, wout, ln2, wg, wu, wd, convst, recst, dnc,
                  pid, ones_f, ones_b, ident, identf)
    out = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=x2)
    return out


# ----------------------------------------------------------------------------------------------
# full-attention layer (+ MLP) over T tokens at positions pos .. pos+T-1, keys 0 .. L-1 (masked)

EARLYKV = True         # TIGHT prefill chunks: the cached keys / values (< pos) are loaded early, this chunk's own
                       # K^T / V are written straight into the attention tiles (no HBM write -> read round trip)


def kv_early(kc, vc, pid):
    """TIGHT chunk (L == pos + T): kT [P, 2, L], vv [P, L / P, 256] SBUF tiles; the cached part (keys < L - T) is
    loaded now (DMA), the chunk's own part is filled on-chip by the caller (kT[:, :, L - T:], vv[:, (L - T) / P:, :])."""
    NB = L // P
    NBO = (L - T) // P
    kT = nl.ndarray((P, 2, L), dtype=BF16, buffer=nl.sbuf)
    vv = nl.ndarray((P, NB, 256), dtype=BF16, buffer=nl.sbuf)
    if NBO > 0:
        for dt in range(2):
            nisa.dma_copy(dst=kT[:, dt, 0:L - T], src=kc[pid, dt, :, 0:L - T], dge_mode=nisa.dge_mode.hwdge,
                          engine=nisa.engine.sync)
        nisa.dma_copy(dst=vv[:, 0:NBO, :], src=vc[pid, 0:L - T, :].reshape((NBO, P, 256)).permute((1, 0, 2)),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    return kT, vv, NBO


def full_body(xs, preg, pf, ln1, wq, wk, wv, qnw, knw, cos, sin, rot, wo, ln2, wg, wu, wd, kc, vc, pid, ones_f,
              selF=None, identf=None):
    """xs [P, KT, T] bf16 SBUF; preg: position register; pf [P, 1] fp32 position (SBUF, every partition);
    core c: KV head c, query heads 4c..4c+3. wq [2, P, 16, 16, 128] (per local head: q, q, gate, gate);
    wk, wv [2, P, 2, 16, 128]; qnw, knw [P, 2] bf16; cos/sin [64, MAXPOS] bf16 (feature-major tables);
    rot [P, P] bf16 (rotate_half as a signed permutation, rot[p, m]); wo [2, P, 8, 16, 128].
    KV cache per core (its KV head), updated in place: kc [2, 2 (dt), P, MAXPOS] bf16 (K^T), vc [2, MAXPOS, 256] bf16."""
    w1 = load_norm_w1(ln1, KT, 'ln1')
    h = rmsnorm_fm(xs, w1, KT, T, D, ones_f, selF, identf, pid, defer=(T > 1 and RSD))

    QH = NQH // 2
    q = nl.ndarray((P, 4 * QH, T), dtype=BF16, buffer=nl.sbuf)      # local head i: tiles 4i, 4i+1 query; +2, +3 gate
    gemm_n(q, wq, pid, h, 4 * QH, KT, T, 4)
    k = nl.ndarray((P, 2, T), dtype=BF16, buffer=nl.sbuf)
    gemm_n(k, wk, pid, h, 2, KT, T, 4)
    v = nl.ndarray((P, 2, T), dtype=BF16, buffer=nl.sbuf)
    gemm_n(v, wv, pid, h, 2, KT, T, 4)
    early = EARLYKV and TIGHT and T > 1
    if early:
        kT, vv, NBO = kv_early(kc, vc, pid)

    qn1 = load_norm_w1(qnw, 2, 'qn')
    kn1 = load_norm_w1(knw, 2, 'kn')
    cs = nl.ndarray((64, T), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cos.ap(pattern=[[MAXPOS, 64], [1, T]], offset=0, scalar_offset=preg, indirect_dim=1))
    sn = nl.ndarray((64, T), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn, src=sin.ap(pattern=[[MAXPOS, 64], [1, T]], offset=0, scalar_offset=preg, indirect_dim=1))
    rots = ld('rots', (P, P), BF16, rot)

    # per-head RMSNorm over 256 dims (fp32, rounded once) followed by partial RoPE on dims 0..63
    qr = nl.ndarray((P, 2 * QH, T), dtype=BF16, buffer=nl.sbuf)     # local query heads: tiles 2i, 2i+1
    kr = kT[:, :, L - T:L] if early else nl.ndarray((P, 2, T), dtype=BF16, buffer=nl.sbuf)
    for hh in range(QH + 1):
        if hh < QH:
            src0 = q[:, 4 * hh:4 * hh + 2, :]
            wn = qn1
            dst = qr[:, 2 * hh:2 * hh + 2, :]
        else:
            src0 = k[:, 0:2, :]
            wn = kn1
            dst = kT[:, 0:2, L - T:L] if early else kr[:, 0:2, :]
        sq = nl.ndarray((P, 2, T), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=src0, data2=src0, op=nl.multiply)
        var = nl.ndarray((P, T), dtype=FP32, buffer=nl.sbuf)
        if T == 1:
            ps2 = nl.ndarray((P, 2), dtype=FP32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps2, stationary=ones_f, moving=sq[:, :, 0])
            s01 = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=s01, op=nl.add, data=ps2, axis=1)
            nisa.tensor_scalar(dst=var, data=s01, op0=nl.multiply, operand0=1.0 / 256, op1=nl.add, operand1=EPS)
        else:
            ps = nl.ndarray((P, T), dtype=FP32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=ones_f, moving=sq[:, 0, :])
            nisa.nc_matmul(dst=ps, stationary=ones_f, moving=sq[:, 1, :])
            nisa.tensor_scalar(dst=var, data=ps, op0=nl.multiply, operand0=1.0 / 256, op1=nl.add, operand1=EPS)
        rstd = nl.ndarray((P, T), dtype=FP32, buffer=nl.sbuf)
        rsqrt_accurate(rstd, var)
        xn = nl.ndarray((P, 2, T), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xn, data1=src0, data2=rstd.expand_dim(1).broadcast(1, 2), op=nl.multiply)
        nisa.tensor_tensor(dst=dst, data1=xn, data2=wn.expand_dim(2).broadcast(2, T), op=nl.multiply)
        # RoPE: d < 64 of tile 0: bf16(bf16(x*cos) + bf16(rotate_half(x)*sin))
        rp = nl.ndarray((P, T), dtype=FP32, buffer=nl.psum)
        nisa.nc_matmul(dst=rp, stationary=rots, moving=dst[:, 0, :])
        a1 = nl.ndarray((64, T), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a1, data1=dst[0:64, 0, :], data2=cs, op=nl.multiply)
        a2 = nl.ndarray((64, T), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a2, data1=rp[0:64, :], data2=sn, op=nl.multiply)
        nisa.tensor_tensor(dst=dst[0:64, 0, :], data1=a1, data2=a2, op=nl.add)

    # write K^T and V of this chunk into the cache (own copy = own KV head)
    for c in range(2):
        nisa.dma_copy(dst=kc[pid, c].ap(pattern=[[MAXPOS, P], [1, T]], offset=0, scalar_offset=preg, indirect_dim=1),
                      src=kT[:, c, L - T:L] if early else kr[:, c, :])
    for s in range(NS):
        v_tm = vv[:, NBO + s, :] if early else nl.ndarray((TS, 2 * P), dtype=BF16, buffer=nl.sbuf)
        for c in range(2):
            tv = nl.ndarray((TS, P), dtype=BF16, buffer=nl.psum)
            nisa.nc_transpose(dst=tv, data=v[:, c, s * TS:(s + 1) * TS])
            nisa.tensor_copy(dst=v_tm[:, c * P:(c + 1) * P], src=tv)
        nisa.dma_copy(dst=vc[pid].ap(pattern=[[256, TS], [1, 256]], offset=s * TS * 256, scalar_offset=preg, indirect_dim=0),
                      src=v_tm)

    NB = (L + P - 1) // P
    ao = xtile(2 * NQH, T) if T == 1 else nl.ndarray((P, 2 * NQH, T), dtype=BF16, buffer=nl.sbuf)
    if not early:
        kT = nl.ndarray((P, 2, L), dtype=BF16, buffer=nl.sbuf)
        for dt in range(2):
            nisa.dma_copy(dst=kT[:, dt, :], src=kc[pid, dt, :, 0:L], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        vv = nl.ndarray((P, NB, 256), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=vv, src=vc[pid, 0:L, :].reshape((NB, P, 256)).permute((1, 0, 2)), dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync)
    if T == 1:
        # one token: keys on the partitions (key = b*128 + p), the QH query heads of this KV head side by side
        qd = nl.ndarray((P, 2, QH), dtype=BF16, buffer=nl.sbuf)
        for hq in range(QH):
            for dt in range(2):
                nisa.tensor_copy(dst=qd[:, dt, hq:hq + 1], src=qr[:, 2 * hq + dt, :])
        kidx = nl.ndarray((P, NB), dtype=FP32, buffer=nl.sbuf)
        nisa.iota(dst=kidx, pattern=[[P, NB]], offset=0, channel_multiplier=1)
        kmask = nl.ndarray((P, NB), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=kmask, data=kidx, op0=nl.greater, operand0=pf, op1=nl.multiply, operand1=NEG)
        sp = nl.ndarray((P, NB, QH), dtype=FP32, buffer=nl.psum)
        for bb in range(NB):
            for dt in range(2):
                nisa.nc_matmul(dst=sp[:, bb, :], stationary=kT[:, dt, bb * P:(bb + 1) * P], moving=qd[:, dt, :])
        sc = nl.ndarray((P, NB, QH), dtype=FP32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=sc, data=sp, op0=nl.multiply, operand0=0.0625, op1=nl.add,
                                  operand1=kmask.expand_dim(2).broadcast(2, QH))
        # max over all keys per head: over the blocks, then across partitions; broadcast back to all partitions
        m1 = nl.ndarray((P, QH), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m1, op=nl.max, data=sc.permute((0, 2, 1)), axis=2)
        m1t = nl.ndarray((QH, P), dtype=FP32, buffer=nl.psum)
        nisa.nc_transpose(dst=m1t, data=m1)
        m2 = nl.ndarray((QH, 1), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m2, op=nl.max, data=m1t, axis=1, negate=True)
        dg = nl.ndarray((QH, QH), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dg, data=identf_small(QH), op0=nl.multiply, operand0=m2)
        nm_ps = nl.ndarray((P, QH), dtype=FP32, buffer=nl.psum)
        nisa.nc_matmul(dst=nm_ps, stationary=ones_f[0:QH, :], moving=dg)
        nm = nl.ndarray((P, QH), dtype=FP32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nm, src=nm_ps)
        pe = nl.ndarray((P, NB, QH), dtype=FP32, buffer=nl.sbuf)
        psums = nl.ndarray((P, QH), dtype=FP32, buffer=nl.sbuf)
        for hq in range(QH):
            nisa.activation(dst=pe[:, :, hq], op=nl.exp, data=sc[:, :, hq], bias=nm[:, hq:hq + 1], scale=1.0,
                            reduce_op=nl.add, reduce_res=psums[:, hq:hq + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        tot = nl.ndarray((P, QH), dtype=FP32, buffer=nl.psum)
        nisa.nc_matmul(dst=tot, stationary=ones_f, moving=psums)
        rsum = nl.ndarray((P, QH), dtype=FP32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rsum, data=tot)
        pb = nl.ndarray((P, NB, QH), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pb, data1=pe, data2=rsum.expand_dim(1).broadcast(1, NB), op=nl.multiply)
        for dt in range(2):
            po = nl.ndarray((P, QH), dtype=FP32, buffer=nl.psum)
            for bb in range(NB):
                nisa.nc_matmul(dst=po, stationary=vv[:, bb, dt * P:(dt + 1) * P], moving=pb[:, bb, :])
            ab16 = nl.ndarray((P, QH), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ab16, src=po)
            for hq in range(QH):
                sg = nl.ndarray((P, 1), dtype=BF16, buffer=nl.sbuf)
                nisa.activation(dst=sg, op=nl.sigmoid, data=q[:, 4 * hq + 2 + dt, :])
                nisa.tensor_tensor(dst=ao[:, pid * 2 * QH + 2 * hq + dt, :], data1=ab16[:, hq:hq + 1], data2=sg, op=nl.multiply)
    else:
        sga = nl.ndarray((P, QH, 2, T), dtype=BF16, buffer=nl.sbuf)       # sigmoid(gate), all heads at once
        for hq in range(QH):
            nisa.activation(dst=sga[:, hq, :, :], op=nl.sigmoid, data=q[:, 4 * hq + 2:4 * hq + 4, :])
        # per local head hq and 128-token sub-chunk s (16 items, head-major), software-pipelined: the scores of item i + 1 are
        # emitted before the softmax / transposes / PV of item i, so the TensorE computes them while the VectorE /
        # ScalarE run the softmax of item i.
        # TIGHT (bucket L == pos + T, guaranteed by the host for these variants): sub-chunk s needs keys
        # < L - (NS - 1 - s) * TS only (the keys left out are fully masked: exact zeros), and of those only the last
        # TS can be masked, by the causal triangle (key j' > query t): one constant [TS, TS] mask; the other scores
        # are q.k / 16 (+ 0 in the general form). Otherwise an additive mask [TS, L], NEG where key j > pos + s*TS + t.
        if TIGHT and ATT_T:
            att_tight_t(qr, kT, vv, ao, sga, qn1, kn1, pid, ones_f)
        elif TIGHT:
            jt = nl.ndarray((TS, TS), dtype=FP32, buffer=nl.sbuf)
            nisa.iota(dst=jt, pattern=[[1, TS]], offset=0, channel_multiplier=-1)          # j' - t
            tri = nl.ndarray((TS, TS), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tri, data=jt, op0=nl.greater, operand0=0.0, op1=nl.multiply, operand1=NEG)
        else:
            jt = nl.ndarray((TS, L), dtype=FP32, buffer=nl.sbuf)
            nisa.iota(dst=jt, pattern=[[1, L]], offset=0, channel_multiplier=-1)          # j - t
        if not (TIGHT and ATT_T):
            NI = NS * QH
            masks = []
            scs = []
            for i in range(NI + 1):
                if i < NI:
                    hq = i // NS
                    s = i % NS
                    Ls = L - (NS - 1 - s) * TS if TIGHT else L
                    if TIGHT:
                        scs.append(att_scores_tight(qr, kT, tri, hq, s, Ls))
                    else:
                        if hq == 0:
                            pfs = nl.ndarray((TS, 1), dtype=FP32, buffer=nl.sbuf)
                            nisa.tensor_scalar(dst=pfs, data=pf[0:TS, :], op0=nl.add, operand0=float(s * TS))
                            mask = nl.ndarray((TS, Ls), dtype=FP32, buffer=nl.sbuf)
                            nisa.tensor_scalar(dst=mask, data=jt[:, 0:Ls], op0=nl.greater, operand0=pfs, op1=nl.multiply,
                                               operand1=NEG)
                            masks.append(mask)
                        scs.append(att_scores(qr, kT, masks[s], hq, s, Ls))
                if i >= 1:
                    j = i - 1
                    hq = j // NS
                    s = j % NS
                    Ls = L - (NS - 1 - s) * TS if TIGHT else L
                    NBs = Ls // P
                    sc = scs[j]
                    mx = nl.ndarray((TS, 1), dtype=FP32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=mx, op=nl.max, data=sc, axis=1, negate=True)
                    pe = nl.ndarray((TS, Ls), dtype=FP32, buffer=nl.sbuf)
                    ssum = nl.ndarray((TS, 1), dtype=FP32, buffer=nl.sbuf)
                    nisa.activation(dst=pe, op=nl.exp, data=sc, bias=mx, scale=1.0, reduce_op=nl.add, reduce_res=ssum,
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                    rsum = nl.ndarray((TS, 1), dtype=FP32, buffer=nl.sbuf)
                    nisa.reciprocal(dst=rsum, data=ssum)
                    pb = nl.ndarray((TS, Ls), dtype=BF16, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=pb, data=pe, op0=nl.multiply, operand0=rsum)
                    pT = nl.ndarray((P, NBs, TS), dtype=BF16, buffer=nl.sbuf)
                    for b in range(NBs):
                        tp = nl.ndarray((P, TS), dtype=BF16, buffer=nl.psum)
                        nisa.nc_transpose(dst=tp, data=pb[:, b * P:(b + 1) * P])
                        nisa.tensor_copy(dst=pT[:, b, :], src=tp)
                    for dt in range(2):
                        po = nl.ndarray((P, TS), dtype=FP32, buffer=nl.psum)
                        for b in range(NBs):
                            nisa.nc_matmul(dst=po, stationary=vv[:, b, dt * P:(dt + 1) * P], moving=pT[:, b, :])
                        ab16 = nl.ndarray((P, TS), dtype=BF16, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=ab16, src=po)
                        nisa.tensor_tensor(dst=ao[:, pid * 2 * QH + 2 * hq + dt, s * TS:(s + 1) * TS], data1=ab16,
                                           data2=sga[:, hq, dt, s * TS:(s + 1) * TS], op=nl.multiply)
                    if s == NS - 1:
                        # head hq done for all sub-chunks: its two tiles go to the other core at once
                        exchange_part(ao, 2 * QH, 2 * hq, 2, pid)
    if T == 1:
        exchange(ao, 2 * QH, pid, T)

    att = xtile(KT, T) if T == 1 else nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.sbuf)
    x1 = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.sbuf)
    if T == 1:
        gemm_fm(att[:, pid * 8:pid * 8 + 8, :], wo, pid, ao, 8, KT, T, 4)
        exchange(att, 8, pid, T)
        nisa.tensor_tensor(dst=x1, data1=xs, data2=att, op=nl.add)
    else:
        gemm_fm(att[:, pid * 8:pid * 8 + 8, :], wo, pid, ao, 8, KT, T, OPG, xfull=att, resid=(xs, x1), kown=2 * QH)
    x2 = mlp_block(x1, ln2, wg, wu, wd, pid, T, ones_f, selF, identf)
    return x2


KVLAST = True          # TIGHT chunks without the head: the last layer only writes K / V (its output is never read)
LASTTOK = True         # fused-head chunks: the last layer's query / attention / out-proj / MLP for the last token only


def gemv_col(dst, w, pid, x, nt_total, kt_total, group, kown, bc=None):
    """one column of gemm_fm's T > 1 path: dst [P, nt_total, 1] = W_core x (x [P, KT, 1]), k-outer per slab of
    `group` output tiles in the order kord(k0, kt_total, kown), evacuated as bf16(bc * PSUM) (bc [P, 1]: the
    deferred rstd of that column) or as a plain copy; every value is computed exactly as in that column of the GEMM"""
    for n0 in range(0, nt_total, group):
        g = min(group, nt_total - n0)
        wt = nl.ndarray((P, g, kt_total, P), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w[pid, :, n0:n0 + g, :, :], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        ps = nl.ndarray((P, g, 1), dtype=FP32, buffer=nl.psum)
        for k0 in range(kt_total):
            k = kord(k0, kt_total, kown, pid)
            for i in range(g):
                nisa.nc_matmul(dst=ps[:, i, :], stationary=wt[:, i, k, :], moving=x[:, k, :])
        if bc is not None:
            nisa.tensor_tensor(dst=dst[:, n0:n0 + g, :], data1=ps, data2=bc.expand_dim(1).broadcast(1, g), op=nl.multiply)
        else:
            nisa.tensor_copy(dst=dst[:, n0:n0 + g, :], src=ps)


def norm_col(x, w1, pid, ones_f):
    """norm_ctx + norm_gram + norm_bcast of one token (x [P, KT, 1]): hx = bf16(x (1 + w)) and rstd broadcast [P, 1],
    the same operations per value as in the chunk's RMSNorm (Gram diagonal = the 1 x 1 Gram, k in kord order)"""
    hx = nl.ndarray((P, KT, 1), dtype=BF16, buffer=nl.sbuf)
    for k0 in range(KT):
        k = kord(k0, KT, KT // 2, pid)
        nisa.activation(dst=hx[:, k, :], op=nl.copy, data=x[:, k, :], scale=w1[:, k:k + 1])
    gr = nl.ndarray((1, 1), dtype=FP32, buffer=nl.psum)
    for k0 in range(KT):
        k = kord(k0, KT, KT // 2, pid)
        nisa.nc_matmul(dst=gr, stationary=x[:, k, :], moving=x[:, k, :])
    vart = nl.ndarray((1, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=vart, data=gr, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
    rst = nl.ndarray((1, 1), dtype=FP32, buffer=nl.sbuf)
    rsqrt_accurate(rst, vart)
    bp = nl.ndarray((P, 1), dtype=FP32, buffer=nl.psum)
    nisa.nc_matmul(dst=bp, stationary=ones_f[0:1, :], moving=rst)            # exact broadcast (1 * r)
    bc = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bc, src=bp)
    return hx, bc


def qknorm_col(src0, wn, dst, cs, sn, rots, ones_f, t):
    """one head of full_body's T > 1 q/k RMSNorm + RoPE loop on t columns (the same operations per value)"""
    sq = nl.ndarray((P, 2, t), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=src0, data2=src0, op=nl.multiply)
    ps = nl.ndarray((P, t), dtype=FP32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=ones_f, moving=sq[:, 0, :])
    nisa.nc_matmul(dst=ps, stationary=ones_f, moving=sq[:, 1, :])
    var = nl.ndarray((P, t), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=var, data=ps, op0=nl.multiply, operand0=1.0 / 256, op1=nl.add, operand1=EPS)
    rstd = nl.ndarray((P, t), dtype=FP32, buffer=nl.sbuf)
    rsqrt_accurate(rstd, var)
    xn = nl.ndarray((P, 2, t), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xn, data1=src0, data2=rstd.expand_dim(1).broadcast(1, 2), op=nl.multiply)
    nisa.tensor_tensor(dst=dst, data1=xn, data2=wn.expand_dim(2).broadcast(2, t), op=nl.multiply)
    rp = nl.ndarray((P, t), dtype=FP32, buffer=nl.psum)
    nisa.nc_matmul(dst=rp, stationary=rots, moving=dst[:, 0, :])
    a1 = nl.ndarray((64, t), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=a1, data1=dst[0:64, 0, :], data2=cs, op=nl.multiply)
    a2 = nl.ndarray((64, t), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=a2, data1=rp[0:64, :], data2=sn, op=nl.multiply)
    nisa.tensor_tensor(dst=dst[0:64, 0, :], data1=a1, data2=a2, op=nl.add)


def full_last(xs, preg, pf, ln1, wq, wk, wv, qnw, knw, cos, sin, rot, wo, ln2, wg, wu, wd, kc, vc, pid, ones_f,
              selF, identf, kv_only=False):
    """the last (full-attention) layer of a TIGHT chunk whose output only feeds the LM head: K / V (and the cache
    writes) for all T tokens as in full_body; query, attention, out-projection and MLP for the last token only, each
    value computed as in that token's column of full_body (gemv_col, norm_col, qknorm_col, one query column of
    att_tight_t). Returns x2 of the last token [P, KT, 1]. kv_only: only the K / V cache writes (a chunk that is not
    the last of its prompt: nothing reads the layer's output), returns None."""
    QH = NQH // 2
    w1 = load_norm_w1(ln1, KT, 'ln1')
    h = rmsnorm_fm(xs, w1, KT, T, D, ones_f, selF, identf, pid, defer=True)
    k = nl.ndarray((P, 2, T), dtype=BF16, buffer=nl.sbuf)
    gemm_n(k, wk, pid, h, 2, KT, T, 4)
    v = nl.ndarray((P, 2, T), dtype=BF16, buffer=nl.sbuf)
    gemm_n(v, wv, pid, h, 2, KT, T, 4)
    early = EARLYKV and not kv_only
    if early:
        kT, vv, NBO = kv_early(kc, vc, pid)
    if not kv_only:
        q = nl.ndarray((P, 4 * QH, 1), dtype=BF16, buffer=nl.sbuf)
        gemv_col(q, wq, pid, h['hx'][:, :, T - 1:T], 4 * QH, KT, 4, KT // 2, bc=h['bcs'][:, T - 1:T])

    qn1 = load_norm_w1(qnw, 2, 'qn')
    kn1 = load_norm_w1(knw, 2, 'kn')
    cs = nl.ndarray((64, T), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cos.ap(pattern=[[MAXPOS, 64], [1, T]], offset=0, scalar_offset=preg, indirect_dim=1))
    sn = nl.ndarray((64, T), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn, src=sin.ap(pattern=[[MAXPOS, 64], [1, T]], offset=0, scalar_offset=preg, indirect_dim=1))
    rots = ld('rots', (P, P), BF16, rot)
    kr = kT[:, :, L - T:L] if early else nl.ndarray((P, 2, T), dtype=BF16, buffer=nl.sbuf)
    qknorm_col(k[:, 0:2, :], kn1, kr, cs, sn, rots, ones_f, T)
    qr = nl.ndarray((P, 2 * QH, 1), dtype=BF16, buffer=nl.sbuf)
    for hq in range(QH if not kv_only else 0):
        qknorm_col(q[:, 4 * hq:4 * hq + 2, :], qn1, qr[:, 2 * hq:2 * hq + 2, :], cs[:, T - 1:T], sn[:, T - 1:T], rots,
                   ones_f, 1)

    # K / V cache writes (all tokens), then the cache up to L (as full_body)
    for c in range(2):
        nisa.dma_copy(dst=kc[pid, c].ap(pattern=[[MAXPOS, P], [1, T]], offset=0, scalar_offset=preg, indirect_dim=1),
                      src=kT[:, c, L - T:L] if early else kr[:, c, :])
    for s in range(NS):
        v_tm = vv[:, NBO + s, :] if early else nl.ndarray((TS, 2 * P), dtype=BF16, buffer=nl.sbuf)
        for c in range(2):
            tv = nl.ndarray((TS, P), dtype=BF16, buffer=nl.psum)
            nisa.nc_transpose(dst=tv, data=v[:, c, s * TS:(s + 1) * TS])
            nisa.tensor_copy(dst=v_tm[:, c * P:(c + 1) * P], src=tv)
        nisa.dma_copy(dst=vc[pid].ap(pattern=[[256, TS], [1, 256]], offset=s * TS * 256, scalar_offset=preg, indirect_dim=0),
                      src=v_tm)
    if kv_only:
        return None
    NB = L // P
    if not early:
        kT = nl.ndarray((P, 2, L), dtype=BF16, buffer=nl.sbuf)
        for dt in range(2):
            nisa.dma_copy(dst=kT[:, dt, :], src=kc[pid, dt, :, 0:L], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        vv = nl.ndarray((P, NB, 256), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=vv, src=vc[pid, 0:L, :].reshape((NB, P, 256)).permute((1, 0, 2)), dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync)

    # attention of the last query (one column of att_tight_t: its causal mask keeps every key of the last block)
    sq = nl.ndarray((P, 2, 2), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq[:, 0, :], data1=qn1, data2=qn1, op=nl.multiply)
    nisa.tensor_tensor(dst=sq[:, 1, :], data1=kn1, data2=kn1, op=nl.multiply)
    mx = nl.ndarray((P, 2), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.max, data=sq, axis=2)
    mxt = nl.ndarray((2, P), dtype=FP32, buffer=nl.psum)
    nisa.nc_transpose(dst=mxt, data=mx)
    m2 = nl.ndarray((2, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m2, op=nl.max, data=mxt, axis=1)
    cps = nl.ndarray((P, 1), dtype=FP32, buffer=nl.psum)
    nisa.nc_matmul(dst=cps, stationary=ones_f[0:2, :], moving=m2)
    nshift = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nshift, data=cps, op0=nl.multiply, operand0=-8.0, op1=nl.add, operand1=64.0)
    ones_b = ones_tile(BF16)
    qa = nl.ndarray((P, 2, QH), dtype=BF16, buffer=nl.sbuf)
    for dt in range(2):
        for hq in range(QH):
            nisa.tensor_copy(dst=qa[:, dt, hq:hq + 1], src=qr[:, 2 * hq + dt, :])
    po0 = nl.ndarray((P, QH), dtype=FP32, buffer=nl.psum)
    po1 = nl.ndarray((P, QH), dtype=FP32, buffer=nl.psum)
    psm = nl.ndarray((P, QH), dtype=FP32, buffer=nl.psum)
    for b in range(NB):
        sp = nl.ndarray((P, QH), dtype=FP32, buffer=nl.psum)
        for dt in range(2):
            nisa.nc_matmul(dst=sp, stationary=kT[:, dt, b * P:(b + 1) * P], moving=qa[:, dt, :])
        pt = nl.ndarray((P, QH), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=pt, op=nl.exp, data=sp, scale=0.0625, bias=nshift)
        nisa.nc_matmul(dst=po0, stationary=vv[:, b, 0:P], moving=pt)
        nisa.nc_matmul(dst=po1, stationary=vv[:, b, P:2 * P], moving=pt)
        nisa.nc_matmul(dst=psm, stationary=ones_b, moving=pt)
    rs = nl.ndarray((P, QH), dtype=FP32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=psm)
    sga = nl.ndarray((P, QH, 2, 1), dtype=BF16, buffer=nl.sbuf)
    for hq in range(QH):
        nisa.activation(dst=sga[:, hq, :, :], op=nl.sigmoid, data=q[:, 4 * hq + 2:4 * hq + 4, :])
    ao = nl.ndarray((P, 2 * NQH, 1), dtype=BF16, buffer=nl.sbuf)
    for dt in range(2):
        o16 = nl.ndarray((P, QH), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o16, data1=(po0, po1)[dt], data2=rs, op=nl.multiply)
        for hq in range(QH):
            nisa.tensor_tensor(dst=ao[:, pid * 2 * QH + 2 * hq + dt, :], data1=o16[:, hq:hq + 1], data2=sga[:, hq, dt, :],
                               op=nl.multiply)
    exchange_part(ao, 2 * QH, 0, 2 * QH, pid)

    # out-projection, residual, MLP of the last token
    att = nl.ndarray((P, KT, 1), dtype=BF16, buffer=nl.sbuf)
    gemv_col(att[:, pid * 8:pid * 8 + 8, :], wo, pid, ao, 8, KT, OPG, 2 * QH)
    exchange_part(att, 8, 0, 8, pid)
    x1 = nl.ndarray((P, KT, 1), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=x1, data1=xs[:, :, T - 1:T], data2=att, op=nl.add)
    w2 = load_norm_w1(ln2, KT, 'ln2')
    hx2, bc2 = norm_col(x1, w2, pid, ones_f)
    m = nl.ndarray((P, FT, 1), dtype=BF16, buffer=nl.sbuf)
    G = 4
    for n0 in range(0, FL, G):
        gate = nl.ndarray((P, G, 1), dtype=BF16, buffer=nl.sbuf)
        gemv_col(gate, wg[:, :, n0:n0 + G, :, :], pid, hx2, G, KT, G, KT // 2, bc=bc2)
        up = nl.ndarray((P, G, 1), dtype=BF16, buffer=nl.sbuf)
        gemv_col(up, wu[:, :, n0:n0 + G, :, :], pid, hx2, G, KT, G, KT // 2, bc=bc2)
        act = nl.ndarray((P, G, 1), dtype=BF16, buffer=nl.sbuf)
        nisa.activation(dst=act, op=nl.silu, data=gate)
        nisa.tensor_tensor(dst=m[:, pid * FL + n0:pid * FL + n0 + G, :], data1=act, data2=up, op=nl.multiply)
    exchange_part(m, FL, 0, FL, pid)
    dn = nl.ndarray((P, KT, 1), dtype=BF16, buffer=nl.sbuf)
    gemv_col(dn[:, pid * 8:pid * 8 + 8, :], wd, pid, m, 8, FT, 2, FL)
    exchange_part(dn, 8, 0, 8, pid)
    x2 = nl.ndarray((P, KT, 1), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=x2, data1=x1, data2=dn, op=nl.add)
    return x2


ATT_T = True           # prefill TIGHT attention in the transposed (keys-on-partitions) layout, see att_tight_t


def att_tight_t(qr, kT, vv, ao, sga, qn1, kn1, pid, ones_f):
    """Prefill attention of a TIGHT chunk with the keys on the partitions: per sub-chunk s and 128-key block b the
    scores of all 4 local query heads at once, S^T [128 keys, 4 * TS queries] = K_b Q^T (one PSUM bank), then
    P^T = bf16(exp(S^T / 16 - c)) straight from PSUM on the ScalarE: P^T is already the moving operand of P V, so no
    transposes, no score copy and no max pass are needed. The softmax sum over the keys is a ones-matmul of P^T (it
    lands broadcast on every partition), and the output is normalised after P V: o = (P~ V) / sum(P~).
    The shift c replaces the row max (softmax is shift-invariant): with q and k RMS-normalised over 256 dims with
    weights (1 + w), |q| <= 16 max|1 + w_q| and |k| <= 16 max|1 + w_k|, so |s| <= 16 Wq Wk <= 8 (Wq^2 + Wk^2) =: c0;
    c = c0 - 64 keeps exp(s - c) <= e^64 (no overflow, also summed over 8192 keys), and the largest probability of a
    row underflows only if its max score is below c0 - 150 (never for this checkpoint: c0 <= 92).
    Rounding differs from the row-max form (fp32 exp of shifted arguments, normalisation after P V)."""
    QH = NQH // 2
    # c from the norm weights: max over partitions of (1 + w)^2 by a PE transpose
    sq = nl.ndarray((P, 2, 2), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq[:, 0, :], data1=qn1, data2=qn1, op=nl.multiply)
    nisa.tensor_tensor(dst=sq[:, 1, :], data1=kn1, data2=kn1, op=nl.multiply)
    mx = nl.ndarray((P, 2), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.max, data=sq, axis=2)
    mxt = nl.ndarray((2, P), dtype=FP32, buffer=nl.psum)
    nisa.nc_transpose(dst=mxt, data=mx)
    m2 = nl.ndarray((2, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m2, op=nl.max, data=mxt, axis=1)
    cps = nl.ndarray((P, 1), dtype=FP32, buffer=nl.psum)
    nisa.nc_matmul(dst=cps, stationary=ones_f[0:2, :], moving=m2)            # Wq^2 + Wk^2 on every partition
    nshift = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nshift, data=cps, op0=nl.multiply, operand0=-8.0, op1=nl.add, operand1=64.0)
    # causal mask of the diagonal block, keys on partitions: keep key j' <= query t
    jt = nl.ndarray((P, TS), dtype=FP32, buffer=nl.sbuf)
    nisa.iota(dst=jt, pattern=[[1, TS]], offset=0, channel_multiplier=-1)            # t - j'
    m01 = nl.ndarray((P, TS), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m01, data=jt, op0=nl.greater_equal, operand0=0.0)
    ones_b = ones_tile(BF16)
    W = QH * TS
    items = []
    for s in range(NS):
        NBs = (L - (NS - 1 - s) * TS) // P
        for b in range(NBs):
            items.append((s, b, NBs))
    acc = []
    qas = []
    pts = []
    n = len(items)
    for i in range(n + 1):
        if i < n:
            s, b, NBs = items[i]
            if b == 0:
                qa = nl.ndarray((P, 2, W), dtype=BF16, buffer=nl.sbuf)       # the 4 heads' queries side by side
                for dt in range(2):
                    for hq in range(QH):
                        nisa.tensor_copy(dst=qa[:, dt, hq * TS:(hq + 1) * TS], src=qr[:, 2 * hq + dt, s * TS:(s + 1) * TS])
                qas.append(qa)
            qa = qas[s]
            sp = nl.ndarray((P, W), dtype=FP32, buffer=nl.psum)
            for dt in range(2):
                nisa.nc_matmul(dst=sp, stationary=kT[:, dt, b * P:(b + 1) * P], moving=qa[:, dt, :])
            pt = nl.ndarray((P, W), dtype=BF16, buffer=nl.sbuf)
            nisa.activation(dst=pt, op=nl.exp, data=sp, scale=0.0625, bias=nshift)
            if b == NBs - 1:
                ptm = nl.ndarray((P, W), dtype=BF16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ptm.reshape((P, QH, TS)), data1=pt.reshape((P, QH, TS)),
                                   data2=m01.expand_dim(1).broadcast(1, QH), op=nl.multiply)
                pt = ptm
            pts.append(pt)
        if i >= 1:
            s, b, NBs = items[i - 1]
            pt = pts[i - 1]
            if b == 0:
                acc.append((nl.ndarray((P, W), dtype=FP32, buffer=nl.psum), nl.ndarray((P, W), dtype=FP32, buffer=nl.psum),
                          nl.ndarray((P, W), dtype=FP32, buffer=nl.psum)))
            po0, po1, psm = acc[s]
            nisa.nc_matmul(dst=po0, stationary=vv[:, b, 0:P], moving=pt)
            nisa.nc_matmul(dst=po1, stationary=vv[:, b, P:2 * P], moving=pt)
            nisa.nc_matmul(dst=psm, stationary=ones_b, moving=pt)
            if b == NBs - 1:
                rs = nl.ndarray((P, W), dtype=FP32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=psm)
                for dt in range(2):
                    o16 = nl.ndarray((P, W), dtype=BF16, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=o16, data1=(po0, po1)[dt], data2=rs, op=nl.multiply)
                    for hq in range(QH):
                        nisa.tensor_tensor(dst=ao[:, pid * 2 * QH + 2 * hq + dt, s * TS:(s + 1) * TS],
                                           data1=o16[:, hq * TS:(hq + 1) * TS], data2=sga[:, hq, dt, s * TS:(s + 1) * TS],
                                           op=nl.multiply)
    exchange_part(ao, 2 * QH, 0, 2 * QH, pid)


def att_scores_tight(qr, kT, tri, hq, s, Ls):
    """as att_scores for a TIGHT variant: only the last TS keys of Ls get the (causal triangle) mask; the others
    are q.k / 16 exactly as q.k / 16 + 0"""
    sc = nl.ndarray((TS, Ls), dtype=FP32, buffer=nl.sbuf)
    for c0 in range(0, Ls, 512):
        cw = min(512, Ls - c0)
        ps = nl.ndarray((TS, cw), dtype=FP32, buffer=nl.psum)
        for dt in range(2):
            nisa.nc_matmul(dst=ps, stationary=qr[:, 2 * hq + dt, s * TS:(s + 1) * TS], moving=kT[:, dt, c0:c0 + cw])
        # unmasked scores scaled on the ScalarE (x / 16 is exact on either engine; the VectorE is the busier one)
        if c0 + cw < Ls:
            nisa.activation(dst=sc[:, c0:c0 + cw], op=nl.copy, data=ps, scale=0.0625)
        else:
            if cw > TS:
                nisa.activation(dst=sc[:, c0:c0 + cw - TS], op=nl.copy, data=ps[:, 0:cw - TS], scale=0.0625)
            nisa.scalar_tensor_tensor(dst=sc[:, c0 + cw - TS:c0 + cw], data=ps[:, cw - TS:cw], op0=nl.multiply,
                                      operand0=0.0625, op1=nl.add, operand1=tri)
    return sc


def att_scores(qr, kT, mask, hq, s, Ls):
    """prefill attention scores of local query head hq, sub-chunk s: sc [TS, Ls] fp32 = q.k / 16 + mask"""
    sc = nl.ndarray((TS, Ls), dtype=FP32, buffer=nl.sbuf)
    for c0 in range(0, Ls, 512):
        cw = min(512, Ls - c0)
        ps = nl.ndarray((TS, cw), dtype=FP32, buffer=nl.psum)
        for dt in range(2):
            nisa.nc_matmul(dst=ps, stationary=qr[:, 2 * hq + dt, s * TS:(s + 1) * TS], moving=kT[:, dt, c0:c0 + cw])
        nisa.scalar_tensor_tensor(dst=sc[:, c0:c0 + cw], data=ps, op0=nl.multiply, operand0=0.0625,
                                  op1=nl.add, operand1=mask[:, c0:c0 + cw])
    return sc


def load_posf(posf, pid):
    """position as fp32 on every partition [P, 1]"""
    pf = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pf, src=posf[pid])
    return pf


@nki.jit
def full_layer(x, pos, posf, ln1, wq, wk, wv, qnw, knw, cos, sin, rot, wo, ln2, wg, wu, wd, kc, vc):
    """One full-attention layer + MLP; x [P, KT, T] bf16 in HBM; pos [2, 1, 1] int32 and posf [2, P, 1] fp32
    (per core, same value). See full_body."""
    pid = nl.program_id(0)
    ones_f = ones_tile(FP32)
    preg = load_pos(pos, pid)
    pf = load_posf(posf, pid)
    xs = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    x2 = full_body(xs, preg, pf, ln1, wq, wk, wv, qnw, knw, cos, sin, rot, wo, ln2, wg, wu, wd, kc, vc, pid, ones_f)
    out = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=x2)
    return out


# ----------------------------------------------------------------------------------------------
# position update (pos += T on both per-core copies); with the head it also stores the token

def advance_pos(pos, posf, pid):
    """pos[pid] += T, posf[pid] += T. Returns the new position as an int32 [1, 1] SBUF tile."""
    pfa = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pfa, src=posf[pid])
    pfn = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pfn, data=pfa, op0=nl.add, operand0=float(T))
    nisa.dma_copy(dst=posf[pid], src=pfn)
    pin = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pin, src=pfn[0:1, :])
    nisa.dma_copy(dst=pos[pid], src=pin)
    return pin


@nki.jit
def zero_states(l_convst, l_recst):
    """Zero the DeltaNet conv and recurrent states of all 18 layers (each core its own half)."""
    pid = nl.program_id(0)
    z = nl.ndarray((P, HL * P), dtype=FP32, buffer=nl.sbuf)
    nisa.memset(dst=z, value=0.0)
    zb = nl.ndarray((P, QL * 3), dtype=BF16, buffer=nl.sbuf)
    nisa.memset(dst=zb, value=0.0)
    for li in range(NLAYER - NLAYER // 4):
        nisa.dma_copy(dst=l_recst[li, pid], src=z.reshape((P, HL, P)), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.dma_copy(dst=l_convst[li, pid], src=zb.reshape((P, QL, 3)), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    out = nl.ndarray((1, 1), dtype=FP32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=z[0:1, 0:1])
    return out


@nki.jit
def adv(pos, posf):
    """pos/posf advanced by T (a prefill chunk without the LM head)."""
    pid = nl.program_id(0)
    pin = advance_pos(pos, posf, pid)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=pin)
    return out


# ----------------------------------------------------------------------------------------------
# final norm + LM head + greedy argmax (ties to the lowest id) for column HEADCOL of x

def head_body(xs, lnf, wlm, vofs, toks, pos, posf, pid, ones_f, pin_pre=None):
    """xs [P, KT, 1] bf16 SBUF (the last token); lnf [P, KT] bf16; wlm [2, P, VTL, 16, 128] bf16
    (core c: vocabulary tiles c*VTL ..); vofs [2, P, 1] fp32 = -c*VTL*128.
    Stores the token at toks[pos + T], advances pos/posf by T, returns the token [1, 1] int32."""
    w1 = load_norm_w1(lnf, KT)
    h = rmsnorm_fm(xs, w1, KT, 1, D, ones_f)
    lg = nl.ndarray((P, VTL), dtype=FP32, buffer=nl.sbuf)
    G = 4
    for c0 in range(0, VTL, 512):
        cn = min(512, VTL - c0)
        ps = nl.ndarray((P, cn), dtype=FP32, buffer=nl.psum)
        for n0 in range(0, cn, G):
            g = min(G, cn - n0)
            wt = nl.ndarray((P, g, KT, P), dtype=BF16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt, src=wlm[pid, :, c0 + n0:c0 + n0 + g, :, :], dge_mode=nisa.dge_mode.hwdge,
                          engine=nisa.engine.sync)
            for i in range(g):
                for k in range(KT):
                    nisa.nc_matmul(dst=ps[:, n0 + i:n0 + i + 1], stationary=wt[:, i, k, :], moving=h[:, k, :])
        lb = nl.ndarray((P, cn), dtype=BF16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lb, src=ps)                 # logits are bf16 in the reference
        nisa.tensor_copy(dst=lg[:, c0:c0 + cn], src=lb)
    # local argmax: id = c*VTL*128 + col*128 + p; score = 2^20 - id
    mp = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mp, op=nl.max, data=lg, axis=1)
    vo = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vo, src=vofs[pid])
    idl = nl.ndarray((P, VTL), dtype=FP32, buffer=nl.sbuf)
    nisa.iota(dst=idl, pattern=[[-P, VTL]], offset=1048576, channel_multiplier=-1)
    idsc = nl.ndarray((P, VTL), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=idsc, data=idl, op0=nl.add, operand0=vo)
    eq = nl.ndarray((P, VTL), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=mp)
    sc = nl.ndarray((P, VTL), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sc, data1=eq, data2=idsc, op=nl.multiply)
    sp = nl.ndarray((P, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sp, op=nl.max, data=sc, axis=1)
    tm = nl.ndarray((1, P), dtype=FP32, buffer=nl.psum)
    nisa.nc_transpose(dst=tm, data=mp)
    ts = nl.ndarray((1, P), dtype=FP32, buffer=nl.psum)
    nisa.nc_transpose(dst=ts, data=sp)
    tms = nl.ndarray((1, P), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tms, src=tm)
    tss = nl.ndarray((1, P), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tss, src=ts)
    # pr[0, c, :] = (max, best score) of core c
    pr = nl.ndarray((1, 2, 4), dtype=FP32, buffer=nl.sbuf)
    nisa.memset(dst=pr, value=0.0)
    nisa.tensor_reduce(dst=pr[:, pid, 0:1], op=nl.max, data=tms, axis=1)
    e2 = nl.ndarray((1, P), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e2, data=tms, op0=nl.equal, operand0=pr[:, pid, 0:1])
    s2 = nl.ndarray((1, P), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s2, data1=e2, data2=tss, op=nl.multiply)
    nisa.tensor_reduce(dst=pr[:, pid, 1:2], op=nl.max, data=s2, axis=1)
    nisa.sendrecv(src=pr[:, pid, :], dst=pr[:, 1 - pid, :], send_to_rank=1 - pid, recv_from_rank=1 - pid,
                  pipe_id=0, dma_engine=nisa.dma_engine.gpsimd_dma)
    M = nl.ndarray((1, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=M, op=nl.max, data=pr[:, :, 0], axis=1)
    e3 = nl.ndarray((1, 2), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e3, data=pr[:, :, 0], op0=nl.equal, operand0=M)
    s3 = nl.ndarray((1, 2), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s3, data1=e3, data2=pr[:, :, 1], op=nl.multiply)
    best = nl.ndarray((1, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=best, op=nl.max, data=s3, axis=1)
    idf = nl.ndarray((1, 1), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=idf, data=best, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1048576.0)
    idi = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idi, src=idf)
    # toks[pos + T] = token; pos += T  (HEAD_ADV False: pos was advanced by the chunk; toks[pos] = token)
    if pin_pre is not None:
        pin = pin_pre                    # the fused prefill chunk advanced pos (its new value, SBUF)
    elif HEAD_ADV:
        pin = advance_pos(pos, posf, pid)
    else:
        pin = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pin, src=pos[pid])
    treg = nisa.register_alloc()
    nisa.register_load(treg, pin)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=treg, indirect_dim=0), src=idi)
    return idi


@nki.jit
def head(x, lnf, wlm, vofs, toks, pos, posf):
    """Final norm + LM head + argmax of column HEADCOL of x [P, KT, T] (HBM); see head_body."""
    pid = nl.program_id(0)
    ones_f = ones_tile(FP32)
    xs = nl.ndarray((P, KT, 1), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x[:, :, HEADCOL:HEADCOL + 1])
    idi = head_body(xs, lnf, wlm, vofs, toks, pos, posf, pid, ones_f)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=idi)
    return out


# ----------------------------------------------------------------------------------------------
# one whole decode step (T = 1): embedding, 24 layers, LM head, argmax, toks[pos + 1] = token, pos += 1.
# Weights of the 18 DeltaNet layers (l_*) and of the 6 full-attention layers (f_*) are stacked along a
# leading layer dimension (same per-layer layouts as lin_layer / full_layer).

def prefetch_layer(l, pid, rot, l_ln1, l_convst, l_wab, l_convw, l_dtb, l_alog, l_recst, l_normw, l_ln2,
                   f_ln1, f_qnw, f_knw, f_ln2):
    """the DMAs (only: no arithmetic, so no compute engine waits for them early) of layer l's small loads in PF_SET"""
    d = {}
    if l % 4 == 3:
        fi = l // 4
        srcs = [('ln1', (P, KT), BF16, f_ln1[fi]), ('qn', (P, 2), BF16, f_qnw[fi]), ('kn', (P, 2), BF16, f_knw[fi]),
                ('rots', (P, P), BF16, rot), ('ln2', (P, KT), BF16, f_ln2[fi])]
    else:
        li = l - l // 4
        srcs = [('ln1', (P, KT), BF16, l_ln1[li]), ('cs', (P, QL, 3), BF16, l_convst[li][pid]),
                ('wabs', (P, KT, 2 * HL), BF16, l_wab[li][pid]), ('cw', (P, QL, 4), BF16, l_convw[li][pid]),
                ('dtt', (1, HL), BF16, l_dtb[li][pid, 0:1, :]), ('alt', (1, HL), FP32, l_alog[li][pid, 0:1, :]),
                ('S', (P, HL, P), FP32, l_recst[li][pid]), ('nw', (P, 1), FP32, l_normw[li]),
                ('ln2', (P, KT), BF16, l_ln2[li])]
    for it in srcs:
        if it[0] in PF_SET:
            pf_load(d, it[0], it[1], it[2], it[3])
    return d


@nki.jit
def step(toks, pos, posf, emb, lnf, wlm, vofs, cos, sin, rot,
         l_ln1, l_wqkv, l_wz, l_wab, l_convw, l_alog, l_dtb, l_normw, l_wout, l_ln2, l_wg, l_wu, l_wd,
         l_convst, l_recst,
         f_ln1, f_wq, f_wk, f_wv, f_qnw, f_knw, f_wo, f_ln2, f_wg, f_wu, f_wd, f_kc, f_vc):
    pid = nl.program_id(0)
    ones_f = ones_tile(FP32)
    ones_b = ones_tile(BF16)
    ident = ident_tile()
    identf = ident_tile(FP32)
    preg = load_pos(pos, pid)
    pf = load_posf(posf, pid)
    # one buffer for every exchange tile of the step (never reused: see exchange)
    xbuf = nl.ndarray((P, NLAYER * (3 * KT + FT), 1), dtype=BF16, buffer=nl.sbuf)
    nisa.memset(dst=xbuf, value=0.0)
    XB.clear()
    XB.extend([xbuf, 0])
    xs = embed_body(toks, preg, emb)


    nxt = prefetch_layer(0, pid, rot, l_ln1, l_convst, l_wab, l_convw, l_dtb, l_alog, l_recst, l_normw, l_ln2, f_ln1, f_qnw, f_knw, f_ln2) if PREFETCH else {}
    for l in range(NLAYER):
        cur = nxt
        nxt = prefetch_layer(l + 1, pid, rot, l_ln1, l_convst, l_wab, l_convw, l_dtb, l_alog, l_recst, l_normw, l_ln2, f_ln1, f_qnw, f_knw, f_ln2) if (PREFETCH and l + 1 < NLAYER) else {}
        PFC.clear()
        PFC.append(cur)
        if l % 4 == 3:
            fi = l // 4
            xs = full_body(xs, preg, pf, f_ln1[fi], f_wq[fi], f_wk[fi], f_wv[fi], f_qnw[fi], f_knw[fi], cos, sin, rot,
                           f_wo[fi], f_ln2[fi], f_wg[fi], f_wu[fi], f_wd[fi], f_kc[fi], f_vc[fi], pid, ones_f)
        else:
            li = l - l // 4
            xs = lin_body(xs, l_ln1[li], l_wqkv[li], l_wz[li], l_wab[li], l_convw[li], l_alog[li], l_dtb[li],
                          l_normw[li], l_wout[li], l_ln2[li], l_wg[li], l_wu[li], l_wd[li], l_convst[li], l_recst[li],
                          None, pid, ones_f, ones_b, ident, identf)
    PFC.clear()
    XB.clear()
    idi = head_body(xs, lnf, wlm, vofs, toks, pos, posf, pid, ones_f)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=idi)
    return out


# ----------------------------------------------------------------------------------------------
# one whole prefill chunk (T = 128): embedding, 24 layers, pos += 128; the hidden state goes to HBM for the head.

def sel_tile(n):
    """bf16 [n, n, P]: sel[k, c, m] = (k == c), the stationary that broadcasts row c of a [n, T] moving tile"""
    idn = nl.ndarray((n, n), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idn, src=identf_small(n))
    s = nl.ndarray((n, n, P), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s, src=idn.expand_dim(2).broadcast(2, P))
    return s


def pchunk_body(toks, pos, posf, emb, cos, sin, rot, dnc,
           l_ln1, l_wqkv, l_wz, l_wab, l_convw, l_alog, l_dtb, l_normw, l_wout, l_ln2, l_wg, l_wu, l_wd,
           l_convst, l_recst, l_cdiag,
           f_ln1, f_wq, f_wk, f_wv, f_qnw, f_knw, f_wo, f_ln2, f_wg, f_wu, f_wd, f_kc, f_vc, pid, ones_f, last_only=False,
           kv_last=False):
    """embedding + 24 layers of the chunk; returns the hidden state [P, KT, T] (SBUF) (last_only: [P, KT, 1] of the
    last token, the last layer by full_last; kv_last: the last layer only writes its K / V cache and the returned
    state is layer 23's output)"""
    ones_b = ones_tile(BF16)
    ident = ident_tile()
    identf = ident_tile(FP32)
    sel16 = sel_tile(2 * HL)
    selF = nl.ndarray((HL, HL, P), dtype=FP32, buffer=nl.sbuf)       # selF[k, c, m] = (k == c)
    nisa.tensor_copy(dst=selF, src=identf_small(HL).expand_dim(2).broadcast(2, P))
    preg = load_pos(pos, pid)
    pf = load_posf(posf, pid)
    xs = embed_body(toks, preg, emb)
    for l in range(NLAYER):
        if l % 4 == 3 and kv_last and l == NLAYER - 1:
            # the chunk's hidden state is not used (a later chunk follows): the last layer only writes its K / V
            fi = l // 4
            full_last(xs, preg, pf, f_ln1[fi], f_wq[fi], f_wk[fi], f_wv[fi], f_qnw[fi], f_knw[fi], cos, sin, rot,
                      f_wo[fi], f_ln2[fi], f_wg[fi], f_wu[fi], f_wd[fi], f_kc[fi], f_vc[fi], pid, ones_f, selF, identf,
                      kv_only=True)
        elif l % 4 == 3 and last_only and l == NLAYER - 1:
            fi = l // 4
            xs = full_last(xs, preg, pf, f_ln1[fi], f_wq[fi], f_wk[fi], f_wv[fi], f_qnw[fi], f_knw[fi], cos, sin, rot,
                           f_wo[fi], f_ln2[fi], f_wg[fi], f_wu[fi], f_wd[fi], f_kc[fi], f_vc[fi], pid, ones_f, selF, identf)
        elif l % 4 == 3:
            fi = l // 4
            xs = full_body(xs, preg, pf, f_ln1[fi], f_wq[fi], f_wk[fi], f_wv[fi], f_qnw[fi], f_knw[fi], cos, sin, rot,
                           f_wo[fi], f_ln2[fi], f_wg[fi], f_wu[fi], f_wd[fi], f_kc[fi], f_vc[fi], pid, ones_f, selF, identf)
        else:
            li = l - l // 4
            xs = lin_body(xs, l_ln1[li], l_wqkv[li], l_wz[li], l_wab[li], l_convw[li], l_alog[li], l_dtb[li],
                          l_normw[li], l_wout[li], l_ln2[li], l_wg[li], l_wu[li], l_wd[li], l_convst[li], l_recst[li],
                          dnc, pid, ones_f, ones_b, ident, identf, l_cdiag[li], sel16, selF)
    return xs



@nki.jit
def pchunk(toks, pos, posf, emb, cos, sin, rot, dnc,
           l_ln1, l_wqkv, l_wz, l_wab, l_convw, l_alog, l_dtb, l_normw, l_wout, l_ln2, l_wg, l_wu, l_wd,
           l_convst, l_recst, l_cdiag,
           f_ln1, f_wq, f_wk, f_wv, f_qnw, f_knw, f_wo, f_ln2, f_wg, f_wu, f_wd, f_kc, f_vc):
    pid = nl.program_id(0)
    ones_f = ones_tile(FP32)
    # TIGHT variants are never the last chunk of a prompt (their fused-head twins pchunk*_l*th are used there), so
    # nothing reads their hidden-state output: the last layer stops after its K / V cache writes (KVLAST)
    xs = pchunk_body(toks, pos, posf, emb, cos, sin, rot, dnc,
                     l_ln1, l_wqkv, l_wz, l_wab, l_convw, l_alog, l_dtb, l_normw, l_wout, l_ln2, l_wg, l_wu, l_wd,
                     l_convst, l_recst, l_cdiag,
                     f_ln1, f_wq, f_wk, f_wv, f_qnw, f_knw, f_wo, f_ln2, f_wg, f_wu, f_wd, f_kc, f_vc, pid, ones_f,
                     kv_last=(KVLAST and TIGHT))
    advance_pos(pos, posf, pid)
    out = nl.ndarray((P, KT, T), dtype=BF16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=xs)
    return out


@nki.jit
def pchunkh(toks, pos, posf, emb, cos, sin, rot, dnc, lnf, wlm, vofs,
            l_ln1, l_wqkv, l_wz, l_wab, l_convw, l_alog, l_dtb, l_normw, l_wout, l_ln2, l_wg, l_wu, l_wd,
            l_convst, l_recst, l_cdiag,
            f_ln1, f_wq, f_wk, f_wv, f_qnw, f_knw, f_wo, f_ln2, f_wg, f_wu, f_wd, f_kc, f_vc):
    """the last prefill chunk of a prompt with the final norm + LM head + argmax of its last token fused in (one
    execution less; the LM-head weight stream starts during the chunk's last layer): toks[pos + T] = token"""
    pid = nl.program_id(0)
    ones_f = ones_tile(FP32)
    xs = pchunk_body(toks, pos, posf, emb, cos, sin, rot, dnc,
                     l_ln1, l_wqkv, l_wz, l_wab, l_convw, l_alog, l_dtb, l_normw, l_wout, l_ln2, l_wg, l_wu, l_wd,
                     l_convst, l_recst, l_cdiag,
                     f_ln1, f_wq, f_wk, f_wv, f_qnw, f_knw, f_wo, f_ln2, f_wg, f_wu, f_wd, f_kc, f_vc, pid, ones_f,
                     last_only=(LASTTOK and TIGHT))
    pin = advance_pos(pos, posf, pid)
    idi = head_body(xs if (LASTTOK and TIGHT) else xs[:, :, T - 1:T], lnf, wlm, vofs, toks, pos, posf, pid, ones_f,
                    pin_pre=pin)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=idi)
    return out
