"""throttle micro-benchmark: PE GEMM stream with / without gaps, with / without filler matmuls.
usage: mb.py VARIANT OUTDIR   (variants: A, B, C)"""
import os, sys
V = sys.argv[1]
art = os.path.abspath(sys.argv[2])
os.makedirs(art, exist_ok=True)
os.environ["NKI_ARTIFACTS_DIR"] = art
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
import nki.compiler.driver as _drv
import nki.framework.compiled as _cmp
_orig_run = _drv.run_from_bir
_cmp.run_from_bir = lambda bir, co, inputs, **kw: _orig_run(bir, co, inputs, executor=lambda *a, **k: None)
import numpy as np, ml_dtypes
import nki
import nki.language as nl
import nki.isa as nisa
P = 128
NG = int(os.environ.get("NG", "200"))     # groups
GM = int(os.environ.get("GM", "32"))      # matmuls per group
NF = int(os.environ.get("NFILL", "16"))   # filler matmuls per gap (C)
NV = int(os.environ.get("NV", "6"))       # vector ops per gap chain (B, C)
FW = int(os.environ.get("FW", "512"))     # filler moving columns
BF16 = nl.bfloat16
FD = os.environ.get("FD", "in")
FK = os.environ.get("FK", "f32")
DOFF = int(os.environ.get("DOFF", "2"))
FDV = 1.0 if FD == "one" else 0.0
FP32 = nl.float32

@nki.jit
def kern(w, x):
    pid = nl.program_id(0)
    ws = nl.ndarray((P, 16, P), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w)
    xs = nl.ndarray((P, 512), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    acc = nl.ndarray((P, 512), dtype=FP32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    fsum = nl.ndarray((P, 512), dtype=FP32, buffer=nl.sbuf)
    nisa.memset(dst=fsum, value=0.0)
    cur = xs
    fp = nl.ndarray((P, 512), dtype=FP32, buffer=nl.psum)
    if FD == "in":
        fst, fmv = ws, xs
    else:
        fst = nl.ndarray((P, 16, P), dtype=BF16, buffer=nl.sbuf)
        nisa.memset(dst=fst, value=FDV)
        fmv = nl.ndarray((P, 512), dtype=BF16, buffer=nl.sbuf)
        nisa.memset(dst=fmv, value=FDV)
    f32s = nl.ndarray((P, P), dtype=FP32, buffer=nl.sbuf)
    nisa.memset(dst=f32s, value=0.0)
    f32m = nl.ndarray((P, 512), dtype=FP32, buffer=nl.sbuf)
    nisa.memset(dst=f32m, value=0.0)
    fq = nl.ndarray((P, 512), dtype=FP32, buffer=nl.psum)
    if V in ("C", "D", "E", "F"):
        nisa.nc_matmul(dst=fp, stationary=ws[:, 0, :], moving=xs)
    for g in range(NG):
        ps = nl.ndarray((P, 512), dtype=FP32, buffer=nl.psum)
        for m in range(GM if GM > 0 else 1):
            nisa.nc_matmul(dst=ps, stationary=ws[:, m % 16, :], moving=cur)
        if V == "A":
            nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)
        elif V == "F":
            nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)
            for f in range(NF):
                if FK == "f32":
                    nisa.nc_matmul(dst=fq[:, 0:FW], stationary=f32s, moving=f32m[:, 0:FW])
                elif FK == "tr":
                    nisa.nc_transpose(dst=fq[:, 0:128], data=f32m[:, 0:128])
                elif FK == "same128":
                    nisa.nc_matmul(dst=fp[:, 0:128], stationary=ws[:, 0, :], moving=xs[:, 0:128])
                elif FK == "mix":
                    if f % 2 == 0:
                        nisa.nc_matmul(dst=fq[:, 0:128], stationary=f32s, moving=f32m[:, 0:128])
                    else:
                        nisa.nc_matmul(dst=fp[:, 0:512], stationary=ws[:, f % 16, :], moving=xs[:, 0:512])
            if FK == "tr" or FK == "f32" or FK == "mix":
                nisa.tensor_copy(dst=f32m[:, 0:128], src=fq[:, 0:128])
        elif V == "E":
            nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)
            for f in range(NF):
                nisa.nc_matmul(dst=fp[:, 0:FW], stationary=ws[:, f % 16, :], moving=xs[:, 0:FW])
        else:
            # chain on the VectorE that the next group's moving operand depends on
            t = nl.ndarray((P, 512), dtype=FP32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t, data1=acc, data2=ps, op=nl.add)
            for v in range(NV):
                t2 = nl.ndarray((P, 512), dtype=FP32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=t2, data=t, op0=nl.multiply, operand0=0.5)
                t = t2
                if V == "D" and v == NV - 1 - DOFF:
                    mid = nl.ndarray((P, 512), dtype=BF16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=mid, src=t)
                    for f in range(NF):
                        nisa.nc_matmul(dst=fp[:, 0:FW], stationary=fst[:, f % 16, :], moving=mid[:, 0:FW])
            nisa.tensor_copy(dst=acc, src=t)
            nxt = nl.ndarray((P, 512), dtype=BF16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=nxt, src=t)
            if V == "C":
                for f in range(NF):
                    nisa.nc_matmul(dst=fp[:, 0:FW], stationary=fst[:, f % 16, :], moving=fmv[:, 0:FW])
            cur = nxt
    out = nl.ndarray((P, 512), dtype=FP32, buffer=nl.shared_hbm)
    if V in ("C", "D", "E", "F"):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=fp, op=nl.add)
    if V == "F":
        nisa.tensor_tensor(dst=acc, data1=acc, data2=f32m, op=nl.add)
    nisa.dma_copy(dst=out, src=acc)
    return out

bf = ml_dtypes.bfloat16
kern[2](np.zeros((P, 16, P), dtype=bf), np.zeros((P, 512), dtype=bf))
