"""fp32 transpose via two bf16 half transposes. usage: tr.py VARIANT OUTDIR  (A: fp32 nc_transpose, B: halves)"""
import os, sys
V = sys.argv[1]; art = os.path.abspath(sys.argv[2]); os.makedirs(art, exist_ok=True)
os.environ["NKI_ARTIFACTS_DIR"] = art; os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
RUN = os.environ.get("RUN") == "1"
if not RUN:
    import nki.compiler.driver as _drv
    import nki.framework.compiled as _cmp
    _orig_run = _drv.run_from_bir
    _cmp.run_from_bir = lambda bir, co, inputs, **kw: _orig_run(bir, co, inputs, executor=lambda *a, **k: None)
import numpy as np
import nki, nki.language as nl, nki.isa as nisa
P = 128; H = 8; NR = int(os.environ.get("NR", "50"))
FP32 = nl.float32; BF16 = nl.bfloat16

@nki.jit
def kern(x):
    xs = nl.ndarray((P, H, P), dtype=FP32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    acc = nl.ndarray((P, H, P), dtype=FP32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for r in range(NR):
        ps = nl.ndarray((P, H, P), dtype=FP32, buffer=nl.psum)
        for h in range(H):
            if V == "A":
                nisa.nc_transpose(dst=ps[:, h, :], data=xs[:, h, :])
            else:
                xb = xs[:, h, :].view(BF16)          # [P, 256]
                pb = ps[:, h, :].view(BF16)
                for half in range(2):
                    nisa.nc_transpose(dst=pb[:, half:256:2], data=xb[:, half:256:2])
        nisa.tensor_copy(dst=acc if r == NR - 1 else acc, src=ps)
    out = nl.ndarray((P, H, P), dtype=FP32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=acc)
    return out

rng = np.random.default_rng(0)
x = (rng.standard_normal((P, H, P)) * np.exp(rng.uniform(-30, 30, (P, H, P)))).astype(np.float32)
x[0, 0, :4] = [1e-40, -1e-42, 3.0, -0.0]
r = kern(x) if not RUN else kern(x)
if RUN:
    r = np.asarray(r)
    ref = x.transpose(2, 1, 0)
    print("bitwise equal:", np.array_equal(r.view(np.uint32), np.ascontiguousarray(ref).view(np.uint32)),
          "maxdiff", np.abs(r - ref).max())
