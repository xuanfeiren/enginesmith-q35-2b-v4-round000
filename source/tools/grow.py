import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
import numpy as np
import nki, nki.language as nl, nki.isa as nisa
P=128; H=8; T=128
FP32=nl.float32
@nki.jit
def kern(gneg_h, ut_h):
    gneg = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf); nisa.dma_copy(dst=gneg, src=gneg_h)
    Ut = nl.ndarray((T, T), dtype=FP32, buffer=nl.sbuf); nisa.dma_copy(dst=Ut, src=ut_h)
    gc_ps = nl.ndarray((T, H), dtype=FP32, buffer=nl.psum)
    nisa.nc_matmul(dst=gc_ps, stationary=Ut, moving=gneg)
    gcs = nl.ndarray((T, H), dtype=FP32, buffer=nl.sbuf); nisa.tensor_copy(dst=gcs, src=gc_ps)
    gr_ps = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.psum)
    for h in range(H):
        nisa.nc_matmul(dst=gr_ps[:, h, :], stationary=gneg[:, h:h + 1].broadcast(1, P), moving=Ut)
    grow = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.sbuf); nisa.tensor_copy(dst=grow, src=gr_ps)
    # transpose of gcs -> [H, T]
    tp = nl.ndarray((H, T), dtype=FP32, buffer=nl.psum); nisa.nc_transpose(dst=tp, data=gcs)
    gT = nl.ndarray((H, T), dtype=FP32, buffer=nl.sbuf); nisa.tensor_copy(dst=gT, src=tp)
    o1 = nl.ndarray((T, H), dtype=FP32, buffer=nl.shared_hbm); nisa.dma_copy(dst=o1, src=gcs)
    o2 = nl.ndarray((P, H, T), dtype=FP32, buffer=nl.shared_hbm); nisa.dma_copy(dst=o2, src=grow)
    o3 = nl.ndarray((H, T), dtype=FP32, buffer=nl.shared_hbm); nisa.dma_copy(dst=o3, src=gT)
    return o1, o2, o3
rng=np.random.default_rng(1)
ok=True
for trial in range(3):
    g = -(rng.random((T, H)) * rng.choice([0.01, 0.3, 3.0])).astype(np.float32)
    ut = np.triu(np.ones((T, T), np.float32))     # Ut[t, i] = 1 if t <= i
    o1, o2, o3 = kern(g, ut)
    o1=np.asarray(o1); o2=np.asarray(o2); o3=np.asarray(o3)
    # grow[m, h, i] vs gcs[i, h]
    d = (o2 != o1.T[None, :, :]).sum()
    dT = (o3 != o1.T).sum()
    ref = np.cumsum(g.astype(np.float64), axis=0)
    print('trial', trial, 'grow!=gcs:', d, 'transpose diff', dT, 'rows equal across m:', (o2 == o2[0:1]).all(), 'maxrel vs f64', np.abs(o1-ref).max()/np.abs(ref).max())
