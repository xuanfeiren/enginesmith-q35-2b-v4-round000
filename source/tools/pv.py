"""bitwise: P^T stationary x [V | 1] moving vs V stationary x P^T moving (and ones x P^T); output both"""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
import numpy as np, ml_dtypes
import nki, nki.language as nl, nki.isa as nisa
P=128; NB=8; W=512
BF16=nl.bfloat16; FP32=nl.float32
@nki.jit
def kern(pt_h, v_h):
    pt = nl.ndarray((P, NB, W), dtype=BF16, buffer=nl.sbuf); nisa.dma_copy(dst=pt, src=pt_h)
    vv = nl.ndarray((P, NB, 257), dtype=BF16, buffer=nl.sbuf); nisa.dma_copy(dst=vv, src=v_h)
    ones = nl.ndarray((P, P), dtype=BF16, buffer=nl.sbuf); nisa.memset(dst=ones, value=1.0)
    po0 = nl.ndarray((P, W), dtype=FP32, buffer=nl.psum); po1 = nl.ndarray((P, W), dtype=FP32, buffer=nl.psum)
    psm = nl.ndarray((P, W), dtype=FP32, buffer=nl.psum)
    for b in range(NB):
        nisa.nc_matmul(dst=po0, stationary=vv[:, b, 0:P], moving=pt[:, b, :])
        nisa.nc_matmul(dst=po1, stationary=vv[:, b, P:2 * P], moving=pt[:, b, :])
        nisa.nc_matmul(dst=psm, stationary=ones, moving=pt[:, b, :])
    a0 = nl.ndarray((P, 3, W), dtype=FP32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=a0[:, 0, :], src=po0); nisa.tensor_copy(dst=a0[:, 1, :], src=po1); nisa.tensor_copy(dst=a0[:, 2, :], src=psm)
    o1 = nl.ndarray((P, 3, W), dtype=FP32, buffer=nl.shared_hbm); nisa.dma_copy(dst=o1, src=a0)
    # reoriented: per head hq (128 queries): out[q, 0:257] = sum_b pt[:, b, hq]^T [V | 1]
    a1 = nl.ndarray((P, 4, 257), dtype=FP32, buffer=nl.sbuf)
    for hq in range(4):
        acc = nl.ndarray((P, 257), dtype=FP32, buffer=nl.psum)
        for b in range(NB):
            nisa.nc_matmul(dst=acc, stationary=pt[:, b, hq * P:(hq + 1) * P], moving=vv[:, b, :])
        nisa.tensor_copy(dst=a1[:, hq, :], src=acc)
    o2 = nl.ndarray((P, 4, 257), dtype=FP32, buffer=nl.shared_hbm); nisa.dma_copy(dst=o2, src=a1)
    return o1, o2
rng=np.random.default_rng(3)
bf=ml_dtypes.bfloat16
for trial in range(3):
    pt=rng.random((P, NB, W)).astype(np.float32).astype(bf)*np.float32(rng.choice([1,10,1000])).astype(bf)
    v=(rng.standard_normal((P, NB, 257))).astype(bf); v[:, :, 256]=1
    o1,o2=kern(pt.astype(bf), v)
    o1=np.asarray(o1); o2=np.asarray(o2)
    # o1[d, 0, q] (d<128), o1[d,1,q] (d 128..255), o1[*,2,q] sum ; o2[q, hq, d]
    new_pv0 = np.concatenate([o2[:, h, 0:128].T for h in range(4)], axis=1)   # [d, 512]
    new_pv1 = np.concatenate([o2[:, h, 128:256].T for h in range(4)], axis=1)
    new_sum = np.concatenate([o2[:, h, 256] for h in range(4)])
    print('trial', trial, 'pv0 diff', (new_pv0 != o1[:, 0, :]).sum(), 'pv1 diff', (new_pv1 != o1[:, 1, :]).sum(),
          'sum diff', (new_sum != o1[0, 2, :]).sum(), 'sum rows equal', (o1[:, 2, :] == o1[0:1, 2, :]).all())
