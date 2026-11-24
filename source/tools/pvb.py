import os, sys
V=sys.argv[1]; art=os.path.abspath(sys.argv[2]); os.makedirs(art, exist_ok=True)
os.environ["NKI_ARTIFACTS_DIR"]=art; os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]="trn2"
import nki.compiler.driver as _drv, nki.framework.compiled as _cmp
_o=_drv.run_from_bir; _cmp.run_from_bir=lambda bir, co, inputs, **kw: _o(bir, co, inputs, executor=lambda *a, **k: None)
import numpy as np, ml_dtypes
import nki, nki.language as nl, nki.isa as nisa
P=128; NB=32; W=512; R=40
BF16=nl.bfloat16; FP32=nl.float32
@nki.jit
def kern(pt_h, v_h):
    pt = nl.ndarray((P, NB, W), dtype=BF16, buffer=nl.sbuf); nisa.dma_copy(dst=pt, src=pt_h)
    vv = nl.ndarray((P, NB, 257), dtype=BF16, buffer=nl.sbuf); nisa.dma_copy(dst=vv, src=v_h)
    ones = nl.ndarray((P, P), dtype=BF16, buffer=nl.sbuf); nisa.memset(dst=ones, value=1.0)
    a = nl.ndarray((P, 4, W), dtype=FP32, buffer=nl.sbuf); nisa.memset(dst=a, value=0.0)
    for r in range(R):
        if V == "old":
            po0 = nl.ndarray((P, W), dtype=FP32, buffer=nl.psum); po1 = nl.ndarray((P, W), dtype=FP32, buffer=nl.psum)
            psm = nl.ndarray((P, W), dtype=FP32, buffer=nl.psum)
            for b in range(NB):
                nisa.nc_matmul(dst=po0, stationary=vv[:, b, 0:P], moving=pt[:, b, :])
                nisa.nc_matmul(dst=po1, stationary=vv[:, b, P:2 * P], moving=pt[:, b, :])
                nisa.nc_matmul(dst=psm, stationary=ones, moving=pt[:, b, :])
            nisa.tensor_tensor(dst=a[:, 0, :], data1=a[:, 0, :], data2=po0, op=nl.add)
            nisa.tensor_tensor(dst=a[:, 1, :], data1=a[:, 1, :], data2=po1, op=nl.add)
            nisa.tensor_tensor(dst=a[:, 2, :], data1=a[:, 2, :], data2=psm, op=nl.add)
        else:
            accs = []
            for hq in range(4):
                accs.append(nl.ndarray((P, 257), dtype=FP32, buffer=nl.psum))
            for b in range(NB):
                for hq in range(4):
                    nisa.nc_matmul(dst=accs[hq], stationary=pt[:, b, hq * P:(hq + 1) * P], moving=vv[:, b, :])
            for hq in range(4):
                nisa.tensor_tensor(dst=a[:, hq, 0:257], data1=a[:, hq, 0:257], data2=accs[hq], op=nl.add)
    o = nl.ndarray((P, 4, W), dtype=FP32, buffer=nl.shared_hbm); nisa.dma_copy(dst=o, src=a)
    return o
bf=ml_dtypes.bfloat16
kern[2](np.zeros((P, NB, W), dtype=bf), np.zeros((P, NB, 257), dtype=bf))
