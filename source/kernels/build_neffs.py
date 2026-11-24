"""Build time only: compile every kernel variant to a NEFF through NKI's documented route.

NKI_ARTIFACTS_DIR is set before nki is imported; calling a @nki.jit kernel with numpy arrays compiles it
(kernel.neff lands in that directory) and then runs it once on a NeuronCore. Only the NEFF is used; the run
may fail when another build process holds the core, which does not affect the NEFF. Variants are built in
parallel worker processes and cached under target/neff-cache keyed by the kernel source and the variant.

usage: build_neffs.py OUTDIR            (builds all variants)
       build_neffs.py --one NAME DIR    (worker: builds one variant into DIR)
"""
import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TS = [1]               # per-layer kernels: single-token prefill remainder only (128-token chunks are fused)
BUCKETS = [512, 1024, 2048, 4096, 8192]
STEP_BUCKETS = [512, 1024, 1152, 2048, 4096, 4224, 8192]
PCHUNK_TS = [128, 256, 512]   # fused prefill chunk sizes
# 512-token chunks start at multiples of 512 and get the exact bucket pos + 512 (TIGHT); 256 at position 0: 256
TIGHT_BUCKETS = {512: [512 * k for k in range(1, 17)], 256: [256]}
# last chunks of a prompt with the LM head fused in (pchunkh); otherwise the chunk and headna_t* run separately
HEAD_FUSED = {(256, 256)} | {(512, 512 * k) for k in range(1, 17)}


def variants():
    v = {}
    for t in TS:
        v[f"embed_t{t}"] = ("embed", t, 0)
        v[f"lin_t{t}"] = ("lin_layer", t, 0)
        v[f"head_t{t}"] = ("head", t, 0)
        v[f"adv_t{t}"] = ("adv", t, 0)
        for b in BUCKETS:
            v[f"full_t{t}_l{b}"] = ("full_layer", t, b)
    for b in STEP_BUCKETS:
        v[f"step_l{b}"] = ("step", 1, b)
    v["zero_states"] = ("zero_states", 1, 0)
    for t in PCHUNK_TS:
        v[f"headna_t{t}"] = ("headna", t, 0)
        for b in (BUCKETS if t != 512 else []):
            v[f"pchunk{t}_l{b}"] = ("pchunk", t, b)
        for b in TIGHT_BUCKETS.get(t, []):
            v[f"pchunk{t}_l{b}t"] = ("pchunkt", t, b)
            if (t, b) in HEAD_FUSED:
                v[f"pchunk{t}_l{b}th"] = ("pchunkh", t, b)
    return v


def build_one(name, art):
    os.environ["NKI_ARTIFACTS_DIR"] = art
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    sys.path.insert(0, HERE)
    # compile only: skip the run on a NeuronCore that NKI's route does after writing the NEFF (the build then never
    # touches a device, and parallel builds cannot collide with engines on the same core)
    import nki.compiler.driver as _drv
    import nki.framework.compiled as _cmp
    _orig_run = _drv.run_from_bir
    _cmp.run_from_bir = lambda bir, co, inputs, **kw: _orig_run(bir, co, inputs, executor=lambda *a, **k: None)
    import numpy as np
    import ml_dtypes
    import qwen35 as K
    kind, t, b = variants()[name]
    K.T = t
    K.TS = min(t, 128)
    K.NS = max(1, t // 128)
    K.L = b if b else 512
    K.HEADCOL = t - 1
    bf = ml_dtypes.bfloat16
    P, D = 128, 2048
    z = lambda shape, dt=bf: np.zeros(shape, dtype=dt)
    i32 = np.int32
    f32 = np.float32
    if kind == "embed":
        args = (z((K.MAXPOS + 128,), i32), z((2, 1, 1), i32), z((K.VOCAB, D)))
        fn = K.embed
    elif kind == "lin_layer":
        args = (z((P, 16, t)), z((P, 16)), z((2, P, 24, 16, 128)), z((2, P, 8, 16, 128)), z((2, P, 16, 16)),
                z((2, P, 24, 4)), z((2, P, 8), f32), z((2, P, 8)), z((P, 1), f32), z((2, P, 8, 16, 128)),
                z((P, 16)), z((2, P, 24, 16, 128)), z((2, P, 24, 16, 128)), z((2, P, 8, 48, 128)),
                z((2, P, 24, 3)), z((2, P, 8, 128), f32), z((P, K.NMASK, P), f32))
        fn = K.lin_layer
    elif kind == "full_layer":
        args = (z((P, 16, t)), z((2, 1, 1), i32), z((2, P, 1), f32), z((P, 16)), z((2, P, 16, 16, 128)),
                z((2, P, 2, 16, 128)), z((2, P, 2, 16, 128)), z((P, 2)), z((P, 2)), z((64, K.MAXPOS)), z((64, K.MAXPOS)),
                z((P, P)), z((2, P, 8, 16, 128)), z((P, 16)), z((2, P, 24, 16, 128)), z((2, P, 24, 16, 128)),
                z((2, P, 8, 48, 128)), z((2, 2, P, K.MAXPOS)), z((2, K.MAXPOS, 256)))
        fn = K.full_layer
    elif kind == "head":
        args = (z((P, 16, t)), z((P, 16)), z((2, P, K.VTL, 16, 128)), z((2, P, 1), f32),
                z((K.MAXPOS + 128,), i32), z((2, 1, 1), i32), z((2, P, 1), f32))
        fn = K.head
    elif kind == "step":
        NL, NF = 18, 6
        args = (z((K.MAXPOS + 128,), i32), z((2, 1, 1), i32), z((2, P, 1), f32), z((K.VOCAB, D)), z((P, 16)),
                z((2, P, K.VTL, 16, 128)), z((2, P, 1), f32), z((64, K.MAXPOS)), z((64, K.MAXPOS)), z((P, P)),
                z((NL, P, 16)), z((NL, 2, P, 24, 16, 128)), z((NL, 2, P, 8, 16, 128)), z((NL, 2, P, 16, 16)),
                z((NL, 2, P, 24, 4)), z((NL, 2, P, 8), f32), z((NL, 2, P, 8)), z((NL, P, 1), f32),
                z((NL, 2, P, 8, 16, 128)), z((NL, P, 16)), z((NL, 2, P, 24, 16, 128)), z((NL, 2, P, 24, 16, 128)),
                z((NL, 2, P, 8, 48, 128)), z((NL, 2, P, 24, 3)), z((NL, 2, P, 8, 128), f32),
                z((NF, P, 16)), z((NF, 2, P, 16, 16, 128)), z((NF, 2, P, 2, 16, 128)), z((NF, 2, P, 2, 16, 128)),
                z((NF, P, 2)), z((NF, P, 2)), z((NF, 2, P, 8, 16, 128)), z((NF, P, 16)), z((NF, 2, P, 24, 16, 128)),
                z((NF, 2, P, 24, 16, 128)), z((NF, 2, P, 8, 48, 128)), z((NF, 2, 2, P, K.MAXPOS)), z((NF, 2, K.MAXPOS, 256)))
        fn = K.step
    elif kind == "headna":
        K.HEAD_ADV = False
        args = (z((P, 16, t)), z((P, 16)), z((2, P, K.VTL, 16, 128)), z((2, P, 1), f32),
                z((K.MAXPOS + 128,), i32), z((2, 1, 1), i32), z((2, P, 1), f32))
        fn = K.head
    elif kind in ("pchunk", "pchunkt", "pchunkh"):
        K.TIGHT = kind in ("pchunkt", "pchunkh")
        K.FIRST = K.TIGHT and b == t          # the exact bucket pos + t == t: position 0
        NL, NF = 18, 6
        args = (z((K.MAXPOS + 128,), i32), z((2, 1, 1), i32), z((2, P, 1), f32), z((K.VOCAB, D)),
                z((64, K.MAXPOS)), z((64, K.MAXPOS)), z((P, P)), z((P, K.NMASK, P), f32),
                z((NL, P, 16)), z((NL, 2, P, 24, 16, 128)), z((NL, 2, P, 8, 16, 128)), z((NL, 2, P, 16, 16)),
                z((NL, 2, P, 24, 4)), z((NL, 2, P, 8), f32), z((NL, 2, P, 8)), z((NL, P, 1), f32),
                z((NL, 2, P, 8, 16, 128)), z((NL, P, 16)), z((NL, 2, P, 24, 16, 128)), z((NL, 2, P, 24, 16, 128)),
                z((NL, 2, P, 8, 48, 128)), z((NL, 2, P, 24, 3)), z((NL, 2, P, 8, 128), f32), z((NL, 2, P, 96, P)),
                z((NF, P, 16)), z((NF, 2, P, 16, 16, 128)), z((NF, 2, P, 2, 16, 128)), z((NF, 2, P, 2, 16, 128)),
                z((NF, P, 2)), z((NF, P, 2)), z((NF, 2, P, 8, 16, 128)), z((NF, P, 16)), z((NF, 2, P, 24, 16, 128)),
                z((NF, 2, P, 24, 16, 128)), z((NF, 2, P, 8, 48, 128)), z((NF, 2, 2, P, K.MAXPOS)), z((NF, 2, K.MAXPOS, 256)))
        fn = K.pchunk
        if kind == "pchunkh":
            K.HEAD_ADV = False
            args = args[:8] + (z((P, 16)), z((2, P, K.VTL, 16, 128)), z((2, P, 1), f32)) + args[8:]
            fn = K.pchunkh
    elif kind == "zero_states":
        args = (z((18, 2, P, 24, 3)), z((18, 2, P, 8, 128), f32))
        fn = K.zero_states
    elif kind == "adv":
        args = (z((2, 1, 1), i32), z((2, P, 1), f32))
        fn = K.adv
    try:
        fn[2](*args)
    except Exception as e:  # the run after compilation may fail (core busy); the NEFF is what we need
        if not os.path.exists(os.path.join(art, "kernel.neff")):
            raise
        print(f"[{name}] run after compile failed ({type(e).__name__}); NEFF kept", file=sys.stderr)


def main():
    if sys.argv[1] == "--one":
        build_one(sys.argv[2], sys.argv[3])
        return
    out = os.path.abspath(sys.argv[1])
    only = sys.argv[2:] or None
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(os.path.dirname(out), "neff-cache")
    os.makedirs(cache, exist_ok=True)
    src = open(os.path.join(HERE, "qwen35.py"), "rb").read() + open(__file__, "rb").read()
    names = [n for n in variants() if not only or n in only]

    def job(name):
        key = hashlib.sha256(src + name.encode()).hexdigest()[:20]
        cached = os.path.join(cache, f"{name}-{key}.neff")
        if not os.path.exists(cached):
            art = os.path.join(cache, f"build-{name}")
            shutil.rmtree(art, ignore_errors=True)
            os.makedirs(art)
            with open(os.path.join(art, "log.txt"), "w") as log:
                r = subprocess.run([sys.executable, __file__, "--one", name, art], stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0 or not os.path.exists(os.path.join(art, "kernel.neff")):
                tail = open(os.path.join(art, "log.txt")).read()[-3000:]
                return f"FAILED {name}\n{tail}"
            shutil.copy(os.path.join(art, "kernel.neff"), cached + ".tmp")
            os.replace(cached + ".tmp", cached)
            shutil.rmtree(art, ignore_errors=True)
        shutil.copy(cached, os.path.join(out, f"{name}.neff"))
        return f"ok {name}"

    with ThreadPoolExecutor(max_workers=int(os.environ.get("NEFF_JOBS", "12"))) as ex:
        res = list(ex.map(job, names))
    bad = [r for r in res if not r.startswith("ok")]
    for r in res:
        print(r.splitlines()[0])
    if bad:
        print("\n".join(bad), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
