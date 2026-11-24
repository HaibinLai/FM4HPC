"""Bit-for-bit validation of fastmodel.emulate_fast against compare.emulate (forward) on given .bin inputs.
Usage: python3 validate_fastmodel.py BAD_POWERS.txt INPUT.bin [INPUT.bin ...]
Runs both the libm variant and the GPU-style log2 variant.
"""
import sys, re, math, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C, align_gpu as A, fastmodel as F

class GpuMath:
    def __init__(self, f): self.f = f
    def __getattr__(self, n): return getattr(math, n)
    def log2(self, x): return self.f(x)

def load(path):
    import struct
    b = Path(path).read_bytes(); m, n, k = struct.unpack('<III', b[:12]); d = np.frombuffer(b, dtype='<f8', offset=12)
    return d[:m*k].reshape((m, k), order='F').copy(), d[m*k:].reshape((k, n), order='F').copy()

def main():
    bad = {int(x) for x in re.findall(r'^BAD (-?\d+)$', Path(sys.argv[1]).read_text(), re.M)}
    gl = F.make_gpu_log2(bad); orig = C.math; allok = True
    for p in sys.argv[2:]:
        a, b = load(p)
        for tag, fn, mod in (('libm', math.log2, orig), ('gpu_style', gl, GpuMath(gl))):
            t0 = time.time(); fast, meta = F.emulate_fast(a, b, 12, fn); t1 = time.time()
            C.math = mod
            try: slow = C.emulate(a, b, 12)[0]['forward']
            finally: C.math = orig
            t2 = time.time(); same = bool(np.array_equal(fast.view(np.uint64), slow.view(np.uint64))); allok &= same
            print(f'{Path(p).name:34} {tag:9} bit-identical={same}  fast {t1-t0:.3f}s  slow {t2-t1:.1f}s')
    print('ALL IDENTICAL' if allok else 'MISMATCH FOUND')

if __name__ == '__main__': main()
