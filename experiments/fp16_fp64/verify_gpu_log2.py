"""Check the 1-ULP explanation: GPU double log2 is inexact on some exact powers of two, so
tau=ceil(log2(rowmax)) is one too large when a residual row max is exactly 2^e (e in the measured bad set).
Usage: python3 verify_gpu_log2.py INPUT.bin GPU_OUTPUT.bin BAD_POWERS.txt
BAD_POWERS.txt = output of log2probe (lines 'BAD e'), measured on V100 / nvcc 12.0.140 / driver 580.178.04.
The 'GPU-style log2' is a measured lookup of THAT toolkit's libdevice behaviour, not a model of the library.
"""
import sys, re, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C, align_gpu as A

class GpuMath:
    def __init__(self, bad): self.bad = bad
    def __getattr__(self, n): return getattr(math, n)
    def log2(self, x):
        m, e = math.frexp(x)
        return (e - 1) + 2.0**-45 if m == 0.5 and (e - 1) in self.bad else math.log2(x)

def main():
    inp, outp, badf = sys.argv[1:4]
    bad = {int(x) for x in re.findall(r'^BAD (-?\d+)$', Path(badf).read_text(), re.M)}
    a, b, cu, oz = A.read_pair(Path(inp).read_bytes(), Path(outp).read_bytes())
    orig = C.math
    for tag, mod in (('model, exact ceil(log2)', orig), ('model, GPU-style log2', GpuMath(bad))):
        C.math = mod
        try: s = A.compare_arrays(oz, C.emulate(a, b, 12)[0]['forward'])
        finally: C.math = orig
        print(f"{Path(inp).name}: {tag:26} {s['bit_identical']}/{s['total']} bit-identical, max ULP {s['max_ulp']}")

if __name__ == '__main__': main()
