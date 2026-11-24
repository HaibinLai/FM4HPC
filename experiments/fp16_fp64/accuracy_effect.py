"""Does the GPU log2 discrepancy change ACCURACY? For multiseed jobs where GPU != libm model (small shapes, all elements
exact via Fractions), count which of GPU / libm-model is closer to the exact result, and compare error statistics.
Usage: python3 accuracy_effect.py [--max-elems 4096]   (needs gpu_results/explore/multiseed/ from `explore.py multiseed`)
"""
import sys, math, json, argparse
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore as E, gpubatch as G, fastmodel as FM

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--max-elems', type=int, default=4096); a = ap.parse_args()
    here = Path(__file__).resolve().parent; out = here / 'gpu_results/explore/multiseed'
    jobs = E.suite_jobs('multiseed', 6)
    tot = {'jobs': 0, 'differing_jobs': 0, 'elems_diff': 0, 'gpu_closer': 0, 'libm_closer': 0, 'tie': 0}
    maxerr = {'gpu': [], 'libm': [], 'cublas': []}; per_family = {}
    for j in jobs:
        M, N, K = j['M'], j['N'], j['K']
        if M * N > a.max_elems or j['meta']['fa'] not in ('gaussian', 'spread30', 'spread100', 'uniform'): continue
        p = out / 'out' / f"{j['name']}.bin"
        if not p.exists(): continue
        cu, oz = G.read_out(p, M, N); A, B = j['A'], j['B']
        lm, _ = FM.emulate_fast(A, B, 12, math.log2)
        exact = [[sum((Fr(float(x)) * Fr(float(y)) for x, y in zip(A[i], B[:, c])), Fr(0)) for c in range(N)] for i in range(M)]
        scale = max(abs(v) for r in exact for v in r)
        err = lambda arr: np.array([[float(abs(Fr(float(arr[i, c])) - exact[i][c]) / scale) for c in range(N)] for i in range(M)])
        eg, el, ec = err(oz), err(lm), err(cu)
        fam = j['meta']['fa']; d = per_family.setdefault(fam, {'jobs': 0, 'differing_jobs': 0, 'elems_diff': 0, 'gpu_closer': 0, 'libm_closer': 0, 'tie': 0, 'mean_maxerr_gpu': [], 'mean_maxerr_libm': [], 'mean_maxerr_cublas': []})
        diff = (oz + 0.0).view(np.uint64) != (lm + 0.0).view(np.uint64)
        d['jobs'] += 1; d['mean_maxerr_gpu'].append(eg.max()); d['mean_maxerr_libm'].append(el.max()); d['mean_maxerr_cublas'].append(ec.max())
        if diff.any():
            d['differing_jobs'] += 1; d['elems_diff'] += int(diff.sum())
            d['gpu_closer'] += int((eg[diff] < el[diff]).sum()); d['libm_closer'] += int((eg[diff] > el[diff]).sum()); d['tie'] += int((eg[diff] == el[diff]).sum())
    print(f"{'family':10} {'jobs':>5} {'differing':>9} {'elems':>6} {'GPU closer':>10} {'libm closer':>11} {'tie':>4}  mean(max err): GPU / libm-model / cuBLAS")
    for fam, d in per_family.items():
        m = lambda k: float(np.mean(d[k]))
        print(f"{fam:10} {d['jobs']:5d} {d['differing_jobs']:9d} {d['elems_diff']:6d} {d['gpu_closer']:10d} {d['libm_closer']:11d} {d['tie']:4d}  {m('mean_maxerr_gpu'):.3e} / {m('mean_maxerr_libm'):.3e} / {m('mean_maxerr_cublas'):.3e}")
    (here / 'gpu_results/explore/accuracy_effect.json').write_text(json.dumps({k: {kk: (vv if not isinstance(vv, list) else float(np.mean(vv))) for kk, vv in v.items()} for k, v in per_family.items()}, indent=1))

if __name__ == '__main__': main()
