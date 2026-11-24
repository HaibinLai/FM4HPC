"""Accuracy on ill-conditioned dot products (Ogita-Rump-Oishi GenDot) for OzBLAS D_H vs cuBLAS DGEMM vs the CPU model.
Usage: python3 illcond.py [--device 0] [--seeds 12] [--maxsplit 12]
Matrix A has rows x1,x2 (ill-conditioned with y1,y2), B has columns y1,y2: the DIAGONAL outputs are the ill-conditioned dots.
Metric: relative error |computed-exact|/|exact| per diagonal element (exact via Fractions). cond(dot)=2*sum|x_i y_i|/|x.y| is reported.
"""
import sys, math, json, argparse
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpubatch as G, explore as E, fastmodel as FM, analysis as AN

def exact_dot(x, y): return sum((Fr(float(a)) * Fr(float(b)) for a, b in zip(x, y)), Fr(0))

def gendot(n, log2cond, rng):
    """ORO05 Algorithm 6.1: vectors x,y (binary64) whose dot product has condition ~ 2^log2cond."""
    n2 = n // 2; b = log2cond; x = np.zeros(n); y = np.zeros(n)
    e = np.rint(rng.uniform(0, 1, n2) * b / 2).astype(int); e[0] = int(round(b / 2)) + 1; e[-1] = 0
    for i in range(n2):
        x[i] = (2 * rng.uniform() - 1) * 2.0**e[i]; y[i] = (2 * rng.uniform() - 1) * 2.0**e[i]
    e2 = np.rint(np.linspace(b / 2, 0, n - n2)).astype(int)
    for i in range(n2, n):
        x[i] = (2 * rng.uniform() - 1) * 2.0**e2[i - n2]
        y[i] = float((Fr(float((2 * rng.uniform() - 1) * 2.0**e2[i - n2])) - exact_dot(x[:i], y[:i])) / Fr(float(x[i])))
    perm = rng.permutation(n); return x[perm], y[perm]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='0'); ap.add_argument('--seeds', type=int, default=12); ap.add_argument('--maxsplit', type=int, default=12); a = ap.parse_args()
    K = 64; jobs = []; meta = {}
    for lc in (20, 30, 40, 53, 60, 80, 100, 120, 140):
        for s in range(a.seeds):
            rng = np.random.default_rng(424242 + lc * 100 + s); x1, y1 = gendot(K, lc, rng); x2, y2 = gendot(K, lc, rng)
            A = np.vstack([x1, x2]); B = np.vstack([y1, y2]).T; name = f'ill_{lc}_{s}'
            jobs.append({'name': name, 'M': 2, 'N': 2, 'K': K, 'A': A, 'B': B, 'ta': 'N', 'tb': 'N', 'alpha': 1.0, 'beta': 0.0, 'c0': None, 'maxsplit': a.maxsplit, 'repeat': 1, 'meta': {'fa': 'gendot', 'fb': 'gendot', 'seed': lc}}); meta[name] = lc
    res, local = G.run_jobs(jobs, f'illcond_ms{a.maxsplit}', a.device, Path(__file__).resolve().parent / 'gpu_results/explore')
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list)); mism = 0; ntot = 0
    for j in jobs:
        rs = res[j['name']]
        if rs['out'] is None: continue
        cu, oz = G.read_out(rs['out'], 2, 2); lc = meta[j['name']]
        try: mo, _ = FM.emulate_fast(j['A'], j['B'], a.maxsplit, E.GPU_TABLE)
        except Exception as ex: mo = None
        if mo is not None: ntot += 1; mism += int(not AN.compare(oz, mo)['identical'])
        for d in (0, 1):
            ex = exact_dot(j['A'][d], j['B'][:, d]); cond = float(2 * sum(abs(Fr(float(u)) * Fr(float(v))) for u, v in zip(j['A'][d], j['B'][:, d])) / abs(ex)) if ex != 0 else float('inf')
            rel = lambda v: float(abs(Fr(float(v)) - ex) / abs(ex)) if ex != 0 else float('nan')
            agg[lc]['cond'].append(cond); agg[lc]['oz'].append(rel(oz[d, d])); agg[lc]['cu'].append(rel(cu[d, d]))
            if mo is not None: agg[lc]['model'].append(rel(mo[d, d]))
    print(f'maxsplit={a.maxsplit}  GPU vs model(table) mismatching matrices: {mism}/{ntot}')
    print('log2cond  median cond   OzBLAS rel err (median / max)   cuBLAS rel err (median / max)   model rel err (max)')
    for lc in sorted(agg):
        g = agg[lc]; f = lambda v: f'{np.median(v):.1e} / {np.max(v):.1e}'
        print(f'{lc:<9} {np.median(g["cond"]):<12.1e} {f(g["oz"]):<31} {f(g["cu"]):<31} {np.max(g["model"]) if g["model"] else float("nan"):.1e}')
    (Path(__file__).resolve().parent / f'gpu_results/explore/illcond_ms{a.maxsplit}.json').write_text(json.dumps({str(k): {kk: vv for kk, vv in v.items()} for k, v in agg.items()}, indent=1))

if __name__ == '__main__': main()
