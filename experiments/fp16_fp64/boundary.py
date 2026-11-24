"""Where do large finite inputs turn into NaN? Scan the max-entry exponent e (entries uniform*0.9*2^e, so tau=e) for several K.
Compares the GPU (OzBLAS) with cuBLAS finiteness and with the model's 'sigma overflow' prediction (rho+tau).
Usage: python3 boundary.py [--device 0]
"""
import sys, math, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpubatch as G, explore as E, fastmodel as FM

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='0'); a = ap.parse_args()
    jobs = []; meta = {}
    for K in (8, 64, 1024, 16384):
        for e in range(940, 1024):
            r = np.random.default_rng(100000 + K + e); M = N = 4
            A = r.uniform(-1, 1, (M, K)) * 0.9 * 2.0**e; A[:, 0] = 0.9 * 2.0**e; B = r.uniform(-1, 1, (K, N)) * 2.0**-60   # scale B down: exact outputs stay finite
            name = f'b_K{K}_e{e}'; jobs.append({'name': name, 'M': M, 'N': N, 'K': K, 'A': A, 'B': B, 'ta': 'N', 'tb': 'N', 'alpha': 1.0, 'beta': 0.0, 'c0': None, 'maxsplit': 12, 'repeat': 1, 'meta': {'fa': f'K{K}', 'fb': 'u', 'seed': e}}); meta[name] = (K, e)
    res, local = G.run_jobs(jobs, 'boundary', a.device, Path(__file__).resolve().parent / 'gpu_results/explore')
    first_nan = {}; first_model_fail = {}; agree = {}
    for j in jobs:
        K, e = meta[j['name']]; rs = res[j['name']]
        if rs['out'] is None: continue
        cu, oz = G.read_out(rs['out'], 4, 4); gnan = int((~np.isfinite(oz)).sum()); cnan = int((~np.isfinite(cu)).sum())
        try: FM.emulate_fast(j['A'], j['B'], 12, FM.make_gpu_log2({}) ); mfail = False
        except Exception: mfail = True
        # overflow in intermediate arithmetic shows as non-finite/ValueError; model raising ~ sigma/sum overflow
        if gnan and K not in first_nan: first_nan[K] = e
        if mfail and K not in first_model_fail: first_model_fail[K] = e
        agree.setdefault(K, []).append((e, gnan > 0, mfail, cnan > 0))
    print('K      rho  | first e with NaN in OzBLAS | first e where model raises | cuBLAS NaN anywhere? | agreement(model raises <=> GPU NaN)')
    for K in (8, 64, 1024, 16384):
        rho = math.ceil(53 - (24 - math.log2(K)) / 2); rows = agree[K]
        ag = all((g == m) for e, g, m, c in rows); cu_any = any(c for e, g, m, c in rows)
        print(f'{K:<6} {rho:<4} | {first_nan.get(K, "none (up to 1023)")!s:>26} | {first_model_fail.get(K, "none")!s:>26} | {str(cu_any):>20} | {ag}   (1023 - rho = {1023 - rho})')

if __name__ == '__main__': main()
