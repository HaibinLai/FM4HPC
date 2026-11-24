"""Special-value behaviour of the original OzBLAS D_H path vs cuBLAS DGEMM (and vs the model when inputs are finite).
Usage: python3 specials.py [--device 0] [--out DIR]
Each case is 4x4x8 unless noted. Output: per-case table of NaN/Inf counts and, for finite inputs, model agreement.
Non-finite behaviour is documented, not judged: cuBLAS is the reference semantics (IEEE per-element), OzBLAS may differ.
"""
import sys, math, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpubatch as G, explore as E

def U(r, s, sc=1.0): return r.uniform(-1, 1, s) * sc

def cases():
    r = np.random.default_rng(777); M, N, K = 4, 4, 8; out = []
    def add(name, A, B, M=M, N=N, K=K, **kw): out.append(E.make_job(name, M, N, K, 'uniform', 'uniform', 1) | {'A': A, 'B': B, 'name': name} | kw)
    # finite special values
    add('zero_A', np.zeros((M, K)), U(r, (K, N)))
    add('zero_B', U(r, (M, K)), np.zeros((K, N)))
    a = U(r, (M, K)); a[2] = 0; add('zero_row_A', a, U(r, (K, N)))
    b = U(r, (K, N)); b[:, 1] = 0; add('zero_col_B', U(r, (M, K)), b)
    a = np.zeros((M, K)); a[0, 0] = 1.5; add('single_entry_A', a, U(r, (K, N)))
    add('subnormal_A', U(r, (M, K), 2.0**-1070), U(r, (K, N)))
    add('subnormal_both', U(r, (M, K), 2.0**-1060), U(r, (K, N), 2.0**-1060))
    add('near_max_A', U(r, (M, K), 2.0**1020), U(r, (K, N)))
    add('overflow_out', np.full((M, K), 2.0**1023), np.ones((K, N)))
    add('tiny_products', U(r, (M, K), 2.0**-600), U(r, (K, N), 2.0**-600))
    add('col_scaled_B', U(r, (M, K)), U(r, (K, N)) * np.ldexp(1.0, np.array([-400, 0, 400, 100]))[None, :])
    add('K1', U(r, (M, 1)), U(r, (1, N)), K=1)
    a = U(r, (M, K)); a[1, :4] = -0.0; add('neg_zero_entries', a, U(r, (K, N)))
    # non-finite inputs
    for nm, pos, val in (('nan_A', (1, 2), np.nan), ('posinf_A', (1, 2), np.inf), ('neginf_A', (1, 2), -np.inf)):
        a = U(r, (M, K)); a[pos] = val; add(nm, a, U(r, (K, N)))
    b = U(r, (K, N)); b[3, 0] = np.inf; add('posinf_B', U(r, (M, K)), b)
    a = U(r, (M, K)); a[0, 0] = np.inf; b = U(r, (K, N)); b[0, 0] = 0.0; add('inf_times_zero', a, b)
    a = U(r, (M, K)); a[0, 0] = np.inf; b = U(r, (K, N)); b[0, :] = 0.0; b[1, 1] = np.inf; add('inf_row_zero_B_row', a, b)
    # beta handling with non-finite C0
    add('nan_C0_beta0', U(r, (M, K)), U(r, (K, N)), beta=0.0, c0=np.full((M, N), np.nan))
    add('nan_C0_beta1', U(r, (M, K)), U(r, (K, N)), beta=1.0, c0=np.full((M, N), np.nan))
    add('inf_C0_beta0', U(r, (M, K)), U(r, (K, N)), beta=0.0, c0=np.full((M, N), np.inf))
    c0 = np.zeros((M, N)); c0[1, 1] = np.nan; add('one_nan_C0_beta0', U(r, (M, K)), U(r, (K, N)), beta=0.0, c0=c0)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='0'); ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'gpu_results/explore'))
    a = ap.parse_args(); jobs = cases()
    res, local = G.run_jobs(jobs, 'specials', a.device, a.out)
    rows = []
    print(f"{'case':22} {'inputs':>9} | {'OzBLAS nan/inf':>14} {'cuBLAS nan/inf':>14} | {'nonfinite pattern equal':>23} | finite-in: GPU==model(table)  note")
    for j in jobs:
        rs = res[j['name']]; row = {'case': j['name']}
        if rs['out'] is None: print(f"{j['name']:22} NO OUTPUT json={rs['json']}"); continue
        cu, oz = G.read_out(rs['out'], j['M'], j['N']); fin = bool(np.all(np.isfinite(j['A'])) and np.all(np.isfinite(j['B'])) and (j['c0'] is None or np.all(np.isfinite(j['c0']))))
        cn = lambda x: (int(np.isnan(x).sum()), int(np.isinf(x).sum()))
        pat = bool(np.array_equal(np.isnan(oz), np.isnan(cu)) and np.array_equal(np.isinf(oz), np.isinf(cu)))
        both = np.isfinite(oz) & np.isfinite(cu); rel = float(np.max(np.abs(oz[both] - cu[both]) / np.maximum(np.abs(cu[both]), 1e-300))) if both.any() else None
        ev = E.evaluate(j, rs) if fin else {}
        tab = ev.get('gpu_table', {}); note = ''
        if fin: note = 'model_error: ' + tab['model_error'][:60] if 'model_error' in tab else (f"identical" if tab.get('identical') else f"ndiff={tab.get('ndiff')} maxULP={tab.get('max_ulp')}")
        else: note = f"finite elems max rel diff vs cuBLAS: {rel}"
        print(f"{j['name']:22} {'finite' if fin else 'NON-FINITE':>9} | {str(cn(oz)):>14} {str(cn(cu)):>14} | {str(pat):>23} | {note}")
        rows.append({'case': j['name'], 'finite_inputs': fin, 'ozblas_nan_inf': cn(oz), 'cublas_nan_inf': cn(cu), 'pattern_equal': pat, 'note': note, 'library_status': (rs['json'] or {}).get('status')})
    (Path(a.out) / 'specials_summary.json').write_text(json.dumps(rows, indent=1))

if __name__ == '__main__': main()
