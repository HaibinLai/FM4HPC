"""Random fuzz campaign: random shapes/families/transposes/alpha-beta/split caps/library modes/padding.
GPU (original OzBLAS D_H) vs CPU fast model with the measured GPU-style log2 table. Any non-identical job is listed with its full spec.
Usage: python3 fuzz.py [--device 0] [--n 1500] [--seed 1] [--batch 500] [--maxdim 128] [--maxk 600]
Specs are fully determined by (seed, index), so a mismatch can be re-created.  Finite-sample evidence, not proof.
"""
import sys, math, json, argparse, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore as E, gpubatch as G

AFAM = ['uniform', 'gaussian', 'f16', 'f32', 'pow2', 'dyadic', 'ints', 'spread30', 'spread100', 'rowscale', 'neg1000', 'wide500']
BFAM = ['uniform', 'gaussian', 'f16', 'spread30', 'dyadic']
AB = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.5, 0.0), (-1.0, 0.0), (1.0, 1.0), (0.3, 0.7), (-2.0, 0.5), (1e-3, 1e3)]

def spec(seed, idx, maxdim=128, maxk=600):
    r = np.random.default_rng([seed, idx]); lu = lambda lo, hi: int(round(math.exp(r.uniform(math.log(lo), math.log(hi)))))
    M, N, K = lu(1, maxdim), lu(1, maxdim), lu(1, maxk); al, be = AB[int(r.integers(len(AB)))]
    if (al != 1.0 or be != 0.0) and M * N > 4096: al, be = 1.0, 0.0
    fa = AFAM[int(r.integers(len(AFAM)))]; fb = BFAM[int(r.integers(len(BFAM)))]
    if fa == 'wide500': fb = 'uniform'        # keep products finite
    return {'idx': idx, 'M': M, 'N': N, 'K': K, 'fa': fa, 'fb': fb, 'ta': 'NT'[int(r.integers(2))], 'tb': 'NT'[int(r.integers(2))], 'alpha': al, 'beta': be,
            'c0kind': 'uniform' if be != 0.0 else None, 'maxsplit': int(r.choice([1, 2, 3, 4, 6, 8, 12, 12, 12, 16, 20, 30])),
            'sumMode': int(r.choice([0, 2])), 'fast': int(r.random() < 0.25), 'batched': int(r.random() < 0.25), 'epsmode': int(r.choice([0, 0, 9])),
            'pad': int(r.choice([0, 0, 0, 1, 3, 5])), 'jobseed': int(r.integers(1, 2**31))}

def build(sp):
    j = E.make_job(f"fz{sp['idx']}", sp['M'], sp['N'], sp['K'], sp['fa'], sp['fb'], sp['jobseed'], ta=sp['ta'], tb=sp['tb'], alpha=sp['alpha'], beta=sp['beta'], c0kind=sp['c0kind'], maxsplit=sp['maxsplit'])
    for k in ('sumMode', 'fast', 'batched', 'epsmode', 'pad'): j[k] = sp[k]
    return j

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='0'); ap.add_argument('--n', type=int, default=1500); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--maxdim', type=int, default=128); ap.add_argument('--maxk', type=int, default=600); a = ap.parse_args()
    out = Path(__file__).resolve().parent / 'gpu_results/explore'; t0 = time.time(); specs = [spec(a.seed, i, a.maxdim, a.maxk) for i in range(a.n)]
    tot = ident = ident_libm = noout = merr = splits_bad = 0; bad = []; lib_bad = 0; by_fam = {}
    for b0 in range(0, a.n, a.batch):
        chunk = specs[b0:b0 + a.batch]; jobs = [build(s) for s in chunk]
        res, _ = G.run_jobs(jobs, f'fuzz{a.seed}_{b0}', a.device, out, ws_mb=1024)
        for sp, j in zip(chunk, jobs):
            r = E.evaluate(j, res[j['name']]); tot += 1
            if r.get('status') == 'no_output': noout += 1; bad.append({'spec': sp, 'why': 'no_output', 'json': r.get('json')}); continue
            g = r.get('gpu_table', {}); L = r.get('libm', {})
            if 'model_error' in g: merr += 1; bad.append({'spec': sp, 'why': 'model_error ' + g['model_error'][:80]}); continue
            if r.get('library_status') not in (0, None): lib_bad += 1
            ok = g['identical']; ident += ok; ident_libm += int(L.get('identical', False)); splits_bad += int(not g['splits_match'])
            d = by_fam.setdefault(sp['fa'], [0, 0]); d[0] += 1; d[1] += int(ok)
            if not ok or not g['splits_match']: bad.append({'spec': sp, 'why': f"ndiff={g['ndiff']}/{g['total']} maxULP={g['max_ulp']} splits_match={g['splits_match']}"})
        print(f'[{time.time()-t0:.0f}s] done {tot}/{a.n}: GPU==table {ident}, GPU==libm {ident_libm}, model_error {merr}, no_output {noout}, mismatches so far {len(bad)}', flush=True)
    print(f'\nFUZZ seed={a.seed}: {tot} jobs; GPU==table model bit-identical {ident}; splits mismatch {splits_bad}; model errors {merr}; no output {noout}; library status != 0: {lib_bad}')
    print('GPU==libm model (no GPU-log2 table):', ident_libm); print('per A-family (n, identical):', by_fam)
    for x in bad[:12]: print('MISMATCH', json.dumps(x))
    (out / f'fuzz_seed{a.seed}.json').write_text(json.dumps({'n': tot, 'identical': ident, 'libm_identical': ident_libm, 'model_errors': merr, 'no_output': noout, 'bad': bad}, indent=1, default=str))

if __name__ == '__main__': main()
