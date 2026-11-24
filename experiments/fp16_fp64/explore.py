"""Exploration suites: generate jobs, run on the V100 node (gpubatch), compare GPU with the fast CPU model.
Usage: python3 explore.py SUITE [--device 0] [--seeds 6] [--out DIR]
Suites: multiseed, shapes, maxsplit, transposes, alphabeta, specials
The model is evaluated twice per job: 'libm' (tau=ceil(math.log2 max)) and 'gpu_style' (measured GPU log2 on exact
powers of two, see HANDOFF 11.5). Everything is a finite-sample observation, not a proof.
"""
import sys, re, math, json, time, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpubatch as G, analysis as AN, fastmodel as FM

HERE = Path(__file__).resolve().parent
BAD = {int(x) for x in re.findall(r'^BAD (-?\d+)$', (HERE / 'gpu_results/v100_gpu0_extreme/gpu_log2_bad_powers.txt').read_text(), re.M)}
GPU_LOG2 = FM.make_gpu_log2(BAD)
TAB_PATH = HERE / 'gpu_results/v100_gpu0_extreme/gpu_log2_table.txt'
GPU_TABLE = FM.make_gpu_log2_table(FM.load_gpu_tau_table(TAB_PATH)) if TAB_PATH.exists() else GPU_LOG2

def fam(name, rng, shape):
    if name == 'uniform':   return rng.uniform(-1, 1, shape)
    if name == 'gaussian':  return rng.standard_normal(shape)
    if name == 'f16':       return rng.uniform(-1, 1, shape).astype(np.float16).astype(np.float64)
    if name == 'f32':       return rng.uniform(-1, 1, shape).astype(np.float32).astype(np.float64)
    if name == 'pow2':      return np.ldexp(rng.choice([-1.0, 1.0], shape), rng.integers(-8, 9, shape))
    if name == 'dyadic':    return rng.integers(-8, 9, shape) / 8.0
    if name == 'ints':      return rng.integers(-10, 11, shape).astype(np.float64)
    if name == 'spread30':  return np.ldexp(rng.uniform(-1, 1, shape), rng.integers(-30, 31, shape))
    if name == 'spread100': return np.ldexp(rng.uniform(-1, 1, shape), rng.integers(-100, 101, shape))
    if name == 'neg1000':   return np.ldexp(rng.uniform(-1, 1, shape), rng.integers(-1000, 1, shape))
    if name == 'wide500':   return np.ldexp(rng.uniform(-1, 1, shape), rng.integers(-500, 501, shape))
    if name == 'pos900':    return np.ldexp(rng.uniform(-1, 1, shape), rng.integers(0, 901, shape))
    if name == 'rowscale':  return rng.uniform(-1, 1, shape) * np.ldexp(1.0, rng.integers(-50, 51, (shape[0], 1)))
    raise KeyError(name)
FAMILIES = ['uniform', 'gaussian', 'f16', 'f32', 'pow2', 'dyadic', 'ints', 'spread30', 'spread100', 'rowscale']

def make_job(name, M, N, K, fa, fb, seed, ta='N', tb='N', alpha=1.0, beta=0.0, c0kind=None, maxsplit=12, repeat=1):
    rng = np.random.default_rng(seed)
    A = fam(fa, rng, (M, K) if ta == 'N' else (K, M)); B = fam(fb, rng, (K, N) if tb == 'N' else (N, K))
    c0 = None if c0kind is None else (rng.uniform(-1, 1, (M, N)) if c0kind == 'uniform' else np.full((M, N), np.nan))
    return {'name': name, 'M': M, 'N': N, 'K': K, 'A': A, 'B': B, 'ta': ta, 'tb': tb, 'alpha': alpha, 'beta': beta, 'c0': c0,
            'maxsplit': maxsplit, 'repeat': repeat, 'meta': {'fa': fa, 'fb': fb, 'seed': seed}}

def evaluate(job, res):
    cfg = f"sm{job.get('sumMode', 2)}_f{job.get('fast', 0)}_b{job.get('batched', 0)}_e{job.get('epsmode', 0)}_r{job.get('repro', 1)}"
    r = {'cfg': cfg, 'name': job['name'], 'shape': [job['M'], job['N'], job['K']], 'ta': job['ta'], 'tb': job['tb'], 'alpha': job['alpha'], 'beta': job['beta'],
         'maxsplit': job['maxsplit'], **job['meta']}
    if res['out'] is None: r['status'] = 'no_output'; r['json'] = res['json']; return r
    oj = res['json'] or {}; r['library_status'] = oj.get('status'); r['pad'] = oj.get('pad'); r['pad_clobbered_oz'] = oj.get('pad_clobbered_oz'); r['pad_clobbered_cublas'] = oj.get('pad_clobbered_cublas'); r['inputs_modified'] = oj.get('inputs_modified'); r['repeat_bitwise_diffs'] = oj.get('repeat_bitwise_diffs')
    cu, oz = G.read_out(res['out'], job['M'], job['N'])
    opA = job['A'] if job['ta'] == 'N' else job['A'].T; opB = job['B'] if job['tb'] == 'N' else job['B'].T
    finite_in = bool(np.all(np.isfinite(opA)) and np.all(np.isfinite(opB)) and (job['c0'] is None or np.all(np.isfinite(job['c0']))))
    r['gpu_nonfinite'] = int((~np.isfinite(oz)).sum()); r['cublas_nonfinite'] = int((~np.isfinite(cu)).sum()); r['finite_inputs'] = finite_in
    if not finite_in: return r
    models = {}
    for tag, fn in (('libm', math.log2), ('gpu_style', GPU_LOG2), ('gpu_table', GPU_TABLE)):
        try:
            t, meta = FM.emulate_fast(opA, opB, job['maxsplit'], fn, fast=bool(job.get('fast', 0)), rho_mode=int(job.get('epsmode', 0)))
            out = AN.apply_alpha_beta(t, job['alpha'], job['beta'], job['c0'])
            cmpr = AN.compare(oz, out); cmpr['splits_match'] = (oj.get('splits_A') == max(1, meta['A']['slices']) and oj.get('splits_B') == max(1, meta['B']['slices']))   # the library always performs >=1 split pass (all-zero operand -> 1 zero slice)
            cmpr['model_splits'] = [meta['A']['slices'], meta['B']['slices']]; r[tag] = cmpr; models[tag] = out
        except Exception as ex:
            r[tag] = {'model_error': f'{type(ex).__name__}: {ex}'}
    r['gpu_splits'] = [oj.get('splits_A'), oj.get('splits_B')]
    outs = {'ozblas': oz, 'cublas': cu}; outs.update({f'model_{k}': v for k, v in models.items()})
    r['sampled_rel_err'] = AN.sampled_errors(opA, opB, job['alpha'], job['beta'], job['c0'], outs, nsamp=24, seed=job['meta']['seed'])
    return r

def summarize(rows, key):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows: g[key(r)].append(r)
    print(f"{'group':26} {'n':>4} {'GPU==libm':>10} {'GPU==gpu_style':>14} {'GPU==table':>11} {'maxULP(libm)':>12} {'maxULP(gs)':>10} {'maxULP(tab)':>11} {'splitsOK':>8} {'modelErr':>8} {'noOut':>5}")
    for k, rs in sorted(g.items(), key=lambda t: str(t[0])):
        ok = [r for r in rs if 'libm' in r and 'identical' in r.get('libm', {}) and 'identical' in r.get('gpu_style', {})]
        nl = sum(r['libm']['identical'] for r in ok); ng = sum(r['gpu_style']['identical'] for r in ok)
        ul = max((r['libm']['max_ulp'] for r in ok), default=0); ug = max((r['gpu_style']['max_ulp'] for r in ok), default=0)
        so = sum(r['gpu_style']['splits_match'] for r in ok)
        nt = sum(r.get('gpu_table', {}).get('identical', False) for r in ok); ut = max((r.get('gpu_table', {}).get('max_ulp', 0) for r in ok), default=0)
        me = sum(1 for r in rs if any('model_error' in r.get(t, {}) for t in ('libm', 'gpu_style'))); no = sum(1 for r in rs if r.get('status') == 'no_output')
        print(f"{str(k):26} {len(rs):4d} {nl:5d}/{len(ok):<4d} {ng:9d}/{len(ok):<4d} {nt:6d}/{len(ok):<4d} {ul:12d} {ug:10d} {ut:11d} {so:4d}/{len(ok):<3d} {me:8d} {no:5d}")

def suite_jobs(name, seeds):
    jobs = []; s0 = 7000
    if name == 'multiseed':
        shapes = [(4, 4, 16), (8, 8, 32), (16, 16, 64), (32, 32, 128), (64, 64, 256), (48, 32, 100), (24, 40, 17)]
        for f in FAMILIES:
            for (M, N, K) in shapes:
                for s in range(seeds):
                    jobs.append(make_job(f'{f}_{M}x{N}x{K}_s{s}', M, N, K, f, 'uniform' if f != 'gaussian' else 'gaussian', s0 + len(jobs)))
    elif name == 'shapes':
        for (M, N, K) in [(128, 128, 512), (256, 256, 256), (256, 256, 1024), (512, 512, 512), (128, 128, 4096), (64, 64, 16384), (32, 32, 65536), (1024, 1024, 128), (4, 2048, 64), (2048, 4, 64), (1, 64, 64), (64, 1, 64), (1, 1, 64), (64, 64, 1), (2, 2, 3), (255, 257, 129)]:
            for f in ('uniform', 'gaussian', 'spread30'):
                for s in range(max(1, seeds // 3)):
                    jobs.append(make_job(f'{f}_{M}x{N}x{K}_s{s}', M, N, K, f, 'uniform' if f != 'gaussian' else 'gaussian', s0 + len(jobs)))
    elif name == 'maxsplit':
        for ms in (1, 2, 3, 4, 5, 6, 8, 12):
            for f in ('uniform', 'gaussian', 'spread100', 'f16'):
                for s in range(seeds):
                    jobs.append(make_job(f'{f}_ms{ms}_s{s}', 16, 16, 64, f, 'uniform', s0 + len(jobs), maxsplit=ms))
    elif name == 'transposes':
        for ta in 'NT':
            for tb in 'NT':
                for (M, N, K) in [(8, 8, 32), (16, 24, 64), (33, 17, 100), (64, 64, 256)]:
                    for f in ('uniform', 'gaussian', 'spread30'):
                        for s in range(max(1, seeds // 2)):
                            jobs.append(make_job(f'{f}_{ta}{tb}_{M}x{N}x{K}_s{s}', M, N, K, f, 'uniform', s0 + len(jobs), ta=ta, tb=tb))
    elif name == 'alphabeta':
        combos = [(1.0, 0.0, None), (2.0, 0.0, None), (0.5, 0.0, None), (-1.0, 0.0, None), (1.0, 1.0, 'uniform'), (1.0, -1.0, 'uniform'), (0.3, 0.7, 'uniform'),
                  (2.0, 0.5, 'uniform'), (0.0, 1.0, 'uniform'), (0.0, 0.0, None), (1.0, 0.0, 'uniform'), (1e-3, 1e3, 'uniform')]
        for (al, be, c0k) in combos:
            for (M, N, K) in [(8, 8, 32), (32, 32, 128)]:
                for f in ('uniform', 'gaussian'):
                    for s in range(max(1, seeds // 3)):
                        jobs.append(make_job(f'{f}_a{al}_b{be}_{c0k}_{M}x{N}x{K}_s{s}', M, N, K, f, 'uniform', s0 + len(jobs), alpha=al, beta=be, c0kind=c0k))
    elif name == 'nearpow':
        tab = FM.load_gpu_tau_table(TAB_PATH); pairs = []
        for (ex, k), t in tab.items():
            if not -900 <= ex <= 900 or k == 0: continue
            x = math.ldexp(1.0, ex)
            for _ in range(abs(k)): x = math.nextafter(x, math.inf if k > 0 else 0.0)
            if math.ceil(math.log2(x)) != t: pairs.append((ex, k, x))
        rng0 = np.random.default_rng(5); order = rng0.permutation(len(pairs)); pairs = [pairs[i] for i in order[:max(60, seeds * 60)]]
        print(f'nearpow: {len(pairs)} (e,k) pairs where GPU tau != libm tau (of total differing)', flush=True)
        for (ex, k, x) in pairs:
            M, N, K = 4, 4, 16; rng = np.random.default_rng(s0 + len(jobs))
            A = rng.uniform(-1, 1, (M, K)) * 0.9 * x; A[:, 0] = x; B = rng.uniform(-1, 1, (K, N))
            jobs.append({'name': f'np_e{ex}_k{k}', 'M': M, 'N': N, 'K': K, 'A': A, 'B': B, 'ta': 'N', 'tb': 'N', 'alpha': 1.0, 'beta': 0.0, 'c0': None,
                         'maxsplit': 12, 'repeat': 1, 'meta': {'fa': f'near2^e{"+" if k > 0 else ""}k', 'fb': 'uniform', 'seed': ex}})
    elif name == 'deepexp':
        for f in ('neg1000', 'wide500', 'pos900'):
            for (M, N, K) in [(8, 8, 32), (16, 16, 64), (32, 24, 128)]:
                for s in range(seeds):
                    jobs.append(make_job(f'{f}_{M}x{N}x{K}_s{s}', M, N, K, f, f if f == 'wide500' else 'uniform', s0 + len(jobs)))
    elif name == 'pad':
        for pad in (0, 1, 3, 8, 17):
            for ta in 'NT':
                for tb in 'NT':
                    for (M, N, K) in [(8, 8, 32), (16, 24, 64), (33, 17, 100)]:
                        for f in ('uniform', 'gaussian'):
                            for s in range(max(1, seeds // 3)):
                                j = make_job(f'{f}_p{pad}_{ta}{tb}_{M}x{N}x{K}_s{s}', M, N, K, f, 'uniform', s0 + len(jobs), ta=ta, tb=tb); j['pad'] = pad; jobs.append(j)
        for pad in (0, 5):
            for (al, be) in ((0.7, 0.3), (2.0, 1.0)):
                for s in range(3):
                    j = make_job(f'uniform_p{pad}_ab{al}_{be}_s{s}', 16, 16, 64, 'uniform', 'uniform', s0 + len(jobs), alpha=al, beta=be, c0kind='uniform'); j['pad'] = pad; jobs.append(j)
    elif name == 'modes':
        cfgs = [('base', {}), ('sum0', {'sumMode': 0}), ('batched', {'batched': 1}), ('batched_sum0', {'batched': 1, 'sumMode': 0}), ('fast1', {'fast': 1}),
                ('fast1_sum0', {'fast': 1, 'sumMode': 0}), ('eps2', {'epsmode': 2}), ('eps9', {'epsmode': 9}), ('eps9_fast1', {'epsmode': 9, 'fast': 1}), ('repro0', {'repro': 0})]
        for cname, cfg in cfgs:
            for (M, N, K) in [(8, 8, 32), (16, 16, 64), (32, 32, 128)]:
                for f in ('uniform', 'gaussian', 'spread30', 'spread100'):
                    for s in range(max(1, seeds // 2)):
                        j = make_job(f'{f}_{cname}_{M}x{N}x{K}_s{s}', M, N, K, f, 'uniform', 9000 + 17 * (len(jobs) % 36) + s); j.update(cfg); jobs.append(j)
    else: raise KeyError(name)
    return jobs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('suite'); ap.add_argument('--device', default='0'); ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--out', default=str(HERE / 'gpu_results/explore')); ap.add_argument('--ws', type=int, default=1024)
    a = ap.parse_args(); t0 = time.time()
    jobs = suite_jobs(a.suite, a.seeds); print(f'{a.suite}: {len(jobs)} jobs, device {a.device}', flush=True)
    res, local = G.run_jobs(jobs, f'{a.suite}', a.device, a.out, ws_mb=a.ws); print(f'GPU done in {time.time()-t0:.0f}s; outputs {sum(1 for v in res.values() if v["out"])}/{len(jobs)}', flush=True)
    rows = [evaluate(j, res[j['name']]) for j in jobs]; print(f'analysis done in {time.time()-t0:.0f}s', flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / f'{a.suite}_summary.json').write_text(json.dumps(rows, indent=1, default=str))
    print('--- by family'); summarize(rows, lambda r: r.get('fa'))
    print('--- by shape/ta-tb/maxsplit'); summarize(rows, (lambda r: r['cfg']) if a.suite == 'modes' else (lambda r: ('pad', r['pad'], 'clobbered_oz', sum(x['pad_clobbered_oz'] or 0 for x in rows if x['pad'] == r['pad']), 'clobbered_cublas', sum(x['pad_clobbered_cublas'] or 0 for x in rows if x['pad'] == r['pad']), 'inputs_modified', sum(x['inputs_modified'] or 0 for x in rows if x['pad'] == r['pad']))) if a.suite == 'pad' else (lambda r: (tuple(r['shape']), r['ta'] + r['tb'], r['maxsplit']) if a.suite != 'multiseed' else tuple(r['shape'])))

if __name__ == '__main__': main()
