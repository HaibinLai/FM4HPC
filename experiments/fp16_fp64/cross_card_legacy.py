"""Cross-card check WITHOUT touching other GPUs: rerun the inputs that were run earlier on GPU 6 (original 4 + mid_64 + 33-shape scan)
on GPU 0 and compare with the saved GPU 6 outputs bit-for-bit. Also repeat-run determinism on GPU 0 (--repeat R in one process).
Usage: python3 cross_card_legacy.py SCAN_INPUT_DIR [--device 0] [--repeat 20]
Note: the GPU 6 outputs were produced by gpu_compare.cu (same header + cuBLAS + OzBLAS layout, nSplitMax=12, same library build).
"""
import sys, struct, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpubatch as G

def load(path):
    b = Path(path).read_bytes(); m, n, k = struct.unpack('<III', b[:12]); d = np.frombuffer(b, dtype='<f8', offset=12)
    return m, n, k, d[:m*k].reshape((m, k), order='F').copy(), d[m*k:].reshape((k, n), order='F').copy()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('scan_in'); ap.add_argument('--device', default='0'); ap.add_argument('--repeat', type=int, default=20); a = ap.parse_args()
    here = Path(__file__).resolve().parent; g6 = here / 'gpu_results/v100_gpu6'; pairs = []
    for nm in ('uniform_k16', 'uniform_k64', 'cancellation', 'spread_40'): pairs.append((nm, here / f'{nm}.bin', g6 / f'{nm}.bin'))
    pairs.append(('mid_64', here / 'mid_64.bin', g6 / 'mid_64.bin'))
    for f in sorted(Path(a.scan_in).glob('s_*.bin')): pairs.append((f.stem, f, g6 / 'scan' / f'{f.stem}.out.bin'))
    jobs = []; ref = {}
    for nm, inp, out in pairs:
        if not out.exists(): continue
        m, n, k, A, B = load(inp); jobs.append({'name': nm, 'M': m, 'N': n, 'K': k, 'A': A, 'B': B, 'repeat': a.repeat}); ref[nm] = out
    res, local = G.run_jobs(jobs, 'legacy_vs_gpu6', a.device, here / 'gpu_results/explore')
    n = noz = ncu = rep = 0; diffs = []
    for j in jobs:
        rs = res[j['name']]
        if rs['out'] is None: diffs.append((j['name'], 'no output')); continue
        ca, oa = G.read_out(rs['out'], j['M'], j['N']); cb, ob = G.read_out(ref[j['name']], j['M'], j['N']); n += 1
        d_oz, d_cu = oa.tobytes() != ob.tobytes(), ca.tobytes() != cb.tobytes(); noz += d_oz; ncu += d_cu
        rd = (rs['json'] or {}).get('repeat_bitwise_diffs', None); rep += int(bool(rd))
        if d_oz or d_cu: diffs.append((j['name'], 'oz' if d_oz else '', 'cublas' if d_cu else ''))
    print(f'GPU {a.device} rerun vs saved GPU 6 outputs: compared {n}; OzBLAS differ {noz}; cuBLAS differ {ncu}; jobs with repeat-run diffs ({a.repeat}x): {rep}')
    for d in diffs[:12]: print('  ', d)
    (here / 'gpu_results/explore/cross_card_legacy.json').write_text(json.dumps({'compared': n, 'ozblas_differ': noz, 'cublas_differ': ncu, 'repeat': a.repeat, 'repeat_diff_jobs': rep, 'diff_list': diffs}, indent=1))

if __name__ == '__main__': main()
