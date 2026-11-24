"""Estimate how often the GPU log2 discrepancy can trigger, per input family (CPU only, no GPU run).
Replays the model's row splitting (libm tau trajectory, limit 12) and counts per matrix:
  A) 'gpu_too_large': a residual row max is EXACTLY 2^e with e in the measured bad set (GPU tau = true+1; libm model correct)
  B) 'formula_too_small': ceil(log2(max)) in double differs from the exact ceiling (libm AND GPU too small; model already reproduces)
Events are detected along the model trajectory, so counts after a first divergence are approximate.
Usage: python3 tau_event_rates.py BAD_POWERS.txt [--matrices N] [--rows M] [--k K] [--json OUT.json]
"""
import sys, re, math, json, argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bad'); ap.add_argument('--matrices', type=int, default=300)
    ap.add_argument('--rows', type=int, default=32); ap.add_argument('--k', type=int, default=64); ap.add_argument('--json')
    a = ap.parse_args()
    BAD = {int(x) for x in re.findall(r'^BAD (-?\d+)$', Path(a.bad).read_text(), re.M)}
    rho = math.ceil(53 - (24 - math.log2(max(2, a.k))) / 2)
    rng = np.random.default_rng(42)
    fams = {
      'uniform(-1,1) float64':      lambda: rng.uniform(-1, 1, (a.rows, a.k)),
      'gaussian float64':           lambda: rng.standard_normal((a.rows, a.k)),
      'uniform float32->double':    lambda: rng.uniform(-1, 1, (a.rows, a.k)).astype(np.float32).astype(np.float64),
      'uniform float16->double':    lambda: rng.uniform(-1, 1, (a.rows, a.k)).astype(np.float16).astype(np.float64),
      'multiples of 1/8 in [-1,1]': lambda: rng.integers(-8, 9, (a.rows, a.k)) / 8.0,
      'small integers -10..10':     lambda: rng.integers(-10, 11, (a.rows, a.k)).astype(np.float64),
      '0/1 matrix':                 lambda: rng.integers(0, 2, (a.rows, a.k)).astype(np.float64),
      'powers of two (random e)':   lambda: np.ldexp(rng.choice([-1.0, 1.0], (a.rows, a.k)), rng.integers(-8, 9, (a.rows, a.k))),
      'spread +-100 (float64)':     lambda: np.ldexp(rng.uniform(-1, 1, (a.rows, a.k)), rng.integers(-100, 101, (a.rows, a.k))),
    }
    out = {}
    for name, gen in fams.items():
        mats_A = mats_B = 0; rowsteps = A_ev = B_ev = 0
        for _ in range(a.matrices):
            r = gen().astype(np.float64); hitA = hitB = False
            for s in range(12):
                for i in range(r.shape[0]):
                    if not np.any(r[i]): continue
                    mx = float(np.max(np.abs(r[i]))); m, e = math.frexp(mx)
                    true = e - 1 if m == 0.5 else e; libm = math.ceil(math.log2(mx)); rowsteps += 1
                    if m == 0.5 and (e - 1) in BAD: A_ev += 1; hitA = True
                    if libm != true: B_ev += 1; hitB = True
                    sg = math.ldexp(.75, rho + libm); p = (r[i] + sg) - sg; r[i] = r[i] - p
            mats_A += hitA; mats_B += hitB
        out[name] = {'matrices': a.matrices, 'matrices_with_gpu_too_large': mats_A, 'matrices_with_formula_too_small': mats_B,
                     'row_steps': rowsteps, 'gpu_too_large_events': A_ev, 'formula_too_small_events': B_ev}
        print(f'{name:28} matrices w/ A-event {mats_A:4d}/{a.matrices}  w/ B-event {mats_B:4d}/{a.matrices}   events A={A_ev} B={B_ev} over {rowsteps} row-steps')
    if a.json: Path(a.json).write_text(json.dumps(out, indent=1))

if __name__ == '__main__': main()
