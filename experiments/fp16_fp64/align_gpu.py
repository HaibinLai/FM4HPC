"""Element-wise alignment of original-library GPU output with the CPU arithmetic model.
Usage: python3 align_gpu.py INPUT.bin GPU_OUTPUT.bin [RUN.log] [--limit N] [--json OUT.json]

For every output element compares (bit pattern and ULP distance):
  ozblas_gpu  vs each CPU model summation variant (forward/reverse/pairwise/math_fsum)
  ozblas_gpu  vs correctly rounded exact result
  cublas_gpu  vs correctly rounded exact result (independent baseline)
Also compares slice counts / partial-GEMM counts from RUN.log with the model.

Scope: this is a numerical comparison only. Bit equality with a model variant does
not prove the library implements that variant (different orders can coincide), and
says nothing about Tensor Core execution. The model limit defaults to 12, the value
the GPU entry point uses. Model variants other than 'forward' are controls.
"""
from pathlib import Path
import argparse, json, struct, sys
from fractions import Fraction as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import exact_dot, emulate

VARIANTS = ('forward', 'reverse', 'pairwise', 'math_fsum')

def ordered(x):
    """Map binary64 to integers monotonic in value (-0 and +0 coincide)."""
    i = int(np.float64(x).view(np.int64))
    return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)

def ulp_distance(a, b):
    return abs(ordered(a) - ordered(b))

def bits(x):
    return int(np.float64(x).view(np.uint64))

def read_pair(source, output):
    m, n, k = struct.unpack('<III', source[:12])
    if output[:12] != source[:12]:
        raise ValueError('shape/header mismatch')
    if len(source) != 12 + 8 * (m * k + k * n) or len(output) != 12 + 16 * m * n:
        raise ValueError('file length mismatch')
    d = np.frombuffer(source, dtype='<f8', offset=12)
    a = d[:m * k].reshape((m, k), order='F'); b = d[m * k:].reshape((k, n), order='F')
    d = np.frombuffer(output, dtype='<f8', offset=12)
    cublas = d[:m * n].reshape((m, n), order='F'); oz = d[m * n:].reshape((m, n), order='F')
    return a, b, cublas, oz

def read_runlog(path):
    if path is None or not Path(path).exists():
        return None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    return None

def compare_arrays(x, y):
    """Summary of element-wise difference between two float64 matrices."""
    ud = [ulp_distance(x[i, j], y[i, j]) for i in range(x.shape[0]) for j in range(x.shape[1])]
    same = [bits(x[i, j]) == bits(y[i, j]) for i in range(x.shape[0]) for j in range(x.shape[1])]
    return {'bit_identical': int(sum(same)), 'total': len(same),
            'max_ulp': max(ud), 'all_bit_identical': all(same)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('gpu_output'); ap.add_argument('runlog', nargs='?')
    ap.add_argument('--limit', type=int, default=12)
    ap.add_argument('--json', dest='json_out')
    args = ap.parse_args()

    a, b, cublas, oz = read_pair(Path(args.input).read_bytes(), Path(args.gpu_output).read_bytes())
    m, n = oz.shape
    for name, arr in (('cublas', cublas), ('ozblas', oz)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f'non-finite {name} output: cannot align')

    exact = [[exact_dot(a[i], b[:, j]) for j in range(n)] for i in range(m)]
    rounded = np.array([[float(v) for v in row] for row in exact])
    outs, meta = emulate(a, b, args.limit)

    elements = []
    for i in range(m):
        for j in range(n):
            row = {'index': [i, j], 'exact_rounded_hex': float(rounded[i, j]).hex(),
                   'ozblas_gpu_hex': float(oz[i, j]).hex(), 'cublas_gpu_hex': float(cublas[i, j]).hex(),
                   'ozblas_err_vs_exact': float(abs(F(float(oz[i, j])) - exact[i][j])),
                   'ozblas_ulp_vs_rounded_exact': ulp_distance(oz[i, j], rounded[i, j]),
                   'cublas_ulp_vs_rounded_exact': ulp_distance(cublas[i, j], rounded[i, j]),
                   'ozblas_ulp_vs_model': {v: ulp_distance(oz[i, j], outs[v][i, j]) for v in VARIANTS}}
            elements.append(row)

    summary = {'gpu_vs_model': {v: compare_arrays(oz, outs[v]) for v in VARIANTS},
               'gpu_vs_rounded_exact': compare_arrays(oz, rounded),
               'cublas_vs_rounded_exact': compare_arrays(cublas, rounded)}
    # Variants that match the GPU on every element (coincidence is possible, not proof).
    matching = [v for v in VARIANTS if summary['gpu_vs_model'][v]['all_bit_identical']]

    log = read_runlog(args.runlog)
    structure = {'model': {'splits_A': meta['A']['slices'], 'splits_B': meta['B']['slices'],
                           'partial_gemms': meta['partial_gemms'],
                           'zero_padded_row_steps_A': meta['A']['zero_padded_row_steps'],
                           'zero_padded_row_steps_B': meta['B']['zero_padded_row_steps']}}
    if log:
        structure['gpu'] = {k: log.get(k) for k in ('splits_A', 'splits_B', 'partial_gemms',
                            'sumMode', 'tensor_core_execution_confirmed')}
        structure['slices_match'] = (log.get('splits_A') == meta['A']['slices'] and
                                     log.get('splits_B') == meta['B']['slices'] and
                                     log.get('partial_gemms') == meta['partial_gemms'])
    report = {'input': args.input, 'gpu_output': args.gpu_output, 'model_limit': args.limit,
              'shape_MNK': [m, n, a.shape[1]], 'structure': structure, 'summary': summary,
              'model_variants_matching_gpu_on_all_elements': matching, 'elements': elements,
              'caveat': 'numerical comparison only; says nothing about Tensor Core execution or library internals'}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))

    print(f'{Path(args.input).name}: M,N,K={m},{n},{a.shape[1]}  model limit={args.limit}')
    print('  slices:', structure.get('gpu', 'no run log'), '| model', structure['model'],
          '| match:', structure.get('slices_match'))
    for v in VARIANTS:
        s = summary['gpu_vs_model'][v]
        print(f'  GPU vs model[{v:9}] bit-identical {s["bit_identical"]}/{s["total"]}  max ULP {s["max_ulp"]}')
    for key in ('gpu_vs_rounded_exact', 'cublas_vs_rounded_exact'):
        s = summary[key]
        print(f'  {key:24} bit-identical {s["bit_identical"]}/{s["total"]}  max ULP {s["max_ulp"]}')
    print('  variants matching all elements:', matching or 'none')

if __name__ == '__main__':
    main()
