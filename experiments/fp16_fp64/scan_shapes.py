"""Classify cuBLAS half-GEMM kernels per shape from nvprof outputs, and align with the model.
Usage: python3 scan_shapes.py SCAN_DIR INPUT_DIR [--align-budget N] [--json OUT.json]

SCAN_DIR holds NAME.nvprof.txt (nvprof --print-gpu-summary stdout+stderr of the run) and
NAME.out.bin (GPU output); INPUT_DIR holds NAME.bin. Kernel class is by demangled name:
  tensor_core : name contains tensorop / wmma / s884 / h884 / s1688 / hmma
  cuda_core   : gemmSN / gemv / sgemm / simt
  other       : anything else (needs manual look)
The half-input GEMM kernels are the ones that are NOT cuozblas*/asum/dgemm/memcpy helpers.
Alignment with the CPU model (forward variant) runs only when M*N*K <= N (default 300000),
because the Python model is slow. Bit equality is a numerical observation only.
"""
from pathlib import Path
import argparse, json, re, struct, subprocess, sys

TC = re.compile(r'tensorop|wmma|[sh]884|[sh]1688|hmma', re.I)
CC = re.compile(r'gemmSN|gemv|sgemm|simt', re.I)
HELPER = re.compile(r'cuozblas|asum_kernel|CUDA mem|transpose_readWrite', re.I)
# Only half-input GEMM kernels count; the cuBLAS DGEMM baseline runs double kernels in the same process.
HALF = re.compile(r'__half|f16|fp16', re.I)

def kernels(text):
    rows = []
    for line in text.splitlines():
        m = re.match(r'\s*(?:GPU activities:)?\s*([\d.]+)%\s+([\d.]+[mun]?s)\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(.*)$', line)
        if m:
            rows.append({'calls': int(m.group(3)), 'name': m.group(4).strip()})
    return rows

def classify(name):
    if TC.search(name): return 'tensor_core'
    if CC.search(name): return 'cuda_core'
    return 'other'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scan_dir'); ap.add_argument('input_dir')
    ap.add_argument('--align-budget', type=int, default=300000)
    ap.add_argument('--json', dest='json_out')
    ap.add_argument('--reuse-align', help='earlier summary JSON whose align results are copied instead of recomputed')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    results = []
    old = {x['name']: x for x in json.loads(Path(args.reuse_align).read_text())} if args.reuse_align else {}
    for p in sorted(Path(args.scan_dir).glob('*.nvprof.txt'), key=lambda q: q.name):
        name = p.name[:-len('.nvprof.txt')]
        m, n, k = map(int, re.match(r's_(\d+)x(\d+)x(\d+)', name).groups())
        text = p.read_text(errors='replace')
        rec = {'shape': [m, n, k], 'name': name}
        run = re.search(r'\{"device".*\}', text)
        if run:
            j = json.loads(run.group(0)); rec.update({kk: j.get(kk) for kk in ('splits_A', 'splits_B', 'partial_gemms')})
        gk = [r for r in kernels(text) if HALF.search(r['name']) and not HELPER.search(r['name'])]
        rec['gemm_kernels'] = [{'class': classify(r['name']), 'calls': r['calls'], 'name': r['name'][:200]} for r in gk]
        classes = sorted({r['class'] for r in rec['gemm_kernels']})
        rec['half_gemm_class'] = classes[0] if len(classes) == 1 else ('mixed:' + '+'.join(classes) if classes else 'none_found')
        out = Path(args.scan_dir) / (name + '.out.bin')
        rec['ran'] = out.exists() and run is not None
        if name in old and 'align' in old[name]:
            rec['align'] = old[name]['align']
        elif rec['ran'] and m * n * k <= args.align_budget:
            tmp = Path(args.scan_dir) / (name + '.align.json')
            r = subprocess.run([sys.executable, str(here / 'align_gpu.py'), str(Path(args.input_dir) / (name + '.bin')),
                                str(out), '--json', str(tmp)], capture_output=True, text=True)
            if r.returncode == 0 and tmp.exists():
                rep = json.loads(tmp.read_text()); s = rep['summary']['gpu_vs_model']['forward']
                rec['align'] = {'forward_bit_identical': s['bit_identical'], 'total': s['total'], 'forward_max_ulp': s['max_ulp'],
                                'matching_variants': rep['model_variants_matching_gpu_on_all_elements']}
                tmp.unlink()
            else:
                rec['align'] = {'error': (r.stderr.strip().splitlines() or ['failed'])[-1]}
        results.append(rec)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
    print(f'{"shape":>16} {"splits":>6} {"gemms":>5}  {"half-GEMM kernel class":<24} align(forward)')
    for r in results:
        a = r.get('align')
        al = '-' if a is None else (a.get('error') or f'{a["forward_bit_identical"]}/{a["total"]} maxULP={a["forward_max_ulp"]}')
        sp = f'{r.get("splits_A")}x{r.get("splits_B")}' if r.get('splits_A') else '?'
        print(f'{"x".join(map(str, r["shape"])):>16} {sp:>6} {r.get("partial_gemms", "?")!s:>5}  {r["half_gemm_class"]:<24} {al}')

if __name__ == '__main__':
    main()
