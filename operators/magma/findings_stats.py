# Statistics over FINDINGS.csv: counts by category and defect density per code area (z source files of MAGMA 2.10.0).
# Usage: python3 findings_stats.py <magma-2.10.0 root>   (reads FINDINGS.csv next to this script)
import csv, sys, os, glob, re, collections
here = os.path.dirname(os.path.abspath(__file__)); root = sys.argv[1]
rows = [r for r in csv.DictReader(open(os.path.join(here, 'FINDINGS.csv')))]
real = [r for r in rows if r['evidence'] != 'not-reproduced']
def tab(key, rs=real):
    c = collections.Counter(r[key] for r in rs); t = sum(c.values())
    return [(k, v, 100.0 * v / t) for k, v in c.most_common()]
# code areas: z files only (s/c/d are generated from them)
def loc(files): return sum(sum(1 for l in open(f, errors='ignore') if l.strip()) for f in files)
src = glob.glob(os.path.join(root, 'src', 'z*.cpp')); mb = glob.glob(os.path.join(root, 'magmablas', 'z*.c*'))
isb = lambda f: 'batched' in os.path.basename(f); ism = lambda f: re.search(r'(_m|_mgpu|_mgpu_right|_ooc)\.cpp$', os.path.basename(f)) is not None
mixed_pat = re.compile(r'^(zc|zgerbt|zgerfs_nopiv|zgesv_rbt)')
isx = lambda f: mixed_pat.match(os.path.basename(f)) is not None and not isb(f)
mixed_orig = [os.path.join(root, 'src', f) for f in ['dfgmres_spd_gpu.cpp', 'dshposv_gmres_gpu.cpp', 'dxgesv_gmres_gpu.cpp', 'shpotrf_gpu.cpp', 'xhsgetrf_gpu.cpp', 'xshgetrf_gpu.cpp']]
zsparse = lambda d: [f for f in glob.glob(os.path.join(root, 'sparse', d, '*')) if re.match(r'^(magma_)?z', os.path.basename(f)) and re.search(r'\.(cpp|cu|c)$', f)]
area_files = {
  'dense drivers (src, not batched/multi-GPU/mixed)': [f for f in src if not isb(f) and not ism(f) and not isx(f)],
  'mixed precision / IR (zc*, rbt, gerfs; + 6 non-z originals)': [f for f in src if isx(f)] + mixed_orig,
  'batched / vbatched (src + magmablas)': [f for f in src if isb(f)] + [f for f in mb if isb(f)],
  'magmablas kernels (not batched)': [f for f in mb if not isb(f)],
  'multi-GPU / out-of-core (src *_m, *_mgpu)': [f for f in src if ism(f)],
  'sparse solvers + preconditioners (sparse/src)': zsparse('src'),
  'sparse control / IO / formats (sparse/control)': zsparse('control'),
  'sparse blas (sparse/blas)': zsparse('blas'),
}
area_of = {  # subsystem label -> code area
  'dense-LU': 0, 'dense-Cholesky': 0, 'dense-LDL/Hermitian-indefinite': 0, 'dense-QR/least-squares': 0, 'dense-eigen': 0, 'dense-SVD': 0,
  'dense-reduction (hetrd/gebrd/gehrd/he2hb)': 0, 'dense-inverse/triangular': 0, 'infrastructure': 0,
  'mixed-precision-IR': 1, 'batched/vbatched': 2, 'magmablas-kernels': 3, 'multi-GPU/out-of-core': 4,
  'sparse-solvers': 5, 'sparse-preconditioners': 5, 'sparse-control/IO/format': 6, 'sparse-blas': 7 }
names = list(area_files)
cnt = collections.Counter(names[area_of.get(r['subsystem'], 0)] for r in real)
out = []
out.append(f"rows={len(rows)} reproduced-or-reading={len(real)} not-reproduced={len(rows) - len(real)}")
for key in ['subsystem', 'defect_class', 'trigger', 'impact', 'evidence']:
    out.append(f"\n## {key}"); out += [f"  {k:45s} {v:4d}  {p:5.1f}%" for k, v, p in tab(key)]
out.append("\n## density by code area (z files, non-blank lines)")
for a in names:
    fs = area_files[a]; L = loc(fs); c = cnt[a]
    out.append(f"  {a:50s} files={len(fs):4d} lines={L:7d} findings={c:4d} per-kloc={1000.0 * c / max(L, 1):.2f}")
# status-plumbing vs numerical vs memory
groups = {'status plumbing': {'status-lost-or-overwritten', 'wrong-status-value', 'wrong-argument-number', 'missing-check'},
          'numerical result': {'silent-wrong-result', 'numerical-method'}, 'memory / size arithmetic': {'memory-safety', 'integer-overflow', 'workspace-sizing', 'launch-config'},
          'build / API': {'build-or-API'}, 'performance': {'performance-only'}}
out.append("\n## defect groups")
for g, s in groups.items():
    v = sum(1 for r in real if r['defect_class'] in s); out.append(f"  {g:28s} {v:4d}  {100.0 * v / len(real):5.1f}%")
out.append("\n## trigger x impact (silent-wrong-result only)")
for k, v in collections.Counter(r['trigger'] for r in real if r['impact'] == 'silent-wrong-result').most_common(): out.append(f"  {k:30s} {v}")
print('\n'.join(out))
