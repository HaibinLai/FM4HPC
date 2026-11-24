"""Does the nSplitMax cap (12 vs library default 30) explain accuracy loss at very large K? GPU vs model + accuracy vs cuBLAS.
Usage: python3 bigk_cap.py [--device 0]
"""
import sys, math, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore as E, gpubatch as G

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='0'); a = ap.parse_args(); t0 = time.time()
    jobs = [E.make_job(f'{f}_K{K}_ms{ms}', 2, 2, K, f, 'uniform' if f != 'gaussian' else 'gaussian', 666000 + len(jobs_) , maxsplit=ms)
            for jobs_ in [[]] for K in (2**17, 2**20, 2**22) for f in ('uniform', 'gaussian') for ms in (12, 30)]
    res, _ = G.run_jobs(jobs, 'bigK_ms', a.device, Path(__file__).resolve().parent / 'gpu_results/explore', ws_mb=2048); print(f'GPU {time.time()-t0:.0f}s', flush=True)
    rows = []; fmt = lambda v: 'None' if v is None else f'{v:.2e}'
    print('K        maxsplit  used(A,B)  GPU==model  OzBLAS err  cuBLAS err  family', flush=True)
    for j in jobs:
        r = E.evaluate(j, res[j['name']]); rows.append(r); gs = r.get('gpu_table', {}); se = r.get('sampled_rel_err', {})
        print(f"{r['shape'][2]:<8} {r['maxsplit']:<9} {str(r.get('gpu_splits')):<10} {str(gs.get('identical', gs.get('model_error', '?')))[:10]:<11} {fmt(se.get('ozblas')):<11} {fmt(se.get('cublas')):<11} {r['fa']}   [{time.time()-t0:.0f}s]", flush=True)
    (Path(__file__).resolve().parent / 'gpu_results/explore/bigK_maxsplit_summary.json').write_text(json.dumps(rows, indent=1, default=str))

if __name__ == '__main__': main()
