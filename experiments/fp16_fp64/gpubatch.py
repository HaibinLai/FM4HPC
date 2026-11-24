"""Run many gpu_general jobs on the compute node in one process and fetch the outputs.

A job is a dict: {'name': str, 'M','N','K': int, 'A': ndarray (as stored), 'B': ndarray (as stored),
                  'ta':'N'|'T', 'tb':'N'|'T', 'alpha': float, 'beta': float, 'c0': ndarray|None,
                  'maxsplit': int (default 12), 'repeat': int (default 1)}
Arrays are float64; A/B are given in STORED (column-major) layout shapes:
  A: (M,K) if ta=='N' else (K,M);  B: (K,N) if tb=='N' else (N,K)   (numpy arrays, converted with order='F').
Connection: ssh s82 -> ssh -i <KEY_PATH> haibin@<NODE_IP> (see HANDOFF.md section 2).
Remote workspace: /ssd/haibin/lean4hpc-fp16/batch/TAG/ (deleted after the outputs are fetched unless keep_remote=True).
Requires /ssd/haibin/lean4hpc-fp16/dbg/gpu_general (built from gpu_general.cu).
"""
import io, json, struct, subprocess, tarfile, shutil
from pathlib import Path
import numpy as np

HOP = ['ssh', '-o', 'ConnectTimeout=20', 's82']
INNER = 'ssh -i <KEY_PATH> -o ConnectTimeout=10 haibin@<NODE_IP>'
# Data policy (user instruction 2026-11-23): put data under /ssd/haibin, avoid /home; delete data when done.
REMOTE_ROOT = '/ssd/haibin/lean4hpc-fp16'

def _remote(cmd, stdin=None, capture=True, timeout=3000):
    full = HOP + [f"{INNER} '{cmd}'"]
    return subprocess.run(full, input=stdin, capture_output=capture, timeout=timeout)

def _bin(job):
    M, N, K = job['M'], job['N'], job['K']
    a = np.asarray(job['A'], dtype='<f8'); b = np.asarray(job['B'], dtype='<f8')
    ea = (M, K) if job.get('ta', 'N') == 'N' else (K, M); eb = (K, N) if job.get('tb', 'N') == 'N' else (N, K)
    assert a.shape == ea and b.shape == eb, (a.shape, ea, b.shape, eb)
    return struct.pack('<III', M, N, K) + a.tobytes(order='F') + b.tobytes(order='F')

def run_jobs(jobs, tag, device, outdir, ws_mb=1024, rerun_missing_only=False, keep_remote=False):
    """Returns {name: {'out': Path, 'json': dict|None}}. device = physical GPU index (CUDA_VISIBLE_DEVICES)."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(); lines = []
    with tarfile.open(fileobj=buf, mode='w') as tf:
        def add(name, data):
            ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
        for j in jobs:
            n = j['name']; add(f'in/{n}.bin', _bin(j)); opt = f"--ta {j.get('ta','N')} --tb {j.get('tb','N')} --alpha {j.get('alpha',1.0)!r} --beta {j.get('beta',0.0)!r} --ws {ws_mb} --repeat {j.get('repeat',1)} --maxsplit {j.get('maxsplit',12)}"
            for key, flag in (('sumMode', '--sumMode'), ('fast', '--fast'), ('batched', '--batched'), ('epsmode', '--epsmode'), ('repro', '--repro'), ('pad', '--pad')):
                if j.get(key) is not None: opt += f' {flag} {j[key]}'
            if j.get('c0') is not None:
                add(f'in/{n}.c0', np.asarray(j['c0'], dtype='<f8').tobytes(order='F')); opt += f' --c0 in/{n}.c0'
            lines.append(f'in/{n}.bin out/{n}.bin {opt}')
        add('list.txt', ('\n'.join(lines) + '\n').encode())
        add('run.sh', f'export CUDA_VISIBLE_DEVICES={device}\nmkdir -p out\n{REMOTE_ROOT}/dbg/gpu_general --batch list.txt > results.jsonl 2> stderr.txt\necho $? > rc.txt\n'.encode())
    d = f'{REMOTE_ROOT}/batch/{tag}'
    r = _remote(f'rm -rf {d} && mkdir -p {d} && tar xf - -C {d} 2>/dev/null && cd {d} && bash run.sh', stdin=buf.getvalue())
    if r.returncode != 0: print(f'[gpubatch] WARNING tag={tag}: remote run exited {r.returncode}: {r.stderr.decode(errors="replace")[-300:]}', flush=True)
    r2 = _remote(f'cd {d} && tar cf - out results.jsonl stderr.txt rc.txt')
    if r2.returncode != 0 or not r2.stdout: print(f'[gpubatch] WARNING tag={tag}: download exited {r2.returncode}, {len(r2.stdout)} bytes: {r2.stderr.decode(errors="replace")[-300:]}', flush=True)
    local = outdir / tag
    if local.exists(): shutil.rmtree(local)
    local.mkdir(parents=True)
    subprocess.run(['tar', 'xf', '-', '-C', str(local)], input=r2.stdout, stderr=subprocess.DEVNULL)
    if not keep_remote and r2.returncode == 0 and r2.stdout and d.startswith(REMOTE_ROOT + '/batch/') and len(tag) > 0:
        _remote(f'rm -rf {d}')                      # data produced here is removed once fetched
    res = {}
    jl = {}
    if (local / 'results.jsonl').exists():
        for line in (local / 'results.jsonl').read_text().splitlines():
            try: o = json.loads(line); jl[o['out']] = o
            except Exception: pass
    for j in jobs:
        p = local / 'out' / f"{j['name']}.bin"
        res[j['name']] = {'out': p if p.exists() else None, 'json': jl.get(f"out/{j['name']}.bin")}
    return res, local

def read_out(path, M, N):
    b = Path(path).read_bytes(); assert struct.unpack('<III', b[:12])[0] == M
    d = np.frombuffer(b, dtype='<f8', offset=12)
    return d[:M*N].reshape((M, N), order='F').copy(), d[M*N:2*M*N].reshape((M, N), order='F').copy()
