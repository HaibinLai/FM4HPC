#!/usr/bin/env python3
"""Happens-before model of magma_[zcds]potrf3_mgpu, upper case (src/zpotrf3_mgpu.cpp lines 225-500).

Replays the host program block by block (m = n = M*nb, every block full) and records, for every
operation, the queue it is enqueued on, the events it waits for, and the buffers it touches.
Ordering rules (CUDA semantics):
  * operations on one queue run in order;
  * magma_queue_wait_event(q, e): later work on q is after everything recorded in e;
  * an event wait on a never-recorded event is a no-op;
  * magma_queue_sync(q) on the host: everything later enqueued (and every later CPU step) is after
    everything enqueued on q so far;
  * a CPU step (lapackf77_zpotrf) is after the host-synced set and before everything enqueued later.
Buffers: the pinned host workspace Aup (h column-block slots, rows tracked per block row), and the
device workspaces dlP(d, ., ., buf) (diagonal part D and row part R), and for d/s precision the
dtrsm_work buffers dinvA(d,0), dx(d,0), dx(d,1).
Reports every pair of conflicting accesses (same buffer, overlapping block rows, at least one write)
with no happens-before path in either direction.
usage: potrf_mgpu_hb.py [--real [--no-memset]]   --real: d/s path (dtrsm_work; its trtri_diag issues a cudaMemset
on the legacy default stream of the device, modelled as a device-wide barrier unless --no-memset;
--one-gpu: the barrier spans all virtual devices, as in the single-GPU emulation)   (sweeps ngpu = 1..8, h = 1..4, M = 2..12)
"""
import sys
from itertools import product

class Sim:
    def __init__(self):
        self.anc = []          # ancestor bitset per node
        self.info = []         # description per node
        self.last = {}         # queue -> last node
        self.pending = {}      # queue -> extra predecessor bitset from event waits
        self.events = {}       # event -> bitset (closure of recorded work)
        self.host = 0          # closure of work the host has observed (syncs, CPU steps)
        self.acc = []          # (node, buffer, rows(set), mode)
        self.ndev, self.one_gpu = 1, False

    def _closure(self, n):
        return self.anc[n] | (1 << n)

    def _new(self, preds, desc, accesses):
        n = len(self.anc)
        self.anc.append(preds)
        self.info.append(desc)
        for buf, rows, mode in accesses:
            self.acc.append((n, buf, frozenset(rows), mode))
        return n

    def op(self, q, desc, accesses):
        preds = self.host | self.pending.pop(q, 0)
        if q in self.last:
            preds |= self._closure(self.last[q])
        n = self._new(preds, desc, accesses)
        self.last[q] = n
        return n

    def legacy_barrier(self, dev, nstreams=3):
        """an operation on the legacy default stream (cudaMemset without a stream, as in magmablas_dtrtri_diag):
        after all earlier work on every blocking stream of the device, before all later work on them.  With
        one_gpu set (all virtual devices on one GPU, the round-50 emulation) the device is the whole GPU."""
        devs = range(self.ndev) if self.one_gpu else [dev]
        qs = [(d2, s) for d2 in devs for s in range(nstreams)]
        pre = self.host
        for q in qs:
            if q in self.last:
                pre |= self._closure(self.last[q])
        n = self._new(pre, f'legacy-stream memset dev{dev}', [])
        for q in qs:
            self.pending[q] = self.pending.get(q, 0) | self._closure(n)

    def record(self, ev, q):
        self.events[ev] = self._closure(self.last[q]) if q in self.last else 0

    def wait(self, q, ev):
        if ev in self.events:
            self.pending[q] = self.pending.get(q, 0) | self.events[ev]

    def sync(self, q):
        if q in self.last:
            self.host |= self._closure(self.last[q])

    def cpu(self, desc, accesses):
        n = self._new(self.host, desc, accesses)
        self.host |= self._closure(n)

    def races(self):
        out = []
        by_buf = {}
        for a in self.acc:
            by_buf.setdefault(a[1], []).append(a)
        for buf, lst in by_buf.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    (n1, _, r1, m1), (n2, _, r2, m2) = lst[i], lst[j]
                    if n1 == n2 or not (r1 & r2) or (m1 == 'R' and m2 == 'R'):
                        continue
                    if (self.anc[n2] >> n1) & 1 or (self.anc[n1] >> n2) & 1:
                        continue
                    out.append((buf, self.info[n1], self.info[n2], sorted(r1 & r2)))
        return out


def potrf_upper(g, h, M, real, memset=True, one_gpu=False):
    S = Sim(); S.ndev = g; S.one_gpu = one_gpu
    owns_after = lambda d, k: any(b % g == d for b in range(k + 1, M))
    s1, s2, s3 = 0, 1, 2
    H = lambda slot: ('Aup', slot)
    for k in range(M):
        idd = k % g; buf = k % g
        if k > 0:
            S.op((idd, s1), f'herk k={k} dev{idd}', [])
        S.op((idd, s1), f'diag D2H k={k} dev{idd}.s1', [(H(k % h), {k}, 'W')])
        if k > 0:
            for dd in range(g):
                d = (k + 1 + dd) % g
                if owns_after(d, k):
                    if d == idd:
                        S.wait((d, s2), (d, 4)); acc = []
                    else:
                        S.wait((d, s2), (d, 0)); acc = [(('P.R', d, buf), set(range(k)), 'R')]
                    S.op((d, s2), f'gemm k={k} dev{d}.s2', acc)
                    S.record((d, 2), (d, s2))
        S.sync((idd, s1))
        S.cpu(f'cpu potrf k={k}', [(H(k % h), {k}, 'W')])
        if k + 1 < M:
            for dd in range(g):
                d = (k + 1 + dd) % g
                acc = [(H(k % h), {k}, 'R')]
                if d != idd:
                    acc.append((('P.D', d, buf), {0}, 'W'))
                S.op((d, s1), f'diag H2D k={k} dev{d}.s1', acc)
                S.record((d, 1), (d, s1))
        else:
            S.op((idd, s1), f'diag H2D k={k} dev{idd}.s1', [(H(k % h), {k}, 'R')])
        if k + 1 < M:
            for dd in range(g):
                d = (k + 1 + dd) % g
                pan = [] if d == idd else [(('P.D', d, buf), {0}, 'R')]
                if d == (k + 1) % g:
                    S.wait((d, s1), (d, 2))
                    if real and memset: S.legacy_barrier(d)
                    work = [(('dinvA0', d), {0}, 'W'), (('dx0', d), {0}, 'W')] if real else []
                    S.op((d, s1), f'lookahead trsm k={k} dev{d}.s1', pan + work)
                    S.record((d, 4), (d, s1))
                elif owns_after(d, k):
                    S.wait((d, s2), (d, 1))
                    if real and memset: S.legacy_barrier(d)
                    work = [(('dinvA0', d), {0}, 'W'), (('dx0', d), {0}, 'W')] if real else []
                    S.op((d, s2), f'trsm(all) k={k} dev{d}.s2', pan + work)
            d = (k + 1) % g
            S.wait((d, s3), (d, 4))
            S.op((d, s3), f'rows D2H k={k} dev{d}.s3', [(H((k + 1) % h), set(range(k + 1)), 'W')])
            S.record((d, 3), (d, s3))
            buf2 = (k + 1) % g
            for d2 in range(g):
                if d2 != d:
                    S.wait((d2, s3), (d, 3))
                    S.op((d2, s3), f'rows H2D k={k} dev{d2}.s3',
                         [(H((k + 1) % h), set(range(k + 1)), 'R'), (('P.R', d2, buf2), set(range(k + 1)), 'W')])
                    S.record((d2, 0), (d2, s3))
            if owns_after(d, k + 1):
                pan = [] if d == idd else [(('P.D', d, buf), {0}, 'R')]
                if real:   # dtrsm_work with flag = 0: uses the inverse in dinvA(d,0); the panel is not read
                    S.wait((d, s2), (d, 4)); pan = []
                    work = [(('dinvA0', d), {0}, 'R'), (('dx1', d), {0}, 'W')]
                else:
                    S.wait((d, s2), (d, 1)); work = []
                S.op((d, s2), f'trsm(rest) k={k} dev{d}.s2', pan + work)
    return S.races()


if __name__ == '__main__':
    real = '--real' in sys.argv
    memset = '--no-memset' not in sys.argv
    one_gpu = '--one-gpu' in sys.argv   # all virtual devices on one GPU (the round-50 emulation): the memset orders every stream
    print(f"# potrf3_mgpu upper, {'d/s (DTRSM_WORK)' if real else 'z/c'} path" + (', legacy-stream memset per device' if real and memset else ''))
    print('ngpu,h,M,unordered_pairs,first_example')
    for g, h in product(range(1, 9), range(1, 5)):
        for M in range(2, 13):
            r = potrf_upper(g, h, M, real, memset, one_gpu)
            if r or M == 12:
                ex = '' if not r else f'{r[0][0]}: [{r[0][1]}] vs [{r[0][2]}] rows {r[0][3]}'
                print(f'{g},{h},{M},{len(r)},{ex}')
                break
