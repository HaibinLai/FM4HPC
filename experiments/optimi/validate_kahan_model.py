"""Bit-for-bit validation of kahan_model.py against PyTorch ops and against the REAL optimi AdamW step (CPU, bf16).
Usage: PYTHONPATH=<dir containing optimi/> python validate_kahan_model.py OPTIMI_ROOT
Part A: elementwise bf16 ops vs torch.  Part B: one real optimi AdamW(kahan_sum=True, foreach=False) step vs the model.
"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kahan_model as K

def bits(t): return t.view(torch.int16).numpy().astype(np.int32) & 0xFFFF
def npbits(a): return (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int32)

def main():
    optimi_root = sys.argv[1]; sys.path.insert(0, optimi_root)
    import optimi
    g = torch.Generator().manual_seed(0); n = 20000
    rb = lambda scale: (torch.randn(n, generator=g) * scale).to(torch.bfloat16)
    # ---- Part A: ops
    a, b = rb(1.0), rb(1e-2)
    chk = lambda name, tv, nv: print(f'  {name:22} mismatches: {int((bits(tv) != npbits(nv)).sum())} / {n}')
    print('Part A: bf16 elementwise ops, numpy model vs torch (CPU)')
    chk('add', a + b, K.bf16_add(a.float().numpy(), b.float().numpy()))
    chk('sub', a - b, K.bf16_sub(a.float().numpy(), b.float().numpy()))
    v = rb(1.0).abs(); chk('sqrt', v.sqrt(), K.to_bf16(np.sqrt(v.float().numpy())))
    m = rb(1.0); d = (rb(1.0).abs() + 0.5).to(torch.bfloat16); c = rb(1e-2); lr = 3e-4
    for variant in ('a', 'b'):
        ref = c.clone().addcdiv_(m, d, value=-lr)
        mine = np.array([K.addcdiv_value(ci, mi, di, -lr, variant) for ci, mi, di in zip(c.float().numpy(), m.float().numpy(), d.float().numpy())], dtype=np.float32)
        chk(f'addcdiv variant {variant}', ref, mine)
    # ---- Part B: real optimi step
    print('Part B: real optimi AdamW (kahan_sum=True, foreach=False) one step vs model')
    tot = bad = 0
    for trial in range(60):
        gen = torch.Generator().manual_seed(100 + trial)
        size = int(torch.randint(1, 4000, (1,), generator=gen)); lr = float(10 ** (-5 + 3 * torch.rand(1, generator=gen)))
        p = torch.nn.Parameter((torch.randn(size, generator=gen) * (10 ** torch.randint(-2, 2, (1,), generator=gen).item())).to(torch.bfloat16))
        opt = optimi.AdamW([p], lr=lr, weight_decay=0.0, kahan_sum=True, foreach=False, triton=False)
        p.grad = (torch.randn(size, generator=gen) * 0.1).to(torch.bfloat16); opt.step()                 # initialise state + one step
        st = opt.state[p]; st['kahan_comp'].copy_((torch.randn(size, generator=gen) * 1e-3 * p.detach().float().abs().mean()).to(torch.bfloat16))
        w0 = p.detach().clone(); c0 = st['kahan_comp'].clone(); p.grad = (torch.randn(size, generator=gen) * 0.1).to(torch.bfloat16)
        opt.step()
        m1, v1 = st['exp_avg'].float().numpy(), st['exp_avg_sq'].float().numpy(); w1, c1 = p.detach(), st['kahan_comp']
        eps = 1e-6; ew = np.empty(size, np.float32); ec = np.empty(size, np.float32)
        for i in range(size): ew[i], ec[i] = K.kahan_step(w0.float().numpy()[i], c0.float().numpy()[i], m1[i], v1[i], lr, eps, 'a')
        mw = int((bits(w1) != npbits(ew)).sum()); mc = int((bits(c1) != npbits(ec)).sum()); tot += size; bad += mw + mc
        if mw or mc: print(f'  trial {trial}: size {size} lr {lr:.2e}: weight mismatches {mw}, comp mismatches {mc}')
    print(f'  total elements {tot}, mismatching (w or c) values {bad}')

if __name__ == '__main__': main()
