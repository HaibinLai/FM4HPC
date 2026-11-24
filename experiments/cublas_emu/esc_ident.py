import sys; sys.path.insert(0, '.')
import numpy as np, esc_model as M
small = M.measured_small()
pc = {n: (A, B) for n, A, B in M.probe_cases()}
for nm in ("denormA", "tinyA1000", "rand_s0"):
    A, B = pc[nm]
    print(nm, "measured", small[nm], "true-exp", M.esc_max(A, B, -4195) + 54, "field-exp", M.esc_max(A, B, -4195, "field") + 54)
lc = {n: (A, B) for n, A, B in M.large_cases() if n.startswith("E1 n=1024")}
for blk in (32, 64, 128):
    print("BLK", blk, [M.esc_max(A, B, -4195, blk=blk) + 54 for n, (A, B) in lc.items()], "measured", [M.MEASURED_LARGE[n] for n in lc])
# exact (unblocked) need, for reference: max over nonzero products
def exact_need(A, B):
    ea, eb = M.expo(A, -10**9), M.expo(B, -10**9)
    x, y = ea.max(1), eb.max(0)
    worst = -10**18
    for i in range(0, A.shape[0], 16):
        z = (ea[i:i+16, :, None] + eb[None, :, :]).max(1)
        worst = max(worst, int((x[i:i+16, None] + y[None, :] - z).max()))
    return worst + 54
for nm in ("tri_s60", "zero_lane0", "rand_s60", "rand_s0"):
    A, B = pc[nm]; print(nm, "exact-z need", exact_need(A, B))
