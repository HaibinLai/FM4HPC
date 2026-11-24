# GMSTRS fix for MAGMA 2.10.0 mixed-precision FGMRES refinement

Files:
- `gmstrs_fix.patch`: patch against MAGMA v2.10.0 (`src/dxgesv_gmres_gpu.cpp`, `src/dgmres_plu_gpu.cpp`). Apply it from the MAGMA root with `patch -p1 < gmstrs_fix.patch`. A dry run applies cleanly to the v2.10.0 GitHub sources. **Not yet compiled against MAGMA or run on a GPU.**
- `certify_cpu.cpp`: the certified test `magma_dgmstrs_certify`, copied verbatim from the patched file, with host stubs for the MAGMA types. Build: `clang++ -O2 -std=c++17 -ffp-contract=off -shared -fPIC certify_cpu.cpp -o libcertify.dylib`.
- `port_fixed.py`, `port_fixed_results.txt`: original vs fixed on the faithful CPU port (`../fgmres_port.py`). The certified test is the compiled C++ above, and the documented condition is decided exactly with Fractions.
- Proof: `lean/OptimiKahan/MagmaFixed.lean` (log `check_log_magmafixed.txt`).

What the patch changes:

| Change | Where | Needed for |
|---|---|---|
| Certified post-check after FGMRES: Dot2 residual, outward-rounded test `L < R`, non-finite ⇒ fail; fail ⇒ `goto fallback`, `iter = -(niters+1)` | replaces `goto cleanup` after FGMRES | **soundness** (`fixed_post`, `fixed_nan_fallback`, `fixed_rejects*`) |
| `gmtol = eps·√n·BWDMAX` (dimensionless) instead of `cte = ‖A‖∞·eps·√n` | l.493 | how often the check passes; scale invariance (`fgmresFixed_scale`) |
| `ro = dnrm2(r)`; break only on `ro == 0` instead of `ro ≤ 1e-18` | FGMRES l.151–154 | tiny right-hand sides (no more x = 0) |
| FP16 range check (`max|a_ij| ≤ 65504`) before the half-precision LU, else `iter = -2` | after `dlag2s` | avoids 500 NaN iterations before the fallback |
| FGMRES per column; `trans ≠ NoTrans` ⇒ FP64 path (`iter = -1`) | GMSTRS call | `magma_dfgmres_plu_gpu` ignores `trans`, `nrhs`, `lddb`, `lddx` (see below) |

Soundness does not depend on the last four rows. The Lean theorem `fixed_post` holds for **any** FGMRES behaviour.

The proof assumes three things (the hypothesis `Faithful` in the Lean file):
- the published Dot2 error bound;
- the FP64 row-sum bound;
- that the few operations forming `L` and `R` stay within the `(1 ± 16u)` margin.

Underflow inside Dot2 is excluded, as in the paper. The C++ must not be compiled with `-ffast-math`, because that breaks TwoSum. FMA contraction is harmless here.

CPU result (n = 256, `port_fixed_results.txt`):
- **original:** 17 of 20 calls report success while violating the documented condition;
- **fixed:** 0 of 20.

Details of the fixed version:
- **FP32 LU:** iter = 3 and the same backward error (ratio 0.0815) at every scale 2^-80 … 2^60, so the result is scale invariant.
- **FP16 LU:**
  - k ≥ 20: range check ⇒ fallback;
  - k ≤ −20: FP16 underflow ⇒ check fails ⇒ fallback;
  - k = 0, 10: certified after 13 iterations.

The FP16 stand-in is a plain FP16 LU, whereas MAGMA's `dhgesv` uses FP32 storage with FP16 tensor-core GEMMs, so its small-scale behaviour differs.

New candidate defects, predicted from the code and **not yet confirmed on a GPU**: `magma_dfgmres_plu_gpu` never uses `trans`, `nrhs`, `lddb` or `lddx`. All of its gemv and dsgetrs calls are `MagmaNoTrans` and use one column. Prediction:
- `magma_dsgesv_iteref_gpu(MagmaTrans, …)` returns the solution of A x = b instead of Aᵀ x = b, with info = 0 and iter ≥ 0;
- with `nrhs > 1`, columns 2… stay 0 and success is still reported.

Possible further improvements (not in the patch):
- scale A by a power of two into the FP16 range before the half-precision LU, so the FP16 path also works at small and large scales;
- break FGMRES on a NaN estimate;
- a GPU Dot2 kernel instead of the host pass.
