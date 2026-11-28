# Round 29: sparse iterative solvers (roxy), triage

`probes/round29_sparse.cpp` ran 21 solver names (as mapped by `magma_zparse_opts`), each in its own process with a 600 s limit (`scripts/roxy/run_sparse29.ps1`). Each solver ran on 8 matrices × 3 formats (CSR, ELL, SELLP), and the preconditioned solvers also with NONE, JACOBI and ILU: 942 runs.

The matrices are 1024 × 1024: 2-D Poisson, anisotropic Poisson (1e−4), convection–diffusion, Poisson shifted by −4 (indefinite, zero diagonal), complex symmetric (shift −0.5i), Neumann (singular, consistent b), one zero diagonal entry, and ‖b‖ = 1e−30.

Every run is judged against the true residual computed on the host (`summary.txt`, made by `scripts/sparse29_summary.py`). The verdicts are:
- **OK**: info 0 and the true residual is at most 1e−6.
- **LIE**: info 0 although the true residual is not small.
- **GAP**: info 0, but the reported final_res is much smaller than the true residual.
- **FALSEFAIL**: info ≠ 0 although the true residual is at most 1e−8.
- **FAIL**: an honest failure.

**No new defect was found. Every non-OK cluster is a known finding or expected behaviour:**

| Cluster | Verdict | Known as |
|---|---|---|
| ‖b‖ = 1e−30: every solver returns x0 after 0–1 iterations with info 0 | KNOWN | Absolute default atol (1e−16) accepts immediately (cf. `tiny_rhs_returns_zero`; sweep_results.txt "tiny b") |
| BA, ITERREF: info 0 after 0 iterations on every matrix, x unchanged | KNOWN | M089, the wrapper's default branch returns 0 |
| JACOBI solver: info 0 after 1000 iterations with residual 7e−4 to 4e−2 | KNOWN | `jacobi_any_decrease_all` |
| PBICG, PTFQMR with JACOBI on a zero diagonal: info 0 after 1 iteration, no progress | KNOWN | M083, breakdown overwritten by success |
| IDR, PIDR (IDRMERGE, PIDRMERGE): −202 on every matrix, including Poisson | KNOWN | M103 |
| CG, PCG (CGMERGE): fail on complex data at every size, even n = 16 (`cg_pcg_idr_sizes_complex.txt`, from a first run with complex x) | KNOWN | M084, inner products not conjugated. The main sweep then used real data. |
| CG, PCG, QMR, BICG on shifted_indef: −201 at 1000 iterations with residual 1–2e−10 | NOT A BUG | The residual is above the requested rtol 1e−10, so the failure report is honest. The sweep's FALSEFAIL threshold (1e−8) was too loose. |
| CG on non-Hermitian / indefinite matrices: −205; ILU/Jacobi on a zero diagonal: −3003 / −204 | NOT A BUG | Honest failures |
| BOMBARDMENT: CSR OK, ELL/SELLP −103 (MAGMA_ERR_NOT_SUPPORTED) | NOT A BUG | An unsupported format is rejected honestly |

Negative result: ELL and SELLP give the same verdicts as CSR for every solver, so the SpMV formats added no solver-level defect on these inputs.

**Correction (round 45, 2026-11-27).** The sentence above is wrong for six solvers. The per-solver CSVs contain no SELLP rows for QMR, BICG, LSQR and BOMBARDMENT, and only 19 of 24 CSR rows (no ELL/SELLP) for PQMR and PBICG: those processes died, and the crash was not noticed when the CSVs were summarised. Round 45 reproduced the crashes and found the cause:
- QMR, BICG, LSQR, BOMBARDMENT on SELLP: integer division by zero. The SELL-P blocksize/alignment are lost in magma_zmtranspose / magma_zmconvert. New finding **M251** (`results/round45_sparse_edge/TRIAGE.md`).
- PQMR, PBICG during CSR: the ILU setup on the zerodiag matrix ends in the known stack overflow M098.
