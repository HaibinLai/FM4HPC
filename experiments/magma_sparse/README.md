# MAGMA v2.10.0 sparse solvers: acceptance-logic findings (2026-11-25, read-only audit + CPU port; no GPU run)

Audit table and ranked defects: subagent report summarized in HANDOFF §28.5. Main-session checks of the source: zbicg.cpp l.145–147 and l.197–198; magma_z_solver_wrapper.cpp l.141–142, 153–154; zmergecg.cu l.232, 1012–1022, 1258, 1308–1318; zcg_merge.cpp l.86–170.

`cgmerge_port.py`: the scalar recurrences of magma_zcg_merge (default solver), complex double, with the unconjugated sums of the merged kernels (`cgmerge_port_results.txt`):

| system | port of MAGMA | textbook CG |
|---|---|---|
| A = diag(1, 1.5), b = (1, 0.5i) | **info = 0 after 1 iteration, rel. error 1.14** (Σ r_i² = 0 by cancellation) | 2 iterations, 5e-17 |
| A = diag(2, 3), b = (2, i) | **info = 0, numiter = 1, x = (2, i)**, true residual 2.83 > ‖b‖ = 2.24; exact (1, i/3) | — |
| random complex HPD, cond 10, n = 16 / 64 | no convergence in 1000 iterations (status −201), error 1e3–1e4 | 16 / 33 iterations |
| control: real SPD data in complex storage | converges in 16 / 33 iterations, 3e-16 / 4e-11 | same |

Conclusion (port): the default sparse solver is correct only when the complex data are real; on genuinely complex Hermitian systems it either converges falsely with info = 0 or does not converge. Lean: `MagmaSparse.cg_merge_false_convergence` (exact Gaussian-integer computation of the A = diag(2, 3) case).
