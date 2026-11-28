# Reported Success, Wrong Answer: Formalizing and Fixing the Acceptance Logic of Mixed-Precision Iterative Refinement

*First full draft, 2026-11-25. Markdown for review; convert to LaTeX for submission.*
*Markers: **[TODO-GPU]** needs the s82 V100; **[TODO-USER]** needs the authors (bug filing, AI-use statement, author list); **[CHECK]** a number or claim to re-verify before submission.*

---

## Abstract

Mixed-precision iterative refinement (IR) factorizes a linear system in low precision (FP16, BF16, TF32, FP32) and refines the solution in FP64. Libraries promise the caller that a successful return means FP64-level backward error, and that otherwise they fall back to an FP64 solve. That promise rests on a short piece of *acceptance logic*: a stopping criterion, a norm or max reduction, a fallback path, and a returned status.

We study this acceptance logic across about 25 routines in 18 libraries (16 open source, 2 documentation only). We combine code-level models in Lean 4, predictions written before running, and runs on CPU and GPU.
- In 10 libraries, a run reports success while returning a wrong answer. Examples:
  - MAGMA's FP16 tensor-core IR returns an all-NaN solution with `info = 0` for a finite, FP64-solvable system.
  - cuSOLVER's GMRES-based IR reports convergence with relative error 0.93–23 after a 2^40 scaling, in every release since CUDA 11.8.
  - SLATE's LAPACK-compatible `slate_dsgesv` loses the solution of a well-conditioned 100×100 system and returns `info = 0`.
  - SuperLU_DIST returns error bounds and a backward error that are bit-identical to the clean run while half of the solution is NaN.
- None of the 16 defects is triggered by the libraries' own test suites, and in four libraries the test checks would pass the faulty result even if it were triggered.

We prove in Lean 4 (core library only, no `sorry`) four results:
1. An acceptance contract ("success ⇒ good answer" and "FP64-solvable ⇒ success") holds if and only if four conditions C1–C4 hold. Every observed failure violates one of them.
2. Of six combinations of comparison form and max reduction, exactly one rejects every NaN-contaminated input.
3. The documented threshold √n·u lies below the worst-case rounding error of an FP64 residual, so FP64 residuals cannot certify it.
4. A solver-agnostic acceptance gate (compensated residual, outward rounding, non-finite ⇒ fallback) is sound for *any* output of the inner solver.

The gate certifies all 36 cuSOLVER test cases, including the 13 wrong ones it rejects and re-solves in FP64. The same test with an FP64 residual certifies none.

---

## 1 Introduction

**A motivating example.** We call the LAPACK-compatible entry point of SLATE (commit 878439b), `slate_dsgesv`, on a 100×100 diagonally dominant matrix. It returns `info = 0` and `iter = 23`. The output array `x` is untouched, and the right-hand side `b`, which the routine uses internally as the solution buffer, has been overwritten with zeros. The solution is in neither array. The cause is two lines:
- the internal solution matrix is built on top of `b`;
- `*info = 0; // todo: get a real value for info` (`lapack_api/lapack_gesv_mixed.cc:89-90`).

Nothing in SLATE's tests calls this routine.

This is not an isolated case. Mixed-precision iterative refinement is now the standard way to use low-precision accelerators for FP64 linear systems [Carson–Higham 2018; Haidar et al. 2018; Higham–Mary 2022]. It is shipped by:
- vendors: cuSOLVER IRS, cuDSS, oneMKL;
- dense libraries: LAPACK `dsgesv`, MAGMA, SLATE, PLASMA;
- sparse direct solvers: SuperLU, SuperLU_DIST, MUMPS, PaStiX, STRUMPACK;
- benchmarks: HPL-MxP.

The numerical analysis of IR is mature. Our subject is the code that decides when the refinement has succeeded. We call it the *acceptance logic*: a comparison of a residual norm against a threshold, a reduction that computes that norm, a fallback to FP64, and the status returned to the caller. It is a few dozen lines per library. It is the only thing standing between a failed low-precision factorization and a caller who reads `info = 0`.

**Why tests do not find these defects.** The failures need specific inputs:
- a NaN or Inf in `b`;
- a matrix whose scale is far from 1 (2^±40, 1e-12);
- a zero initial guess;
- pivot growth that overflows FP32 or FP16 while FP64 is exact.

Library tests draw `b` from U(0,1) or set `b = A·1` (§6.3). Moreover, the checks in several test suites reuse the same NaN-dropping max reduction as the code under test.

**Why existing formal work does not cover them.** Recent GPU-kernel verification in Lean (TensorLean 2026; Koska et al., arXiv 2611.04447) works on device code and models floating point as real numbers. The latter states explicitly that it does not reason about numerical error. Code-level proofs of numerical programs (VST/LAProof; VSCL) prove a specific implementation correct; they do not predict how a library fails. The acceptance logic is host-side, and its failures are driven precisely by NaN semantics and rounding.

**Approach.** For each library:
1. We write a *contract card*: the rule from the code, the documented promise, the theorem that can be proved, and the gap between them.
2. Where code is available, we translate the control flow and stopping tests line by line into Lean 4. NaN is modelled; values are otherwise exact rationals.
3. We derive inputs that the model predicts will violate the contract, and run them.

For closed-source cuSOLVER we use a proved metamorphic oracle instead: scaling by 2^k commutes with IEEE arithmetic.

**Contributions.**
1. **An acceptance contract and its characterization** (`IrDriver.contract_iff`). Soundness and completeness of an IR driver hold iff C1–C4 hold (outer check sound, fallback reachable, fallback correct, status not forced). We map eleven defect classes D1–D11 found in practice onto these conditions.
2. **When C1 is achievable in floating point.**
   - A NaN-detection theorem for comparison forms × max reductions (`MaxFold.detects_iff`).
   - A scaling-covariance theorem that yields a zero-false-positive oracle (`IrsContract.eval_scale`).
   - A proof that FP64 residuals cannot certify the documented threshold (`IrsContract.fp64_cannot_certify`).
3. **An empirical study.**
   - About 25 routines in 18 libraries.
   - 10 libraries with confirmed "success but wrong answer" runs; all 7 CPU predictions made from code reading were confirmed.
   - An audit showing that none of the 16 defects is reached by the libraries' default tests.
4. **A certified acceptance gate.** It is proved sound for any inner-solver output (`MagmaFixed.fixed_post`, `IrsContract.wrapper_post`). It is implemented as a cuSOLVER wrapper and as a MAGMA patch.

**What we do not claim.**
- We are not the first to observe that a mixed-precision solver can accept a NaN solution. A series of Reference-LAPACK pull requests (#1499–#1520, Nov 2026) addresses this for `xSGESV/xSPOSV`. §8 compares our results with that series.
- Scaling invariance of floating-point arithmetic is classical [Bauer 1963; Higham 2002 §9.8] **[CHECK]**.

---

## 2 Background: the anatomy of an IR driver

A mixed-precision IR driver has five parts (Fig. 1 **[TODO: figure]**):
1. Convert A (and sometimes b) to low precision; factorize.
2. Compute an initial solution.
3. Refine. *Classical IR* repeats r = b − Ax in FP64, solves the correction with the low-precision factors, and updates x. *GMRES-IR* runs a preconditioned Krylov solver as the inner solver [Carson–Higham 2017].
4. Accept or iterate. LAPACK `dsgesv` stops when `RNRM < SQRT(N)·XNRM·ANRM·EPS·BWDMAX` (∞-norms, BWDMAX = 1). cuSOLVER documents the same rule.
5. If refinement fails, fall back to an FP64 factorization. Then return a status: `info`, `iter` (negative values encode failure reasons, e.g. −2 "overflow in down-cast", −31 "did not converge"), or a library return code.

Documented promises, quoted from the libraries:
- cuSOLVER: "same normwise backward error as the main precision"; with fallback enabled, "always a good solution".
- LAPACK: iter > 0 means the refinement converged; iter < 0 means a fallback happened.

We write the promise as a post-condition:

> **Post(A, b, x, iter):** iter ≥ 0 ⇒ ‖b − Ax‖∞ < c·‖A‖∞·‖x‖∞ and x is finite, with c = √n·u.

---

## 3 The acceptance contract

### 3.1 An abstract driver

We model a driver as `Impl P X` with four parts:
- an outer check `check : P → X → Bool`;
- a flag saying whether the fallback is reachable, `fbReach`;
- a fallback `fb : P → X × Bool`;
- a flag `force` meaning the wrapper forces the status to success.

The inner solver is adversarial: it may return any x. `run` returns `(x, ok)`:
- the inner result, if `check` passes;
- the fallback result, if not and the fallback is reachable;
- otherwise failure.

In every case `force` can overwrite `ok`.

**Sound:** ok ⇒ the returned x is Good. **Complete:** the problem is FP64-solvable ⇒ ok.

### 3.2 Characterization

**Theorem 1 (`contract_iff`).** Assume four things:
- the inner solver can return bad values;
- good answers exist only for solvable problems;
- both solvable and unsolvable problems occur.

Then Sound ∧ Complete holds iff all four conditions hold:
- C1 (check sound);
- C2 (fallback reachable);
- C3 (fallback sound and complete);
- C4 (status not forced).

Sufficiency (`contract_of_conditions`) needs no assumptions.

**Corollaries** (Table 1):

| Corollary | Violates | Libraries where observed |
|---|---|---|
| `trust_inner_unsound` | C1 | MAGMA GMSTRS, `dshposv` (GPU 0), PaStiX GMRES, STRUMPACK mixed |
| `forced_unsound` | C4 | SLATE LAPACK API, STRUMPACK mixed, PaStiX |
| `stub_fallback_fails` | C3 | MAGMA `zchesv` (confirmed on GPU 0: X = B, info = 0) |
| `skipped_fallback_incomplete` | C2 | PLASMA `dsgesv` |
| `c1_forces_accept_propagate` | C1 via §4.1 | LAPACK, MAGMA `dsgesv_gpu`, PLASMA, SuperLU(_DIST), MUMPS |

**Non-vacuity.** `tOk_contract` exhibits a toy instance in which all assumptions hold and the contract is satisfied. Its "forced" and "no-check" variants are proved unsound (`tForced_unsound`, `tTrust_unsound`).

The logic of Theorem 1 is simple. Its value is the complete list of conditions and the mapping of real libraries onto them. The numerical difficulty lies entirely in C1, which §4 addresses.

---

## 4 When can C1 hold in floating point?

### 4.1 Comparison form and max reductions (D1, D11)

A comparison with NaN is false. A *reject-form* test `if (r > thr) continue` therefore accepts NaN. An *accept-form* test `if (r < thr) accept` rejects it.

The reduction that computes r matters as much as the comparison. We model three max reductions over values `V ::= nan | num q`:
- SuperLU's `SUPERLU_MAX(s,v) = (s>v)?s:v`;
- PLASMA/IDAMAX's `if (v>s) s=v`;
- a NaN-propagating max.

**Theorem 2.**
- (`smax_fold_nan_iff_last`) The SuperLU fold is NaN iff the *last* element is NaN.
- (`smax_fold_value`) If finite values follow the last NaN, the result equals the maximum of those finite values alone.
- (`pstep_fold_isNaN`) The PLASMA/IDAMAX fold is NaN iff the *first* element is NaN.
- (`detects_iff`) Over {reject, accept} × {SuperLU, PLASMA, propagate}, "every NaN-containing input is rejected" holds iff the test is accept-form and the reduction propagates. Each of the other five combinations has a two-element counterexample.

`masked_equals_baseline` instantiates the SuperLU_DIST experiment of §6.2: a first block entirely NaN and a second block equal to the clean values. The returned value is *equal* to the clean baseline. This matches the run, where `berr` is 2.96e-16 in both.

`c1_forces_accept_propagate` connects this to §3. Suppose the check has the form criterion(reduction(per-row values)) and every NaN pattern can arise from some bad input. Then C1 forces an accept-form criterion and a propagating reduction. D1 and D11 are therefore special cases of violating C1.

### 4.2 Scaling covariance as an oracle (D2, D3)

`IrsContract` defines a rounding function `rnd` with a p-bit significand, round-to-nearest-even and unbounded exponent.

**Theorem 3.**
- (`rnd_scale`) rnd(2^k q) = 2^k · rnd(q).
- (`eval_scale`, `iterate_scale`) Every program in a small typed language commutes with scaling: the same branches and the same rounding decisions. The language has rounded +, −, ×, ÷; exact neg, abs, max; comparisons; and degree-0 constants. Mixing degrees is a type error. Iterations whose stopping flags and counters have degree 0 also commute.
- (`oracle`) Degree-0 outputs, such as the solution x of Ax = b when A and b have degree 1, are bit-identical under 2^k scaling.
- (`absolute_tol_breaks`) An absolute constant in a comparison makes the branch flip under scaling.

Consequence: an implementation built from these operations, without overflow or underflow, must return bit-identical x and iter for (2^k A, 2^k b). The documented criterion is itself scale-invariant (`accept_scale`). So a change of x under scaling proves that the implementation contains a scale-dependent constant, or that it overflowed or underflowed. This gives a zero-false-positive oracle that needs no source code. Its precondition, and how we check it in practice, is a fixed summation order: repeated runs are bit-identical before any scaling comparison.

### 4.3 FP64 residuals cannot certify the documented threshold (D6)

**Theorem 4 (`wrapper_sound`, `fp64_cannot_certify`).** Suppose error bounds for the computed residual norm (γ_R) and matrix norm (γ_A) are given. If the test

(1+ε)·R̂ + γ_R·(B + X·Â/(1−γ_A)) < c·X·Â/(1+γ_A)

passes, then the exact documented condition holds, whatever the solver returned. Conversely, if γ_R ≥ c, the test can never pass.
- For an FP64 residual, γ_R ≈ (n+1)u, which exceeds c = √n·u. For example, at n = 1024, c = 32u < 1025u (`fp64_n1024`).
- For a Dot2 residual [Ogita–Rump–Oishi 2005], γ_R ≈ n²u².

### 4.4 A certified acceptance gate

The gate treats the inner solver as a black box:
1. Compute the residual with Dot2.
2. Evaluate the test of Theorem 4 with outward rounding (1 ± 16u) on each side.
3. Treat any non-finite value in x, L or R as failure.
4. On failure, fall back to FP64 and set iter = −(niters+1).

**Theorem 5 (`fixed_post`, `fixed_nan_fallback`).** For any FGMRES output (any niters, any vector, NaN included), if the computed quantities satisfy `Faithful`, the returned value satisfies Post. NaN anywhere implies iter < 0.
- `fixed_rejects_tiny` and `fixed_rejects_one_step`: the two counterexamples to the original MAGMA code (§5) are rejected.
- `fixed_accepts_example` and `example_faithful`: the accept branch is reachable, and `Faithful` is satisfiable with n = 1024 constants.
- `fgmresFixed_scale`: the patched exit decision is invariant under (A, b) → s(A, b).

**Assumption.** `Faithful` bundles three things, none of them proved in Lean:
- the published Dot2 error bound;
- an FP64 row-sum bound;
- the (1 ± 16u) enclosure of the few operations that form L and R.

Underflow is excluded. Deriving `Faithful` from `rnd` is future work (§9).

### 4.5 Modelling decisions

- **Values.** Exact rationals plus a NaN constructor; comparisons with NaN are false; Inf is not modelled. This is conservative for the defects we report, which are all about NaN or finite values.
- **No exponent range in `rnd`.** Scaling theorems therefore assume no overflow or underflow. For closed-source libraries this cannot be checked, which is why the oracle reports "scale-dependent constant *or* over/underflow".
- **Hand translation.** Models are translated by hand, line by line, with file:line references. Vector kernels (gemv, dot, getrs) are abstract inputs. §5.2 validates the MAGMA model against execution traces.
- **Summation order.** Theorems 3–5 use worst-case bounds, which hold for any summation order. The scaling oracle additionally needs the *same* order in both runs.

---

## 5 Method: from contract to counterexample

### 5.1 Contract cards

For each automatically chosen parameter or acceptance rule, we fill six fields:
- **rule** (verbatim from code);
- **claim** (verbatim from documentation);
- **provable** (a theorem with all premises);
- **gap** (premises the code does not check);
- **counterexample family** (inputs derived from the gap);
- **measurement**.

### 5.2 A code-level model of MAGMA GMRES-IR, validated by traces

MAGMA 2.10.0's `magma_dsgesv_iteref_gpu` (FP32 LU) and `magma_dhgesv_iteref_gpu` (FP16 tensor-core LU) both take the GMSTRS branch of `magma_dxgesv_gmres_gpu`, which calls `magma_dfgmres_plu_gpu`. From the code:
- `gmtol = ‖A‖∞·u·√n` (l.490–493 of `dxgesv_gmres_gpu.cpp`).
- After FGMRES the code jumps straight to `cleanup`: no residual recheck, no fallback, `info` stays 0 (l.570–600).
- FGMRES (`dgmres_plu_gpu.cpp`):
  - breaks on the absolute test `fabs(ro) <= 1e-18` (l.154);
  - stops on the Givens-recursion estimate `ro <= tol·ro₀` (l.164, l.272);
  - returns `niters = iters` with no negative path (l.297).

`Magma/Fgmres.lean` (364 lines) proves the following:

| Theorem | Statement |
|---|---|
| `gmstrs_always_success` | After a successful low-precision factorization, every run returns `info = 0`, `iter ≥ 0` |
| `nan_runs_to_maxiter` | A NaN residual estimate runs to `maxiter` and reports success |
| `enforced_rule` | The enforced test is \|rs\| ≤ √n·u·‖A‖∞·‖b‖₂, which scales as s² under (A,b) → s(A,b); the documented test scales as s |
| `tol_ge_one_one_step` | ‖A‖∞·u·√n ≥ 1 ⇒ stop after one step, regardless of the factor quality |
| `one_step_violates`, `one_step_returned` | A concrete 2×2 instance (A = 2^60·I, FP16-level preconditioner error) returns (iter, info) = (1, 0) with a residual of about 9.4e-4 that violates Post |
| `tiny_rhs_returns_zero`, `tiny_violates` | 0 < ‖b‖₂ ≤ 1e-18 ⇒ x = 0, iter = 0, which violates Post |

**Trace validation.** We check that the hand translation matches the code:
- We inserted 13 read-only print points into the two MAGMA files (hex-exact `%a` output, enabled only by an environment variable).
- We reran the GPU probe for n = 512, 1024 and 2048 (72 calls). Output with and without instrumentation is bit-identical at n = 1024.
- Each trace is replayed through the Lean model's own `fgmres`/`gmstrs` definitions.
- **72/72** calls match on (niters, early exit, iter, info).
- All seven model premises (A1–A7) hold on every trace.
- Three deliberately wrong model variants are detected (12, 45 and 29 mismatches). This shows the replay has power to detect a wrong model.

This covers scalar control logic only; vector kernels take the traced values.

### 5.3 Predict, then run

For the cross-library survey (§6.1) we read code, assigned defect classes, and wrote down the predicted failing input before running anything. **[CHECK]** The survey (dated 2026-11-24) and the run results (2026-11-25) entered version control in the same commit. Session logs and in-file dates are the evidence that the predictions came first. Future predictions will be committed separately before runs.

---

## 6 Empirical study

### 6.1 RQ1: How common are acceptance-logic defects?

We read the acceptance logic of about 25 routines. The libraries and versions are:

| Library | Version / commit | Library | Version / commit |
|---|---|---|---|
| MAGMA | v2.10.0 (752e63c) | SLATE | 878439b |
| PLASMA | 548e0bd | LAPACK | 3.12.0 (system) |
| SuperLU | a8d98be | SuperLU_DIST | 9.3.2 (239280c) |
| MUMPS | 5.8.1 (read), 5.8.2 (run) | STRUMPACK | 8.1.0 (edc9a72) |
| PaStiX | 6.4.0 (e2ecf99) | hpl-ai | 8c7ab18 |
| rocHPL-MxP | a22a987 | Ginkgo | 796a388 |
| PETSc | 0bedf3a | Trilinos | bc7b016 |
| hipSOLVER / rocSOLVER | a69f369 | IterativeRefinement.jl | — |
| cuDSS, oneMKL | documentation only | | |

The defect classes are:

| Class | Meaning |
|---|---|
| D1 | Reject-form test accepts NaN |
| D2 | Tolerance not degree-1 homogeneous under scaling |
| D3 | Absolute threshold affects acceptance or breakdown |
| D4 | No outer true-residual check after a Krylov inner solve |
| D5 | Reaching maxiter reported as success |
| D6 | Threshold below the rounding error of an FP64 residual |
| D7 | Ignored API parameters (trans, nrhs, ld*, x) |
| D8 | No range check on down-conversion |
| D9 | Fallback unreachable, stubbed or absent |
| D10 | Wrapper loses or forces status |
| D11 | Max reduction drops NaN |

Table 2 gives the full matrix **[TODO: compress `IR_SURVEY_zh.md` table to one page]**. Six patterns emerge:

1. **D1 is nearly universal.** Only rocHPL-MxP, Trilinos Belos, PETSc, Ginkgo, SLATE's `*_mixed_gmres` and hpl-ai's driver check get it right. It has two roots: the reject-form test inherited from LAPACK `dsgesv` by MAGMA, SLATE and PLASMA, and D11 in SuperLU, PLASMA, and MUMPS via IDAMAX. The LAPACK PR series fixes the first root in Reference LAPACK only.
2. **Krylov inner solvers are trusted without an outer check (D4).** This affects MAGMA's two GMRES paths, STRUMPACK mixed, PaStiX GMRES and cuSOLVER's GMRES path (inferred, black box). Only SLATE's GMRES variant and Trilinos recompute the true residual.
3. **Wrappers lose status (D10).** SLATE's LAPACK and ScaLAPACK APIs hard-code `info = 0`; STRUMPACK mixed and PaStiX always return success; hipSOLVER never writes `niters`.
4. **Absolute thresholds (D3) are everywhere:** MAGMA 1e-18 and EPSILON; STRUMPACK atol = 1e-10; SuperLU_DIST GMRES 1e-14; PETSc and PaStiX 1e-50.
5. **Down-conversion is often unchecked (D8).**
6. **The fallback itself breaks (D9):** stubbed in MAGMA `zchesv`, skipped after task failure in PLASMA, unreachable code in MAGMA's GMRES path.

**Safe designs** serve as a contrast:
- rocHPL-MxP: `res < thrsh` on the scaled HPL residual.
- Trilinos Belos FixedPoint: an explicit `NaNError` and a separate `MaxItersReached` code.
- PETSc: `KSP_DIVERGED_NANORINF`, but with abstol 1e-50.
- cuDSS: an explicit `IR_FAILED` status.

### 6.2 RQ2: Do the predicted failures occur?

Table 3 lists the confirmed defects. Each row was predicted from the code or the contract before the run. CPU runs were on a CPU-only node (conda toolchain, each library built from source at the commit read); GPU runs on a V100 (MAGMA) and an RTX 5070 Ti (cuSOLVER).

| Library / routine | Class / condition | Trigger | Observed | Report |
|---|---|---|---|---|
| MAGMA `dhgesv_iteref` (FP16 TC LU + FGMRES) | D8, D4, D5 / C1 | n = 1024, A ∝ 2^k, k ≥ 20 (entries > 65504) | **x all NaN, info = 0, iter = 500** | drafted **[TODO-USER]** |
| MAGMA `dsgesv/dhgesv_iteref` | D3 / C1 | k = −80, −70 (‖b‖₂ ≤ 1e-18) | x = 0, iter = 0, error 1 | same |
| MAGMA `dsgesv/dhgesv_iteref` | D2 / C1 | k = 0, 10–20, 30 | iter 3, 2, 1; backward error / threshold 2.8, 2.2e4, 3.1e8 | same |
| MAGMA `dhgesv_iteref` | D5 / C1 | k = −60, −40 | iter = 500, info = 0, error 3.7e-3, backward error 3.8e8× | same |
| MAGMA `dsgesv_gpu` | D1 / C1 | W(30)·2^120 (FP32 pivot growth) | iter = 0, info = 0, X all NaN (`dgesv` exact) | same |
| cuSOLVER `D{B,X,S}gesv`, IRS GMRES (12.4.1, sm_120) | D2/D3 / C1 | (A, b)·2^40 | DBgesv iter = 1, relative error 0.93–23; DXgesv 0.04–2.1 | drafted **[TODO-USER]** |
| cuSOLVER, CUDA 11.8 → 13.x | same | same | present in all runnable releases (DBgesv errors 3.1, 3.1, 1.24, 0.93) | same |
| LAPACK 3.12.0 `dsgesv` | D1 / C1 | W(40)·2^100, W(30)·2^120 | ITER = 0, INFO = 0, all NaN | prior art: PR #1501 |
| PLASMA `dsgesv` | D1, D11 / C1 | b with NaN; FP32 pivot growth | retval 0, iter 0, X all NaN | drafted |
| PLASMA `dsgesv` | D9 / C2 | FP32-singular A; b overflowing FP32 | retval ≠ 0, iter = −31 (likely read from never-written buffers, inferred), X never written | drafted |
| SLATE `slate_dsgesv` (LAPACK API) | D7, D10 / C4 | well-conditioned 100×100 | info = 0, iter = 23, x untouched, b overwritten (zeros in the printed entries) | drafted |
| SLATE `slate_dsgesv` | D10 / C4 | exactly singular A | info = 0 | drafted |
| SuperLU `dgssvx`+`dgsrfs` | D11 / C1 | two decoupled blocks, b[0] = NaN or Inf | 500 NaN in x, info = 0, berr 6.94e-17 (a clean value); A = I, b = (NaN, 1) gives berr = 0 | drafted |
| SuperLU_DIST `psgssvx_d2` | D11 / C1 | same | info, both error bounds (1.03e-7, the formula's floor) and berr (2.96e-16) bit-identical to baseline; 500 NaN in x | drafted |
| MUMPS 5.8.2 IR | D11-like (per-row comparisons drop NaN, inferred) / C1 | b[0] = NaN | INFOG(1) = 0, omega1 = 1.94e-16 = baseline | drafted |
| PaStiX GMRES refinement | D4, D5, D10 / C1, C4 | x0 = 0 (documented as allowed) | 250 iterations, x ≈ 249·x_true, residual 249, `PASTIX_SUCCESS` | drafted |
| STRUMPACK mixed and default `SparseSolver<double>` | D3, D10 / C1, C4 | (A, b)·1e-12 or 1e-20; b with NaN | x ≡ 0, SUCCESS | drafted |

Not yet run: hpl-ai, Ginkgo, PETSc. (MAGMA `zchesv` and `dshposv` were run on GPU 0; see below.)

**Further MAGMA findings (code-level Lean models + faithful CPU ports, confirmed on GPU 0 unless noted; `experiments/magma_gpu/`).** `Magma/DenseIR.lean` (23 theorems) and the three `Magma/Sparse/` modules (78 theorems) model every remaining MAGMA routine with refinement or acceptance logic:
- *Solution overflow is accepted.* If A and b are in FP32 range but the solution is not (a = 2^-100, b = 2^28), the FP32 solve returns Inf and `Inf > Inf·cte` is false. Result: iter = 0, info = 0, x = Inf in zcgesv and zcposv (`x_overflow_accepted`). The same structure in Accelerate's LAPACK dsgesv/dsposv returns x = inf/nan with ITER = INFO = 0 (run). zcgesv also never checks B's FP32 range: the conversion is commented out and `zclaswp` has no check (`zcgesv_b_overflow_accepted`).
- *zchesv returns X = B with info = 0.* Its FP64 fallback is commented out (`zchesv_stub`). With its non-pivoted LDLᵀ, a cond-3 saddle matrix gives X = B; in another instance the refinement diverges to overflow and is accepted as converged at iter = 29 (`diverge_accepted`; CPU port).
- *zgesv_rbt hides every failure.* `iter` is a local variable and the nopiv info is overwritten (`rbt_info_lost`).
  - The vector butterfly skips size-1 blocks, so n = 1, 2, 3 are solved wrongly with info = 0 (relative errors 6e-2, 3e-2, 3e-3).
  - A cond-3.09, n = 64 matrix built against the glibc rand() butterfly returns a relative error of 3.6e35 with info = 0 (CPU port).
- *dshposv (FP16 Cholesky + FGMRES).* The public entry points `return 0`, and there is no recheck or fallback (`dshposv_always_success`). The absolute threshold is 2^-53 (`spd_tiny_rhs_returns_zero`). With nrhs ≥ 2 the inner loop writes past the `d_w` buffer once it reaches the restart length (`nonconverged_oob`).
- *Sparse solvers.* CG_MERGE, the default solver, uses unconjugated complex inner products. A = diag(2, 3), b = (2, i) stops after one iteration with info = 0 and x = (2, i), while the exact solution is (1, i/3) (`cg_merge_false_convergence`, exact). Random complex HPD systems do not converge at all (CPU port).
  - BiCG/PBiCG/PTFQMR overwrite a detected breakdown with SUCCESS (`bicg_breakdown_success`).
  - The ILU setup never checks numerical zero pivots, so together with PBiCG a zero pivot becomes info = 0 (`ilu_zero_pivot_pbicg_success`).
  - The success status ignores the computed true residual (`template_ignores_final_res`).
  - pcg_merge upgrades to SUCCESS relative to the initial residual (`pcg_merge_upgrade`).
  - The wrapper returns 0 for unhandled options (`wrapper_unhandled_success`).
  - A grep-based classification of all 36 solver files shows that exactly four (BiCG, PBiCG, PTFQMR, PTFQMR_MERGE) overwrite a detected breakdown with SUCCESS. The other thirteen with a "fixed" post-loop keep DIVERGENCE (`fixed_breakdown_divergence`).
  - FGMRES keeps a stale SUCCESS across restarts and exits on a NaN without rechecking (`fgmres_sticky_success`).
  - LOBPCG's parameter check is dead code (`lobpcg_check_dead`).
  - The Jacobi and IC preconditioner setups miss stored-zero diagonals and numerical IC breakdown (`jacobi_stored_zero_passes`, `icc_breakdown_unchecked`).
- *GPU 0 confirmations of the above:*
  - solution overflow accepted (x = inf/nan, iter = info = 0);
  - zchesv X = B (iter = −31, info = 0) and divergence accepted (iter = 27, x with Inf);
  - dshposv: x = 0 for ‖b‖₂ = 1e-17, and a one-step stop at ×2^40 (backward error 5.5e7 × threshold);
  - **dshposv with nrhs = 2 writes one full column (4096 bytes) past its workspace**, detected by an LD_PRELOAD guard;
  - RBT wrong for n = 1–3, and a cond-3.09 matrix solved with relative error 1.6e13 and info = 0;
  - CG_MERGE complex false convergence; BiCG breakdown reported as success; the wrapper returns 0 untouched;
  - pcg_merge SUCCESS at 2.9e4 × rtol.

  Not as predicted: an ILU(0) numerical zero pivot crashes `magma_z_precondsetup` (segfault) instead of passing silently.
- *A status-loss audit of all dense routines* found 15 further defects. The five we ran on GPU 0 all reproduce:
  - `gbtrs_batched_strided` aliases its B pointer array onto A (all solutions NaN, return 0);
  - `heevdx_2stage` with Upper returns wrong eigenvalues (error 12.9) with info = 0;
  - `hesv_nopiv` misses a zero last pivot (info = 0, x = NaN);
  - `potf2` writes a 0-based failure index, so a failure at column 0 reads as success (Upper native potrf and potf2_gpu);
  - `trtri_gpu` loses or mis-offsets singular-block info (info 0, 45, 256 where LAPACK gives 1, 301, 512).
  - A second round confirmed four more:
    - `potrf_vbatched` ignores Upper and factors the lower triangle;
    - `potrf_vbatched` skips matrices whose stale info is non-zero;
    - `getri_outofplace_batched` never writes `info_array`;
    - `gegqr_gpu` reports rank 0 as info = 0 with a non-orthogonal Q.
  - `hgetrf_gpu` with an unaligned leading dimension returns corrupted factors (20,903 entries differ) with info = 0.
  - `Magma/Dense/StatusAudit.lean` models these items. Its `trtri_gpu_instances` reproduces the three GPU info values exactly from the block size nb = 256.
  - One prediction was not reproduced: an IC(0)-breakdown matrix had no visible effect through ICC + PCG.
  - A third round added three more:
    - `getrf_m`, the out-of-core LU behind `zgetrf`/`zgesv` when the matrix exceeds GPU memory, never reports singularity: its final pivot pass resets `info` to 0 (zero columns give 0 where LAPACK gives 101, …, 901; `getrf_m_never_singular`);
    - `gesvj_batched` returns info = 0 with all-NaN singular values;
    - `hgetrf_gpu` returns dA still transposed after a singular panel.
  - A scan of 1152 sparse FGMRES solves found 252 with info = 0 and a non-finite x or a large true residual, consistent with `fgmres_sticky_success`.
- *A sweep of all 32 sparse solver classes × 6 inputs on GPU 0.*
  - With a NaN in b, exactly the four solvers predicted by the source-level classification (BiCG, PBiCG, PTFQMR, PTFQMR_MERGE) report success.
  - With a tiny b, 21 of 32 report success with a wrong answer (info = 0): 5 return the initial guess unchanged (numiter 0), and 16 iterate without converging.
  - An exact initial guess is reported as a failure by five solvers.
  - IDR_MERGE and PIDR_MERGE fail even on a well-conditioned real SPD system that IDR solves in 1–10 iterations.
- *A second pass over the parts not yet audited* (sparse formats, SpMV, scaling and preconditioner setup; dense QR, least squares, generalized and tridiagonal eigensolvers; precision-conversion kernels). On GPU 0, each of the following returns success with a wrong or missing result:
  - `magma_z_spmv` returns 0 without computing when x is shorter than A's column count (`spmv_mismatch_success`). With a DENSE matrix it computes Aᵀx, because the conversion stores row-major and the kernel reads column-major (`dense_spmv_transpose`). With CSC from `magma_zmconvert` the result is wrong. With SELL-P at alignment 1, 44 of 300 rows of a tall matrix are never written: the grid is sized by columns (`sellp_tall_rows_skipped`).
  - The sync-free ILU triangular solve. On CUDA ≥ 11 a macro passes the CSC output arrays to `cusparseCsr2cscEx2` in the legacy order, so the n+1-entry row-pointer array receives the nnz row indices; on GPU 0 it holds 127 entries in an array allocated for 65 (`syncfree_csc_swapped`). Setup returns 0. With a guard allocator, the corruption shifts and PBiCGSTAB reports success at relative residual 1.
  - `magma_zmscale(UNITDIAG)` returns 0 with every entry NaN for a negative or purely imaginary diagonal (`mscale_nonfinite_success`).
  - `magma_dsygvdx` for n ≤ 128 ignores the requested range: il = iu = 50 returns all 50 eigenpairs, smallest first (`hegvdx_small_ignores_range`).
  - `magma_dgels_gpu` on a rank-deficient matrix returns info = 0 and a finite x with ‖b − Ax‖/‖b‖ = 95; LAPACK returns info = 1. The diagonal-block inverse discards `trtri`'s singular status and keeps R itself (`geqrf_singular_block_used`).
  - `magma_dgeqp3_gpu` never factors a fixed (pre-pivoted) column: info = 0 and the column comes back bit-identical (`geqp3_fixed_unreduced`). A fixed column that must be swapped segfaults, because a host `zswap` is applied to device memory. The documented LWORK is rejected with −8 (`geqp3_doc_lwork_rejected`).
  - `magma_dstedx` returns info = 0 with 10 NaN eigenvalues, because a later block's `dlascl` clears the failure of an earlier block (`stedx_info_erased`).
  - `magmablas_dlag2s`/`slag2h` let NaN pass the range check (`lag2s_nan_passes`, `lag2h_nan_passes`). `slag2h` passes the host shadow of its device flag to the kernel: any out-of-range entry causes an illegal memory access, and info is still 0 (`lag2h_info_always_zero`).
  - Two failures are reported, but caused by the library:
    - complex ICC applies L⁻ᵀ instead of L⁻ᴴ, so PCG+ICC fails after 1000 iterations on a Hermitian tridiagonal system that unpreconditioned CG solves in 17 (`icc_transpose_not_conjugate`);
    - `shpotrf_gpu` reports an SPD matrix scaled by 2^40 as not positive definite (info = 1025), because its FAST_16F GEMM overflows without a range check.
  - Two predictions were not confirmed:
    - the ILU(k) symbolic-storage overflow is reported (−3009) rather than silent;
    - the shpotrf allocation-failure path that crashes is compiled only for HIP.
  - One finding is specific to HIP builds and has only code-reading evidence (no AMD GPU was available): after argument checking, `gesv_rbt` sets `refine = MagmaFalse` (an author TODO cites failures on AMD GPUs). A caller who asks for refinement gets the unrefined no-pivot solution, and `info` does not say so (`rbt_hip_refine_dropped`, `rbt_hip_refine_irrelevant_all`). On CUDA, that unrefined solution had relative error 1.6e13. Of the 126 source files with HIP-specific branches, we read the status-relevant ones (the rest only swap BLAS calls or tune launch sizes); in all of them except `gesv_rbt` and `shpotrf`, the status logic matches the CUDA path.
  - A systematic workspace sweep (then checked by an independent reviewer, who confirmed every item) compared, for all 83 z drivers with an `lwork`, `lrwork` or `liwork` argument, three sizes: the one returned by a query, the minimum the check accepts, and what the code and its sub-calls use. It found new defects beyond the five lwork items above (code-reading evidence only; `Magma/Dense/Workspace.lean`):
    - `zhegvdx_2stage(_m)` reports and accepts `5n + n·nb` without eigenvectors, but passes the same lwork to `zheevdx_2stage`, which needs `5n + 2·nb·n`. For every n > 128, calling with the size the routine itself reports fails with −14 (the inner routine's argument number) after A and B have been overwritten (`hegvdx_2stage_query_rejected_all`). With eigenvectors, the shortfall occurs exactly when n < 2·nb (`hegvdx_2stage_vec_short_iff`). The real version `dsygvdx_2stage` is short by 4n + nb·n (`sygvdx_2stage_real_short_all`).
    - In `zgesdd` path 6to-slow (m < n, `jobz = O`, lwork between the minimum and the fast-path size), the chunked VT buffer takes the space `zungbr(P)` needs. Once min(m,n) ≥ 1024, `zungbr` returns −9, the status is never read, and the right singular vectors are wrong with `info = 0` (`gesdd_slow_ungbr_short_all`; smallest case m = 1024, n = 1087).
    - `zhetrd_he2hb_mgpu` repeats the single-GPU lwork ≥ 1 check: a host underrun, and at lwork = nb² − 1 a silent workspace query in place of the panel QR (`he2hb_mgpu_underrun_all`, `he2hb_mgpu_silent_query`). In both versions even the size the routine reports (n·nb) underruns whenever 0 < n < nb (`he2hb_query_underrun_all`).
    - `zgbtrf_native`/`zgbsv_native` never clear `*info`, so a stale nonzero value makes them return at once without factoring (`gbtrf_native_stale_info_all`).
    - Three workspace queries are rejected by the routine's own check or not answered: `zgeqlf` with min(m,n) = 0, the batched band-LU query on its quick return, and `zhegvr` with `lrwork = −1`.
    - In ten routines the documented minimum is below the accepted minimum, so a caller following the documentation is rejected (`doc_minimum_rejected_all`).
  - The sweep also turned up two defects that are not about workspace size:
    - `magma_[sdcz]ungtr` (exported, but not declared in any public header and never called in the tree) shifts the hetrd reflectors with the wrong loop bounds compared with LAPACK ZUNGTR. In the upper case, one entry per column is not moved; in the lower case, the diagonal is overwritten, the last row is not moved, and A(n−1,0) is not cleared (`ungtr_upper_shift_short`, `ungtr_lower_shift_wrong`, `ungtr_first_col_stale`). A CPU check ran the loops copied verbatim, followed by reference LAPACK (`experiments/magma_dense/orgtr_shift.c`). At n = 50, it gives ‖QᵀQ − I‖ = 9.2e19 in the upper case and 1.8 in the lower case. The same code with LAPACK's loop bounds reproduces LAPACK exactly.
    - In an ILP64 build, the device info slot of `zgbtrf_native_work` is at byte offset 4, so the kernels' 8-byte write to it is misaligned (`gbtrf_info_alignment_all`).
  - A third scan evaluated every workspace-size expression with a product of two sizes in 32-bit `magma_int_t`. In all 14 divide-and-conquer eigensolver drivers, `lrwmin = 1 + 5n + 2n²` wraps negative for every n in [32767, 46339]. The routine's own check then accepts any lrwork, and the workspace query reports a negative size, although no 32-bit lrwork is large enough. The call proceeds through the tridiagonal reduction, which overwrites A, and fails inside `dstedx` with −12. In zheevd's numbering, −12 names `liwork` (`heevd_lrwmin_wraps_all`). This generalizes the multi-GPU instance confirmed in round 24.
  - In the batched solvers, the per-matrix failure is returned only when the library is built with `CHECK_INFO`. That option cannot be compiled in three of the four drivers that have it (`zposv_batched`, `zgesv_batched`, `zgesv_nopiv_batched`): they call the seven-argument `magma_getvector` macro with six arguments, and `zgesv_nopiv_batched` also reads an undeclared array. Running `cc -E` on the copied lines reports exactly these three calls (`check_info_build_broken`; `experiments/magma_dense/check_info_pp.c`).
  - A scan of kernel launches whose y or z grid dimension depends on the batch count found one launch without chunking. `magma_zgbtrf_set_fillin` puts the full batchCount in gridDim.z (limit 65535), whereas its sibling kernels and the strided interface chunk at 65534. The pointer-array `magma_zgbtrf_batched` reaches it whenever the band is too wide for the shared-memory kernels: for example kl = ku = 32 on a V100, about 155 KB per matrix and 10 GB for 65,536 of them. The launch then fails unchecked, and the band LU continues with an unwritten `ju_array` (`set_fillin_launch_invalid_all`, `gbtrf_generic_path_reached`).
  - Kernel offsets are computed in 32-bit `int`. An independent read of all 123 z kernel files found that 89 form a matrix-wide column offset `j·ld` in `int`. These include lacpy, laset, geadd, lascl, lange, the magmablas gemm/hemm/gemv/hemv templates, trtri and the native laswp variants. Only 4 files use 64-bit offsets, among them transpose and the main laswp kernel (`offset_audit_covers_all`). The default hybrid `getrf_gpu` uses only the 64-bit-safe kernels. `getrf_native` swaps rows through `laswp_columnserial`, whose `i1·lda` wraps (`getrf_native_laswp_wraps`). `potrf_gpu` computes its host column pointers as `j·ldda` in 32-bit `magma_int_t`, whereas `getrf_gpu` casts to `size_t`. In the default LP64 build, every column pointer beyond j = 42,950 of a 50,000-order matrix therefore wraps (`potrf_gpu_host_offset_wraps`). The same audit found two unrelated defects. `magmablas_ztranspose_batched_stride` offsets each chunk of 65,534 matrices by `i` instead of `i·stride` (`transpose_stride_chunk_wrong_all`). `magmablas_zswapblk` stores pivot targets in `short` (`swapblk_short_truncates`). Lacpy was traced line by line (`lacpy_offset_negative_all`).
  - **Second GPU and a differential sweep.** On a second machine (RTX 5070 Ti, sm_120, Windows), MAGMA was built from the same tarball with CUDA 13.5 and OpenBLAS 0.3.34.
    - Round 26 reproduced every code-reading finding above that fits in its memory: the zhegvdx_2stage query rejected with −14, the zgesdd slow path, zungtr, stale gbtrf_native info, the hegvx JOBZ/RANGE swap, the zcposv_gpu uplo path, the he2hb query underrun, the zgeqlf query, and the failed 65,536-batch launch with return value 0.
    - Round 27 compared most dense operators with LAPACK on 17,327 adversarial cases: block-boundary sizes, zero columns, zero matrices, extreme scaling, NaN and Inf. Every non-OK cluster was triaged; the harness false positives are documented.
    - The sweep found a defect that code reading had missed. `zgeqp3` returns a wrong factorization (A·P ≠ Q·R, residual 0.2–6) with info = 0 on rank-deficient matrices, in 18 of 78 random rank-n/2 cases. There are two root causes in `magma_zlaqps`. After a host-side column swap, the GPU copy is restored only below row offset+nb, but the rows above it stay in the device trailing matrix when the panel stops early (`geqp3_swap_restore_short_all`). Flagged column norms are recomputed from host data that was not updated, with an off-by-one row split (`geqp3_norm_rows_off_by_one`). A patched copy of `magma_zlaqps` linked in front of the library reduced the failures from 18 to 0. A third sweep, on 5,125 inputs with prescribed spectra, showed that the defect is systematic. Every rank-1, cond-10¹², and row-graded matrix of order ≥ 63 is factorized wrongly (92 of 238 cases); with the patch, 714 of 714 cases are correct. The same sweep found that `zgeqp3_gpu` requires a device `tau` although `magma_z.h` declares a host pointer; a host array gives an illegal memory access on every call (`geqp3_gpu_tau_space_mismatch`).
    - The sweep also found five smaller defects. `magmablas_zgemm` with k = 0 does not scale C by beta. `magmablas_zgemv`/`zhemv` read y when beta = 0, contrary to their own documentation. `zgels` returns a NaN solution with info = 0 for singular or zero A. `zheevdx` with n = 1 leaves `mout` unset. Six GPU-interface routines reject n = 0 because their small-matrix path passes lda = n to LAPACK.
  - **A platform-specific defect.** The second machine runs Windows, and there a sweep of the batched Jacobi eigensolver and SVD found that their sorting step swaps 64-bit values as one `long`, which is 32 bits under LLP64 (`magmablas/swap_scalar.cuh`). Only the low halves are exchanged while the index array is swapped correctly, so sorted eigenvalues and singular values are perturbed and paired with the wrong vectors, with info = 0. This affected 313 of 784 heevj cases and 1016 of 1568 gesvj cases. With sorting disabled through the expert interface, 0 of 64 cases are wrong (`swap_long_llp64_wrong`). The same sweep found that the blocked gesvj scales U by 1/σ with no zero guard, so it returns NaN for zero or rank-deficient input (`gesvj_zero_sigma_nan`). The first defect cannot be triggered on Linux.
  - **Where the defects are.** All 238 distinct findings are collected in `operators/magma/FINDINGS.csv`; 224 of them were reproduced or are code-reading results, and 83 % of those were confirmed on a GPU.
    - About half are status plumbing: missing checks, lost or overwritten status, wrong status values, wrong argument numbers. 27 % are wrong numerical results, and 22 % concern memory or size arithmetic.
    - 52 % of the defects give a silently wrong result.
    - 62 % need an input or path away from the main one: singular or rank-deficient input, non-finite input, boundary sizes, scaling, allocation failure, or build options.
    - Defect density rises from the compute kernels to the composition layers. It is 0.48 per kLOC in the magmablas kernels, 1.7 in the dense drivers, 2.2–2.8 in the sparse solver and utility layers, and 8.4 in the mixed-precision refinement code. Audit effort was concentrated on the last of these.
    - The recurring mechanisms are code copied from LAPACK without adapting it to the signature, quick-return and small-size paths, host/device copies that diverge in hybrid algorithms, status lost between layers, workspace and 32-bit size arithmetic, NaN that slips through comparisons, and build-configuration branches (`operators/magma/BUG_DISTRIBUTION_zh.md`).
  - Two mechanical scans covered all of src and magmablas.
    - Functions that read `*info` before writing it: apart from the band LU above, only three public copy helpers that never set `info` (`htod_returns_caller_value_all`).
    - Argument checks whose reported −k is not the position of the argument tested: 58 checks in 33 routines, each confirmed against the signature by hand (`argnum_all_wrong`). Most come from LAPACK checks copied into a signature with extra block-size, offset or batch parameters. In the generalized eigensolvers `zhegvx`, `zhegvr`, `zhegvdx` and `zhegvdx_2stage`, the JOBZ and RANGE codes are swapped (`hegv_jobz_range_swapped`). `zcposv_gpu` never checks `uplo`: an invalid value is reported as a precision-conversion failure (`iter = −2`) before the FP64 fallback returns −1.
  - A last round ran the remaining code-reading findings (`MagmaSparse` §14, `MagmaAudit` §19):
    - `magma_z_spmv` on CPU data returns 0, frees the caller's arrays, and loses the result;
    - CSR→CSRD with a row missing its diagonal writes past the array, and glibc aborts at a later `malloc`; ELLD stores column −1;
    - `zmdiff` reports 0 for I₂ against a matrix with no common entry, and for 1+5i against 1−5i;
    - `zmfrobenius` adds the caller's initial `*norm`;
    - `zmdiagadd` skips rows with no stored diagonal;
    - `zmtranspose` of a 1×3 CPU matrix returns 1×3;
    - an out-of-range index in a `.mtx` file segfaults the reader;
    - `magma_z_applyprecond` with ILU or ICC returns 0 and never writes x, and `applyprecond_right(ConjTrans)` does the same;
    - `zhegvx` with a non-positive-definite B leaks 21 MB of device memory per call at n = 512;
    - the CPU `magma_dgels`, given its own queried workspace, makes LAPACK `dormqr` reject parameter 12 when nrhs = 16384, and reference-LAPACK `xerbla` ends the process;
    - `dgeqrs3_gpu` reports a bad `lddb` as argument 8 (it is argument 9);
    - `magma_zgels_batched` is declared in the header but not exported.
  - A ninth round covered the 31 multi-GPU and out-of-core drivers, run with ngpu = 1, and the last three dense files not yet audited. An inventory of all 202 `src/z*.cpp` files is in `operators/magma/INVENTORY.md`: 127 audited earlier, 31 in this round, 3 newly read, and 41 helpers with no status decision.
    - Range-V eigenvalue selection compares the internally rescaled eigenvalues with the unscaled bounds. On A = 1e-150·diag(1..1000), all four `zheevdx` variants return m = 10 instead of 100 with info = 0 (`rangev_unscaled_bounds`).
    - A NaN in A ends the calling process inside every `zheevdx` variant: `dlascl` rejects a NaN scale factor and reference-LAPACK `xerbla` stops. LAPACK `zheevd` returns instead (`dlascl_nan_rejected`).
    - `zunmtr_m` swallows `zunmqr_m`'s NOT_IMPLEMENTED for side = Right: info = 0 with C unchanged (`unmtr_m_not_implemented_lost`).
    - `zhetrd_he2hb` checks only lwork ≥ 1. With lwork = 1056 the band differs by 2.5e119 with info = 0; with lwork = 100 the heap is corrupted (`he2hb_small_lwork_underflow`).
    - `zgeqrf_m` accepts ngpu = 0 and dies with SIGFPE (`geqrf_m_ngpu0_accepted`).
    - `zcgeqrsv_gpu` and `zgglse` return a non-finite x with info = 0 where LAPACK reports info = 1 (`cgeqrsv_nan_accepted`, `gglse_singular_unreported`).
    - `zggrqf` accepts its documented minimum workspace, overwrites A and B, then returns the inner −7 (`ggrqf_lwork_inconsistent`).
    - `magma_zmscale_generate(UNITCOL)` double-frees (`mscale_generate_double_free`).
    - `zpotrf_mgpu_right` is declared but not exported.
  - A tenth round tested allocation failures by fault injection. An LD_PRELOAD shim fails the k-th `cudaMalloc`/`cudaHostAlloc`, k = 1…all, inside 11 drivers, and each result is compared with LAPACK. That is 55 failure runs:
    - **6 silently wrong (info = 0):**
      - `zhegvd` and `zhegvd_m` return the eigenvalues of A instead of (A, B) when `zhegst`'s workspace allocation fails (relative error 599);
      - `zheevdx_2stage` is wrong when `zhetrd_he2hb`'s allocation fails (error 0.498);
      - `zheevdx_2stage_m` is wrong when `dlaex0_m`'s allocation inside `zstedx_m` fails; the next two stages reset `info`, which confirms the predicted stedx overwrite (error 1);
      - `zgeev` and `zgeev_m` are wrong when `zgehrd(_m)` fails (error 0.067);
      - modelled by `early_alloc_failure_hidden`.
    - **8 end the process** via `dlascl`/`xerbla`, because `zhetrd`'s status is not read (`hetrd_failure_reaches_dlascl`).
    - **3 segfault**, because `zgetrf_gpu` and `zpotrf_gpu` do not check their pinned allocations (`pinned_alloc_unchecked`).
    - The other 38 report the failure (19) or are unaffected (19).
  - Out-of-core `zgeqrf_ooc` accepts lwork = n, far below the documented n·nb, and segfaults (`geqrf_ooc_lwork_short`). The predicted infinite loop at NB = 0 did not occur; the routine returns −113 instead.
  - With ngpu = 2 and one visible GPU, `zpotrf_m` and `zheevd_m` compute correct results but leave a sticky "invalid device ordinal" CUDA error.
  - `zhegvd`'s mapping of a negative `zpotrf` status is unreachable in 2.10.0, because `zpotrf_gpu` never returns an allocation error.
  - *An independent review.* A fresh agent re-checked the inventories against the source tree, re-read 18 randomly chosen "clean" verdicts, and compared nine numbers in the coverage table with the result files.
    - The listings were complete, apart from one kernel file whose status variable our search had missed (it has no defect).
    - Two counts were wrong and are corrected above: 21 of 32 tiny-b solves, not 23, and 6 silent fault-injection results, not 7.
    - It disputed four "clean" verdicts. Three were confirmed on GPU 0, and the fourth (`zcsrsplit`) was confirmed later, in the twenty-third round (below):
      - `zungbr` drops the status of its inner `zungqr2`/`zunglq`, returning info 0 with relative error 52 (Q) or 8.6 (P) after an injected allocation failure;
      - `zlascl_2x2` checks the wrong arguments (`lascl2x2_args`);
      - `zresidual` computes only the first of several residuals (`residual_multi_first_only`).
    - It also found 18 non-z files that no inventory covered: 15 on the z call graph, plus `dlaln2`, `dlaqtrsd` and `dsidi`; `dsiinertia`/`ssiinertia`, `magma_trisolve.cpp` and `mmio.cpp` were added after the second review. These are now classified file by file. The `dlaex*` merge routines, `dlaln2` and `dlaqtrsd` agree with LAPACK. Ten new candidates were found, including range-V eigenvectors selected against unscaled bounds and `adjust_ipiv` silently doing nothing for m > 1024. The GPU round was queued until GPU 0 was idle. Five of its six findings were confirmed:
      - range-V eigenvectors from `zheevdx` and `zheevdx_2stage` are wrong with info = 0, at max|AV − VΛ|/max|A| = 13.5 and max|VᴴV − I| = 296, while the eigenvalue count is right (`stedx_rangev_unscaled`);
      - the range is closed at vl, so 11 eigenvalues come back where LAPACK returns 10 (`rangev_closed_interval`);
      - native LU with nb = 600 returns 1200 invalid pivots and max|PA − LU|/max|A| = 52.7 with info = 0 (`adjust_ipiv_noop_large`);
      - blocked `gesvj` turns a zero column into a non-finite U with info = 0 (`gesvj_blocked_zero_sigma`);
      - unscaled column norms give S = Inf near 1e160 and a non-finite U near 1e-170.
    - A further round ran the remaining candidates (four in round 21, #6 in round 24):
      - **The tridiagonal divide-and-conquer driver loses the block of a failure.** We forced one LAPACK `dsteqr_` call to fail with an interposed library. Failures in the first leaf of block 1 and of block 2 both return info = 226 (`stedx_blocks_indistinguishable`).
      - **Batched `gesvj` reads another problem's convergence flags past the first chunk.** In a batch of 70,000 matrices, all 47 sampled matrices beyond index 65,534 have wrong singular values (relative error up to 0.49), all with info = 0 (`gesvj_chunk_flags_not_offset`).
      - `dsidi` returns a wrong inverse for its documented input with info = 0: max|A·X − I| = 0.13 even for an SPD matrix without 2×2 pivots (`dsidi_inverse_wrong`).
      - The batched sort's rank method gives a NaN the same rank as the maximum, so its index array is not a permutation (`sort_nan_rank_collision`).
      - The 32-bit size arithmetic of the multi-GPU divide-and-conquer path (#6) was reached with ngpu = 1 by keeping Z in host memory (`dc_sizes_wrap`).
        - At n = 38,000 the call fails cleanly with −12. The real workspace requirement (2.89 × 10⁹) exceeds INT_MAX, so no 32-bit `lrwork` can be valid. The error is raised by the inner `dstedx_m`, after the outer check was passed by the wrapped minimum.
        - At n = 53,600 the process segfaults. Several 32-bit size and index expressions overflow on this path (the GPU workspace size wraps from 4.3 × 10⁹ to 1.46 × 10⁷ doubles), and we did not establish which one causes the crash.
  - *A second review* checked that the first review's corrections had landed. The Lean totals and all sampled numbers matched. Most of the roughly 18 stale or contradictory passages it found were fixed; a third pass found and fixed the remainder. Two round-19 "not reproduced" results had used inputs that never reached the code in question. Rerun with suitable inputs, both bugs are confirmed:
    - `zresidual` with a zero-row A and a non-empty b dies with SIGFPE (`residual_div_by_zero`);
    - `zcsrsplit` with an unsorted row and block size 2 writes past the row and corrupts the heap (`csrsplit_unsorted_overflow`).
    - The remaining review items were also confirmed:
      - `zdiinertia` and `zheinertia` count a NaN pivot as a zero eigenvalue (`diinertia_nan_as_zero`);
      - `zmcsrcompressor` silently does nothing on a matrix that does not own its arrays (`compressor_not_owned_noop`);
      - `zmlumerge` double-frees when its 4th host allocation fails, inferred from where the abort falls in the output (`lumerge_double_free`);
      - three routines have mismatched header declarations and library definitions: `magma_zdiagcheck_cpu`, `magma_zmlumerge` and `magmablas_zdiinertia` (`declaration_mismatches`).
  - The ngpu ≥ 2 paths cannot be run here, because only one GPU may be used. Their status bookkeeping is instead proved correct for every ngpu, nb and n (`Magma/MultiGpu.lean`). The block-cyclic distribution of `zgetrf2_mgpu` assigns every column exactly once (`distribution_covers`). The multi-GPU LU reports the first failing column with its global offset (`getrf_mgpu_first_global`), and the multi-GPU Cholesky reports `j + local info` (`potrf_mgpu_global`). These index and status results are positive: no defect is predicted there. The ordering of the multi-GPU Cholesky is not: a happens-before replay of its host program finds two buffer-reuse races, which we confirmed by running the unmodified source with virtual devices on one GPU and delaying a single operation (M255: the device diagonal buffer, z/c, ngpu ≥ 2; M256: the host workspace, whose hard-coded h = 2 is safe only for ngpu ≤ 2, and `potrf_mgpu_host_slot_covered_iff` proves h ≥ ngpu is the exact condition). A layout convention is also not preserved across the multi-GPU Hessenberg pair: `gehrd_m` rounds ilo down and stores T by global column, while `unghr_m` reads it by sub-problem column, so for ilo > 1 every block uses the wrong T (M259, `unghr_m_inconsistent`). This needs no second GPU: `zgeev_m` and `dgeev_m` return eigenvectors with residual 8.6–15 and info 0 whenever balancing permutes the matrix, while the single-GPU pair is consistent for every ilo (`unghr_single_consistent`). With pinned host work, as MAGMA's own tester allocates it, `gehrd_m` also lets `zgehd2` overwrite `work` while the panel device's last `Y = work → dY` copy is still queued, because only the devices holding the R ≤ 2 remainder blocks are synced (M260, `gehrd_m_panel_unsynced`).
  - The helpers `zunmtr`, `zunmbr`, `zunghr` and `zungbr` discard their inner routine's status. Under the same fault injection they return C unchanged, or a wrong Q, with info = 0 (`helper_iinfo_dropped`).
- *The status logic inside the GPU kernels.* All 39 `magmablas` z-files that write an info value or a flag were read; the per-file verdicts are in `operators/magma/COVERAGE_zh.md` §3e (the reviews added three more: `zgetf2_nopiv_devicefunc.cuh`, and the two inertia kernels `zdiinertia.cu` and `zheinertia.cu`; 42 status-writing files and 123 without status in all). Ten findings were run on GPU 0, and all ten were confirmed:
  - the variable-size batched LU erases a zero pivot outside the first panel: A(40,40) = 0 gives info 0, where LAPACK gives 40 (`vbatched_panel_info_erased`);
  - the generic batched band LU keeps only the last column's status: A(5,5) = 0 gives 0, and A(200,200) = 0 gives 1 (`gb_generic_info_last_column`);
  - `zgetf2_gpu` never reports singularity, and its factor has 30 non-finite entries (`getf2_gpu_never_singular`);
  - `zpotf2_gpu` and `zpotrf_native(Upper)` miss an exact 0 or a NaN diagonal; `zpotrf_batched` misses NaN when n = 12, but not when n = 16 (`potf2_zero_nan_pass`);
  - `zgetrf_native` misses an all-zero column when m = 16384, because columns longer than 15360 go to cuBLAS `Izamax` and info is never written (`izamax_long_column_no_info`);
  - non-square batched band LU reports a spurious info of 4 where LAPACK gives 0, or hides a real zero pivot and returns ipiv = 5 > m (`gb_km_one_extra_row`);
  - `zheevj_batched_small_sm` reports a NaN matrix as converged with info = 0, and all 8 eigenvalues come back finite (the NaN is silently absorbed) (`heevj_nan_converged`);
  - `zgetrf_nopiv_expert_vbatched` never counts replaced pivots, because `*info++` increments the pointer (`nopiv_count_not_incremented`);
  - `zgesv_batched` overwrites B with non-finite values for a singular matrix that it correctly reports (`gesv_small_solves_singular`).
  - A thirteenth round ran the remaining kernel items:
    - the native recursive LU's missing panel offset gives info 18 where LAPACK gives 50, the value the model predicts (`native_recursive_offset`);
    - fused batched getf2 with m < n gives a spurious info of 5 (`fused_mlt_n_spurious_info`);
    - a NaN in the pivot column makes the tree argmax return 0, so `zgetrf_batched` reports a spurious info of 1 for m ≥ 16 (`argmax_nan_hides_max`);
    - `zlascl` reports the wrong argument numbers.
  - The suspected race on the atomic info write was not reproduced: 300 of 300 runs were correct.
  - A fourteenth round ran every item that had been backed only by code reading. Four were confirmed:
    - **The single global `zlag2c` overflow flag races.** With two host threads making 2000 calls each, the overflowing matrix was reported clean in 1889 calls, and the clean matrix was reported as overflowing in 7 (`lag2c_shared_flag_race`).
    - `zheevj` with max_sweeps = 0 reports success for all 64 matrices, although every eigenvalue set is wrong (`heevj_zero_sweeps_success`).
    - The CPU transpose of a 1×1000 matrix segfaults (`transpose_cpu_heap_overflow`).
    - A truncated `.mtx` file is read as if complete, and the missing entry is duplicated from the previous one (`mtx_truncated_phantom`).
  - Three did not reproduce: the `zlascl2` overread (the compiled code moves the load inside the bounds check), the `zpotf2` warp reduction (1000 identical runs), and the `zheevj` convergence-flag race (12,800 correct solves).
- *A per-file pass over the 66 sparse control/blas z files* (`operators/magma/SPARSE_INVENTORY.md`): 14 have known findings, 31 have new candidates, 13 are clean (5 of them later revised by the reviews), 8 have no status logic. Thirteen new candidates were run on GPU 0; 11 were confirmed and 2 were not:
  - **Confirmed:**
    - the CSR, ELL and SELL-C SpMV kernels ignore `transA` (`csrmv_trans_ignored`);
    - the complex shuffle reduction behind `zgemvmdot_shfl`, which IDR_MERGE uses, returns its input, so 64 ones sum to 1 (`mdot_shfl_reduction_noop`). This plausibly explains why IDR_MERGE and PIDR_MERGE failed in the sweep;
    - multi-vector CSR SpMV strides the output by the column count (`mgecsrmv_stride`);
    - multi-vector SELL-P computes only half the vectors (`sellp_multi_half`);
    - `zvread` cannot read the two-column format the library itself writes, and overflows the heap (`vread_two_columns_overflow`);
    - `zcsrget` returns pointers into a freed copy (`csrget_dangling`);
    - the LOBPCG `zcompact` treats a NaN norm as converged (`compact_nan_converged`);
    - `zmcsrcompressor_gpu` empties a CPU matrix (`compressor_cpu_emptied`);
    - `zfrobenius` uses real parts only (`frobenius_real_only`);
    - `zmdiagdom` returns avg = NaN for an empty row;
    - `zsort` never terminates on NaN input (`zsort_nan_no_progress`).
  - **Not confirmed:** the ELLPACKT padding test; the NaN we saw comes from complex Inf arithmetic. The `zmdotc` test was inconclusive (its output was never written) and was confirmed in the sixteenth round.
  - A sixteenth round ran 13 of the remaining candidates. Eleven were confirmed:
    - `zmdotc` and `zgemvmdot` are unconjugated, returning −1 for v = r = (i) (`mdotc_unconjugated`);
    - `zcsr_sort` sorts the column indices but not the values (`csr_sort_values_not_permuted`);
    - `zcsrcoo_transpose` keeps the dimensions;
    - `zbitonic_sort` drops its own error and returns an unsorted result (`bitonic_error_dropped`);
    - `zvget_dev` returns a freed device pointer;
    - `zcsr_sort_gpu` leaves a device pointer in a CPU matrix;
    - `zge3pt` with m = 1 reads `dx[-1]` (`ge3pt_reads_before_start`);
    - the ELLRT SpMV launch fails with "invalid configuration", yet returns 0 with y unchanged (`ellrt_launch_unchecked`);
    - `zjaccard_weights` gives 1/3 for identical rows (`jaccard_identical_rows`);
    - `zlobpcg_shift` shifts only row 0 (`lobpcg_shift_row0_only`);
    - `zcuspmm` on CPU matrices segfaults in its cleanup (`cuspmm_cpu_uninit_destroy`).
  - Two were not reproduced: `zmbdiagdom` with oversized block sizes, and `zcuspaxpy` (cuSPARSE asked for no buffer, so the uninitialised buffer was never used).
  - A seventeenth round called the four remaining routines through their exported symbols, since they are absent from the public headers. All four were confirmed:
    - **The sync-free triangular solve is wrong even with correctly laid-out CSC arrays.** Its dependency wait is commented out, so a lower bidiagonal system with n = 4096 has 4095 of 4096 entries wrong, and the call returns 0 (`syncfree_wait_disabled`).
    - `zcsrget_gpu` returns a freed device pointer.
    - `zorderstatistics` reports NaN only when it is not in the last slot (`orderstat_last_unchecked`).
    - `zrowentries` misses purely imaginary entries.
    - `ztrisolve` returns a finite "solution" with status 0 for a structurally singular factor (`trisolve_singular_success`).
  - All 31 ids of the per-file pass have now been run; 29 were confirmed.
  - Eleven of the 12 public dense helpers that allocate were fault-injected one by one, and all eleven report the failure. `zunmtr`, `zunmbr`, `zunghr` and `zungbr` do not allocate themselves, but the routines they call do; under the same injection they return wrong results with info 0. `zhetrd_hb2st` was not injected on its own. `zunglq` reports an injected failure correctly. Its declaration in `magma_z.h` takes (…, dT, nb), but the definition takes (…, work, lwork), and a call written against the header segfaults (`unglq_header_mismatch`). Per-file verdicts for all 41 helpers are in `operators/magma/INVENTORY.md`.

Observations:
- **Three runs were worse than predicted.** SuperLU_DIST masked *all four* status quantities, not only berr. STRUMPACK's default double-precision solver has the same defect as its mixed one. SLATE loses the solution even on an easy system.
- **cuSOLVER's classical IR conforms to its documentation.** It is scale-invariant across k = −20…20 with constant niters, and has 15/15 bit-identical niters and error across scalings (R3IRS). The defect is specific to the GMRES path and to the one-line `D?gesv` APIs that use it.
- **The cuSOLVER result is robust.** It reproduces across 3 rounds: 10 seeds; 4 matrix families × 2 sizes × 3 seeds; and one process per call with exact-rational residuals. Decimal scalings (1e6, 1e12, 1e−18) trigger it too.
  - Over R1 and R2: 140 of 306 scaled calls give a relative error above 1e-6 (42/90 in R1, 98/216 in R2), and 89 exceed the documented threshold by more than 100× (19 and 70).
  - Every accepted solution's exact residual matches its FP64 residual to within 1%. The violation is not a rounding artifact.

### 6.3 RQ3: Why do the libraries' tests miss them?

We read the test suites, default test lists and CI scripts of the 8 libraries in Table 3 (excluding cuSOLVER and LAPACK).

| Finding | Evidence |
|---|---|
| **0 of 16** defects is triggered by any default test or CI run | No test puts NaN/Inf in b, scales (A, b) below 1e-12, or passes x0 = 0 to PaStiX. b is U(0,1) or A·1 |
| 6 routines have no checked test at all | MAGMA `zchesv`; SLATE `slate_dsgesv` and `pdsgesv`; SuperLU_DIST's d2 path (needs `-B n -i 2`, never registered); MUMPS refinement (examples never set ICNTL(10)); STRUMPACK mixed (`test_SPD_mixedPrecision` only prints and returns 0) |
| Test oracles share the defect | SuperLU's checkers use `SUPERLU_MAX` (`dgst02.c:121`, `dgst07.c:129,151,193`) and fail only if `result >= THRESH` (`ddrive.c:473`), so a NaN passes. SuperLU_DIST's `psinf_norm_error` uses `SUPERLU_MAX`. STRUMPACK's `max_scaled_residual` uses `std::max` with `if (res > tol)`. MUMPS examples check only `INFOG(1) < 0` |
| Hazard-capable generators are never used for IR | MAGMA `_ofl/_large`, SLATE `gfpp` and `--matrixB` are available options, but no list applies them to IR routines; neither generator has NaN/Inf kinds |
| MAGMA's test metric is not a backward error | ‖r‖/(N‖A‖) has no ‖x‖ term, so x = 0 for a tiny b passes |
| Positive example | SuperLU_DIST's `ptest_ci.c` is written so that "a NaN anywhere fails the test", but no registered run exercises the d2 path |

The third row generalizes the contract argument. A test oracle is itself an acceptance check, and it can violate C1 in the same way as the code it tests.

### 6.4 RQ4: Does the certified gate work, and what does it cost?

- **cuSOLVER wrapper** (Dot2 residual; test evaluated exactly with rationals; n = 1024; seeds 1–3; DB/DX/DS × {2^0, 2^20, 2^40, 2^−60}; 36 calls):
  - All 13 "converged but wrong" results are rejected and re-solved with DDgesv, then certified.
  - 11 further results that are small but above the threshold are also rejected.
  - 12 are accepted directly.
  - **36/36** are certified, with final error ≤ 1.5e-11.
  - The same test with an FP64 residual certifies **0/36**, as `fp64_cannot_certify` predicts.
- **MAGMA patch** (`gmstrs_fix.patch`, 163 lines; applies cleanly to v2.10.0). In a faithful CPU port with the patch's certification function compiled verbatim (n = 256, k ∈ {−80…60}):
  - reported-success-but-violating results drop from **17/20 to 0/20**;
  - FP32 LU gives iter = 3 and ratio 0.0815 at every scale.
- **On GPU 0 (V100).** The first version of the patch did not compile: two includes followed MAGMA's `min`/`max` macros. `gmstrs_fix_v2.patch` fixes only that.
  - On the 24 original calls the original reports success on 24 and violates the documented condition on 20. The fixed version reports success on 15 and violates it on **0**; the other 9 fall back correctly.
  - The fixed dsgesv_iteref takes iter = 4 at every scale k = −80…60.
  - With trans = T, the original returns the solution of Ax = b (error 6.6e-2) with info = 0. With nrhs = 2 it returns a zero second column. The fixed version handles both.
  - Cost. The first timings linked the reference CPU LAPACK and were not representative, so MAGMA was rebuilt against OpenBLAS. Correctness is unchanged there: the original violates the condition on 23 of 24 calls, and the fixed version on 0.
    - Certifying on the host (copy of A plus a serial Dot2) costs 11–16× at n = 8192 (3.3 s against 0.21 s).
    - `gmstrs_fix_v3.patch` moves the Dot2 residual and the row sums into a one-thread-per-row GPU kernel. The arithmetic sequence is the same, and the outward-rounded comparison stays on the host. At n = 8192 the fixed FP32-LU solver then takes 0.23–0.30 s, against 0.21–0.26 s unpatched and 0.29–0.43 s for FP64 `dgesv_gpu`. At n = 16384 the figures are 0.90 s, 0.85 s and 1.09 s.
    - At n ≤ 4096 the fix adds 0.03–0.17 s. This comes from one extra iteration and the certification's launch and copies. That extra iteration also makes the answer 1000× more accurate (1e-14 vs 1e-11).
    - These are ranges because the node was shared: repeated runs differ by up to 30%.
    - The V100's strong FP64 leaves little room for mixed precision in the first place. The measurement shows only that a certified acceptance gate costs no more than the factorization it guards.

### 6.5 RQ5: Is the scaling oracle free of false positives?

- By Theorem 3, any change under 2^k scaling (without over/underflow) proves a scale-dependent constant.
- Empirically, cuSOLVER's classical IR, which has no such constant, is bit-identical across all scalings (15/15, one process per call). This is the negative control.
- The oracle's precondition is a deterministic summation order. We check it by repeating each unscaled call and confirming bit-identical output before comparing scalings.

### 6.6 RQ6 (planned): A reference-free cross-hardware check **[TODO]**

Suppose a building block (GEMM, dot product, reduction) promises an order-independent forward bound B. Then two platforms with different summation orders must agree to within 2B. A larger difference proves that one of them violates its claim, without any reference answer. We have not run this study; see `PAPER_OUTLINE_zh.md` RQ6. Candidate for a later version or a separate short paper.

---

## 7 Discussion, limitations, trusted base

**Trusted base.**
- The Lean kernel.
- The faithfulness of hand-translated models to the source. Evidence: trace replay for MAGMA (72/72), and agreement of CPU/GPU results with every prediction. There is no mechanized extraction.
- The `Faithful` assumption for the gate.
- The mapping of libraries to conditions C1–C4, which comes from code reading plus runs.

**Scope.**
- Square systems; mostly nrhs = 1 and no transpose.
- MAGMA and cuSOLVER were run on different GPUs (V100, sm_120).
- Closed-source mechanisms (cuSOLVER's stopping rule) are inferred. We could not identify the rule cuSOLVER actually enforces.
- MUMPS was run on 5.8.2 while 5.8.1 was read; the masking mechanism (per-row comparisons in `dsol_aux.F:889–899`) is inferred from code and may depend on the compiler's `max` with NaN.

**What is new and what is not.**
- New:
  - the defects in MAGMA, cuSOLVER, SLATE, PLASMA, SuperLU, SuperLU_DIST, MUMPS, PaStiX and STRUMPACK (no public report found);
  - the contract characterization;
  - the comparison-form × reduction theorem;
  - the explicit statement that FP64 residuals cannot certify √n·u;
  - the machine-checked solver-agnostic gate;
  - the test-oracle audit.
- Not new:
  - LAPACK `dsgesv` accepting NaN (PR #1501);
  - the observation that bitwise reproducibility differs across platforms;
  - scaling invariance itself;
  - the Dot2 bound.

**AI usage statement [TODO-USER].** AI coding assistants were used extensively: to read library source code, draft Lean models and proofs, write probes and harnesses, run experiments, and draft text. The authors must state which artefacts they reviewed by hand: models against source lines, every reported number against raw result files, and the bug reports.

---

## 8 Related work

**GPU-kernel verification.**
- TensorLean (Columbia/CertiK, 2026; paper not released) verifies PTX of production kernels against tensor-level specs over the reals with ε tolerances.
- Koska et al. (arXiv 2611.04447) formalize PTX in Lean and prove functional correctness of hand-written kernels in exact real arithmetic. They explicitly leave numerical error out.
- Both target device code. We target host-side acceptance logic with NaN semantics.

**Verified numerical libraries.**
- VST/LAProof (Coq) prove accuracy of specific implementations.
- VSCL (Lean 4, 2026) mechanizes Higham-style bounds, including any-order summation trees.
- They prove programs correct. We use models to *predict failures* across many libraries, and prove a gate that does not depend on the solver.

**Floating-point defect detection.**
- KLEE-Float, ESBMC, GPU-NBDetect (ISSTA 2025), and Di Franco et al. 2017 find exceptions and numerical bugs below the contract level, with no notion of "reported success".
- FLiT, Verificarlo and CADNA localize platform-induced variability.

**Testing with principled tolerances.**
- "How to grade the accuracy of the BLAS" (Demmel et al., 2026) grades GEMM implementations, including FP64 emulation, against proved bounds.
- numoracle (Shi, 2026) derives per-element acceptance regions for low-precision GPU kernels.
- Both test kernels against a reference. We test the solver's own acceptance decision.

**Iterative refinement theory.**
- Carson–Higham 2017/2018 and Higham–Mary 2022 give convergence and attainable-accuracy analysis.
- LAWN 165 and LAWN 175 are the origin of the √n·u criterion.
- Ogita–Rump–Oishi 2005 give compensated dot products; INTLAB `verifylss` gives verified enclosures.
- Our contribution is at the level of the library's accept/reject decision, not the iteration.

**The LAPACK PR series** (#1499–#1520, Nov 2026) proposes detecting non-finite single-precision solutions in `xSGESV/xSPOSV` via `DLANGE('M')` and falling back. Beyond it, we report:
- a second trigger (FP32 pivot growth);
- the D11 path through max reductions, which that fix does not cover in SuperLU, PLASMA or MUMPS;
- the inheritance of the reject-form test by MAGMA, SLATE and PLASMA;
- wrong *finite* answers (MAGMA, cuSOLVER, PaStiX, STRUMPACK, SLATE) that no NaN check would catch.

**Cross-implementation consistency in HPC.** Climate models accept ports by ensemble statistical tests (CESM-ECT, Baker et al. 2015; Milroy et al. 2018). Lattice QCD's reliable updates and HPL-MxP's backward-error check are a-posteriori residual checks in the same spirit as our gate. We make that check machine-checked and solver-agnostic.

---

## 9 Conclusion and future work

The accuracy promise of mixed-precision iterative refinement is enforced by a few dozen lines of acceptance logic, and in 10 of the libraries we examined those lines can report success with a wrong answer. The failures are logic errors, not rounding errors:
- a comparison form that admits NaN;
- a max that drops NaN;
- a scale-dependent tolerance;
- a missing outer check;
- a fallback that cannot be reached;
- a status that is hard-coded.

A short contract (C1–C4) characterizes correct acceptance. The one numerically hard condition, C1, is achievable with a compensated residual and outward rounding, and we prove such a gate sound for any inner-solver output.

Future work:
1. Derive `Faithful` (the Dot2 and row-sum bounds, the outward-rounding enclosure) from the rounding model in Lean, removing it from the trusted base.
2. Mechanize model extraction from C/CUDA.
3. Apply the same contract audit to other numerical decision points, e.g. the acceptance gates of training/inference consistency in LLM reinforcement learning.

---

## Appendix A: Lean development (Table 4)

All files use core Lean 4 v4.36.0 only (no Mathlib) and contain no `sorry`. File paths below are relative to `lean/OptimiKahan/`. Most MAGMA defects are stated twice: once as the concrete instance observed on the GPU, and once as a general theorem over all inputs that trigger the defect (the `_all` theorems and the "General forms" section of each `Magma/` module). `#print axioms` reports only `propext`, `Classical.choice` and `Quot.sound`; `Classical.choice` enters through core `Rat` lemmas. The whole development builds with one `lake build` in `lean/` (see `lean/OptimiKahan/README.md`); the axioms of every theorem are in `lean/logs/axioms.txt`.

| File | Lines | `theorem`s | Main results | Non-vacuity |
|---|---|---|---|---|
| `IrsContract.lean` | 611 | 52 | `accept_scale`, `post_scale`, `absolute_rule_flips`, `rnd_scale`, `rnd_rel_err`, `eval_scale`, `iterate_scale`, `oracle`, `absolute_tol_breaks`, `wrapper_sound`, `wrapper_post`, `fp64_cannot_certify`, `fp64_n1024` | `expOf_spec` (exponent function exists) |
| `Magma/Fgmres.lean` | 364 | 23 | `single_cycle`, `gmstrs_always_success`, `nan_runs_to_maxiter`, `enforced_rule`, `givens_nonincreasing`, `tol_ge_one_one_step`, `one_step_violates`, `tiny_rhs_returns_zero`, `tiny_violates` | concrete 2×2 and 1×1 instances |
| `Magma/Fixed.lean` | 311 | 16 | `fixed_post`, `fixed_nan_fallback`, `fixed_rejects`, `fixed_rejects_tiny`, `fixed_rejects_one_step`, `fgmresFixed_scale` | `fixed_accepts_example`, `example_faithful`, `example_post` |
| `MaxFold.lean` | 277 | 28 | `smax_fold_nan_iff_last`, `smax_fold_value`, `masked_equals_baseline`, `pstep_fold_isNaN`, `nmax_fold_isNaN`, `detects_iff` | two-element counterexamples for the five unsafe combinations |
| `IrDriver.lean` | 258 | 19 | `contract_of_conditions`, `contract_iff`, `trust_inner_unsound`, `forced_unsound`, `stub_fallback_fails`, `skipped_fallback_incomplete`, `c1_forces_accept_propagate` | `tOk_contract`, `tForced_unsound`, `tTrust_unsound` |
| `Magma/DenseIR.lean` | 400 | 23 | `accept_inf`, `diverge_accepted`, `zchesv_stub`, `rbt_info_lost`, `x_overflow_accepted`, `zcgesv_b_overflow_accepted`, `dshposv_always_success`, `spd_tiny_rhs_returns_zero`, `nonconverged_oob` | concrete 1×1 instances |
| `Magma/Sparse/Solvers.lean` | 419 | 28 | solver acceptance logic, preconditioner setup: `bicg_breakdown_success`, `cg_merge_false_convergence`, `dotu_eq_dotc_real`, `wrapper_unhandled_success`, `early_exit_returns_x0` | concrete instances and code-level models |
| `Magma/Sparse/Control.lean` | 334 | 37 | format conversion, scaling, I/O, preconditioner apply: `syncfree_csc_swapped`, `spmv_mismatch_success`, `dense_spmv_transpose`, `sellp_tall_rows_skipped`, `mscale_nonfinite_success` | concrete instances and code-level models |
| `Magma/Sparse/Blas.lean` | 214 | 28 | SpMV kernels, dot products, sorting, LOBPCG helpers: `csrmv_trans_ignored`, `mdot_shfl_reduction_noop`, `mgecsrmv_stride`, `sellp_multi_half`, `vread_two_columns_overflow` | concrete instances and code-level models |
| `Magma/Dense/StatusAudit.lean` | 290 | 26 | LU/Cholesky/LDLᵀ/trtri status, FP16 conversion: `gbtrs_alias`, `gbtrs_overread`, `heevdx_upper_success`, `last_pivot_untested`, `nopiv_info_wrong` | concrete instances and code-level models |
| `Magma/Dense/QrEigen.lean` | 409 | 44 | QR, least squares, eigensolvers, multi-GPU drivers: `hegvdx_small_ignores_range`, `geqp3_fixed_unreduced`, `geqp3_doc_lwork_rejected`, `geqrf_singular_block_used`, `stedx_info_erased` | concrete instances and code-level models |
| `Magma/Dense/FaultInjection.lean` | 108 | 9 | allocation-failure paths, dropped sub-call status: `early_alloc_failure_hidden`, `hetrd_failure_reaches_dlascl`, `pinned_alloc_unchecked`, `geqrf_ooc_lwork_short`, `helper_iinfo_dropped` | concrete instances and code-level models |
| `Magma/Kernels.lean` | 521 | 58 | status logic in magmablas kernels: `vbatched_panel_info_erased`, `gb_generic_info_last_column`, `getf2_gpu_never_singular`, `potf2_zero_nan_pass`, `izamax_long_column_no_info` | concrete instances and code-level models |
| `Magma/Dense/NonZ.lean` | 240 | 23 | non-z files, review findings, 32-bit sizes: `lascl2x2_args`, `stedx_rangev_unscaled`, `stedx_subblock_info_unmapped`, `stedx_blocks_indistinguishable`, `adjust_ipiv_noop_large` | concrete instances and code-level models |
| `Magma/Dense/Workspace.lean` | 248 | 20 | workspace queries and checks: `hegvdx_2stage_query_rejected_all`, `hegvdx_2stage_vec_short_iff`, `gesdd_slow_ungbr_short_all`, `he2hb_mgpu_underrun_all`, `gbtrf_native_stale_info_all`, `geqlf_query_rejected_all`, `doc_minimum_rejected_all` | concrete instances (`hegvdx_2stage_query_rejected`, `gesdd_6to_slow_instance`) |
| `Magma/Dense/ArgChecks.lean` | 87 | 4 | argument numbering: `hegv_jobz_range_swapped`, `argnum_all_wrong`, `argnum_counts`, `cposv_bad_uplo_reported_as_conversion` | table checked entry by entry against the signatures |
| `Magma/Dense/Differential.lean` | 457 | 35 | differential-sweep findings: `geqp3_swap_restore_short_all`, `geqp3_patch_covers_all`, `geqp3_norm_rows_off_by_one`, `gemm_k0_beta_ignored`, `gemv_beta0_reads_y`, `gels_singular_nan_success`, `heevdx_n1_mout_unset`, `small_path_lda_n_rejects_n0`, `swap_long_llp64_wrong`, `getf2_native_aligned_no_share_all`, `getf2_native_stale_read`, `sgesv_fallback_race`, `batched_k0_beta_ignored_all`, `herk_batched_diag_tile_breaks_contract_all`, `trsm_batched_alpha0_nan`, `hemm_fetch_base_wrong_all`, `trevc3_blocked_without_backtransform`, `rbt_size1_inconsistent`, `sellp_params_lost`, `coo_rows_lost`, `ellrt_reduction_loses_lanes`, `csrlist_empty_row_overflows`, `trtri_diag_not_async` | general forms; GPU reproductions on roxy |
| `Magma/MultiGpu.lean` | 434 | 36 | `distribution_covers`, `nlocal_eq_cols_on_all`, `owned_from_formula`, `getrf_mgpu_first_global`, `potrf_mgpu_global`, `potrf_mgpu_host_slot_covered_iff`, `getrf_m_loop_exceeds_alloc`, `unghr_m_inconsistent`, `unghr_m_fix_consistent`, `gehrd_m_panel_unsynced` | general (all ngpu, nb, n, h) |
| **Total** | **5,982** | **509** | | |

Trace-replay files (`experiments/magma_fgmres/trace/`): `TraceReplay.lean` (72/72) and `TraceMutants.lean` (3 mutants detected).

## Appendix B: Artifact

- Probes and raw results: `experiments/ir_survey/`, `experiments/magma_fgmres/`, `experiments/cusolver_irs/`, `experiments/ir_accept/`.
- Build scripts: each library from source at the stated commit.
- Bug-report drafts: `operators/MAGMA_REPORT_fgmres_ir.md`, `operators/NVIDIA_REPORT_cusolver_irs.md`, `operators/upstream_reports/`.
