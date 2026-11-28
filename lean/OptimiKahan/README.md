# Lean models (`lean/OptimiKahan/`)

Core Lean 4 v4.36.0 only (no Mathlib, no network), built as one Lake package.

## Build and check

```sh
TC=$PWD/.toolchain/elan/toolchains/leanprover--lean4---v4.36.0/bin   # from the project root
cd lean && PATH=$TC:$PATH lake build                                 # builds and checks every module
```

Expected: `Build completed successfully`, and the `#print axioms` output lists only `propext`, `Classical.choice` and
`Quot.sound` (never `sorryAx`). The axioms of every printed theorem are kept in `lean/logs/axioms.txt`; regenerate it with

```sh
cd lean && PATH=$TC:$PATH lake build 2>&1 | grep -E "depends on axioms|does not depend on any axioms" | sed 's/^info: //' > logs/axioms.txt
```

The Lean toolchain (~3 GB) lives in the project's hidden `.toolchain/` directory; `.lake/` is the build directory. Both are
ignored by git.

## Modules

| Module | Subject |
|---|---|
| `Kahan.lean` | optimi's Kahan update |
| `OzSlice.lean`, `OzRecombine.lean` | OzBLAS slicing lemma and FP64 recombination (import `Kahan`) |
| `OzimmuAuto.lean` | ozIMMU automatic slice selection |
| `Gemmul8Range.lean` | GEMMul8 CRT range |
| `CublasEsc.lean`, `AdpEsc.lean` | cuBLAS ADP coarsened-ESC contract; ADP estimate and per-element error bound |
| `IrsContract.lean` | cuSOLVER IRS contract, scaling oracle and post-check |
| `MaxFold.lean`, `IrDriver.lean` | NaN-dropping max reductions (D11); the IR acceptance contract C1–C4 (imports `MaxFold`) |
| `Magma/Fgmres.lean`, `Magma/Fixed.lean` | MAGMA FGMRES-IR (GMSTRS) code model; the fix and its soundness proof |
| `Magma/DenseIR.lean` | the other MAGMA mixed-precision dense solvers (zcgesv, zcposv, zchesv, RBT, dshposv) |
| `Magma/Dense/StatusAudit.lean` | dense status-loss audit: LU, Cholesky, LDLᵀ, trtri, batched/vbatched, FP16 conversion |
| `Magma/Dense/QrEigen.lean` | QR, least squares, generalized and tridiagonal eigensolvers, multi-GPU drivers |
| `Magma/Dense/FaultInjection.lean` | allocation-failure paths and helpers that drop a sub-call status |
| `Magma/Dense/NonZ.lean` | non-z files on the z call graph, review findings, 32-bit size arithmetic |
| `Magma/Dense/Workspace.lean` | workspace sizes (lwork/lrwork/liwork): queried vs. accepted vs. used, over all 83 z drivers |
| `Magma/Dense/ArgChecks.lean` | argument checks: reported `−k` vs. position of the tested argument (58 checks, 33 routines); unchecked `uplo` in zcposv/zchesv_gpu |
| `Magma/Dense/Differential.lean` | findings of the round-27 differential sweep against LAPACK: zgeqp3/zlaqps host-device inconsistency, gemm k = 0, gemv beta = 0, gels singular, heevdx n = 1, n = 0 small paths |
| `Magma/Kernels.lean` | status logic inside the magmablas GPU kernels |
| `Magma/Sparse/Solvers.lean` | sparse Krylov solvers and preconditioner setup |
| `Magma/Sparse/Control.lean` | sparse control layer: conversion, scaling, I/O, preconditioner apply |
| `Magma/Sparse/Blas.lean` | sparse BLAS layer: SpMV kernels, dot products, sorting, LOBPCG helpers |
| `Magma/MultiGpu.lean` | multi-GPU status bookkeeping, proved for every ngpu, nb, n |

Namespaces are unchanged by the reorganisation: theorem `MagmaAudit.x` and `MagmaSparse.x` keep their names.

## Where the old files went (reorganisation of 2026-11-26)

Older documents (`HANDOFF.md` up to §29.26 and the review reports in `operators/magma/reviews/`) cite the earlier flat
layout. `MagmaFgmres.lean` → `Magma/Fgmres.lean`, `MagmaFixed.lean` → `Magma/Fixed.lean`, `MagmaDense.lean` →
`Magma/DenseIR.lean`, `MagmaMultiGpu.lean` → `Magma/MultiGpu.lean`, and the per-file `check_log_*.txt` → `lean/logs/axioms.txt`.
The two large files were split by subject:

| Old file and sections | New module |
|---|---|
| `MagmaAudit.lean` §1, §2, §3, §4, §5, §6, §7, §8, §9, §6b, §10, §15, §16, §17 | `Magma/Dense/StatusAudit.lean` |
| `MagmaAudit.lean` §18, §19, §20 | `Magma/Dense/QrEigen.lean` |
| `MagmaAudit.lean` §21, §22, §26 | `Magma/Dense/FaultInjection.lean` |
| `MagmaAudit.lean` §23, §24, §25, §30 | `Magma/Kernels.lean` |
| `MagmaAudit.lean` §27, §28, §29, §31 | `Magma/Dense/NonZ.lean` |
| `MagmaSparse.lean` §1, §2, §3, §4, §5, §6, §7, §8, §9, §10, §11, §12 | `Magma/Sparse/Solvers.lean` |
| `MagmaSparse.lean` §13, §14, §15, §16, §20, §21, §22 | `Magma/Sparse/Control.lean` |
| `MagmaSparse.lean` §17, §18, §19 | `Magma/Sparse/Blas.lean` |
