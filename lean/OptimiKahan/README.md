# Lean models: optimi's Kahan update (`Kahan.lean`), OzBLAS's slicing lemma (`OzSlice.lean`), FP64 recombination (`OzRecombine.lean`) ozIMMU's auto slice selection (`OzimmuAuto.lean`) GEMMul8's CRT range (`Gemmul8Range.lean`) the cuBLAS ADP coarsened-ESC contract (`CublasEsc.lean`) the ADP estimate plus per-element error bound (`AdpEsc.lean`) the cuSOLVER IRS contract, scaling oracle and post-check (`IrsContract.lean`) a code-level model of MAGMA's FGMRES-IR (`MagmaFgmres.lean`) and the proof of a fix for it (`MagmaFixed.lean`), max reductions that drop NaN (`MaxFold.lean`, defect class D11) and the necessary-and-sufficient acceptance contract of iterative refinement (`IrDriver.lean`)

Core Lean only (no Mathlib, no network). From the project root:

```sh
L=.toolchain/elan/toolchains/leanprover--lean4---v4.36.0/bin
cd lean/OptimiKahan
../../$L/lean Kahan.lean                                  # optimi Kahan update; log: check_log.txt
../../$L/lean -o Kahan.olean Kahan.lean                   # build the module that OzSlice imports
LEAN_PATH=$PWD ../../$L/lean OzSlice.lean                 # OzBLAS slicing lemma; log: check_log_ozslice.txt
LEAN_PATH=$PWD ../../$L/lean OzRecombine.lean             # OzBLAS FP64 recombination; log: check_log_ozrecombine.txt
../../$L/lean OzimmuAuto.lean                             # ozIMMU fp64_int8_auto (standalone); log: check_log_ozimmuauto.txt
../../$L/lean Gemmul8Range.lean                           # GEMMul8 CRT range (standalone); log: check_log_gemmul8range.txt
../../$L/lean AdpEsc.lean                                 # cuBLAS ADP / ESC estimate (standalone); log: check_log_adpesc.txt
../../$L/lean IrsContract.lean                            # cuSOLVER IRS contract / oracle / post-check (standalone, uses core Rat); log: check_log_irscontract.txt
../../$L/lean -o IrsContract.olean IrsContract.lean       # build the module that MagmaFgmres imports
LEAN_PATH=$PWD ../../$L/lean MagmaFgmres.lean             # MAGMA FGMRES-IR (GMSTRS) code model; log: check_log_magmafgmres.txt
LEAN_PATH=$PWD ../../$L/lean -o MagmaFgmres.olean MagmaFgmres.lean   # build the module that MagmaFixed imports
LEAN_PATH=$PWD ../../$L/lean MagmaFixed.lean              # the GMSTRS fix and its soundness proof; log: check_log_magmafixed.txt
../../$L/lean CublasEsc.lean                              # cuBLAS ADP coarsened-ESC contract (standalone); log: check_log_cublasesc.txt
../../$L/lean MaxFold.lean                                # NaN-dropping max reductions, D11 (standalone); log: check_log_maxfold.txt
../../$L/lean -o MaxFold.olean MaxFold.lean               # build the module that IrDriver imports
LEAN_PATH=$PWD ../../$L/lean IrDriver.lean                # IR acceptance contract: C1-C4 iff; log: check_log_irdriver.txt
```

Expected: only `#print axioms` lines (no `sorryAx`), exit code 0 (the word "error" appears in the theorem name `recombine_error`). `Kahan.olean`, `IrsContract.olean`, `MagmaFgmres.olean` and `MaxFold.olean` are build artifacts (not kept; it depends on the exact Lean version).
The Lean toolchain (~3 GB) lives in the project's hidden `.toolchain/` directory; exclude it when archiving the project.

See HANDOFF.md section 13 (Kahan.lean), section 14 (OzSlice.lean), section 15 (OzRecombine.lean) section 16 (OzimmuAuto.lean) section 17 (Gemmul8Range.lean) section 20 (CublasEsc.lean) section 21 (AdpEsc.lean) section 22.6 (IrsContract.lean) section 24 (MagmaFgmres.lean, MagmaFixed.lean) and section 26 (MaxFold.lean, IrDriver.lean) for what is proved, what is assumed, and how the models relate to PyTorch / CUDA.
