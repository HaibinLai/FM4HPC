> Historical review report. File paths and line numbers refer to the layout before the reorganisation of 2026-11-26; see `lean/OptimiKahan/README.md` for where the Lean files went and `experiments/magma_gpu/README.md` for the new `results/`, `probes/`, … directories.

# Second independent review of the MAGMA audit docs (2026-11-25)

Root: /Users/haibin/Documents/Codex/2026-11-22/new-chat/lean4HPC. Source: scratchpad/magma_full/magma-2.10.0. Read-only; only this file was written.

## 1. Status of each item in operators/magma/reviews/1_INDEPENDENT_REVIEW.md

| item | status | where / what is missing |
|---|---|---|
| Unlisted non-z files | PARTLY | operators/magma/NONZ_INVENTORY.md covers 18 files (dlaex0/1/3 and their _m versions, dstedx(_m), dmove_eig, dlaln2, dlaqtrsd, dsidi, getrf_setup_pivinfo.cu, gesvj_kernels.cuh, sort.cu, vbatched_*, magma_templates.h). Still missing: `magmablas/dsiinertia.cu`, `ssiinertia.cu`, `sparse/blas/magma_trisolve.cpp`, `sparse/control/mmio.cpp`. The dlaex3 OpenMP `*info = iinfo` last-writer note is not recorded. |
| zgetf2_nopiv_devicefunc.cuh | CORRECTED | INVENTORY.md:238; COVERAGE_zh.md:215, :252. The section header and count were not updated (see 2a). |
| 23/32 → 21/32 | CORRECTED | COVERAGE_zh.md:51, :216; README.md:182; HANDOFF.md:1803; DRAFT.md:394, :450. No remaining "23 of 32" or "23/32" anywhere except the correction notes. |
| 7 → 6 silent, 37 → 38 | CORRECTED | COVERAGE_zh.md:125, :129, :132; HANDOFF.md:1869; DRAFT.md:436, :444; README.md:265-276 table sums to 6. No stale 7 or 37 remains. |
| zlascl_2x2 | CORRECTED, one stale line left | INVENTORY.md:241; COVERAGE_zh.md:222; Lean `lascl2x2_args`; round19 shows info=-4 for ldda and 0 for lddw=1. But COVERAGE_zh.md:162 still lists lascl_2x2 as clean. |
| zungbr | CORRECTED in the inventory, MISSING from summary lists | INVENTORY.md:185, :315; COVERAGE_zh.md:221; MagmaAudit.lean:436-437; DRAFT.md:452. zungbr is still absent from the helper-drops-iinfo lists at COVERAGE_zh.md:251, :275 and DRAFT.md:468, :525. |
| zresidual / zresidualvec | PARTLY. The multi-RHS part is corrected; the "✖ no division by zero" result was tested on the wrong input | See 2b. |
| magma_zcsrsplit | Annotated, but the GPU test cannot trigger the defect | SPARSE_INVENTORY.md:10 and COVERAGE_zh.md:224 record "✖ no visible effect". round19 used bsize = 1. See 2b. |
| zmlumerge (double free, U rows) | NOT CORRECTED | SPARSE_INVENTORY.md:25 still says "read, clean". round22 tried a probe, but the build failed (`magma_zmlumerge` was not declared). |
| zmcsrcompressor (ownership==false; CPU-branch free) | NOT CORRECTED | SPARSE_INVENTORY.md:16 still says "read, clean". |
| zmconvert `magma_z_csr_compressor` frees `valn` (a `**`) | NOT CORRECTED | SPARSE_INVENTORY.md:15 does not mention it. |
| zdiinertia | NOT IN ANY DOC; Lean claims a GPU result with no result file | Only MagmaAudit.lean:642-647 (`diinertia_nan_as_zero`) has it, with "GPU 0: diag (1,−1,NaN,2) ⇒ 2/1/1". round22_results.txt shows the probe never ran ("./round22_probe: No such file"). It is not in INVENTORY, COVERAGE, README, HANDOFF or DRAFT. |
| zgesvj_heevj_vector_updates / gesvj_kernels.cuh skip | PARTLY | NONZ_INVENTORY.md:21 mentions the "heevj-converged skip" but does not flag that F9 turns it into a silent skip. zgesvj_heevj_vector_updates.cu is in no table. |
| zdiagcheck_cpu (declared, not defined; NaN diagonal passes) | NOT CORRECTED | SPARSE_INVENTORY.md:48 still says "read, clean". round22 confirms 0 symbols for `magma_zdiagcheck_cpu`, but no doc records this. |
| zvpass (zvcopy leaks v_CPU) | NOT CORRECTED | SPARSE_INVENTORY.md:37 still says "read, clean". |
| IDR "128 = 2n" evidence | CORRECTED in COVERAGE only | COVERAGE_zh.md:56 is corrected. README.md:185 ("this also confirms the maxiter cap of `idr_params`") and HANDOFF.md:1806 ("也在 GPU 上证实了 `idr_params` 的 maxiter 截断") still state it without saying that the evidence comes from the MERGE variants only. |
| F9 "0 non-finite eigenvalues" | CORRECTED in COVERAGE only | COVERAGE_zh.md:154 is corrected. README.md:301 still says "info 0", and DRAFT.md:476 says nothing about the NaN being absorbed. |

## 2. Remaining stale numbers, contradictions and test-validity problems

### 2a. Text fixes

1. COVERAGE_zh.md:162 is stale.
   - Old: `其余文件：QR 系列（geqr2*、geqrf_batched_smallsq）、laswp 系列、hetrs、lascl_2x2、lascl_diag、getrf_batched_smallsq_noshfl 为干净或无状态逻辑。`
   - New: remove `lascl_2x2、` and append `lascl_2x2 经独立复核更正（参数编号错、lddw 不查，见 §3h）。`
2. COVERAGE_zh.md:242 contradicts its own table rows :238-:240, which mark #2, #8 and #10 as ✔ in round 21.
   - Old: `#2（需 NaN，会先在 dlascl 处结束进程）、#6、#8（需极大输入）、#10（需 n > 2048 且含 NaN）只有读码证据。`
   - New: `只有 #6 没有 GPU 证据（有害形式需 n > 53,500）。`
3. COVERAGE_zh.md:211 header. Old: `（第十九、二十轮；` → New: `（第十九至二十一轮；`.
4. COVERAGE_zh.md:244 header. Old: `截至第十七轮` → New: `截至第二十一轮`. The section already cites rounds 18-21.
5. COVERAGE_zh.md:251 does not list zungbr.
   - Old: `本身不分配、但内部调用会分配的 zunmtr、zunmbr、zunghr 注入后 info 0、结果错误`
   - New: `本身不分配、但内部调用会分配的 zunmtr、zunmbr、zunghr（第十一轮）与 zungbr（第十九轮）注入后 info 0、结果错误`
6. COVERAGE_zh.md:275. Old: `zunmtr/zunghr/zungtr/zunmbr` → New: `zunmtr/zunghr/zungtr/zunmbr/zungbr`.
7. INVENTORY.md:210 and :212 disagree with the table, which now has 40 rows. 165 − 40 = 125.
   - Line 210, old: `写状态的 39 个 z 文件` → new: `写状态的 39 个 z 文件（另 1 个经独立复核补入，共 40 行）`.
   - Line 212, old: `其余 126 个` → new: `其余 125 个`.
   - Line 212 also needs a qualifier: zdiinertia.cu and zheinertia.cu write `dneig` (see 2c), and zgesvj_heevj_vector_updates.cu consumes heevj info.
8. DRAFT.md:469.
   - Old: `All 39 `magmablas` z-files that write an info value or a flag were read`
   - New: `All 39 `magmablas` z-files that write an info value or a flag were read (a 40th, `zgetf2_nopiv_devicefunc.cuh`, was added by the independent review)`
9. DRAFT.md:451 overstates the zcsrsplit result. zcsrsplit is only suspected, and its GPU test was invalid (2b).
   - Old: `Four "clean" verdicts were wrong. Three were then confirmed on GPU 0:`
   - New: `It disputed four "clean" verdicts. Three were confirmed on GPU 0; the fourth (`zcsrsplit` on unsorted rows) remains suspected:`
10. DRAFT.md:468.
    - Old: `The helpers `zunmtr`, `zunmbr` and `zunghr` discard`
    - New: `The helpers `zunmtr`, `zunmbr`, `zunghr` and `zungbr` discard`
    - Make the same change at DRAFT.md:525 (`zunmtr`, `zunmbr` and `zunghr` do not allocate themselves`).
11. DRAFT.md:504 says "2 were not" but names only one.
    - Old: `**Not confirmed:** the ELLPACKT padding test; the NaN we saw comes from complex Inf arithmetic.`
    - New: append `; `zmdotc` was inconclusive (the output buffer was not written) and was confirmed in the sixteenth round.`
12. DRAFT.md:455 and HANDOFF.md:1999 say "18 non-z files on the z call graph". NONZ_INVENTORY.md:17-19 says dlaln2, dlaqtrsd and dsidi are not reached from z. The same wording is at COVERAGE_zh.md:226 ("在 z 调用链上").
    - New: `18 non-z files (15 on the z call graph, plus dlaln2, dlaqtrsd and dsidi)`.
13. DRAFT.md:396. Old: `that IDR solves in 4 iterations` → new: `that IDR solves in 1–10 iterations`. sweep_results.txt gives numiter 1 at restart 30; sweep_idr_restart.txt gives 4 at restart 4 and 10 at restart 50.
14. HANDOFF.md:1944 miscounts. Round 16 ran 13 probes: zmdotc plus 12 others. Of the 12, 10 were confirmed (the 10 bullets) and 2 were not reproduced.
    - Old: `另 13 条中 11 条确认：` → New: `另 12 条中 10 条确认：`
15. README.md:301. Old: `| info 0 | — |` → New: `| info 0; 0 of 8 eigenvalues non-finite (NaN silently absorbed) | — |`.
16. README.md:185 and HANDOFF.md:1806: the IDR cap evidence comes from IDRMERGE/PIDRMERGE only.
    - README, add: `(the 128-iteration evidence comes from IDRMERGE/PIDRMERGE; IDR/PIDR converge before the cap)`.
17. Superseded "not run / uncertain" statements. These are chronological, so the fix is a forward pointer:
    - HANDOFF.md:2000: `…没有运行` → append `（后已运行，见 §29.22）`.
    - HANDOFF.md:2010: `#5 不确定` → append `（第二十一轮已确认，见 §29.23）`.
    - README.md:421: `inconclusive (dsidi expects the LINPACK factor format)` → append `; superseded by round 21`.
    - README.md:159: `Not run: 10, 11, 15, FGMRES sticky` → append `(all run in the third round)`.
    - The README round 15 zmdotc "inconclusive" at :342 is already resolved at :360. No fix needed.
18. Round 22 is undocumented. round22_results.txt exists with a failed build, and there is no HANDOFF §29.24 and no README section. Meanwhile MagmaAudit.lean already contains `diinertia_nan_as_zero`, citing a GPU result that is not in any result file. Either rerun round 22 and record it, or remove the "GPU 0: …" clause from MagmaAudit.lean:646.
19. dsidi root cause is inconsistent.
    - NONZ_INVENTORY.md:19, :33 and Lean `dsidi_lower_reads_upper` say the defect is "uplo=Lower inverse uses upper triangle".
    - round21 shows Upper is also wrong: SPD with no 2×2 pivots gives max|AX−I| = 0.131.
    - COVERAGE_zh.md:235 maps both Lean theorems to one finding without explaining the Upper failure.
    - Fix: state in NONZ #5 that the Upper path is wrong too and its cause is undetermined.

### 2b. GPU tests that do not exercise the claimed defect (most important)

- **zresidual division by zero (round 19, `residual0`)** cannot reach the division.
  - round19_probe.cpp:31-33 calls it with A 0×0 and b 0×1, so `A.num_rows == b.num_rows` (0 == 0).
  - That takes the first branch (zresidual.cpp:68), where `num_vecs` (l.64) is never used. The compiler may legally drop the division.
  - The independent review's trigger is A.num_rows == 0 with b.num_rows > 0. That reaches `% A.num_rows` at l.74 (and zresidualvec.cpp:73).
  - So "✖ 0×0 未除零" (COVERAGE_zh.md:223), "A 0 行时的除零未复现" (INVENTORY.md:282), README.md:407 and HANDOFF.md:1995 are not valid negative results.
  - New text: `未按复核给出的条件测试（b 也是 0 行，走第一分支）` / "not tested as specified (b also had 0 rows)".
- **zcsrsplit unsorted rows (round 19)** cannot trigger the out-of-bounds write with bsize = 1.
  - round19_probe.cpp:40-42 uses `magma_zcsrsplit(0, 1, …)`, so bsize = 1.
  - With bsize = 1 the lower entry (col 0 < i = 1) goes to R (magma_zcsrsplit.cpp:221), never to the `nnz_diag+1` slot (l.241-245).
  - The in-block lower entry that causes the out-of-bounds write needs bsize ≥ 2, with a row like [diag, lower] inside the block.
  - So "✖ 无可见后果" (COVERAGE_zh.md:224, SPARSE_INVENTORY.md:10, README.md:408, HANDOFF.md:1995) is not a valid negative result. It should be marked "untested (bsize 1 cannot reach the path)".

### 2c. New item not in either review

- `magmablas/zheinertia.cu` is a z-file (`@precisions normal z -> s d c`) with the same `diag > 0 / diag < 0 / else zeig++` classification (l.46-51, 73-78, 126-131, 153-158). A NaN diagonal is therefore counted as a zero eigenvalue in `dneig`.
- It is in neither the 39-file table nor the first review. No src caller was found.

### 2d. Lean totals: no problem found

`^theorem` counts and line counts match DRAFT.md Appendix A row for row:

| file | lines | theorems |
|---|---|---|
| IrsContract | 611 | 52 |
| MagmaFgmres | 364 | 23 |
| MagmaFixed | 311 | 16 |
| MaxFold | 277 | 28 |
| IrDriver | 258 | 19 |
| MagmaDense | 400 | 23 |
| MagmaSparse | 777 | 73 |
| MagmaAudit | 713 | 70 |
| MagmaMultiGpu | 96 | 4 |
| **total** | **3,807** | **308** |

- The totals also match HANDOFF.md §29.23 and DRAFT.md:346 (23 / 73).
- There are 0 `sorry`. check_log_magmaaudit.txt has the same timestamp as MagmaAudit.lean and includes `diinertia_nan_as_zero`.
- Minor: §29.22 says MagmaAudit had 66 theorems. 66 + the 4 from round 21 = 70, which already includes the later-added `diinertia_nan_as_zero`. So the 66 was probably off by one. This is historical only.

## 3. Spot-check of rounds 15–21 against the result files: all match

1. R15 `zmgecsrmv`: Y = 1 2 7 4 (COVERAGE:174, README:344) ✓. `zcsrget` val[0] = 4.94e-310 ✓.
2. R16 `zge3pt`: y = 998 ✓. `zjaccard`: 1/3 ✓. `zmdotc`: −1 for k = 1 and 2 ✓.
3. R17 sync-free solve: n = 2 gives x = (1, 2); n = 4096 gives 4095 wrong entries, x = 1 2 2 2 ✓.
4. R18 `zunglq`: header-style call gives exit 139; defined-style call gives −113 ✓.
5. R19 `zungbr`: 52.2 / 8.58 (docs: 52 / 8.6) ✓. `zlascl_2x2`: −4 and 0 ✓. `zresidual`: res = 2, −1 ✓.
6. R20 range V: m = 145, 13.5 / 296 ✓. Closed interval: m = 11 vs 10 ✓. `adjust_ipiv`: 1200 invalid pivots, 52.7 ✓. gesvj: U non-finite n / 16384 ✓.
7. R21 `zstedx`: 226 / 226 / 11626 ✓. `dsidi`: 0.367 / 1.06 / 0.131 / 1.16 ✓. dsort: 1 duplicated, 1 missing ✓. gesvj: 47/47, 0.491 ✓.
