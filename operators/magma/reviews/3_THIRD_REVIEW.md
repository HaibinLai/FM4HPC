> Historical review report. File paths and line numbers refer to the layout before the reorganisation of 2026-11-26; see `lean/OptimiKahan/README.md` for where the Lean files went and `experiments/magma_gpu/README.md` for the new `results/`, `probes/`, … directories.

# Third independent review of the MAGMA audit docs (2026-11-25)

Root: /Users/haibin/Documents/Codex/2026-11-22/new-chat/lean4HPC. This review was read-only; only this file was written.
Line numbers are as of this review.

## A. Status of each operators/magma/reviews/2_SECOND_REVIEW.md item

### §1 table (carried over from the first review)

| item | status | evidence / what remains |
|---|---|---|
| Unlisted non-z files (dsiinertia/ssiinertia, magma_trisolve.cpp, mmio.cpp) | CORRECTED in NONZ; PARTLY elsewhere | NONZ:26-28 add all four. COVERAGE:226 and DRAFT:455 still say "18 non-z files" and do not mention the 4 added after the second review. |
| dlaex3 OpenMP last-writer note | CORRECTED (wording vague) | NONZ:32. It says "a shared value", not `*info = iinfo`. "no visible effect was found and it was not run" contradicts itself. |
| zgetf2_nopiv_devicefunc.cuh header and count | CORRECTED | INVENTORY:210, :212 (40 rows, 125). |
| zlascl_2x2 at COVERAGE:162 | CORRECTED | Removed from the clean list at COVERAGE:162. The suggested pointer was not added, but §3h:222 has the correction. |
| zungbr in the summary lists | PARTLY | Done at COVERAGE:255, :279 and DRAFT:533. **Not done at DRAFT:476.** |
| zresidual division by zero | CORRECTED | COVERAGE:223, INVENTORY:282, Lean `residual_div_by_zero`. **README:407 and HANDOFF:1995 are unchanged, with no forward pointer.** |
| zcsrsplit | CORRECTED in COVERAGE/SPARSE; NOT in README/HANDOFF; DRAFT now contradicts itself | COVERAGE:224, SPARSE:10. **README:408 and HANDOFF:1995 are unchanged. DRAFT:451 says it is "still only suspected", but DRAFT:469 says it is confirmed.** |
| zmlumerge double free | CORRECTED | SPARSE:25, COVERAGE:244, README:443, HANDOFF:2026, DRAFT:473. |
| zmcsrcompressor ownership | CORRECTED | SPARSE:16, COVERAGE:244, README:442, DRAFT:472. |
| zmconvert valn | CORRECTED | SPARSE:15. |
| zdiinertia | CORRECTED (documented, with a GPU result) | COVERAGE:244, README:441, HANDOFF:2024, DRAFT:471, round22_results.txt:8. **Not in operators/magma/INVENTORY.md: INVENTORY:212 still counts zdiinertia.cu among the 125 files that "do not write info or a flag", although it writes dneig.** |
| gesvj_kernels.cuh heevj skip / zgesvj_heevj_vector_updates.cu | NOT CORRECTED | NONZ:21 is unchanged. zgesvj_heevj_vector_updates.cu appears in no doc (grep finds 0 hits). |
| zdiagcheck_cpu | CORRECTED | SPARSE:48, README:440, HANDOFF:2027. |
| zvpass leak | CORRECTED | SPARSE:37. |
| IDR 128 = 2n evidence | CORRECTED | README:185, HANDOFF:1806, COVERAGE:56. |
| F9 "0 non-finite eigenvalues" | PARTLY | README:301 and COVERAGE:154 are done. **DRAFT:484 is unchanged** (this is the old :476; the text has shifted). |

### §2a text fixes 1-19

| # | status | where |
|---|---|---|
| 1 | CORRECTED | COVERAGE:162 |
| 2 | CORRECTED | COVERAGE:242 |
| 3 | CORRECTED, then outdated | COVERAGE:211 says "第十九至二十一轮", but §3h now also holds rounds 22-23 (:244, :246). |
| 4 | NOT CORRECTED | COVERAGE:248 still says "截至第二十一轮", and §3f:256 cites round 23. |
| 5 | CORRECTED | COVERAGE:255 |
| 6 | CORRECTED | COVERAGE:279 |
| 7 | PARTLY | INVENTORY:210 and :212 have the counts. The zheinertia note at :212 is inside a sentence that says these files do not write status. zdiinertia and zgesvj_heevj_vector_updates are missing. |
| 8 | BROKEN EDIT | DRAFT:477: the inserted clause landed inside the file name: `` `operators/MAGMA_COVERAGE_zh (a 40th, `zgetf2_nopiv_devicefunc.cuh`, was added by the independent review).md` `` |
| 9 | CORRECTED, then outdated | DRAFT:451 says zcsrsplit is "still only suspected"; round 23 and DRAFT:469 confirm it. |
| 10 | PARTLY | DRAFT:533 is done; DRAFT:476 is not. |
| 11 | CORRECTED | DRAFT:512 |
| 12 | PARTLY | COVERAGE:226 and DRAFT:455 are done. **HANDOFF:1999 still says "18 个在 z 调用链上".** |
| 13 | CORRECTED | DRAFT:396 ("1–10 iterations") |
| 14 | CORRECTED | HANDOFF:1944 |
| 15 | CORRECTED | README:301 |
| 16 | CORRECTED | README:185, HANDOFF:1806 |
| 17 | CORRECTED | HANDOFF:2000, :2010; README:421, :159 |
| 18 | CORRECTED | HANDOFF §29.24 (:2022-2027), README:436-448, COVERAGE:244. round22_results.txt now has a full run (line 8 = 2/1/1), so the Lean GPU clause is backed by a result file. |
| 19 | PARTLY | NONZ:19 and COVERAGE:235 are done. **NONZ:38 (findings index #5) still says only "dsidi Lower inverse uses upper triangle".** The Lean docstring of `dsidi_lower_reads_upper` (MagmaAudit.lean:606-609) is also unchanged. |

### §2b invalid negative results
- COVERAGE:223-224, INVENTORY:282 and SPARSE:10 are CORRECTED. Each records round 23.
- README:407-408 and HANDOFF:1995 are NOT CORRECTED.

### §2c zheinertia
- CORRECTED in COVERAGE:246, :256; README:446; HANDOFF:2038; DRAFT:471.
- PARTLY in INVENTORY:212: the note sits in a contradictory sentence and is not a table row.

## B. Round 22 / 23 numbers against the result files

- round22_results.txt matches every doc for these results:
  - diinertia 2/1/1 (l.8);
  - compressor ret 0, nnz 3 → 3 (l.10);
  - lumerge: k = 1, 2, 3, 5 give −112 (l.13-26);
  - nm counts (l.3-6).
- Caveat, not a mismatch: the "k=4" header line is missing from the file. It was presumably lost in the stdout buffer when the process aborted. "k = 4 ⇒ double free" (COVERAGE:244, README:443, HANDOFF:2026, DRAFT:473, Lean `lumerge_double_free`) is inferred from where the abort falls between k = 3 and k = 5. That is reasonable, but the docs do not say it is an inference.
- round23_results.txt matches:
  - zresidual exit 136 = SIGFPE (l.3-5);
  - zcsrsplit col 681594464, then `free(): invalid pointer` (l.7-11);
  - zheinertia: the first run gives −1/−1/−1 (l.13, host ipiv, output not written); the second run gives 2/1/1 (l.23, device ipiv).
- README:446 and HANDOFF:2038 describe both runs correctly. COVERAGE:246 gives only the second run, which is fine.
- Numeric mismatches: **none**.
- Note: round24_probe.cpp and run_round24.sh exist, but there is no round24_results.txt and no doc mentions round 24.

## C. Lean totals (DRAFT.md Appendix A, lines 683-694)

| file | wc -l | ^theorem | DRAFT |
|---|---|---|---|
| IrsContract | 611 | 52 | 611/52 |
| MagmaFgmres | 364 | 23 | 364/23 |
| MagmaFixed | 311 | 16 | 311/16 |
| MaxFold | 277 | 28 | 277/28 |
| IrDriver | 258 | 19 | 258/19 |
| MagmaDense | 400 | 23 | 400/23 |
| MagmaSparse | 820 | 78 | 820/78 |
| MagmaAudit | 722 | 71 | 722/71 |
| MagmaMultiGpu | 96 | 4 | 96/4 |
| total | 3,859 | 314 | 3,859/314 |

- Every row matches, and DRAFT:346 (23 / 78) matches too. None of the 9 files contains `sorry`.
- Mismatches: **none**.
- Minor: HANDOFF's last Lean total is still 3,807 / 308 at §29.23 (:2020). §29.24 and §29.25 add 6 theorems and 52 lines but record no new total.
- Minor: DRAFT:474 says "three routines have mismatched header declarations" (`declaration_mismatches`), but the theorem (MagmaSparse.lean:719-723) models only two: diagcheck_cpu and zmlumerge. zdiinertia is not in it.

## D. Superseded "not run / inconclusive / not reproduced" statements without a forward pointer

| file:line | text | settled in |
|---|---|---|
| README:139-143 | "## Not run on GPU": LOBPCG n > m, IC(0) breakdown, FGMRES sticky, dense items 6-15 | second and third GPU rounds (README:147ff, :161ff) |
| README:342 | "output buffer not written (inconclusive)" | round 16, README:360 |
| README:407 | "ret 0, no division by zero" | round 23, README:444 |
| README:408 | "ret 0, no visible effect" | round 23, README:445 |
| HANDOFF:1764-1773 | items 6, 7, 10-15 marked "未测" | §29.2, §29.3 |
| HANDOFF:1789 | "未测：第 10、11、15 条，FGMRES 粘住的成功" | §29.3 |
| HANDOFF:1847 | "未测：zhegvd 系负 info 映射…、zmscale_generate UNITCOL 双重释放" | §29.9 (:1861) and §29.10 |
| HANDOFF:1863 | "未复现或未测：2-stage 中 stedx 失败被清零…；分配失败路径…；zgeqrf_ooc" | §29.10 |
| HANDOFF:1893 | "未测：native 递归 LU 的偏移…、两处疑似竞争、两条轻微问题" | §29.13, §29.14 |
| HANDOFF:1938 | "zmdotc 共轭（结果未写回，不确定）" | §29.16 (:1943) |
| HANDOFF:1956 | "未运行的 #3、#5、#14 …，#17 …" | §29.17 |
| HANDOFF:1970 | "至此剩下：zunglq 的单独故障注入（dT 格式未确认）" | §29.18 |
| HANDOFF:1995 | "zresidual 0×0 除零 ✖；zcsrsplit 未排序行 ✖" | §29.25 |
| NONZ:35 | "#2 … CONFIRMED-BY-READING (trigger SUSPECTED)" | round 21 |
| NONZ:34, 36-43 | findings index is reading-only (#1, #3, #4, #5, #7-#10 were confirmed on GPU in rounds 20-21; #6 cannot be triggered) | rounds 20-21 |
| DRAFT:451 | "the fourth (`zcsrsplit`) is still only suspected" | DRAFT:469 (round 23) |

## E. Other new problems

- SPARSE_INVENTORY:82-90 (Counts), COVERAGE:166 and DRAFT:499 still give "13 clean / 31 new". Five of the 13 "read, clean" files have since got findings: zcsrsplit, zmcsrcompressor, zmlumerge, zvpass and zdiagcheck.
- COVERAGE:246, HANDOFF:2034 and DRAFT:467 claim that the ~18 text problems from the second review "已改" / "were fixed". The items above that are not corrected or only partly corrected show that this is overstated.
