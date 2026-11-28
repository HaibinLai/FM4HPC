> Historical review report. File paths and line numbers refer to the layout before the reorganisation of 2026-11-26; see `lean/OptimiKahan/README.md` for where the Lean files went and `experiments/magma_gpu/README.md` for the new `results/`, `probes/`, … directories.

# Fourth independent review (2026-11-26), read-only

Root: /Users/haibin/Documents/Codex/2026-11-22/new-chat/lean4HPC. Line numbers are current.

operators/magma/reviews/3_THIRD_REVIEW.md does not number its problems. HANDOFF:2048 says there are "17 处剩余问题". I took these 17 from the items the third review marks NOT CORRECTED, PARTLY or BROKEN, plus its sections B, C and E.

## 1. The 17 remaining problems

| # | problem | status | evidence |
|---|---|---|---|
| 1 | "18 non-z files" without the 4 later additions (COVERAGE, DRAFT) | FIXED | COVERAGE:226 and DRAFT:455 both now say "dsiinertia/ssiinertia, magma_trisolve.cpp, mmio.cpp were added after the second review". |
| 2 | NONZ dlaex3 note vague and self-contradictory | FIXED | NONZ:33: "...it was not run, so no effect has been observed (the last writer of `*info = iinfo` wins)." |
| 3 | zungbr missing from the DRAFT helper list | FIXED | DRAFT:478: "`zunmtr`, `zunmbr`, `zunghr` and `zungbr` discard their inner routine's status". |
| 4 | zresidual: README/HANDOFF without a forward pointer | FIXED | README:407 "...round 23: SIGFPE"; HANDOFF:1995 "...第二十三轮重测均确认，见 §29.25". |
| 5 | zcsrsplit: README/HANDOFF without a pointer; DRAFT "still only suspected" | FIXED | README:408 "...round 23: heap corruption"; HANDOFF:1995 (same line as #4); DRAFT:451 "the fourth (`zcsrsplit`) was confirmed later, in the twenty-third round". No hits remain for "still only suspected". |
| 6 | INVENTORY:212 counts zdiinertia among the 125 files that "do not write info" | PARTLY | INVENTORY:212: "其余 125 个 `magmablas/z*` 文件不写 info 或标志，但有两个例外：zdiinertia.cu 与 zheinertia.cu 写 dneig ... （BLAS 类计算核：gemm、gemv、...）". Both exceptions are now named. But the 125 still includes them, and the "(BLAS 类计算核...)" gloss for the 125 now comes after the exception clause, so the sentence reads badly. |
| 7 | gesvj_kernels heevj skip; zgesvj_heevj_vector_updates.cu absent from every doc | FIXED | NONZ:27 now has a row: "magmablas/zgesvj_heevj_vector_updates.cu (z file, listed here because it consumes heevj status) ... skips the update when heevj reports convergence ...". NONZ:21 mentions "with heevj-converged skip". |
| 8 | DRAFT F9 "0 non-finite eigenvalues" | FIXED | DRAFT:486: "all 8 eigenvalues come back finite (the NaN is silently absorbed)". |
| 9 | COVERAGE §3h heading "第十九至二十一轮" | FIXED, then outdated again | COVERAGE:211 now says "第十九至二十三轮", but §3h now holds the round-24 #6 row (COVERAGE:241-242, "#6 在第二十四轮确认"). |
| 10 | COVERAGE §3f "截至第二十一轮" | FIXED | COVERAGE:248: "截至第二十四轮". |
| 11 | zheinertia note sits inside a contradictory sentence; not a table row | NOT FIXED | Same sentence as #6 (INVENTORY:212). The table heading INVENTORY:210 still says "39 个 z 文件（另 1 个经独立复核补入，共 40 行", and the table has 40 data rows with no zheinertia or zdiinertia row. |
| 12 | DRAFT broken insertion inside the file name | FIXED | DRAFT:479: "`operators/magma/COVERAGE_zh.md` §3e (a 40th, `zgetf2_nopiv_devicefunc.cuh`, was added by the independent review, and a 41st, `zheinertia.cu`, by the second)". |
| 13 | HANDOFF:1999 "18 个在 z 调用链上" | FIXED | HANDOFF:1999: "18 个写状态但不由 z 生成的文件（15 个在 z 调用链上，另 dlaln2、dlaqtrsd、dsidi）". The 4 later additions are not mentioned here (minor). |
| 14 | NONZ findings index #5 and the Lean `dsidi_lower_reads_upper` docstring | FIXED | NONZ:39 "...GPU round 21: the inverse is also wrong for Upper (0.131), cause undetermined"; MagmaAudit.lean:606-608 docstring now gives the Upper result and the undetermined cause. |
| 15 | "k = 4 ⇒ double free" is not marked as an inference | PARTLY | Marked in README:443 ("inferred: the abort falls between the k = 3 and k = 5 lines...") and COVERAGE:244 ("由中止发生在 k = 3 与 k = 5 之间推断"). Not marked in HANDOFF:2026 ("zmlumerge 第 4 次主机分配失败 ⇒ double free ✔。"), DRAFT:475 ("`zmlumerge` double-frees when its 4th host allocation fails"), or the Lean docstring MagmaSparse.lean:715-716 ("the 4th of 5 host allocations failing gives a double free (GPU 0: glibc ...)"). |
| 16a | HANDOFF Lean total stale at 3,807 / 308 | FIXED | HANDOFF:2039 "3,859 行、314 条" (§29.25) and HANDOFF:2049 "3,877 行、315 条" (§29.26). |
| 16b | DRAFT says "three routines" but `declaration_mismatches` models two | FIXED | MagmaSparse.lean:722-725 adds `zdiinertiaDecl`, and the theorem now has 3 conjuncts. DRAFT:476 names `magma_zdiagcheck_cpu`, `magma_zmlumerge`, `magmablas_zdiinertia`. |
| 17a | "13 clean / 31 new" despite 5 revised verdicts | PARTLY | COVERAGE:166 "读码干净 13（其中 5 个经两次复核改判...）" and DRAFT:501 "13 are clean (5 of them later revised by the reviews)" are annotated. The SPARSE_INVENTORY Counts table (SPARSE:84-90) still says "read, clean \| 12 \| 1 \| 13" with no note. The per-row verdicts at SPARSE:10/16/25/37/48 still start with "read, clean；..." followed by the correction. |
| 17b | "~18 text problems fixed" overstated | PARTLY | COVERAGE:246 and HANDOFF:2034 now say "大部分已改，余下的由第三次复核找出并改正"; DRAFT:469 says "a third pass found and fixed the remainder". HANDOFF:2048 says "指出的 17 处剩余问题 ... 已改". That is still overstated: items 6, 9, 11, 15 and 17a above are open. |

## 2. Section D: missing forward pointers

| location | status |
|---|---|
| README:139 "## Not run on GPU" | FIXED: "(all run in the second and third GPU rounds)" |
| README:342 zmdotc "inconclusive" | FIXED: "; confirmed in round 16" |
| README:407 / :408 | FIXED (see #4, #5) |
| HANDOFF:1764-1773 items 6, 7, 10-15 "未测" | FIXED: each row has "（见 §29.2–29.3）" |
| HANDOFF:1789 | FIXED: "（第三轮已运行，见 §29.3）" |
| HANDOFF:1847 | FIXED: "（见 §29.9、§29.10：...）" |
| HANDOFF:1863 | FIXED: "（见 §29.10：故障注入确认了 stedx 失败被清零...）" |
| HANDOFF:1893 | FIXED: "（见 §29.13、§29.14：...）" |
| HANDOFF:1938 | FIXED: "（zmdotc 第十六轮已确认，见 §29.16）" |
| HANDOFF:1956 | FIXED: "（第十七轮已运行，见 §29.17）" |
| HANDOFF:1970 | FIXED: "（zunglq 见 §29.18）" |
| HANDOFF:1995 | FIXED |
| NONZ:36 (#2) | FIXED: "GPU-confirmed in round 21 (info 226 ...)" |
| NONZ:35-44 index | FIXED: every item has a GPU suffix, and #6 has round 24 |
| DRAFT:451 | FIXED |

New pointers still missing:
- HANDOFF:2019 "#6 有害形式需 n > 53,500（45 GB），超出 V100 显存，无法触发。" has no forward pointer to §29.26. Only HANDOFF:2047 retracts it.
- NONZ:10 (dlaex0_m table row) still says "finding #6 (SUSPECTED, 32-bit size overflow)", while NONZ:40 says GPU ✔ round 24. NONZ:40 itself still reads "SUSPECTED. — GPU ✔ round 24".

## 3. Round-24 records against round24_results.txt

Numbers. Every doc matches the result file:
- n = 4000: info 0, error 4.49e-14, orthogonality 3.89e-15.
- n = 38000: workspace 2166114000 → −2128853296; lrwmin 2888152001 → −1406815295; info −12.
- n = 53600: workspace 4309600800 → 14633504; lrwmin 5746134401 → 1451167105; exit 139.

I checked these in HANDOFF §29.26 (:2041-2047), README:450-460, COVERAGE:241, DRAFT:466-468 and Lean `dc_sizes_wrap` (MagmaAudit.lean:662-663). The wrap32 arithmetic in Lean is correct. Numeric mismatches: none.

**Mechanism: new inconsistency.** I checked this against the MAGMA 2.10.0 source in the scratchpad (src/zstedx_m.cpp l.181-192, l.214-215; src/dstedx_m.cpp l.193-201).

- **n = 38,000.** All five records say the −12 comes from lrwmin wrapping negative (COVERAGE:241 "lrwmin 溢出为负，任何 32 位 lrwork 都不合法，返回 −12"; HANDOFF:2045; README "no 32-bit lrwork can be valid"; DRAFT:467 "the minimum workspace wraps negative, so ... fails with −12"; Lean docstring :660). That is not what happens.
  - A negative lrwmin makes zstedx_m's own check `lrwork < lrwmin` pass for INT_MAX.
  - The −12 comes from magma_dstedx_m. The result file line itself says "On entry to magma_dstedx_m, parameter 12". zstedx_m passes it `lrwork - n*n` = 2147483647 − 1444000000 = 703,483,647, and dstedx_m checks that against lwmin = 1 + 4n + n² = 1,444,152,001, which does not wrap.
  - So the rejection is a correct, loud error. The conclusion "no 32-bit lrwork can be valid" is still true, because the true requirement of 2.89e9 is above INT_MAX. The stated reason is wrong.
  - The 32-bit GPU workspace wrap (dlaex0_m l.151) is never reached at n = 38,000.
- **n = 53,600.** The docs blame the segfault on the GPU workspace wrapping to 1.46e7 doubles. The result file shows only "dumped core".
  - On this path, zstedx_m also computes `rwork+n*n` and `lrwork-n*n` with n*n = 2.87e9, which overflows 32-bit (pointer offset −1,422,007,296). Its copy loop also indexes `Z+i+ldz*j` and `rwork+i+n*j` in 32-bit.
  - Any of these could cause the crash. Which one does is not established.
  - The dstedx_m check passes here only because both sides wrap negative.
- **DRAFT structure.** DRAFT:461 says "A further round ran the remaining four candidates:" and then lists five bullets. The fifth (#6, :466) is from round 24, not that round.

## 4. Lean totals

| file | wc -l | ^theorem | DRAFT App. A |
|---|---|---|---|
| IrsContract | 611 | 52 | 611/52 |
| MagmaFgmres | 364 | 23 | 364/23 |
| MagmaFixed | 311 | 16 | 311/16 |
| MaxFold | 277 | 28 | 277/28 |
| IrDriver | 258 | 19 | 258/19 |
| MagmaDense | 400 | 23 | 400/23 |
| MagmaSparse | 822 | 78 | 822/78 |
| MagmaAudit | 738 | 72 | 738/72 |
| MagmaMultiGpu | 96 | 4 | 96/4 |
| total | 3,877 | 315 | 3,877/315 |

- Mismatches: none. DRAFT:346 (23 / 78) and HANDOFF:2049 (72 / 78; 3,877 / 315) agree.
- `sorry` / `admit` in the 9 files: none.

## 5. Other new inconsistencies

- **Count of status-writing magmablas files.** The docs give three different tallies:
  - INVENTORY:210-212: 40 rows, and the 125 others include both zdiinertia and zheinertia as "exceptions".
  - COVERAGE:256 and DRAFT:479: 41 (39 + zgetf2_nopiv + zheinertia), with zdiinertia not mentioned, though it also writes dneig and is GPU-confirmed in round 22.
  - Consistent count: 42 status-writing files and 123 without status (39 + 1 + 2 + 123 = 165).
- COVERAGE §3h heading (:211) is outdated again (see #9).
- HANDOFF:2048 claims all 17 are fixed (see #17b).
