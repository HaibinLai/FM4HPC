> Historical review report. File paths and line numbers refer to the layout before the reorganisation of 2026-11-26; see `lean/OptimiKahan/README.md` for where the Lean files went and `experiments/magma_gpu/README.md` for the new `results/`, `probes/`, … directories.

# Fifth independent review (2026-11-26), read-only

Root: /Users/haibin/Documents/Codex/2026-11-22/new-chat/lean4HPC. Line numbers are current.

## A. Fourth-review items

| item | status | evidence |
|---|---|---|
| #6 / #11 INVENTORY 125 count, missing inertia rows | FIXED | INVENTORY:210 "先前 39 个，复核补入 3 个，共 42 个"; :212 "其余 123 个"; table has 42 data rows, incl. zdiinertia.cu and zheinertia.cu rows |
| #9 COVERAGE §3h heading | FIXED | COVERAGE:211 "第十九至二十四轮" |
| #15 k = 4 inference qualifiers | FIXED | HANDOFF:2026, DRAFT:475, MagmaSparse.lean:715-717, README:443, COVERAGE:244 all mark it as inferred. (Minor, not in the fourth review's list: SPARSE:25 "分配失败时 double free（第二十二轮 ✔" has no qualifier.) |
| #17a SPARSE Counts-table note | PARTLY | SPARSE:88 note added as a 5th cell in a 4-column table: `\| read, clean \| 12 \| 1 \| 13 \| (5 of the 13 later revised ...)`. In GFM, cells beyond the header count are dropped, so the note does not render. |
| #17b HANDOFF "17 处 … 已改" | FIXED | HANDOFF:2049 "中 12 处改好；其余由第四次复核（…）找出后改正". 12 is right (1-5, 7, 8, 10, 12, 13, 14, 16). The "其余 … 改正" part holds, except for the open items below. |
| NONZ #6 "SUSPECTED" | FIXED | NONZ:10 and NONZ:40 no longer say SUSPECTED. The only remaining SUSPECTED (NONZ:21, gesvj launch error) is a different item. |
| HANDOFF "#6 无法触发" pointer | FIXED | HANDOFF:2019 "（判断有误：第二十四轮已触发，见 §29.26）" |
| DRAFT "remaining four candidates" | FIXED | DRAFT:461 "the remaining candidates (four in round 21, #6 in round 24)" |
| Round-24 cause explanation | NOT FIXED in NONZ:40 (see B); fixed elsewhere | |
| 42/123 count | NOT FIXED in COVERAGE:256 (see C) | |

## B. Round-24 cause wording

The result file says: n = 38000 "On entry to magma_dstedx_m, parameter 12 ... info=-12"; n = 53600 "dumped core", exit 139.

Consistent with it: COVERAGE:241, HANDOFF:2045-2047, README:457-458, DRAFT:467-468, MagmaAudit.lean:660-665, NONZ:10.

Inconsistent:
- NONZ:40. The corrected parenthesis was added, but the old causes were left at the end of the line, so it contradicts itself.
  - old: `GPU round 24: the overflow is real (−12 at n = 38,000 from dstedx_m; segfault at n = 53,600, crash site not established): n = 38,000 ⇒ −12 (lrwmin wraps negative), n = 53,600 ⇒ segfault (workspace wraps to 1.46e7 doubles)`
  - new: `GPU round 24: the overflow is real (−12 at n = 38,000 from dstedx_m, which receives lrwork − n² = 703,483,647 against its minimum 1,444,152,001; segfault at n = 53,600, crash site not established).`

## C. 42/123 count

Consistent: INVENTORY:210/212 (42 / 123, 42 rows); DRAFT:479 ("42 status-writing files and 123 without status in all").

History-only uses are fine: COVERAGE:140 (§3e heading, round 12), COVERAGE:214 (what the independent review counted), HANDOFF:1880 (§29.12 heading).

Inconsistent:
- COVERAGE:256 (§3f table). It names only 2 additions (41 in total), omits zdiinertia, and gives no 42/123 count.
  - old: `写状态的 39 个全部逐文件判定（§3e），另有 zgetf2_nopiv_devicefunc.cuh（无缺陷）与 zheinertia.cu（NaN 计零，✔ 第二十三轮）经两次复核补入；其余无状态输出`
  - new: `写状态的 39 个全部逐文件判定（§3e），另有 zgetf2_nopiv_devicefunc.cuh（无缺陷）、zdiinertia.cu（NaN 计零，✔ 第二十二轮）与 zheinertia.cu（NaN 计零，✔ 第二十三轮）经两次复核补入，共 42 个；其余 123 个无状态输出`

No 125/126 remain in the nine files. 40/41 now appear only for the 41 helper routines (a different count).

## D. Lean totals

| file | wc -l | ^theorem |
|---|---|---|
| IrsContract | 611 | 52 |
| MagmaFgmres | 364 | 23 |
| MagmaFixed | 311 | 16 |
| MaxFold | 277 | 28 |
| IrDriver | 258 | 19 |
| MagmaDense | 400 | 23 |
| MagmaSparse | 823 | 78 |
| MagmaAudit | 742 | 72 |
| MagmaMultiGpu | 96 | 4 |
| total | 3,882 | 315 |

- DRAFT Appendix A (:687-696) matches every row and the total.
- DRAFT:346 (23 / 78) and HANDOFF:2050 (72 / 78; 3,882 / 315) also match.
- `sorry` / `admit`: none.

## E. Remaining problems

1. NONZ:40: the old round-24 causes are still appended (see B).
2. COVERAGE:256: zdiinertia is missing and there is no 42/123 count (see C).
3. SPARSE:88: the Counts-table note is an excess cell and does not render.
   - old: `| read, clean | 12 | 1 | 13 | (5 of the 13 later revised by the reviews: zcsrsplit, zmcsrcompressor, zmlumerge, zvpass, zdiagcheck)`
   - new: `| read, clean | 12 | 1 | 13 |`, plus a line after the table: `Of the 13 "read, clean", 5 were later revised by the reviews: zcsrsplit, zmcsrcompressor, zmlumerge, zvpass, zdiagcheck.`
4. (minor) SPARSE:25: the k = 4 inference is not marked.
   - old: `分配失败时 double free（第二十二轮 ✔，`lumerge_double_free`）`
   - new: `第 4 次主机分配失败时 double free（第二十二轮 ✔，由中止位置推断，`lumerge_double_free`）`
