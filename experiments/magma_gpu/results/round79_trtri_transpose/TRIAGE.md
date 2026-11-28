# 第 79 轮：trtri_diag_batched / vbatched 与 get/setmatrix_transpose（2026-11-28，roxy；没有新缺陷，两条观察）

写探针前按文件名和例程名都搜了 FINDINGS、COVERAGE 和 HANDOFF，这几个例程都没有出现过。unm2r_batched 已在第 57 轮经 unmqr_batched 覆盖（M265）；transpose_conj_inplace 在第 63 轮测过；mgpu 版的 get/setmatrix_transpose 经 getrf_m 在第 52 轮的模糊测试中覆盖。

探针：`probes/round79_trtri_transpose.cpp`，用例表 `scripts/roxy/r79_cases.txt`，共 62 例。

## magmablas_ztrtri_diag_batched / _vbatched

**设置**：
- A 为三角矩阵，对角 2 + 随机，非对角 0.3·随机；
- 另一侧三角和 ld 余量填 NaN，diag = Unit 时对角也填 NaN；
- dinvA（128 行，roundup(n, 128) 列）预先填哨兵，resetozero = 1；
- batched 每批 6 个，n ∈ {1, 15, 16, 17, 100, 128, 129, 300}；vbatched 每批 10 个，尺寸取自 [0, nmax]，nmax ∈ {17, 130, 300}；
- U/L × N/U 全部组合。

**结果**：
- 44 种组合中，每个 128 × 128 对角块（最后一块只取有效的 r × r）都满足 inv · A_kk = I，误差 ≤ 7.2e-16；
- 另一侧三角和 Unit 时的对角都没有被引用（填了 NaN，结果仍然有限）。

**观察 1（潜在问题，不计入缺陷）**：n 不是 16 的倍数时，dinvA 中有效块之外有非零值。
- 对角线上的 1：这是有意的设计，把最后一块补成单位阵。
- Upper 情形的耦合项：例如 n = 17 时，第 0–15 行、第 20、24、28 列。
- 原因：triple_zgemm16_part1_upper_device 用 `if (col++ < n) rA[k] = A[...]` 装载 A12，条件不满足时 rA[k] 没有清零，保留的旧值 A(:,16) 又与 B22 的补齐单位阵行（第 20、24、28 行）相乘。
- 这些项只落在超出 n 的补齐列里。库内唯一的使用者 trsm_batched 每块只取 jb = min(NB, 剩余) 的有效部分，读不到它们，所以没有可观察的影响。

## magmablas_zgetmatrix_transpose / zsetmatrix_transpose（单卡）

**设置**：
- m × n ∈ {10×10、37×50、64×1、5×100、200×130、1000×700}，nb 取 4–128；
- lda、ldda、lddw 都留余量，余量填哨兵；
- getT 分别用可分页和锁页主机内存。

**结果**：18 例在同步之后全部逐位正确，余量也没有被写。

**观察 2（文档）**：
- 对锁页主机内存，getmatrix_transpose 返回时 hA 中还有大量元素没到，例如 1000 × 700 时 444,000 个（63%），200 × 130 时 26,000 个（全部）。函数只在两条队列上排了异步拷贝，没有同步。
- 按 MAGMA 的命名习惯，不带 `_async` 的 getmatrix 是同步的，文档也没有说明需要调用者同步。
- 不过库内调用者 getrf_m（mgpu 版本）每次调用后都对两条队列做了同步，所以这是按异步设计、只是没有写进文档，不计入缺陷。可分页内存时，返回时数据已经到位。

## 文件

- r79.csv、r79.err：62 例。
