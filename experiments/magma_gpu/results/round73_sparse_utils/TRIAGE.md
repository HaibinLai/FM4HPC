# 第 73 轮：复数稀疏工具函数（2026-11-28，roxy；新缺陷 M277、M278，扩充 M137/M138）

`probes/round73_sparse_utils.cpp` 用随机复数 CSR 矩阵（对角加 3，非对角密度约 15%）测试以前从没测过的稀疏工具函数，参考值按文档在 CPU 上计算。`probes/round67_trans_rect.cpp` 加了 conjcpu / abscpu / structcpu 三个分支，用来测试 magma_z_mtrans_template 的其余调用者。所有用例都是一个用例一个进程（`scripts/roxy/r73_cases.txt`、`r73t_cases.txt`）。

## 正确的函数

| 函数 | 用例 | 结果 |
|---|---|---|
| magma_zmatrix_tril / _triu | n = 1, 10, 100 | 与稠密参考逐位相同 |
| magma_zmdiagadd（add = 0.25 − 0.5i） | n = 1, 10, 100 | 逐位相同 |
| magma_zindexsort / magma_zindexsortval | len = 1 … 10000，键有重复，值与键绑定 | 排序正确，值跟随键 |
| magma_zm_5stencil | n = 2, 3, 8 | 结构正确；复数版把整个模板乘以 (1 + i)，这是有意的，参考值已按此缩放 |
| magma_zmtransposeconj_cpu / _abs_cpu / _struct_cpu（方阵 40 × 40） | 3 | 正确（重跑 7 次 struct 都正确；清单里第一次的 WRONG 来自加 struct 判据之前的旧探针二进制） |

## M277：magma_z_spmm（稀疏 × 稀疏）在 CUDA ≥ 12 上每次都返回 −3003

- `sparse/blas/magma_zcuspmm.cpp` 的 CUDA ≥ 12 分支调用 `cusparseSpGEMM_workEstimation` 时 beta 传的是 NULL。cuSPARSE 打印 "parameter number 7 (beta) had an illegal value: NULL pointer"，函数返回 MAGMA_ERR_CUSPARSE_INVALID_VALUE（−3003）。
- r73.csv：n = 10、100、500，每次都是 −3003。所以在 roxy 的 CUDA 13 上，稀疏矩阵乘法这条路径完全不能用。
- 另外，magma_z_spmm 的文档写的是 "C = alpha * A * B"，但无论 CUDA < 12 还是 ≥ 12 分支，alpha 都没有传给 zcuspmm，算的是 A·B。这一点来自读代码；CUDA < 12 分支在这里跑不了，所以 alpha 被忽略还没有在 GPU 上看到。
- 稠密 B 的用法（第一版探针）被拒绝，返回 −103，这符合文档（B 必须是 CSR），不算缺陷。
- Lean：`Kernels.lean` §73 `spmm_alpha_ignored`。

## M278：magma_zm_27stencil 在 y 方向的边界上回绕

- 生成器只清除跨越 x 边界的耦合（按 row mod n 判断）。偏移 n、n ± 1、n² − n ± 1、n² + n ± 1 及其镜像也会跨越 y 边界（例如 (0, n−1, z) 连到 (0, 0, z+1)），这些项被保留下来了。
- 预测：n = 3 时有 84 个多余耦合，n = 5 时 312 个。实测（r73.err）分别是 84 和 312，完全吻合。
- n = 2 时几个偏移重合，结果反而丢了 30 个耦合。
- 5 点模板（测试程序实际用的那个）是正确的。
- Lean：`Kernels.lean` §73 `stencil27_wrap_instance`（用 `gp`、`nbr` 对具体网格点算出回绕的耦合）。

## M137/M138 的扩充：转置模板的其他调用者

magma_z_mtrans_template 按 A.num_rows 分配 row_ptr / last_rowel / B->row，却用列号去索引它们，而且把 B 的行数、列数设成和 A 一样。所以：
- 宽矩阵（30 × 47、10 × 11）：堆越界写，进程崩溃（0xC0000005）；10 × 11 只越界一个元素，有时崩溃有时不崩溃（三次运行：structcpu 崩溃、abscpu 崩溃、三种都不崩溃；第三次是用当前源码重新构建之后）。
- 高矩阵（47 × 30、11 × 10）：不崩溃，但返回的维度没有交换，结果是错的。
- conj / abs / struct 三个 _cpu 变体表现相同，已写入 M137/M138。

## 文件

- r73.csv、r73.err：第一组 8 种操作，28 个用例。
- r73t.csv：转置变体，15 个用例（重跑之后的干净版本）。
