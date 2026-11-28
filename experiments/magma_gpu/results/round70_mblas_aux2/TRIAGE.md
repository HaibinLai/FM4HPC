# 第 70 轮：剩余的 magmablas 辅助内核（2026-11-28，roxy，库版本）

## 方法

`probes/round70_mblas_aux2.cpp`，与按文档写的 CPU 参考实现比较。用例表为 `scripts/roxy/r70_cases.txt`（214 例）和 `r70ls_cases.txt`（laswp_sym，跑两遍）。

| 例程 | 参数 | 结果 |
|---|---|---|
| magmablas_zherk_small_reduce | n ∈ {1, 7, 16, 31, 32}，k ∈ {100, 5000}，U/L，N/C，beta 0/≠0，1/32 个线程块 | 160/160 正确，另一侧三角未被改动 |
| magmablas_zhemv、zsymv（非 work 接口） | n 1–1000，U/L，incx 1/2，另一侧三角放 1e6 作哨兵 | 40/40 正确 |
| magmablas_zsymmetrize_tiles | 5 种 tile 配置 × U/L | 8/8 正确；ntile = 1 且步长为 0 时返回 −7，因为违反了文档的 "mstride ≥ m 或 nstride ≥ m"，是我的探针误用 |
| **magmablas_zlaswp_sym** | n 8–200，k2 1–100 | **M275、M276** |
| zlacpy_sym_in/out | — | 文档只是照抄 ZLACPY，没有描述实际行为（按 rows[] 交换行），无法依文档测试，没有测 |

## laswp_sym

- 语义：只有一个主元时，结果与"复对称、不取共轭"的参考完全一致（err2 = 0），与 Hermitian 参考不一致（err 约 1.4）。这与文档 "symmetric pivoting on a symmetric matrix" 相符。
- **M275**：
  - 启动函数按 MAX_PIVOTS = 32 分块，按块内的相对下标填 `params.ipiv[j]`，却把 j0 = k、npivots = k + npivots 按绝对下标传给内核。内核读 `params.ipiv[ii]`（ii 从 j0 到 npivots），所以从第二块起都越过 32 项的数组。
  - 实测 k2 ≥ 33 时相对误差约 200，每次都能复现。
- **M276**：
  - 同一次启动中，每个线程依次处理这一块的所有主元，线程之间没有同步。可是哪个线程负责某个下三角元素，会随主元变化：行交换时由线程 c 负责，行列交换时 (tid, i1) 由线程 r、(i2, tid) 由线程 c 负责，列交换时由线程 r 负责。所以相邻主元之间存在线程间（以及块间，NTHREADS = 64）的竞争。
  - 实测：n = 64、k2 = 10 时误差 1.19；k2 = 32 时两次运行分别为 1.19 和 1.28，结果不确定；n = 40、k2 = 32 时 1.51；n = 200、k2 = 5 时 0.33。
  - 小规模（n = 8、k2 = 4；n = 33、k2 = 16）碰巧正确，可能是因为线程都在一两个 warp 内，几乎同步执行。
- 库内没有调用者（Aasen 用的是别的代码），只影响直接调用这个公开接口的外部用户。
- Lean：`Magma/Kernels.lean` §70 中的 `laswp_sym_second_chunk_out_of_bounds`、`laswp_sym_race_instance`（主元 (0,5) 之后接 (2,7)：元素 (5,2) 先由线程 2、再由线程 5 移动，两者之间没有任何顺序约束）。
