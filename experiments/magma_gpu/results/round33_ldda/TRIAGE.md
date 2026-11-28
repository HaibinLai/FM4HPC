# 第 33 轮：未对齐 ldda 下的分解——分诊

程序：`probes/round33_ldda.cpp`（roxy，sm_120）。s 和 d 两种精度，ldda ∈ {n, n+1, n+3, n+5, roundup(n,32)}。批量调用用 3 个矩阵首尾相接存放（步长 ldda·n），所以每个矩阵的起始地址也不对齐。

| 例程 | 尺寸 | 用例 | 结果 |
|---|---|---|---|
| getrf_batched、getrf_nopiv_batched、potrf_batched、geqrf_batched、getrs_batched | n ∈ {5, 13, 31, 33, 64, 100, 257} | 330 | 全部正确；最大误差 0.53（`batched.csv`） |
| getrf_gpu（混合版）、potrf_gpu、potrf_native、geqrf2_gpu | n ∈ {100, 257, 513, 1000} | 162 | 全部正确（`single.csv`） |
| getrf_native | 同上 | 38 | **19 个错**：s 精度误差 61–16,600，d 精度误差 2e10–3e12，info 都是 0。这就是 M242 |

误差的定义：
- LU：‖PA−LU‖₁ /(‖A‖₁·n·eps)
- Cholesky：‖A−LLᵀ‖₁ /(‖A‖₁·n·eps)
- QR：‖AᵀA−RᵀR‖₁ /(‖A‖₁²·n·eps)
- getrs：x 的范数型后向误差

结论：在 LU、Cholesky、QR 三族中，只有 getrf_native 受对齐影响。融合的 getf2_native 面板内核是唯一依赖跨块通信的分解内核（与第 32 轮的扫描结论一致）。
