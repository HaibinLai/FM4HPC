# 第 63 轮：从未覆盖的 magmablas 辅助内核（2026-11-28，roxy，库版本）

## 方法

`probes/round63_mblas_aux.cpp`：16 个在 COVERAGE、FINDINGS 和第 27 轮中都没出现过的 magmablas_z* 内核，各自与按其文档写的 CPU 参考实现比较。涉及 m ≠ n、1×1、ld 填充、步长、各种 trans/uplo。共 466 例（`scripts/roxy/r63_cases.txt`），全部 exit 0，无 CUDA 错误。

| 内核 | 用例 | 结果 |
|---|---|---|
| geadd2、gemv_conj（incx 1/2）、laset_band（U/L，k 1/2/5）、laswpx、laswp2（行存储） | 110 | 全部正确 |
| transpose_conj、transpose_inplace、transpose_conj_inplace | 38 | 全部正确 |
| trmv（U/L × N/T/C × incx 1/2，n 1–512） | 168 | 全部正确 |
| symmetrize（U/L；行为是 Hermitian：另一侧取共轭，对角取实部） | 28 | 全部正确 |
| symv_work（复对称，另一侧三角放哨兵） | 14 | 全部正确，未读另一侧 |
| larfg、swapdblk、lacpy_conj | 14 | 全部正确 |
| **geam** | 90 | **N/T 组合 40/40 正确；只要有 C 就全错，50/50（M268）** |
| **gemm_reduce** | 4 | **与文档的 AᵀB 不符，与 AᴴB 一致（M269）** |

## 新缺陷

- **M268 magmablas_[zc]geam 的 ConjTrans 没有取共轭**：
  - 文档写明 "MagmaConjTrans: op( A ) = A**H"，但分派（l.386–399）把所有非 NoTrans 的情况都交给转置内核 nc/cn/cc，这几个内核都不取共轭，ConjTrans 实际按 Trans 计算。
  - 误差：只有 B 取共轭时约 0.55（beta = 0.4），A 取共轭时 0.9–1.0（alpha = 0.7）。
  - MAGMA 自己的 `testing_zgeam` 用 `h_A(j,i)` 构造 CPU 参考，同样不取共轭，与缺陷一致，所以测不出来。库内没有调用者。
- **M269 magmablas_[zc]gemm_reduce 计算的是 AᴴB**：
  - 文档写 "C := alpha*A^T*B + beta*C"，内核累加的是 `conj(dA[j])*dB[j]`（l.52）。
  - 实测与 AᴴB 一致（误差 ≤ 3e-15），与 AᵀB 相差 1.1–1.4。
  - 库内唯一的调用（zgeqr2x_gpu-v3.cpp l.35）已被注释掉。

## Lean

`Magma/Kernels.lean` §63：`geam_conjtrans_is_trans`、`geam_conjtrans_wrong`、`geam_n_t_agree`、`gemm_reduce_conjugates`。
