# 第 56 轮：Hermitian / 对称不定分解与其求解器（2026-11-28，roxy，库版本，1 张 GPU）

## 方法

`probes/round56_ldl.cpp <op> <n> <nrhs> <uplo> <pad> <kind>`：

- 生产方与消费方的组合：
  - 驱动：hesv_nopiv_gpu、sysv_nopiv_gpu、hesv（Bunch–Kaufman）；
  - GPU 上分解后再求解：hetrf_nopiv_gpu → hetrs_nopiv_gpu，sytrf_nopiv_gpu → sytrs_nopiv_gpu；
  - CPU 接口的 hetrf_nopiv，上传后用 hetrs_nopiv_gpu 求解；
  - hetrf、hetrf_gpu（Bunch–Kaufman，文档称 ipiv 与 LAPACK 相同），用 LAPACK zhetrs 求解，以此检查输出格式的兼容性；
  - hetrf_aasen（cpu_panel 0、1）：只看状态码。
- 输入：
  - kind 0：正定；
  - kind 1：对角占优的不定矩阵，对角元正负交替，不选主元也稳定；
  - kind 2：随机 Hermitian，必须选主元，只用于 Bunch–Kaufman。
- 参数：n ∈ {1, 33, 300, 513}，nrhs ∈ {1, 7}，U/L，lda = n + {0, 5}。
- 判据：res = max|Ax − b| / (max|A|·max|x|·n) < 1e-13。
- 用例由 `scripts/roxy/r56_cases.txt` 给出，共 528 个，全部 exit 0，无 CUDA 错误。

## 结果

- 求解类 512 例全部正确：每个 op 64 例，误差在 1e-18 到 4e-16 之间。Bunch–Kaufman 的 ipiv 确实与 LAPACK zhetrs 兼容，CPU 接口的 nopiv 输出也能被 GPU 接口的 hetrs_nopiv 使用。
- **M264（新，轻微）hetrf_aasen Upper**：
  - 文档与 Lower 并列描述了 Upper（A = U·D·Uᴴ），但参数检查写的是 `if (uplo != MagmaLower) *info = MAGMA_ERR_NOT_IMPLEMENTED`（l.109–112），文档里的 −1 检查被注释掉了。
  - 实测 Upper 8/8 返回 −117，Lower 8/8 返回 info 0。
  - 会明确报错，不是静默错误；归入"拒绝合法调用"。
  - Lean：`aasen_rejects_documented_upper`（`Magma/Dense/QrEigen.lean` §56）。
