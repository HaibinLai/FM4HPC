# 第 66 轮：稀疏 LOBPCG 与稠密 zheevd 比较（2026-11-28，roxy，库版本）

## 方法

`probes/round66_lobpcg.cpp`：二维磁 Laplacian，Hermitian 正定，n = 256 和 1024；θ = 0 时为实矩阵，作对照。调用路径与库内一致：magma_zparse_opts（`--solver LOBPCG --ev k --rtol 1e-8`）、magma_zeigensolverinfo_init、magma_zlobpcg。k ∈ {4, 8, 16}，预条件为 NONE/JACOBI，共 24 例。比较特征值与 LAPACK zheevd 的差，以及每个向量的残差 ‖Av − λv‖。

## 结果：M271（新）

- 原版（`r66.csv`）：24/24 返回 info 0，但最大特征向量残差为 5.8e-6 到 6.4e-4，远高于要求的 1e-8。实矩阵与复矩阵表现一样，所以与复数无关。
- 逐个向量看（n = 1024，k = 8）：残差随序号增大，依次为 3.8e-9、4.6e-8、5.8e-8、5.8e-7、4.1e-7、6.3e-6、2.5e-5、8.9e-5。只有第 0 个达标。
- 原因：
  - 除了逐向量的判据（zcompact 去掉残差 ≤ rtol 的向量，全部去掉后才结束），zlobpcg.cpp l.496–508 还有一个判据：只要 `residualNorms(0, iter)`，也就是第 0 个 Ritz 对的残差，低于 init_res × rtol，就跳出主循环并返回 MAGMA_SUCCESS。
  - 结果是第 0 对一收敛就停止，其余 k − 1 对都没有收敛。
- 补丁 `probes/zlobpcg_fixed.cpp`（去掉这个跳出，由 `scripts/roxy/build_fix66.bat` 链接在库之前）：24/24 收敛，残差 ≤ 1.6e-8，迭代步数是原来的 2–4 倍（`r66fix.csv`）。
- 与已有条目的关系：M092（参数检查失效）、M093（NaN 视为收敛）、M204（shift 只移第 0 行）讲的都是别的问题。
- Lean：`Magma/Kernels.lean` §66 中的 `lobpcg_stops_on_first`、`lobpcg_instance`。
