# 第 77 轮：LSQR 与复数矩阵上的转置预条件（2026-11-28，roxy；新缺陷 M285）

`probes/round77_precond_trans2.cpp` 是第 76 轮探针的扩充，新增：
- 复数矩阵选项（R77_CPX=1）：二维五点模板，对角 4.05 + 0.8i，下方邻居 −1 + 0.3i，上方邻居 −1 − 0.2i，复数且非 Hermitian；
- Hermitian 恒等式 uᴴ(M⁻¹v) = (M⁻ᴴu)ᴴv（成立当且仅当转置分支给出的是 M⁻ᴴu）；
- Jacobi 预条件；
- LSQR。

用例表：`scripts/roxy/r77r_cases.txt`（实矩阵）、`r77c_cases.txt`（复数矩阵）。

## MagmaTrans 的约定

- LSQR 和 PBiCG 都用 magma_zmtransposeconjugate 构造 Aᴴ。
- ILU 的转置 setup 也用 transposeconjugate。
- r77c.csv：ILU 的转置分支在复数矩阵上满足 Hermitian 恒等式（2e-16），不满足 T 恒等式（0.38–1.45）。

所以这个 wrapper 里 MagmaTrans 的意思是共轭转置。

## M285：Jacobi 的转置分支没有取共轭

magma_z_applyprecond_left 的 MagmaTrans 分支对 Jacobi 仍然乘 D⁻¹，而不是 D⁻ᴴ = conj(D⁻¹)。

| 复数矩阵 | g = 10 | g = 30 |
|---|---|---|
| Hermitian 恒等式，Jacobi | 0.388 | 0.388 |
| LSQR，无预条件 | 64 步 | 91 步 |
| LSQR + Jacobi | −201，200 步后残差 0.27 | −201，1800 步后残差 0.49 |
| PBiCG，无预条件 | 33 步 | 47 步 |
| PBiCG + Jacobi | −202，残差 2.3 | −202，残差 13.2 |

**补丁确认**：`probes/magma_z_precond_wrapper_fix77.cpp` 只在 MagmaTrans 的 Jacobi 分支里改用 conj(D⁻¹)，由 `scripts/roxy/build_fix77.bat` 链接（标记 R77_MARK，打印了 245 次）。r77cf.csv：
- Hermitian 恒等式误差 1.5e-16；
- LSQR + Jacobi 为 64/91 步，PBiCG + Jacobi 为 33/47 步。这与无预条件时相同，因为对角元全是同一个常数，Jacobi 只是标量缩放。

z 和 c 版本受影响；实数 d/s 版本里取共轭不改变任何值。

Lean：`Kernels.lean` §77 `jacobi_trans_adjoint_iff`（标量模型：伴随恒等式成立当且仅当 d 是实数）、`jacobi_trans_instance`。

## 实矩阵上的 LSQR（r77r.csv）

| LSQR | g = 10 | g = 30 |
|---|---|---|
| 无预条件 | 59 步 | 457 步 |
| Jacobi | 59 步 | 457 步 |
| ILU | 23 步 | 69 步 |
| ICC | 38 步 | 82 步 |

- ICC 在 LSQR 里同样受 M284 影响（恒等式误差 0.032），只是 LSQR 仍能收敛，步数比 ILU 多（38 对 23）。
- 所有收敛的用例都是真实收敛：报告的残差与真实残差一致。

## 文件

- r77r.csv / .err：11 例。
- r77c.csv / .err：20 例。
- r77cf.csv / .err：8 例补丁版。
