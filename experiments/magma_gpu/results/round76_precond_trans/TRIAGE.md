# 第 76 轮：PBiCG / PQMR 用到的转置预条件（2026-11-28，roxy；新缺陷 M284，扩充 M106）

## 背景

`magma_zcustomspmv` / `magma_zcustomprecond` 是留给用户替换的模板：custom precond 是空函数，custom spmv 调用 zge3pt。它们在设计上就是占位实现，不算缺陷。

读 `magma_z_precond_wrapper.cpp` 时发现，MagmaTrans 分支里 ILU 用专门的 `_transpose` 求解，ICC 却调用和 NoTrans 一样的求解。PBiCG 和 PQMR 的影子系统要用 M⁻ᵀ，所以会走到这条路。

## 探针

`probes/round76_precond_trans.cpp`：
- 矩阵：A 为 g × g 网格上的实二维拉普拉斯矩阵加 0.05 I，以复数存储。M 是实矩阵，所以 M⁻ᵀ = M⁻ᴴ，也避开了 M129（复数 ICC 用 Lᵀ）。
- ident：对随机复向量 u、v，检验双线性恒等式 uᵀ(M⁻¹v) = (M⁻ᵀu)ᵀv。这里 M⁻¹v = right(N, left(N, v))，即 PBiCG 处理 r 的顺序；M⁻ᵀu = left(T, right(T, u))，即 PBiCG 处理 rt 的顺序。这个恒等式对任何可逆 M 都成立，当且仅当后者确实等于 M⁻ᵀu。同时也打印"左右对调"那个候选的结果。
- solve：PBiCG、PQMR 配 ILU / ICC / 无预条件，以及 BiCG、QMR；b = A·x_true，rtol 1e-10，maxiter 2n。

## M284：ICC 在 MagmaTrans 下左右求解对调

ICC 的 M = L Lᵀ，左求解为 L⁻¹，右求解为 L⁻ᵀ。MagmaTrans 分支仍然是左调用 applycumicc_l、右调用 applycumicc_r，于是影子向量得到的是 L⁻¹L⁻ᵀu，而不是 M⁻ᵀu = L⁻ᵀL⁻¹u。四种精度的代码相同。

r76.csv：

| 检查 | g = 4 | g = 10 | g = 30 |
|---|---|---|---|
| 恒等式，ILU | 1.9e-16 | 1.1e-16 | 1.7e-15 |
| 恒等式，ICC | 0.22 | 0.032 | 0.031 |
| 恒等式，ICC 左右对调后 | 6.6e-17 | 1.1e-16 | 7.6e-16 |

| 求解 | g = 10（n = 100） | g = 30（n = 900） |
|---|---|---|
| PBiCG + ILU | 15 步 | 29 步 |
| PBiCG + ICC | −201，200 步后残差 4.3e-6 | 30 步 |
| PQMR + ILU | 15 步 | 28 步 |
| PQMR + ICC | −201，200 步后残差 3.4e-9 | 370 步 |

**补丁确认**：`probes/magma_z_precond_wrapper_fixed.cpp` 只在两个 MagmaTrans 分支里交换 ICC 的调用，由 `scripts/roxy/build_fix76.bat` 链接在库前面。r76f.csv：
- ICC 的恒等式误差降到 1e-16；
- PBiCG / PQMR + ICC 分别以 15/29、15/28 步收敛，与 ILU 完全相同（SPD M-矩阵上 IC(0) 与 ILU(0) 等价）；
- ILU 的结果不变。

补丁里的链接标记放在了一个没被调用的函数里，所以没有打印出来。但 ICC 的结果变了，说明补丁确实链接上了。

对复 Hermitian 矩阵，还要叠加 M129（applycumicc_r 用的是 Lᵀ 而不是 Lᴴ）。

Lean：`Kernels.lean` §76 `icc_trans_swapped`（2 × 2 模型，L = [[1, 0], [a, 1]]：a ≠ 0 时 L⁻¹L⁻ᵀ ≠ L⁻ᵀL⁻¹）、`icc_trans_instance`。

## M106 的补充

不加预条件的 magma_zqmr 在这个实 SPD 矩阵上同样跑满 maxiter（200、1800 步），返回 −201，但真实残差已经是 3.2e-9、4.1e-10。同一矩阵上 PQMR（Magma_NONE）分别用 35、85 步收敛，BiCG 用 35、86 步收敛。已写入 M106。

## 文件

- r76.csv、r76.err：28 例。
- r76f.csv、r76f.err：20 例补丁版。
