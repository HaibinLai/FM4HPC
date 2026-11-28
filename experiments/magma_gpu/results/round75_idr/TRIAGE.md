# 第 75 轮：直接调用 IDR(s) 求解器与稀疏 QR 包装（2026-11-28，roxy；新缺陷 M282、M283，扩充 M103）

MAGMA 的测试程序把 `--solver IDR` 和 `PIDR` 都映射到 PIDRMERGE，所以 magma_zidr、magma_zpidr、magma_zidr_strms、magma_zpidr_strms 以前从没跑过。`probes/round75_idr.cpp` 直接调用 6 个变体：

- 矩阵（n = 200）：
  - gen：随机复数 CSR，每行约 5 个元，对角 4 + 随机虚部；
  - rotθ：e^{iθ}·S，其中 S = tridiag(−1, 2.1, −1) 是 Hermitian 正定矩阵，θ = 0°、45°、90°、135°。
- 设置：b = A·x_true，x0 = 0，rtol 1e-10，maxiter 2n，s = 1、2、4、8；pidr 系列用 Magma_NONE 预条件。
- 判据：真实残差 |b − Ax| / |b| < 1e-8 且 info = 0。

同一个探针还测了 magma_zqr（IDR 用它来正交化影子空间）。

## M282：IDR(s) 的角度保护只用了内积的实部

- 6 个复数变体（z、c 各自的 idr、pidr、merge、strms）都计算 `rho = |Re(t^H r)| / (|t| |r|)`。注释写的、以及 van Gijzen 的 IDR(s) 原算法用的都是 |t^H r|。
- A 乘一个相位 e^{iθ} 时，t^H r 会旋转 e^{−iθ}：模长不变，实部却变了。A = iS 时 t^H r 几乎是纯虚数，代码看到 rho ≈ 0 < angle = 0.7，就把 ω 放大 angle/rho 倍，迭代因此停滞。
- r75.csv（idr 和 pidr 结果相同）：

| 矩阵 | s = 1 | s = 2 | s = 4 | s = 8 |
|---|---|---|---|---|
| rot0（= S） | 40 步收敛 | 27 | 15 | 9 |
| rot45 / rot135 | 43 | 27 | 16 | 9 |
| rot90（= iS） | −201，400 步后残差 6.9e-2 | −201，6.8e-2 | −201，2.7e-2 | −201，2.3e-2 |
| gen | 9 | 6 | 4 | 2 |

- **补丁确认**：`probes/zidr_fixed.cpp` 只改这一行，改成 |t^H r|，由 `scripts/roxy/build_fix75.bat` 链接在库前面（标记 R75_MARK）。r75f.csv：20/20 全部收敛，所有相位都与 rot0 完全相同（40/27/15/9 步，残差逐位相同），说明修正后的方法对相位不变。
- 实数 d/s 版本不受影响。
- Lean：`Kernels.lean` §75 `idr_rotation_rotates_dot`（t ↦ it 把 t^H r 映到 (im, −re)，这是一个旋转）、`idr_rho_instance`（Hermitian 正定时实部变为 0）。

## M103 的扩充：strms 变体与 merge 变体一样全部失败

idr_strms 和 pidr_strms 走的是和 MERGE 变体相同的 zgemvmdot_shfl 路径，结果与 IDRMERGE / PIDRMERGE 逐位相同：
- 所有 20 种输入都失败，包括 rot0 这种 Hermitian 正定矩阵；
- s = 1 时跑满 400 步，残差 0.16–0.21；
- s ≥ 2 时 11–14 步后返回 −202，残差为 NaN。

已写入 M103。

## M283：magma_zqr 在设备上原地改写输入 A

- 文档把 A 标为 [in]。但对设备矩阵，代码直接令 dA = A.dval 并在其上做 geqrf / ungqr，调用后 A 变成了 Q（r75q.csv：最大变化 0.80–1.16）。CPU 矩阵则先拷贝到设备，A 保持不变。
- Q、R 本身正确：|QᴴQ − I| ≤ 9e-16，|QR − A| / |A| ≤ 1.6e-15，R 的下三角为 0，lda 加余量的情况也正确。
- 4 × 8（m < n）被 zungqr_gpu 以 −2 拒绝。这与 Q 是 m × n 正交矩阵的要求一致，不算缺陷。
- 库内调用者（IDR）调用后立即释放这个矩阵，所以只影响直接调用者。

## 说明

第一次运行时 QR 用例的参数顺序错了：zsh 的 `set -- $c` 不会拆分单词，所以所有 QR 用例都按 CPU、ldpad 0 运行。已改正用例表，重跑结果在 r75q.csv，r75.csv 里删掉了那 12 行无效结果。

## 文件

- r75.csv、r75.err：120 例求解器。
- r75f.csv：20 例补丁版 zidr。
- r75q.csv、r75q.err：12 例 QR。
