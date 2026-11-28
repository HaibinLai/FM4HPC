# 第 65 轮：稀疏迭代求解器在真正复数数据上的表现（2026-11-28，roxy，库版本）

## 起点

第 29 轮发现 CG/PCG 在复数上失败（M084）之后，主扫描改用实数数据，只保留了一个复对称矩阵。本轮补上真正的复数系统，探针为 `probes/round65_sparse_complex.cpp`（由 round29_sparse.cpp 复制，加了 `cplx` 模式）：

- herm_magnetic：二维磁 Laplacian，Hermitian 正定，非对角元为 −e^{±iθ}（每条边的 θ 不同），对角为 4.2；
- herm_magnetic_indef：同一矩阵，对角取 1.0，Hermitian 不定；
- nonherm_complex：复系数对流，一个方向的非对角元为 −1 − 0.4i，另一方向为 −1 + 0.1i，对角为 4。

1024×1024，CSR 格式，x 为复随机向量，以主机上计算的真实残差判定。共 21 个求解器（`scripts/roxy/r65_cases.txt`，运行脚本 `run_multi.ps1`）。

## 结果

| 求解器 | herm_magnetic | herm_indef | nonherm_complex | 归类 |
|---|---|---|---|---|
| BICG、BICGSTAB、PBICG、PBICGSTAB、CGS、PCGS、GMRES、PGMRES、QMR、PQMR、LSQR、BOMBARDMENT | OK | FAIL | OK | 正常；不定矩阵上 info ≠ 0，属于如实报告的失败 |
| JACOBI | OK | FAIL | FAIL | 非对角占优时发散，如实报告 |
| CG、PCG | FAIL | FAIL | FAIL | 已知 M084 |
| IDR、PIDR | FAIL | FAIL | FAIL | 实数数据上也失败，已知 M103 |
| BA、ITERREF | LIE | LIE | LIE | 实数数据上也如此，已知 M089 |
| **TFQMR、PTFQMR（NONE/JACOBI/ILU）** | OK | FAIL | **FAIL** | **新缺陷 M270** |

## M270：TFQMR 系列把影子内积的顺序弄反了

- `magma_zdotc(x, y)` 计算的是 xᴴy。TFQMR 需要 ρ = r̃ᴴr、σ = r̃ᴴv，zcgs.cpp 就是这样写的。
- 四个变体的写法：
  - ztfqmr.cpp、ztfqmr_merge.cpp（`--solver TFQMR` 不带预条件时实际运行的就是它）、zptfqmr_merge.cpp：ρ、σ 都用了 `zdotc(r, r̃)`、`zdotc(v, r̃)` 这种反序；
  - zptfqmr.cpp：σ 是对的，ρ 是反的。
- 反序得到的是正确值的共轭，所以 α、β 都被取了共轭。
- 当 A 为 Hermitian 且 r̃ = r₀ 时，每个 r̃ᴴAᵏr₀ 都是实数，取共轭不影响结果，所以 Hermitian 矩阵上能收敛。非 Hermitian 的复系统（包括第 29 轮的复对称矩阵）则不收敛。
- 证据：
  - 原版在 nonherm_complex 上 1000 步后返回 −201，真实相对残差 0.013–0.043；同类的 CGS 39 步收敛。
  - 补丁 `probes/ztfqmr_merge_fixed.cpp`、`probes/zptfqmr_merge_fixed.cpp`（另有 `ztfqmr_fixed.cpp`）改为 r̃ᴴx 的顺序，构建脚本为 `scripts/roxy/build_fix65.bat`，用标记确认补丁版确实被链接。补丁后 NONE/JACOBI/ILU 分别在 39/39/14 步收敛，相对残差 7e-11。Hermitian 上的结果不变，其余 19 个求解器的结果与原版逐行相同（`r65fix.csv`）。
- 注意：`--solver TFQMR` 经 magma_zparse_opts 映射为 PTFQMRMERGE，再按预条件转成 TFQMRMERGE。第一次只给 ztfqmr.cpp 打补丁时，结果完全不变，原因就在这里。
- Lean：`Magma/Kernels.lean` §65 中的 `tfqmr_dotc_reversed`（反序得到的是共轭）、`tfqmr_rho_wrong_iff`（两者不同，当且仅当虚部不为 0）。
