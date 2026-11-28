# 第 37 轮：两阶段三对角化链路与 RBT——分诊

程序：`probes/round37_2stage.cpp`（d 精度）。

## dsyevdx_2stage（链路：sy2sb → hetrd_hb2st 的 bulge chasing → bulge_back / applyQ 回代变换）

- 共 450 例：n ∈ {1, 2, 63–65, 127–129, 200, 255–257, 511, 513, 1000}；谱分为 random、clustered（间距 1e−10）、twovalues、zero、diagonal、tridiag 六类；range 取 A/I/V，uplo 取 L/U。检查 m、与 LAPACK dsyevd 比较的特征值、残差和正交性（`eig.csv`）。
- 356 例正确。94 例 BAD，全部有归属：

| 归属 | 例数 |
|---|---|
| n = 1 时 mout 没有被设置（已知 M236，其中包含 heevdx_2stage） | 36 |
| uplo = Upper 且 n ≥ 256：sy2sb 打印 “not yet implemented” 后返回 NOT_IMPLEMENTED，返回值被丢弃，于是在没有约化的 A 上继续计算（已知 M065） | 31 |
| range V 且矩阵有重复特征值（twovalues、diagonal）：区间端点正好落在重复值上，期望的个数有歧义（测试程序问题） | 23 |
| range V，n = 256/257，tridiag 和 clustered：特征向量错（残差最大 3.5e13，正交性最大 6.8e15） | 4 |

- 最后一类单独追查过（`v256.csv`）：
  - 1 阶段的 dsyevdx 在同样的用例上同样出错，说明问题在两者共用的 dstedx；
  - tridiag 乘以 0.5（max|T| = 1，dstedx 内部的缩放不起作用）后正确；
  - clustered 不论乘不乘 3 都错，因为特征值间距只有 1e−10，哪怕极小的缩放偏差也会让特征值越过未缩放的区间端点。
  
  这正是已知的 M216（dstedx 用未缩放的端点选取特征向量），本轮补充了证据：1 阶段驱动同样受影响。
- 两阶段链路本身（Lower、n ≥ 256、各种谱、range A/I/V）没有发现新问题。

## dgesv_rbt / gerbt：新缺陷 M250

- n ∈ {1, 2, 3, 5, 7, 31, 33, 63, 65, 127, 129, 255, 513, 1001}，refine 取 0/1，nrhs 取 1/3：n = 1、2、3 的 12 例后向误差为 1e12–1e14，info = 0；其余全部正确（`rbt.csv`）。
- 定位（`rbtiso.csv`）：gesv_nopiv_gpu 本身正确。“gerbt 变换 → 主机端精确求解 → prbt_mv 回代”这一往返在 n ≤ 3 时出错，n ≥ 4 时正确。
- 根因：`magmablas_zapply_transpose_vector_devfunc`（Uᵀb）开头是 `if(n < 2) return;`，而矩阵变换（elementary multiplication，没有提前返回）和 V·y 变换（`if(n < 1) return;`）对大小为 1 的子蝶形都会乘上 u/v。n ≤ 3 时有一层出现大小为 1 的子块（n = 1 时两层都是，n = 2 拆成 1+1，n = 3 拆成 2+1）。
- 验证：`probes/dgerbt_kernels_fixed.cu` 改为 `n < 1`（`scripts/roxy/build_fix37.bat`），rbtiso 6/6、rbt 56/56 正确（`*_fixed.csv`）。gesv_rbt_batched 用的是同一个 device 函数。
- 附带说明：zgesv_rbt.cpp 第 83 行的 `nn = n;//magma_roundup( n, 4 );`，补齐到 4 的倍数这一步被注释掉了；按推理，若保留补齐，n ≤ 3 会被补成 4，子块大小都 ≥ 2，缺陷就不会暴露（未实测）。
