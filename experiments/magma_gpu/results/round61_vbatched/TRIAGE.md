# 第 61 轮：可变尺寸批量分解与 LAPACK 逐矩阵比较（2026-11-28，roxy，库版本）

## 方法

`probes/round61_vbatched.cpp <op> <set> <pad> <sing>`：
- 例程：getrf_vbatched、getrf_vbatched_max_nocheck_work（工作区按查询值分配）、getrf_nopiv_vbatched、potrf_vbatched（Lower）。
- 尺寸集：
  - 0：方阵 1、7、32、33、100、257；
  - 1：长方阵，含 0×5、5×0、1×1；
  - 2：512、1000×200、200×1000；
  - 3：混合尺寸。
- ldda = m + {0, 3}。sing = 1 时，每隔一个矩阵把第 min(m,n)/2 列置零（potrf 改为把该对角元设为 −1）。
- 逐矩阵比较 info、因子和 ipiv，并检查填充行未被改写。

共 64 例（`scripts/roxy/r61_cases.txt`）。之后用 `probes/round61_potrf_info.cpp` 做针对性测试：`r61pi*.csv`，共 85 例。

## 结果

- 非奇异矩阵：4 个例程在所有尺寸集上都正确（误差 ≤ 1.5e-13，ipiv 与 LAPACK 相同，填充行未被改写）。
- 奇异矩阵上的 info：
  - getrf_vbatched / max_nocheck_work：257×257、零列在 128 时报 0，第一面板内的零列报得正确。这是已知的 **M164**，补了证据。
  - getrf_nopiv_vbatched：所有奇异矩阵都报 0。这是已知的 **M173**（`*info++` 自增的是指针而不是计数），补了证据。
- **M266（新）potrf_vbatched 大规模路径的 info 是面板内的局部值**：
  - max_n 超过交叉点（Z 448、C 384、D 480、S 704）时，走 `magma_zpotrf_lg_vbatched`，它调用面板时传的是 gbstep = 0（l.173），而固定尺寸的 zpotrf_batched 传的是 j（l.154）。
  - 结果是第 c 列（从 0 计）的非正主元被报成 (c mod 128) + 1：500 → 117，999 → 104。
  - 同一批里的小矩阵也一样，因为整批都走大路径：200×200 中 150 → 23，199 → 72。
  - 第一面板内的主元报得正确；对照组 zpotrf_batched 22/22 正确。
- **M267（新，仅 Z）360 ≤ max_n ≤ 448 时整批都不做分解**：
  - 小于交叉点 448 时走 lpout 路径。它需要 16·(max_n + 8)·8 字节共享内存，超过 47,000 就在任何内核启动之前返回 −33（zpotf2_kernels_var.cu l.102–108），也就是从 max_n = 360 起就会失败。
  - 实测 n = 360、361、384、448：返回 −33，矩阵与输入逐字节相同，info_array 仍为 0（即使对角元为负，LAPACK 报 101）；xerbla 打印 "parameter 33"，可该例程只有 7 个参数。
  - 批 {400, 100} 中两个矩阵都没动。n = 352、359、449 正确。
  - C/D/S 的 lpout 上限分别为 727、727、1460，都高于各自的交叉点，所以没有这个空档。
  - 固定尺寸版的 `zpotf2_kernels.cu` 用设备的 opt-in 上限（`magma_getdevice_shmem_block_optin()`）作比较，只有 vbatched 版（`*_kernels_var.cu`）把 47000 写死了。固定尺寸的 zpotrf_batched 在 448、449、1000 时都正确。
- Lean（`Magma/Dense/QrEigen.lean` §61）：
  - `potrf_vbatched_lg_info_local`、`_first_panel`、`_instances`（M266）；
  - `potrf_vbatched_z_gap`、`_z_gap_all`、`_cds_no_gap`（M267）。

## 补测：c/d/s 精度（`probes/round61_potrf_info_{c,d,s}.cpp`，由 round61_potrf_info.cpp 用 sed 生成）

- M266 在三种精度下都成立，各自在交叉点（C 384、D 480、S 704）以上：第 128、200、n−1 列的非正主元都报成 (c mod 128) + 1，9/9 例符合。
- 与 n = 1000 同批的 200×200 矩阵，第 151 号主元在三种精度下都报成 23；第一面板内的主元报得正确；对照组 potrf_batched 正确。
- 在交叉点处（384/480/704），c/d/s 走 lpout 路径，返回 0，主元位置正确。M267 确实只有 Z 才有，与 Lean 的 `potrf_vbatched_cds_no_gap` 一致。
