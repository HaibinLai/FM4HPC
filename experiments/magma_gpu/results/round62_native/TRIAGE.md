# 第 62 轮：按尺寸分派的批量与原生路径，资源上限与回退（2026-11-28，roxy；没有新缺陷）

## 起点

M267 说明，按尺寸选路径时，小路径的资源上限可能低于交叉点。本轮把会因资源不足返回负值的内核（`arginfo = -100`、`-33`、`-1`）全部列出，逐个检查调用者是否检查返回值并回退：

| 内核 | 调用者 | 处理 |
|---|---|---|
| zgetf2_fused_vbatched | zgetrf_vbatched l.39、zgetf2_vbatched l.163 | 返回非 0 时回退到逐块路径 |
| zgetf2_fused_batched | zgetf2_batched l.174 | 同上 |
| zgesv_batched_small | zgesv_batched l.124 | 返回非 0 时回退到 getrf_batched |
| zgeqr2_fused_sm_batched、zunm2r_sm_batched | zgeqrf_panel_batched、zunmqr_batched | 保存返回值并据此分派 |
| zgbtrf_batched_fused_sm | zgbtrf_batched l.175 | 同上 |
| zpotf2 lpout（vbatched，阈值写死 47000） | zpotrf_vbatched | **没有回退：即 M267** |
| zgetf2_native_fused（m > ZGETF2_FUSED_MAX_M 时返回 −1） | zgetf2_native | 返回值被忽略，但调用前已判断 `m > ZGETF2_FUSED_MAX_M`，此时改走 blocked 路径（zgetf2_native.cpp l.265）；n 不会超过 SM 数，因为递归到 n ≤ nb ≤ 32 才调用 |

## 实测（`probes/round62_getrf_native_tall.cpp`）

magma_zgetrf_native 与 magma_zgetrf_gpu（对照组），m ∈ {1000, 7168, 7169, 8000, 12000} × n ∈ {32, 100}，以及 8000×300，共 22 例。全部正确：info 为 0，ipiv 与 LAPACK 相同，残差 ≤ 6.4e-16。跨过 7168 也没有问题。

## 补充：vbatched 内核的共享内存上限从哪里来

- zgetf2_kernels_var.cu（l.359–373）和 zgetf2_nopiv_kernels_var.cu（l.217–241、369–391）都用 `cudaDevAttrMaxSharedMemoryPerBlockOptin` 查设备的真实上限（不满足时退回默认上限），超出时返回 −100，调用者会回退。
- vbatched 系列中，只有 zpotf2_kernels_var.cu（l.104）把上限写死为 47000，并且调用者不回退，即 M267。

## 结论

在这一类"资源不足返回负值"的分派中，只有 vbatched 的 lpout 路径没有回退，即 M267。
