# 第 68 轮：公开的面板 QR geqr2x{,2,3,4}_gpu 的工作区（2026-11-28，roxy，库版本）

## 起点

这四个例程从来没测过。它们的文档给出的 dwork 大小与代码明显矛盾：
- v1 写的是 "COMPLEX_16 array, dimension (N)"，但签名里是 double*，并且代码把 dwork + 2n 当作复数区使用；
- v2–v4 写的是 "DOUBLE PRECISION array, dimension (3 N)"，但 v2 的 dnorm 放在 dwork + 4n，v3/v4 的复数区从 dwork + 2n 开始。

## 方法

`probes/round68_geqr2x.cpp`：dwork 按文档大小分配（v1 为 2N 个 double，v2–v4 为 3N 个 double），其后紧跟哨兵保护区。参数 m ∈ {64, 200, 1000}，n ∈ {1, 16, 32, 64, 128}，另外在 m = 1000、n = 64 时试了 4n、5n、6n 三种分配。检查保护区是否被写，并把 tau 和 R 块（ddA）与 LAPACK zgeqrf 比较。共 68 例，全部 exit 0。

## 结果：M274（新）

- 文档大小下，56 例中有 47 例的保护区被写：凡是 n ≥ 16 的都被写，另外 v2 在 n = 1 时也被写。实际用量（以 double 计）：

| 版本 | 文档 | 实际 |
|---|---|---|
| v1 | 2n | 4n |
| v2 | 3n | 5n − 4 |
| v3、v4 | 3n | n ≤ 32 时 6n − 4，n = 64 时 34n，n = 128 时 50n |

- 例如 n = 16 时越界写 v1 32、v2 28、v3/v4 44 个 double；n = 128 时 v3/v4 越界写 6,016 个。
- tau 和 R 仍与 LAPACK 一致（误差 ≤ 3e-13），因为越界的写入落在了保护区里。在真实程序中，它们会落到相邻的分配上。
- MAGMA 自己的 testing_zgeqr2x_gpu 分配 max(5·min_mn, (2·64 + 2)·min_mn) 个 double，所以测不出来。这与 M261（unmqr_gpu 的 dT）是同一种模式：文档给的工作区比代码实际用的小，自带测试多分配了。
- Lean：`Magma/Kernels.lean` §68 中的 `geqr2x_overruns`（3 ≤ n ≤ 128 时每个版本都超出文档大小）、`geqr2x_instances`。
