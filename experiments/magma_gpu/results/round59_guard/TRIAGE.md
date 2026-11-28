# 第 59 轮：设备工作区按文档大小分配，并加哨兵保护区（2026-11-28，roxy，库版本；没有新缺陷）

## 起点

M261 是"按文档大小分配的设备工作区被写出界"。以前的轮次检查过主机端 lwork 的取值（M049 等），但从未对设备工作区做越界检查。本轮对所有带文档化设备工作区的公开例程统一检查：工作区取文档给的大小，其后紧跟 65,536 个元素的哨兵（每字节 0xA5）。调用后统计被改写的哨兵元素数，并在能便宜比较的地方核对结果。

## 方法

程序为 `probes/round59_guard.cpp`，用例表为 `scripts/roxy/r59_cases.txt`，共 89 例，每例一个进程，全部 exit 0，无 CUDA 错误。

| 例程 | 设备工作区（文档大小） | 参数 | 结果 |
|---|---|---|---|
| geqrf_gpu、geqrf3_gpu | dT：(2·min(m,n) + roundup(n,32))·nb | 6 种形状，含 31×31、600×70 | 12/12 保护区未被写 |
| geqrs_gpu、geqrs3_gpu | dT：2·min·nb + roundup(n,32)·max(nb, nrhs)（geqrs 自己的文档） | m ≥ n 的 5 种形状 × nrhs 1/7/100 | 30/30 保护区未被写，X 与 LAPACK zgels 一致 |
| hetrd_he2hb | dT：n·nb | n 100/300 × nb 32/64 | 4/4 |
| hetrd2_gpu | dwork：ldda·⌈n/64⌉ + 2·ldda·nb | n 33/300/513 | 3/3，d 与 LAPACK 一致 |
| getri_gpu | dwork：n·nb | n 33/300/513 | 3/3，‖A·A⁻¹ − I‖ ≤ 4.4e-15 |
| zstedx | dwork：3n²/2 + 3n | n 33/300/513 | 3/3，特征值与 dsterf 一致 |
| gegqr_gpu | dwork：n²、3n² + min + 2、无、n²（ikind 1–4）；主机 work 按文档 | 300×50、1000×30、100×100 | 12/12，‖QᴴQ − I‖ ≤ 1.2e-13 |
| magmablas_ztrsm_work | dX：lddx·n；d_dinvA：⌈dim/128⌉·128²（flag 1） | 3 种形状 × L/R | 6/6，与 blas ztrsm 一致 |
| magmablas_zhemv_work | dwork：ldda·⌈n/64⌉ | n 33/300/513/1000 | 4/4 |
| magma_zlarfb_gpu | dwork：ldwork·k，ldwork = n（L）/ m（R） | 3 种 (m, n, k) × L/R | 6/6，与 LAPACK zlarfb 一致 |
| geqp3_gpu | dwork：(n+1)·nb（[cz] 的文档值） | 6 种形状 | 6/6 返回 −8，是已知的 M120 |

## 结论

- 除 M261（unmqr_gpu）外，带文档化设备工作区的公开例程都没有写出界。
- geqp3_gpu 的代码（l.139）对所有精度都要求 (n+1)·nb + 2n，而文档给 [cz] 的是 (n+1)·nb。M120 原先只有 d 精度的证据，本轮补上 z 的实测。
