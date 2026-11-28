# 第 51 轮：接收 queue、却在别处做事的例程（2026-11-27，roxy）

起点是第 50 轮发现 `trtri_diag` 里不带流的 `cudaMemset` 掩盖了 d/s 的 M256。全库扫描不带流的 `cudaMemset/cudaMemcpy/…Symbol`（testing、example 除外），以及不带流的内核启动，命中以下几组（四种精度各一份）：

| 文件 | 不带流的操作 | 结论 |
|---|---|---|
| `[zcds]trtri_diag.cu` l.104 | `cudaMemset(d_dinvA, …)` | **新登记 M257** |
| `[zcds]gecscsyncfreetrsm.cu` | `cudaMemset`、`cudaMemcpy` 和两个内核都在默认流；消费者的等待被注释掉 | 已有 M123，本轮补补丁证据 |
| `slag2h.cu` | 内核拿到 `&magma_flag`（主机影子地址）；batched 只拷 1 个标志 | 已有 M108、M110，本轮补补丁证据 |
| `[zc]lag2c`、`[zc]lat2c`、`[ds]lag2s`、`[ds]lat2s` | 用 `cudaMemcpyFromSymbol` 读回溢出标志 | 已有 M115（全局标志跨线程竞争）；非阻塞 queue 下本平台未复现，只记为观察 |
| `interface_cuda/alloc.cpp` `magma_memset` | `cudaMemset` | 同步接口，不是缺陷 |
| `dgmres_plu_gpu.cpp` | `cudaMemset`，仅在 `MAGMA_USE_CUDAMEMSET` 下编译 | 默认不编译 |

**教训**：写探针之前应该先在 FINDINGS.csv、SPARSE_INVENTORY.md、NONZ_INVENTORY.md 里按文件名搜一遍。这一轮的四个候选中有三个早已登记（HANDOFF §29.7、§29.14、§29.17）。这条已写进交接文档的流程。

## 方法

`probes/round51_stream.cpp`，一个进程一个用例，`scripts/roxy/run_r51.ps1` 运行，共 99 个进程，全部 exit 0、无缺行。CUDA 错误状态见 `r51.err`（每个用例前有 `==` 行）。

- queue 有三种：`magma_queue_create`；`magma_queue_create_from_cuda` 配阻塞流；`magma_queue_create_from_cuda` 配非阻塞流。文档写明后者接受任意流（"CUDA stream to use, even if NULL"）。
- 补丁版 `round51_stream_fix.exe` 先链接以下三个文件，再链接库：
  - `dgecscsyncfreetrsm_fixed.cu`、`zgecscsyncfreetrsm_fixed.cu`：恢复等待。用 volatile 读入度，等它降到 1，之后 `__threadfence`，再用 `__ldcg` 读部分和。
  - `slag2h_fixed.cu`：用 `cudaGetSymbolAddress` 取设备地址，batched 版拷入、拷出整块标志。

## 结果

- **M257（新）trtri_diag**：另一条阻塞流先忙 150 ms，然后调用 `magmablas_dtrtri_diag`。
  - 主机调用耗时：MAGMA queue 156.4 ms，阻塞流 164.4 ms，非阻塞流 160.2 ms。另一条流空闲时只要 0.4–0.6 ms。
  - 6 种情况结果都正确（|A·inv − I| = 1.1e-16）。
  - 解释：`cudaMemset` 走 legacy 默认流，要等设备上所有阻塞流空闲；在 roxy（Windows）上主机调用本身也被阻塞到那时。dtrsm.cu 的文档说 `magmablas_dtrsm_work()` 和 `magmablas_dtrsm_outofplace()` 是异步的，这一点不成立。
  - 正因为主机被阻塞，memset 不会晚于后面的内核执行，所以 CUDA 规范下"非阻塞 queue 上 memset 与内核无序"的危险在本平台不会出现，这一点没有验证。
  - 同一个 memset 在第 50 轮的单卡模拟里把所有虚拟设备排了序，掩盖了 d/s 的 M256。
- **M123（补证据）sync-free CSC 三角求解**：d、z，下双对角和随机下三角（每列约 5 个非零元），m = 1000 和 20000，各跑 3 次。
  - 原版 24/24 出错，错误分量数为 999/1000、19999/20000、978/1000、19663–19664/20000，最大误差到 266。
  - 补丁版 24/24 精确，误差 ≤ 5.7e-14。
  - s、c 的源码里有同样被注释掉的等待（l.82、l.90），没有运行。
- **M108、M110（补证据）slag2h**：
  - 原版只要有超过 65504 的元素，3 种 queue 都出现 "an illegal memory access was encountered"，且 info = 0；没有溢出的对照组没有错误。
  - batched 版无论溢出在哪个矩阵，都是 info = [0 0 0 0] 加非法访问。
  - 补丁版：单个调用 info = 1；batched 版溢出在第 k 个矩阵时 info_array = e_k（k = 0…3），没有 CUDA 错误。
- **观察（不登记）lag2c/lat2c/lag2s/lat2s**：
  - 3 种 queue，各在 queue 头部延迟 0 和 100 ms，24/24 都返回 info = 1。
  - 按 CUDA 规范，legacy 默认流上的 `cudaMemcpyFromSymbol` 不等待非阻塞流，这里本该读到 0。本平台（WDDM）上这个同步拷贝实际等待了整个设备。Linux 上是否会出错，没有测。

## Lean

`Magma/Dense/Differential.lean` §36k：
- `legacyReturn busy t`：调用返回主机的时刻，即 t 与各阻塞流排空时刻中的最大值。
- `trtri_diag_waits_unrelated`：返回时刻不早于任何一条无关流的排空时刻。
- `trtri_diag_not_async`：只要有一条流在 t 之后才空闲，返回时刻就不等于 t，即不满足文档所说的异步。

## 文件

`r51.csv`、`r51.err`；探针 `probes/round51_stream.cpp`、`probes/[dz]gecscsyncfreetrsm_fixed.cu`、`probes/slag2h_fixed.cu`；脚本 `scripts/roxy/build_r51.bat`、`run_r51.ps1`。
