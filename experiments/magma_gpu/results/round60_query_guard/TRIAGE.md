# 第 60 轮：工作区查询接口按查询值分配，并加保护区（2026-11-28，roxy，库版本；没有新缺陷）

## 方法

`probes/round60_query_guard.cpp`。按 MAGMA 自己的包装函数（例如 `magma_zgetrf_gpu_expert`）的写法调用：
- 先以 lwork = −1 查询；
- 再按查询得到的字节数分配主机端（锁页）和设备端工作区，各自后面紧跟 256 KB 的 0xA5 哨兵。

调用后统计两段哨兵中被改写的字节数，并核对结果。

| 接口 | 用例 | 结果 |
|---|---|---|
| zgeqrf_expert_gpu_work（Hybrid；dT 取 geqrf_gpu 文档大小，也加保护区） | 4 | 保护区未被写 |
| zgetrf_expert_gpu_work（Hybrid / Native，recnb 32） | 8 | 全部正确，P·A = L·U 残差约 1e-17 |
| zpotrf_expert_gpu_work（U/L × Hybrid/Native） | 12 | 全部正确 |
| zgetri_expert_gpu_work（Hybrid/Native） | 6 | 全部正确 |
| zgetrs_expert_gpu_work（N/T/C × nrhs 1/7，Hybrid） | 12 | 全部正确 |
| zpotrs_expert_gpu_work（U/L × nrhs 1/7） | 8 | 全部正确 |
| ztrtri_expert_gpu_work（U/L） | 6 | 全部正确 |
| zgegqr_expert_gpu_work（ikind 1–4） | 12 | 全部正确 |
| zgeqrf_batched_work（3 个矩阵） | 3 | 保护区未被写 |
| zgbtrf_native_work、zgbsv_native_work（kl/ku ∈ {1/1, 3/7, 10/2}，n 100/1000） | 12 | 全部正确，设备工作区 8 字节 |

各用例的保护区都没有被写。

## 不登记的现象

- **Native 模式**：geqrf_expert（4 例）返回 −8，getrs_expert（12 例）返回 −10。两者的文档都写明只支持 Hybrid（"Only mode = MagmaHybrid is currently supported"、"currently not implemented"），所以不算缺陷。geqrf 的提示 `printf` 没有换行，于是本程序的结果行被接在同一行，造成缺行。这是测试程序的问题。
- **gb*_native_work 的查询**：查询时会解引用 `info`（与 M044"入口读取调用者的 *info"是同一处），所以查询也必须传有效的 info 指针。`magma_zgbsv_native` 本身就是这样传的。第一版探针传了 NULL，导致访问违例，已修正（`r60gb.csv`）。
- **测试程序问题**：run_list.ps1 原先只认 `^r5` 开头的行，导致 r60 的结果行全部被判为缺行。已改为 `^r[0-9]`。
