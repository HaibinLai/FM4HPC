# 第 69 轮：范数与 geqr2 的工作区，以及 lanhe 的文档约定（2026-11-28，roxy；没有新缺陷）

起因：M274 那一类问题是"MAGMA 测试程序分配的工作区比文档写的大"。逐个检查测试程序中的设备工作区，凡是第 59 轮没覆盖到的，这一轮补测。程序为 `probes/round69_norms.cpp`，用例表为 `scripts/roxy/r69_cases.txt`、`r69d_cases.txt`。dwork 按文档大小分配，其后放哨兵。

| 例程 | 参数 | 用例 | 结果 |
|---|---|---|---|
| magmablas_zlange（M/O/I/F） | 1×1、33×17、17×33、300×200、64×1000 | 20 | 全部正确，保护区未被写 |
| magmablas_zlanhe（M/O/I × U/L，另一侧三角放 1e6 作哨兵） | n 1–1000 | 48 | 全部正确，没有读另一侧，保护区未被写 |
| zlanhe，F 范数 | 同上 | 16 | 返回 −1。文档写明 "not yet supported"，所以不算缺陷 |
| zlanhe，对角元虚部为 1e3（文档："need not be set and are assumed to be zero"） | 同上 | 48 | 全部正确，虚部被忽略 |
| magma_zgeqr2_gpu（dwork 为文档的 N 个 double） | 5 种形状 | 5 | 全部正确，保护区未被写 |

结论：测试程序中其余的设备工作区都与文档一致。目前只有 M261 和 M274 两处是"文档写小了"。
