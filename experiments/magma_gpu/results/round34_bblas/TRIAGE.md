# 第 34 轮：批量与 vbatched BLAS 的差分测试——分诊

程序：`probes/round34_bblas.cpp`（d 精度；z/c/s 由相同的模板生成），roxy sm_120。参照：主机上的朴素循环，遵循参考 BLAS 的契约：
- beta = 0 时不读 C/y；alpha = 0 时不读 A、B；k = 0 时 C := beta·C；
- syrk/syr2k 只引用并修改 uplo 所指的三角，另一侧三角必须逐位不变；
- trsm/trmm 在 alpha = 0 时把 B 置零；A 只读 uplo 所指的三角（另一侧填入 NaN 作为检测）；
- gemv 在 m = 0 或 n = 0 时快速返回，trans = T 时也一样（y 不乘 beta）。

尺寸包括 0、1、奇数和 > 64；ld = 行数 + {0..3}；batched 每批 3 个矩阵（所有矩阵 ld 相同），vbatched 每批 5 个混合尺寸（含 0）。

## 测试程序问题（已修正，不计入结果）

1. 第一版给 batched 调用中的每个矩阵设了不同的 ld，但 batched 接口只接受一个 ldda。改为统一的 ld。
2. gemv 在 m = 0 或 n = 0 时，参考 BLAS 不缩放 y；第一版的参照按缩放处理。已修正。

## 结果

| 例程 | 结论 | 归属 |
|---|---|---|
| dgemm_batched | k = 0（m = 65，n = 7）的 24 个用例：C 没有乘 beta（beta = 0 时 NaN 留在 C 里，beta = 0.5 时 C 不变）。其余 228 例正确 | **新 M245**（gemm_batched_core.cu:202 的 `k <= 0` 提前返回）|
| dgemm_vbatched | 全部正确，包括 k = 0 的矩阵 | — |
| dgemv_batched / vbatched | beta = 0 时 y 中的 NaN 被保留（n = 31 和 65 时；n = 1 时正确） | 已知 M234（同一个 gemv_template_device.cuh），扩充了证据 |
| dsyrk_batched、dsyr2k_batched | k = 0 时同 M245（herk_batched_core.cu:272；her2k 先过了自己正确的快速返回检查，再调用 herk_batched_core）| **新 M245** |
| dsyrk/dsyr2k batched，dsyrk_vbatched | 另一侧三角被改写：n = 31 时 3 个矩阵的 1,395 个严格上（下）三角元素全部被改写 | **新 M246** |
| dtrsm、dtrmm 的 batched 和 vbatched | alpha = 0 且 B 含 NaN 时结果为 NaN（160 + 32 例），应为 0；其余正确，A 另一侧三角中的 NaN 从未被读到 | **新 M247** |
| magmablas_dtrsm（非批量）、magma_dtrmm（cuBLAS） | alpha = 0 时正确置零 | — |

## M246 的后果：potrf_batched 和 potrf_vbatched 改写“不引用”的上三角（`contract.csv`、`potrf_sizes.csv`）

- 测法：下三角放 SPD 矩阵，严格上三角填哨兵 7.0，分解后检查哨兵。
- potrf_batched：n ≤ 100（融合内核）时不改写。n ≥ 257（递归面板调用 `magmablas_zherk_batched_core`）时被改写：n = 257 改了 19,632/98,688 个，n = 513 改了 43,824/393,984 个，n = 1024 改了 92,208/1,571,328 个。分解结果本身正确，info = 0。
- potrf_vbatched：在 n = 500、511、520、600 时改写（13,098 个、14,979 个、84 个、9,180 个）。n = 512、513、1024 时不改写，因为按 `magma_zrecommend_cublas_gemm_stream` 的选择，这些尺寸改用逐矩阵的 cuBLAS herk。
- 两个例程的文档都写着 “the strictly upper triangular part of A is not referenced”。

## 顺带复现的已知问题

- potrf_vbatched 不清零 info_array（M078）：info 预填 5 时，8 个尺寸全部不做分解，并原样返回 info = 5；预填 0 时全部正确。第一版测试程序用的是未初始化的 dinfo，出现过 info = 0x401C0000（正好是 7.0 的高 32 位）这样的垃圾值，就是这个原因。
