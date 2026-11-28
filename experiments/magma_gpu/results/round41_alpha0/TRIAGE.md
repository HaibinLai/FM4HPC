# 第 41 轮：alpha = 0 且 A/B 含 NaN——分诊（记为观察，不登记）

程序：`probes/round41_alpha0.cpp`。参考 BLAS 的实现在 alpha = 0 时不读 A 和 B（gemm/symm/syrk/syr2k 直接执行 C := βC，gemv/symv 直接执行 y := βy）。以前各轮的 alpha = 0 用例都用的是有限值的 A，所以这一点从未检验过。

| 例程 | 结果（n ∈ {1, 31, 64, 200}） |
|---|---|
| magmablas_dgemm、dgemm_batched、dgemm_vbatched | 每个输出元素都是 NaN |
| magmablas_dgemv、dgemv_batched、dsymv | 每个输出元素都是 NaN |
| magmablas_dsyrk_batched、dsyr2k_batched | 下三角的每个元素都是 NaN |
| magmablas_dsyrk（非批量，内部调用 cuBLAS 的 magma_dgemm） | 正确 |

判定：MAGMA 这些例程的文档只承诺 “beta = 0 时 C/y need not be set”（这一条不成立的情况已登记为 M234）。“alpha = 0 时不引用 A、B need not be set”这句话，只在 trmm/trsm 的文档里出现（违反它的情况已登记为 M247）。参考 BLAS 的 gemm/gemv 文档也没有把 alpha = 0 时不读 A 写成契约，只是实现上这样做；cuBLAS 的实际行为也是不读 A（非批量 dsyrk 经 cuBLAS，结果正确）。

所以这一条是“与参考实现和 cuBLAS 的行为不同、但不违反成文契约”的差异：记为观察，不登记为缺陷。如果调用者依赖“alpha = 0 时 A 可以是未初始化内存”，换用 magmablas 版本后就会得到 NaN。
