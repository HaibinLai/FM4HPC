# 第 71 轮：此前未覆盖的面向用户的 batched 例程（2026-11-28，roxy；没有新缺陷）

程序为 `probes/round71_batched_misc.cpp`，每批 3 个问题，strided 版的步长比数据多出 7（或 3），这样步长用错就会暴露。复数数据，与 CPU 参考或 LAPACK 比较。用例表为 `scripts/roxy/r71_cases.txt`。

| 例程 | 参数 | 用例 | 结果 |
|---|---|---|---|
| magmablas_zgemm_batched_strided | 3 种形状 × tA, tB ∈ {N,T,C}² | 27 | 全部正确 |
| magmablas_zgemv_batched_strided | 3 种形状 × {N,T,C} | 9 | 全部正确 |
| magmablas_zgeadd_batched、ztranspose_conj_batched | 4 种形状，ld 填充 | 8 | 全部正确 |
| magmablas_zlacpy_batched、zlaset_batched | 4 种形状 × U/L/F（其余部分不应改动） | 24 | 全部正确 |
| magmablas_zhemv_batched | n 1–300 × U/L，另一侧三角放哨兵 | 10 | 全部正确 |
| magma_zgbsv_batched、zgbsv_batched_strided | n 10/100/500 × (kl, ku) ∈ {1/1, 2/3, 5/1, 0/2}，非对角占优，需要选主元 | 24 | 与 LAPACK zgbsv 一致 |

共 102 例，全部 exit 0。
