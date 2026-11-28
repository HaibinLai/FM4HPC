# 第 40 轮：按第 34–37 轮的缺陷写法扫描全库——分诊

方法：把已发现缺陷的写法当作检索模式，扫描 magmablas 中所有 BLAS 驱动的快速返回条件，再与参考 BLAS 的语义逐条比对。参考 BLAS 的规定是：
- gemm/herk/syr2k 只在 `(alpha == 0 || k == 0) && beta == 1` 时才提前返回；
- trsm/trmm 在 alpha = 0 时把 B 置零。

可疑的写法在 GPU 上逐一检验（`probes/round40_nb.cpp`，`nb.csv`）。

| 扫描结果 | 判定 |
|---|---|
| zgemm.cu:249、zgemm_batched_core.cu:202、zherk_batched_core.cu:272/433 | 已知 M233、M245 |
| **zherk.cpp:97/145**（非批量的 magmablas_zherk/zsyrk，`if (n <= 0 \|\| k <= 0) return`） | GPU 证实：magmablas_dsyrk 在 k = 0 时 16/16 例错，C 没有乘 beta。归入 M245（已补充） |
| **magmablas_zherk/zsyrk（非批量）的另一侧三角** | GPU 证实：每个 n 下，另一侧三角都被**整块**改写（n = 200 时 19,900/19,900 个元素）。原因是 zherk_internal 按宽度 nb = 512 的块列调用 gemm，覆盖了完整的对角块，所以 n ≤ 512 时就是整个另一侧三角。所引用的三角本身数值正确。归入 M246（已补充） |
| **非批量的 magmablas_dtrmm，alpha = 0** | GPU 证实：B 为 NaN 时 6/6 例结果仍是 NaN（与 trmm_template_device.cuh 的 `rb *= alpha` 相同）。归入 M247（已补充） |
| zgemm_batched_smallsq.cu:130（k ≤ 0 时返回） | 只在 m = n = k 时使用，所以 k = 0 意味着 m = n = 0，不受影响 |
| zgemm_vbatched_core（`max_k < 0`）、zher2k/zsyr2k(_batched/_vbatched)、zgemv_fermi、zhemv*（快速返回条件） | 与参考 BLAS 一致 |
| ztrmm/ztrsm 及其批量版的 `m <= 0 \|\| n <= 0` | 与参考 BLAS 一致 |

库内部没有调用非批量的 magmablas_zherk、zsyrk、ztrmm（已用 grep 确认），所以受影响的是直接调用公开接口的用户。FINDINGS 没有新增条目，仍为 249 条；M245、M246、M247 的范围和证据已扩充。
