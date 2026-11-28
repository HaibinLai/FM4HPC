# 第 72 轮：batched Jacobi 的 strided 接口（2026-11-28，roxy；没有新缺陷）

`probes/round72_jacobi_strided.cpp`：同一批矩阵分别交给指针数组接口和 strided 接口（步长比数据多 5），两者的输出逐位比较，并与 LAPACK 比较。
- magma_zgesvj_batched 与 magma_zgesvj_expert_batched_strided（工作区按查询值分配）：7 种形状，每批 4 个。
- magma_zheevj_batched_small_sm 与 _strided_small_sm：n ∈ {2, 7, 16, 31, 32} × U/L，每批 4 个。

结果：
- **两种接口的输出逐位相同**（diff = 0），所以 strided 接口本身没有问题。
- 两者与 LAPACK 都不一致：gesvj 约 5e-7（8×8 时 0.56），heevj 在 n ≥ 7 时为 0.6–0.86，n = 2 时正确。这正是已知的 M240：Windows（LLP64）上 swap_scalar.cuh 把 64 位值当作 32 位的 long 来交换，排序后的值被扰动了约 2⁻²⁰，并且与向量错配。已把本轮结果补进 M240，说明 strided 接口同样受影响。
