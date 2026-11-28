# 第 78 轮：可变尺寸批量 BLAS（vbatched）（2026-11-28，roxy；没有新缺陷，扩充 M246、M248）

## 为什么测这一组

按文件名扫描，magmablas 里 zgemm_vbatched、zgemv_vbatched、zhemm_vbatched、zhemv_vbatched、zherk / zsyrk / zher2k / zsyr2k_vbatched、ztrmm_vbatched、ztrsm_vbatched 都没有在记录里出现过。不过 M245、M246、M248 是按例程名登记的，其中已经包含了部分 vbatched 结论。

## 方法

`probes/round78_vbatched_blas.cpp`：
- 每批 12 个矩阵，尺寸随机取自 [0, maxsz]，保证 0 和 maxsz 都出现，ld = 行数 + 3；
- 复数数据，α = 0.7 − 0.3i，β = −0.4 + 0.2i；
- ld 余量和不该写的另一侧三角填哨兵值，不该引用的另一侧三角（hemm / hemv / trmm / trsm 的 A）也填哨兵值；
- 参考值在 CPU 上按 BLAS 定义计算，trsm 用残差 op(T)X − αB 检验。

用例表 `scripts/roxy/r78_cases.txt`，共 174 例。

## 结果（r78.csv）

| 例程 | 选项 | 用例 | 结果 |
|---|---|---|---|
| gemm_vbatched | tA, tB ∈ {N, T, C}，maxsz 32 / 100 | 18 | 全部正确（含 k = 0，与 M245 记录的"vbatched 正确"一致） |
| gemv_vbatched | N/T/C × incx 1/2，maxsz 32 / 300 | 12 | 全部正确 |
| hemv_vbatched | U/L × inc 1/2 | 8 | 全部正确，另一侧三角没有被读 |
| trmm_vbatched、trsm_vbatched | L/R × U/L × N/T/C × 单位 / 非单位对角，maxsz 16 / 64 | 96 | 全部正确 |
| herk / syrk / her2k / syr2k_vbatched | U/L × N/C（或 N/T） | 32 | 计算的那一侧三角正确（≤ 7e-16），另一侧三角每次都被改写（每批 3035–19403 个元素），这就是 M246；ld 余量没有被写 |
| hemm_vbatched | L/R × U/L | 8 | 每例 12 个矩阵中有 10 个出错，相对误差 2e3–7e3（读到了哨兵值）；剩下 2 个是有一维为 0 的空矩阵。这就是 M248 |

**补丁验证（M248）**：用第 35 轮修过 fetch 的 `probes/hemmfix/hemm_template_device.cuh` 重新编译 zhemm_vbatched_core.cu（`scripts/roxy/build_fix78.bat`）。r78h.csv：8/8 正确（≤ 2.6e-16），这也说明参考值是对的。

## 说明

- 第一版参考值在 gemv 的 m = 0 或 n = 0 时仍然计算 y := βy。参考 BLAS 在这种情况下直接返回、不改动 y，MAGMA 的行为与它一致，所以已改正参考值。
- 第一版判定把 her2k / syr2k 错标为 WRONG（判定条件只匹配了 "rk"），已重跑这 16 例（r78k.csv）并合并进 r78.csv。

## 文件

- r78.csv、r78.err：174 例。
- r78k.csv：her2k / syr2k 的 16 例重跑。
- r78h.csv：hemm 的补丁版 8 例。
