# 第 31 轮：混合精度迭代精化（IR）求解器的差分扫描——分诊

平台：roxy（RTX 5070 Ti，sm_120，Windows 11，CUDA 13.5，MAGMA 2.10.0 静态库）。程序：`probes/round31_mixed.cpp`（扫描），`probes/round31_race.cpp`（追查），`probes/dsgesv_patched.cpp`、`probes/[sd]getf2_native_kernel_ldcg.cu`（验证补丁），构建脚本 `scripts/roxy/build_fix31.bat`、`build_fix31b.bat`。

## 用例

- 矩阵 A = U·diag(σ)·Vᵀ，σ 从 1 几何递减到 1/cond，cond ∈ {1e1, 1e4, 1e8, 1e12, 1e15}；cond ≤ 1e4 时 A 再乘以 {1, 2⁻³⁰, 2²⁰, 2¹⁰⁰}（越出 FP16 / FP32 的范围）；另加一个 b 中含 NaN 的用例。
- n ∈ {64, 200, 513}，nrhs ∈ {1, 2}。
- 判定（`berr_ratio = max_j ‖b−Ax‖∞ / ((‖A‖∞‖x‖∞+‖b‖∞)·n·eps)`）：
  - info = 0 且 berr_ratio ≤ 100 → OK；
  - info = 0 但 berr_ratio > 100，或 x 非有限 → WRONG；
  - b 含 NaN 而 info = 0 且 iter ≥ 0 → NAN-ACCEPTED。
- 分组：ge（dsgesv_gpu、dsgesv_iteref_gpu、dhgesv_iteref_gpu、dxgesv_gmres_gpu 的 6 种分解 × 3 种精化）、po（dsposv_gpu、dshposv_gpu、dshposv_native）、z（zcgesv_gpu、zcposv_gpu）、ls（dsgeqrsv_gpu）。
- 共 1,944 次求解：OK 1,369，WRONG 413，NAN-ACCEPTED 126，NAN-FALLBACK-OK 36。

注：dxgesv_gmres_gpu 的名字里带逗号（如 `dxgesv_gmres_gpu(SS,IRSTRS)`），所以 ge.csv 中这些行有 13 个字段，解析时要把前两个字段合并。

## 结论汇总

| 求解器 | WRONG | 归属 |
|---|---|---|
| dsgesv_iteref_gpu / dhgesv_iteref_gpu | 42 / 43 | 已知：nrhs = 2 第二列返回 0（M006）；scale 2¹⁰⁰、2²⁰ 时 eps1 = tol·ro0 随 s² 缩放，一步就停（M002）；dhgesv 在 2²⁰ 时 FP16 溢出得到 NaN 并跑满迭代（M007 + M004） |
| dshposv_gpu / dshposv_native | 33 / 33 | 已知：nrhs ≥ 2 时越界写、没有回退（M024、M020）；2¹⁰⁰ 时一步就停（M022） |
| dxgesv_gmres_gpu（18 种组合） | 255 | iter ≥ 0 的部分是已知问题：FP16 溢出接受 NaN（M007）、nrhs 被忽略（M006）、cond ≥ 1e8 时报成功但不复核残差（M001）。**iter < 0 的部分是新缺陷 M243**（FP64 回退中的流间竞争，见下） |
| dsgesv_gpu | 5 | **新缺陷 M242 + M243**（只出现在 n = 513） |
| zcgesv_gpu | 2 | **新缺陷 M243**（n = 200，cond 1e8 / 1e12 正常回退后，结果却是错的） |
| dsposv、zcposv、dsgeqrsv | 0 | 各回退 18 次，结果全部正确 |
| NAN-ACCEPTED（每个求解器 6 例） | — | 已知：拒绝式判据 `Rnrm > Xnrm·cte` 放过 NaN（M008）；GMRES 类遇到 NaN 跑满迭代（M004 / M023）；dsgeqrsv 中 NaN 通过（M017） |

## 新缺陷 M242：`?getf2_native_kernel` 跨线程块读取时命中过时的 L1 数据

- **现象**：dsgesv_gpu 在 n = 513 时，22 个 cond 用例中有 18 个退回了 FP64（其中 iter = −3 有 4 例，即单精度 LU 报告奇异；其余 14 例 iter = −31）。n = 64 和 n = 200 时只有 4 例回退，都是 cond ≥ 1e8 的。cond = 10 的良态矩阵本不该回退。
- **定位**：回退次数完全确定（重复运行，结果都是 14 + 4），与是否加同步无关，所以问题出在单精度分解本身。直接测分解的后向误差 `‖PA−LU‖₁/(‖A‖₁·n·eps)`（`ld_sweep.txt`）：
  - `sgetrf_native` 在 ldda = 100、101、201、301、302、385、500、502、511、513、514、516（n = 513 时）、1025 上为 300–14,000，ldda = 2001 上为 8.4，info = 0；
  - `dgetrf_native` 在 ldda = 101、201、301、385、511、513、514 上为 1e10–1e12，结果完全是垃圾，info = 0；
  - 同样的 n，换成 32 字节对齐的 ldda 后，误差都是 0.06 左右；
  - `sgetrf_gpu`（混合版）在所有 ldda 上都正确。
  
  **问题取决于 ldda，与 n 无关**（例如 n = 512 时 ldda = 513 错、ldda = 520 对）。
- **根因**（`magmablas/zgetf2_native_kernel.cu`，s/d/c 版由它生成）：
  - 融合面板内核让每个线程块负责一列。块 i 写出缩放后的第 i 列，执行 `__threadfence()`，再置 `update_flag[i] = 1`。块 bx > i 自旋等待这个 volatile 标志，然后**用普通指针 `da` 读第 i 列**（第 249–262 行，共 4 处）。
  - 在 sm_70 及以后的架构上，普通的全局内存加载会进 L1，而各 SM 的 L1 之间不保持一致。读取方先前读第 i−1 列（或者自己那一列）时，已经把与第 i 列列首共享的 32 字节扇区装进了 L1。当 `ldda·sizeof(T)` 不是 32 的倍数时，列首不落在扇区边界上，于是后来读到的是更新前的值。
  - MAGMA 自带的测试程序用 `ldda = magma_roundup(n, 32)`，所以从来没测到这种情况。
- **验证**：只把这 4 处读改成 `__ldcg(&da[...])`（缓存在全局层，绕过 L1），重新编译 s/d 两个内核，链接到库之前（`build_fix31b.bat`）。原来错的 16 个配置全部恢复正常（误差约 0.06），原来对的 36 个不受影响（`ld_sweep_ldcg.txt`）。把同样的内核链接进 round31 扫描后，dsgesv n = 513 只剩 cond ≥ 1e8 的 6 例回退，16 个 cond ≤ 1e4 的用例全部收敛。
- **影响范围**：
  - 公开接口 `magma_?getrf_native` 和 `magma_?getrf_gpu_expert(mode = MagmaNative)` 都受影响，只要 ldda 未对齐（float 需 ldda % 8 ≠ 0，double 需 ldda % 4 ≠ 0，这是必要条件，是否真的出错还取决于 m 和偏移）。
  - **dsgesv_gpu 和 zcgesv_gpu 内部固定用 `lddsa = n`**，所以 float 在 n % 8 ≠ 0、complex-float 在 n % 4 ≠ 0 时，单精度分解总会出错。结果是 IR 失败、性能下降，并且总要进入下面那个有竞争的回退路径。
  - 这只在 sm_120（Windows）上做过验证。不过机制（普通加载经过 L1、L1 跨 SM 不一致）在 Volta 及以后的架构上是通用的。
- **Lean**：`getf2_native_aligned_no_share_all`、`getf2_native_unaligned_shares`、`getf2_native_stale_read`、`getf2_native_ldcg_fix`、`dsgesv_lddsa_n_unaligned`。

## 新缺陷 M243：FP64 回退中，dX 在另一个流上被读取，两者没有同步

- **代码**（zcgesv_gpu.cpp 第 357–360 行；dsgesv 由它生成；zcposv_gpu 和 dxgesv_gmres_gpu 第 806–809 行同样如此）：
  ```
  magma_zgetrf_gpu( n, n, dA, ldda, ipiv, info );
  magmablas_zlacpy( MagmaFull, n, nrhs, dB, lddb, dX, lddx, queue );   // 在调用者的 queue 上异步执行
  magma_zgetrs_gpu( trans, n, nrhs, dA, ldda, ipiv, dX, lddx, info );  // 新建一个 queue（cudaStreamCreate），没有 event，也没有同步
  ```
  两个流之间没有任何先后约束，zgetrs 可能在复制完成之前就开始对 dX 做行交换和三角求解，于是从 dX 里残留的 IR 结果出发求解，info 却是 0。
- **证据**：
  - 只跑 dsgesv、n = 513 的扫描：正常运行 4/4 次都有 WRONG（每次 2–8 例）；设置 `CUDA_LAUNCH_BLOCKING=1`（所有启动串行）后 0/4 次有 WRONG。
  - 带开关的 dsgesv 副本（`dsgesv_patched.cpp`，环境变量 `R31_SYNC`）：只在复制后加 `magma_queue_sync(queue)`（bit 2）时，3 次运行都是 WRONG = 0。只在 IR 循环中加同步（bit 1）时，WRONG 仍有 2–8 例。回退次数在各种设置下都不变。
  - dxgesv_gmres n = 200（`dxgesv_n200_normal.csv` 与 `dxgesv_n200_blocking.csv`）：iter < 0（已经回退）却 WRONG 的，正常运行 21 例，串行后 0 例；iter ≥ 0 的 WRONG 在两种运行下都是 22 例（已知问题，不变）。
  - 单独、反复调用 dsgesv（`race_isolated.csv`，先 sync 再调用，良态矩阵不会回退）：120 次全部正确。所以这个竞争只在回退路径上出现。
- **同类代码，但本轮没有观察到出错**：zcposv/dsposv 的回退（zlacpy 之后是 zpotrs_gpu，同样新建 queue）共回退 36 次，全部正确。
- **Lean**：`sgesv_fallback_race`、`sgesv_fallback_synced`。

## 已撤回的 M244：反方向（被调用者 → 调用者）的交接

读码时曾怀疑 IR 循环中的 zcgetrs 和 zcgeqrsv 回退中的 zgeqrs 也有竞争：它们在自己的 queue 上写数据，然后不同步就 destroy。
第 32 轮实测（`round31_race qd`）表明，`magma_queue_destroy` 会通过 `cublasDestroy` 同步整个设备（在 Q 上放一个长 dgemm，destroy 另一个 queue 之后 Q 已经空闲，3/3 次都是这样）；而 `magma_queue_create` 不会同步。
所以“被调用者 → 调用者”方向的交接是有序的，不构成缺陷；只有“调用者 → 被调用者”方向（M243）会竞争。

## 误报 / 测试程序问题

- 无。上表所有 WRONG 都已归入已知条目或新条目。
