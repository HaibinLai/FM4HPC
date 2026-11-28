# MAGMA 2.10.0：建模与 GPU 实测覆盖表（2026-11-25）

列说明：
- **Lean**：`lean/OptimiKahan/` 中对应的定理（MAGMA 部分在 `Magma/` 子目录，模块划分见 `lean/OptimiKahan/README.md`）；
- **GPU 0**：`experiments/magma_gpu/results/` 中的结果文件（表中只写文件名）；
- 结论标“✔”为与预测一致，“✖”为与预测不符（如实记录）。

所有 Lean 文件只用 core Lean，无 sorry；`cd lean && lake build` 一次检查全部模块。多数缺陷同时有“GPU 上观察到的具体实例”和“对所有触发输入成立的一般形式”（各 `Magma/` 模块末尾的 General forms 一节，名字以 `_all` 结尾的定理）。GPU 实测都只用 eightsocket 的 GPU 0（V100），节点数据每轮用完即删。

## 1. 稠密混合精度 / 精化例程（全部例程都已覆盖）

| 例程 | 发现 | Lean | GPU 0 |
|---|---|---|---|
| dsgesv/dhgesv_iteref（GMSTRS） | 无复核无回退、s² 容差、1e-18 绝对阈值、NaN 跑满报成功、忽略 trans/nrhs | `MagmaFgmres.*` | ✔ `fgmres24_original.txt`（24 次中 20 次违约）、`fixprobe_original.txt`（trans、nrhs） |
| 同上，修复补丁 v2 | 带证明的后检 | `MagmaFixed.fixed_post` 等 | ✔ `fgmres24_fixed.txt`（0 次违约）、`cost_results.txt` |
| 同上，补丁 v3（GPU 上做 Dot2 认证） | 同一认证、开销 | 同上（`Faithful` 假设涵盖同一运算序列） | ✔ `perf_v3_results.txt`（OpenBLAS 构建：0/24 违约）；`perf_large_results.txt`：n = 8192 为 0.23–0.30 s，原版 0.21–0.26 s，v2 3.3 s，FP64 0.29–0.43 s |
| zcgesv（dsgesv_gpu） | 解溢出被接受；B 不查范围 | `MagmaDense.x_overflow_accepted`、`zcgesv_b_overflow_accepted` | ✔ `dense_results.txt` |
| zcposv | 解溢出被接受；B 溢出正确回退 | `x_overflow_accepted`、`zcposv_b_overflow_fallback` | ✔ `dense_results.txt` |
| zchesv | X = B 空壳回退；发散被接受 | `zchesv_stub`、`diverge_accepted` | ✔ `dense_results.txt` |
| zcgeqrsv | 不相容最小二乘永不通过判据（性能） | — | 未测（只是性能问题） |
| dshposv_gpu/_native + dfgmres_spd | 返回 0；无回退；2^-53 绝对阈值；NaN 跑满；nrhs ≥ 2 越界写 | `dshposv_always_success`、`spd_tiny_rhs_returns_zero`、`spd_nan_runs_to_maxiter`、`nonconverged_oob` | ✔ `dense_results.txt`、`guard_results.txt`（越界一整列） |
| zgesv_rbt + zgerfs_nopiv | iter 被丢、info 被覆盖、无回退；n ≤ 3 向量变换缺缩放 | `rbt_info_lost`、`gerfs_nopiv_silent` | ✔ `dense_results.txt`（n = 1–3 错；cond 3.09 矩阵误差 1.6e13） |
| zgesv_rbt_batched | info 只在 CHECK_INFO 下检查；内存泄漏 | `rbt_batched_default_build_silent` | ✔ `round5_results.txt`（奇异矩阵返回 0、误差 0.85；200 次调用泄漏显存 25 MB、主机 53 MB） |
| zposv_batched / zgesv_batched / zgesv_nopiv_batched 的 CHECK_INFO | 唯一能把逐矩阵失败作为返回值的编译选项编译不了：对 7 参数宏 `magma_getvector` 只传 6 个参数（漏 queue）；zgesv_nopiv_batched 还引用未声明的 dinfo_array。默认编译下逐矩阵 info 留在设备数组中，后续 getrs 不覆盖 | `check_info_build_broken` | ✔ CPU 预处理（`experiments/magma_dense/check_info_pp_results.txt`：恰好这 3 处报 “too few arguments”，rbt_batched 那处通过） |
| magma_zgbtrf_batched（指针数组接口）通用路径 | `magma_zgbtrf_set_fillin` 把整个 batchCount 放进 gridDim.z（上限 65535），不分块；同文件其他内核与 strided 接口都按 65534 分块。batchCount ≥ 65536 且带宽超出共享内存内核时启动失败且不检查，ju_array 未写入 | `set_fillin_launch_invalid_all`、`gbtrf_generic_path_reached` | ✔ roxy 第 26 轮：65536 个指针（别名到 1024 个矩阵，因空闲显存只有约 4 GB）时启动报 “invalid argument”，函数返回 0；65535 时无错。数值结果未比对（别名） |
| magmablas_zlacpy（及很可能更多内核） | 内核参数为 `int ldda`，`ind + iby·ldda` 以 int 计算；宿主端按约 419 万列分块，不限制这个乘积。元素数超过 2³¹ 的矩阵偏移量回绕为负，ILP64 编译同样如此；没有文档说明这一限制 | `lacpy_offset_wraps`、`lacpy_offset_negative_all` | 未测：读码，只逐行核对了 lacpy（z 内核共 203 处 `int lda/ldda`）；可在 GPU 0 上测 slacpy 50000²（两份共 20 GB） |
| magmablas z 内核的 32 位列偏移（全部 123 个文件） | 89 个在内核里以 int 计算 `j·ld`：48 个非批量、34 个只在批量下、7 个用 magma_int_t（只在 LP64 下回绕）；4 个用 64 位，6 个有界，24 个不适用。默认 hybrid getrf_gpu 只走 64 位安全的内核；getrf_native 经 laswp_columnserial 回绕 | `offset_audit_covers_all`、`getrf_native_laswp_wraps` | 读码（子代理全量核对，主会话抽查 lacpy、laset、gemm 模板、columnserial） |
| zpotrf_gpu 宿主端列指针 | `dA(i,j) = dA + i + j*ldda` 以 32 位 magma_int_t 计算（getrf_gpu 用 size_t）：LP64 下 n = 50,000 时 j ≥ 42,950 的指针回绕 | `potrf_gpu_host_offset_wraps` | 读码；需约 10 GB 单精度矩阵，可在 eightsocket 的 GPU 0 上测 |
| magmablas_ztranspose_batched_stride（公开、库内无调用者） | 每块 65,534 个矩阵按 `dA_array + i` 偏移，应为 `+ i*stride`：批量超过 65,534 时第二块起全部错位 | `transpose_stride_chunk_wrong_all` | 读码 |
| magmablas_zswapblk（公开、库内无调用者） | 主元目标存为 `short`：距块起点超过 32,767 行即截断 | `swapblk_short_truncates` | 读码 |
| **zgeqp3（经 magma_zlaqps）** | 秩亏矩阵上 A·P ≠ Q·R（残差 0.2–6），info = 0；秩也没被揭示。两个根因：主机端交换列后，设备副本只恢复 offset+nb 以下的行，面板提前结束时尾部矩阵里留着过时行；重算范数用的是未更新的主机数据，行区间还差一 | `geqp3_swap_restore_short_all`、`geqp3_patch_covers_all`、`geqp3_norm_rows_off_by_one` | ✔ roxy 第 27 轮（`round27_focus_results.txt`）：重复列 n = 63/65/128/255/256 出错；78 个随机秩 n/2 矩阵中 18 个出错；链接打了补丁的 zlaqps 后为 0。✔ 第 28 轮（`round28_sweep/`）：病态矩阵同样出错，n ≥ 63 的秩 1、条件数 1e12、分级矩阵全部出错，Kahan 矩阵部分出错，共 92/238；打补丁后 714/714 正确 |
| magma_zgeqp3_gpu | tau 必须是设备数组（MAGMA 自带测试用 dtau），但 magma_z.h 把它声明为主机指针类型，文档也没说明：按头文件传主机数组时每次调用都非法访问内存，info = 0，CUDA 上下文报废 | `geqp3_gpu_tau_space_mismatch` | ✔ roxy 第 28 轮（`round28_sweep/qp3small.txt`）；改用设备 tau 后全部正确，GPU 面板 zlaqps2_gpu 没有 zlaqps 的缺陷 |
| **swap_scalar.cuh 的使用者**：[zcd]heevj/syevj_batched_small_sm、[zcd]gesvj_batched_small_sm，以及 sort.cu（分块 gesvj 用的排序） | 64 位值按 `long` 做异或交换；Windows（LLP64，`long` 为 32 位）上只交换低 32 位，所以排序后的特征值/奇异值被扰动，并且与特征向量/奇异向量错位，info = 0（Linux 不受影响；s 精度不受影响） | `swap_long_llp64_wrong` | ✔ roxy 第 30 轮（`round30_batched_eig/`）：heevj 784 例中 313 例错、gesvj 1568 例中 1016 例错；expert 接口 sort_flag = 0 时 0/64 错，sort_flag = 1 时几乎全错；`sizeof(long) = 4` |
| zgesvj_batched 分块路径（n ≥ 15） | 用 1/σ 缩放 U 前没有判断 σ 是否为 0：零矩阵或秩亏矩阵上 U 出现 NaN，info = 0 | `gesvj_zero_sigma_nan` | ✔ roxy 第 30 轮：零矩阵在 n ≥ 15 时每个尺寸 7/7 错；n ≤ 8 的融合内核正常 |
| magmablas_zgemm_batched；zherk/dsyrk_batched；zher2k/dsyr2k_batched（经 herk_batched_core）；非批量的 magmablas_zherk/zsyrk（第 40 轮） | k = 0 时直接返回，不执行 C := beta·C（参考 BLAS 只在 beta = 1 时才提前返回）；vbatched 版正确 | `batched_k0_beta_ignored_all` | ✔ roxy 第 34 轮（`round34_bblas/gemm.csv`、`syrk.csv`）：k = 0 的 24 + 16 例全错，beta = 0 时 NaN 留在 C 里 |
| **herk/syrk、her2k/syr2k 的 batched 和 vbatched**，以及由此波及的 potrf_batched（n ≥ 257）和 potrf_vbatched；非批量的 magmablas_zherk/zsyrk（第 40 轮：n ≤ 512 时整个另一侧三角都被改写） | 跨对角线的 tile 按完整的 gemm tile 写回，C 中“不引用”的另一侧三角被改写；potrf_(v)batched 因此改写了 A 的严格上三角 | `herk_batched_diag_tile_breaks_contract_all` | ✔ roxy 第 34 轮：syrk n = 31 时 3 个矩阵的 1,395 个元素全部被改；potrf_batched n = 257 时改了 19,632/98,688 个，n = 1024 时改了 92,208/1,571,328 个；potrf_vbatched 在 n = 500/511/520/600 时改写；分解本身正确，info = 0 |
| magmablas_ztrsm/ztrmm 的 batched 和 vbatched，以及非批量的 magmablas_ztrmm（第 40 轮） | alpha = 0 时计算 alpha·B 而不是 B := 0，B 中的 NaN/Inf 被保留（非批量的 magmablas_dtrsm 正确） | `trsm_batched_alpha0_nan`、`trsm_batched_alpha0_finite` | ✔ roxy 第 34 轮：alpha = 0 的 192 例全部是 NaN；alpha ≠ 0 时全部正确 |
| **magmablas_[zcsd]hemm/symm**（非批量、batched、vbatched） | hemm_template_device.cuh 把内核的基指针 A 传给 `fetch()`（这个宏期望的是每个线程的 `offs_dA`）：所有线程读同一个元素，`offs_dA` 从来没被读取，所以 C 对任何输入都是错的 | `hemm_fetch_base_wrong_all`、`hemm_fetch_threads_collide` | ✔ roxy 第 35 轮（`round35_bblas2/`）：A = I 时 4 种精度都错（误差 1.5–13.7），cuBLAS 正确；局部修正 fetch 后，40/40 个 beta ≠ 0 的用例正确 |
| **magma_[ds]trevc3、[ds]trevc3_mt**（只有实数版；复数版的回代 GEMM 在 over 分支里，第 39 轮 ztrevc3 正确） | 选择分块版本 2 的条件漏掉了 LAPACK 的 `OVER`：在 howmany = A/S 且 lwork 取文档推荐的最优值时，“回代变换”GEMM 用调用者的 VR 去乘，改写了已经算好的特征向量（geev 用回代变换模式，不受影响） | `trevc3_blocked_without_backtransform`、`trevc3_lapack_selection_correct` | ✔ roxy 第 36 轮（`round36_helpers/`）：36 个特征对中有 26 个残差为 1e13–1e15（LAPACK ≤ 0.54）；lwork = 3n 时正确；加 `over &&` 后 36/36 正确；trevc3_mt 表现相同 |
| **magma_[zcds]gesv_rbt**（以及 gesv_rbt_batched） | gerbt 的 Uᵀb 内核对 n < 2 提前返回，大小为 1 的子蝶形因此不乘 u，与矩阵变换、V·y 变换不一致：n ≤ 3 时 x 错误，info = 0 | `rbt_size1_inconsistent` | ✔ roxy 第 37 轮（`round37_2stage/`）：n = 1、2、3 的后向误差为 1e12–1e14；改为 `n < 1` 后 56/56 正确 |
| **magma_zmconvert（设备端回退路径）、magma_zmtranspose/transposeconjugate**；因而 QMR、BICG、LSQR、BOMBARDMENT、PQMR、PBICG 在 SELL-P 格式下 | SELL-P 的 blocksize/alignment 在临时结构中丢失，CSR→SELL-P 时出现 `256 % 0`，整数除零导致进程崩溃 | `sellp_params_lost`、`sellp_params_forwarded` | ✔ roxy 第 45 轮（`round45_sparse_edge/`）：6 个求解器 × 7 个问题在 SELLP 下全部崩溃（0xC0000094）；直接调用 zmconvert(设备 CSR→SELLP) 也崩溃；两处补丁后全部正常 |
| **magma_zmconvert CSR→COO**（主机和设备） | 行下标写进 row/drow，而 zmtransfer(COO) 和 COO→CSR 读的是 rowidx/drowidx：往返后所有元素都进了第 0 行，info = 0 | `coo_rows_lost` | ✔ roxy 第 46 轮（`round46_formats/`）：3×3 例子往返后行指针为 [0 5 5 5]；补丁后 28/28 正确 |
| **ELLRT SpMV（zgeellrtmv；zmergecg 中的同一代码）** | 每行的 warp 内树形归约没有 volatile 和 syncwarp：T = 8 时第 6、7 号线程的贡献丢失，一行超过 6 个非零元就算错 | `ellrt_reduction_loses_lanes` | ✔ roxy 第 46 轮：40 个非零元的行只得到 j mod 8 ∈ {0…5} 的部分和（误差 24.8，确定性复现）；加 `__syncwarp` 后误差为 1.4e−14 |
| **magma_zmconvert CSR↔CSRLIST**（主机） | 空行：`list[-1]`，以及 do-while 对每个空行多复制一个元素，导致堆越界写 | `csrlist_empty_row_overflows` | ✔ roxy 第 46 轮：堆损坏（0xC0000374），emptyrow/emptyrow0/zero 往返出错；补丁后正确 |
| sparse 其余格式（ELLPACKT、ELL、CSC、CSRL/CSRU、CSRCOO、CUCSR、CSR5、BCSR） | — | — | ✔ 第 46 轮：主机往返全部正确；SpMV 不支持的格式如实返回 NOT_SUPPORTED |
| dsyevdx_2stage 链路（sy2sb、hb2st、bulge_back） | — | — | ✔ 第 37 轮：Lower 下没有新问题；Upper、n = 1、range V 的失败分别归入已知的 M065、M236、M216（M216 同样影响 1 阶段的 dsyevdx） |
| 辅助例程：ormqr/lq/ql/rq、ormqr2/ql2_gpu、ormtr(_gpu)、ormbr、orgqr2、orgbr、larfb_gpu | — | — | ✔ 第 36 轮没有出错：405 个用例与 LAPACK 逐元素一致（larfb_gpu 要求显式存储 V 中的 0 和 1，这是文档里写明的约定） |
| dtrsv_batched、dpotrs_batched | — | — | ✔ 第 35 轮没有出错：trsv 112/112 正确；potrs 16/16 正确，A 不被改动 |
| **[sdz]getrf_native（融合面板内核 ?getf2_native_kernel；第 39 轮 c 精度未复现）**；dsgesv_gpu、zcgesv_gpu（内部用 lddsa = n 调用它） | 线程块之间交接数据：标志是 volatile，但数据用普通的、经过 L1 的加载读取。当 `ldda·sizeof(T)` 不是 32 的倍数时，读到 L1 里过时的扇区，LU 错误，info = 0 | `getf2_native_aligned_no_share_all`、`getf2_native_unaligned_shares`、`getf2_native_stale_read`、`getf2_native_ldcg_fix`、`dsgesv_lddsa_n_unaligned` | ✔ roxy 第 31 轮（`round31_mixed/ld_sweep.txt`）：sgetrf_native 误差 300–14,000，dgetrf_native 误差 1e10–1e12；n = 512 时 ldda = 513 错、ldda = 520 对。4 处读改成 `__ldcg` 后，16/16 恢复正常（`ld_sweep_ldcg.txt`）。dsgesv n = 513 的回退从 18/22 降到 6/22（剩下的都是 cond ≥ 1e8）。第 33 轮（`round33_ldda/`）：在 LU、Cholesky、QR 各族的单矩阵和批量版中，只有 getrf_native 受未对齐 ldda 影响（19/38 例错），其余 492 例全部正确 |
| **dsgesv_gpu/zcgesv_gpu、dxgesv_gmres_gpu 的 FP64 回退**（zcposv/dsposv 是同样的写法，但未观察到出错） | 在调用者的 queue 上执行 `lacpy(dB→dX)` 后，`?getrs_gpu` 在新建的 queue 上直接使用 dX，中间没有同步：求解从残留的 dX 出发，info = 0 | `sgesv_fallback_race`、`sgesv_fallback_synced` | ✔ roxy 第 31 轮：dsgesv n = 513 正常运行时每次 2–8 例 WRONG，`CUDA_LAUNCH_BLOCKING=1` 时 0 例；只加一处 `magma_queue_sync`（R31_SYNC bit 2）也是 0 例。dxgesv_gmres n = 200 回退后 WRONG 的，正常运行 21 例，串行后 0 例 |
| zcgesv/zcposv 的 IR 循环；zcgeqrsv 的回退（已撤回） | 原先怀疑：被调用者在自己的 queue 上写 dX/dR/dB，不同步就 destroy。实测 `magma_queue_destroy` 会通过 cublasDestroy 同步整个设备，所以这个方向的交接是有序的，不是缺陷 | — | ✖ 预测不成立（第 32 轮 `round31_race qd`：destroy 另一个 queue 之后，Q 上的长 dgemm 已经完成，3/3） |
| zheevd/zheevdx/zheevx/zheevr_gpu、dsyevd_gpu 的 lascl → hetrd；zgesv_rbt 的 copy → gesv_nopiv（第 32 轮扫描找到的“调用者 → 被调用者”交接） | 被调用者第一步是把 dA 拷到主机的同步 D2H 拷贝（或者只读同一份数据），竞争窗口很小 | — | ✔ 没有出错（第 32 轮 `round32_xqueue/`）：heev 系列 ‖A‖ 缩放 1e±150，27 组 × 5 次全部正确；rbt n ≤ 2000，15/15 正确；与 `CUDA_LAUNCH_BLOCKING=1` 的结果逐位相同 |
| magmablas_zgemm | k = 0 时直接返回，不执行 C := beta·C | `gemm_k0_beta_ignored` | ✔ roxy 第 27 轮：12 个尺寸全错；beta = 0.5 时 C 不变，beta = 0 时 NaN 留在 C 里 |
| magmablas_zgemv、magmablas_zhemv | beta = 0 时仍然计算 beta·y，y 中的 NaN 被保留（与自身文档 “Y need not be set” 矛盾） | `gemv_beta0_reads_y` | ✔ roxy 第 27 轮：n ≥ 1 全部输出 NaN；第 34 轮：gemv_batched/vbatched（同一模板）在 beta = 0 时也保留 NaN；第 35 轮：symv batched/vbatched（hemv_template_device.cuh:53）和 hemm 所有变体（第 252 行）在 beta = 0 时同样保留 NaN |
| magma_zgels、zgels_gpu | 不检查奇异：零列或零矩阵时返回 NaN 解，info = 0（LAPACK 报 info = i，零矩阵给零解）；文档只承诺 info ≤ 0 | `gels_singular_nan_success` | ✔ roxy 第 27 轮 |
| zheevdx、zheevdx_2stage | n = 1 的快速返回不设置 mout | `heevdx_n1_mout_unset` | ✔ roxy 第 27 轮：mout 保持调用者的 −77；zheevr、zheevx 正常 |
| zheevd_gpu、zheevdx_gpu、zheevr_gpu、zheevx_gpu、ztrtri_gpu、zpotrf_mgpu | n = 0 被拒：小矩阵路径以 lda = n 调用 LAPACK | `small_path_lda_n_rejects_n0` | ✔ roxy 第 27 轮：−5/−6/−6/−6/−5/−4；zpotrf_gpu 有提前返回，为 0 |
| zgetrf_native | NaN/Inf 输入报 info > 0，而 zgetrf_gpu（hybrid）和 LAPACK 报 0 | — | ✔ roxy 第 27 轮（严重度低：同一接口两种模式状态不一致） |
| [sdcz]gesv_rbt（HIP 编译） | 参数检查后 `refine = MagmaFalse`（作者 TODO：“AMD 上有失败”），调用者要求的精化被静默丢弃，info 不提示 | `rbt_hip_refine_dropped`、`rbt_hip_refine_irrelevant_all` | ✖ 只有读码证据（无 AMD 卡）；同一个不精化的解在 CUDA 上误差 1.6e13 |
| zhegvdx_2stage(_m) | 自己查询和接受的 lwork（无向量 5n + n·nb）小于内层 zheevdx_2stage 的要求（5n + 2·nb·n）；n > 128 时用自己报告的大小调用必失败，返回内层的参数号 −14，此时 A、B 已被改写；有向量时恰在 n < 2·nb 时不足 | `hegvdx_2stage_query_rejected_all`、`hegvdx_2stage_vec_short_iff`、`sygvdx_2stage_real_short_all`（实数版缺 4n + nb·n） | ✔ roxy 第 26 轮（`round26_results.txt`）：nb = 128（16 线程），n = 129、200 按自身查询值调用返回 −14，A、B 已改写；按 5n + 2·nb·n 调用正常。MAGMA 自带测试用 `magma_zheevdx_getworksize` 的大小，所以测不出 |
| zgesdd 路径 6to-slow | m < n、jobz = O、lwork 介于最小值与快路径之间时，分块 VT 占掉 zungbr(P) 需要的空间；min(m,n) ≥ 1024 时 zungbr 返回 −9 但 ierr 不读，右奇异向量错误而 info = 0 | `gesdd_slow_ungbr_short_all`、`gesdd_6to_slow_instance` | ✔ roxy 第 26 轮：二分得到最小可接受 lwork 恰为 1,181,696；按它调用时 zungbr 报 −9，info = 0，‖VT·VTᴴ − I‖ = 400；按查询值（1,182,688，快路径）为 5.5e-15 |
| zhetrd_he2hb_mgpu | 与单卡版相同，只查 lwork ≥ 1：主机下溢；lwork = nb² − 1 时面板 QR 变成查询；单卡和多卡在 0 < n < nb 时连查询值 n·nb 都会下溢 | `he2hb_mgpu_underrun_all`、`he2hb_mgpu_silent_query`、`he2hb_query_underrun_all` | 未测：读码（单卡版已 ✔）；单卡版 ✔ roxy 第 26 轮：n = 16、nb = 32 按查询值 512 调用，work 之前恰好被写 512 个元素 |
| zgbtrf_native / zgbsv_native(_work) | 入口不清零 `*info`，调用者变量里残留非零值就直接返回、不做分解；zgbsv_native_work 的 lwork 错误码 −10 实为 info 的参数号 | `gbtrf_native_stale_info_all`、`gbsv_native_work_argnum` | ✔ roxy 第 26 轮：info 残留 7 时返回 7（xerbla 打印 “function-specific error”），带状矩阵未改；info = 0 时正常分解 |
| zgeqlf、zgbtrf_batched 查询、zhegvr | 查询值被自己的检查拒绝（geqlf，min(m,n) = 0）；批量带状 LU 在快速返回时不回答查询（lwork 留为 −1）；zhegvr 只把 lwork = −1 当查询 | `geqlf_query_rejected_all`、`gbtrf_batched_query_unanswered`、`hegvr_lrwork_query_ignored` | zgeqlf ✔ roxy 第 26 轮（m = 0、n = 5 与 m = 7、n = 0：查询返回 1，按它调用得 −7）；其余读码 |
| 10 个例程的文档最小 lwork | zgesvd、zgesdd、zgehrd_m、zungbr(P)、zheevx、zheevr、zhegvx、zhegvr、zstedx、zhegvdx_2stage：按文档分配会被拒（负 info） | `doc_minimum_rejected_all` | 未测：读码 |
| [sdcz]ungtr（导出、不在公开头文件中、库内无调用者） | 反射向量移位循环边界与 LAPACK ZUNGTR 不同：上三角每列少移一个；下三角改写对角、末行不移、A(n−1,0) 不清零 | `ungtr_upper_shift_short`、`ungtr_lower_shift_wrong`、`ungtr_first_col_stale` | ✔ CPU（`experiments/magma_dense/orgtr_shift_results.txt`：照抄循环 + 参考 LAPACK，n = 50 上三角 ‖QᵀQ − I‖ = 9.2e19、下三角 1.8；改成 LAPACK 边界后与 LAPACK 完全一致）；未在 GPU 上调库本身；✔ roxy 第 26 轮调用库本身（上三角）：n = 8 时 ‖Q − Q_LAPACK‖ = 2.7、n = 50 时正交误差 6e30 |
| zgbtrf_native_work（ILP64 编译） | device_info 放在 device_work + 4 字节，内核对它做 8 字节写，不对齐 | `gbtrf_native_info_misaligned_ilp64`、`gbtrf_info_alignment_all` | 未测：读码（节点上是 LP64 编译，不受影响） |
| magma_zhtodpo / zdtohpo / zhtodhe（公开头文件 magmablas_z.h） | 从不写 `*info`，却 `return *info`：外部调用者拿回自己变量里的旧值；库内调用者忽略 iinfo，不受影响 | `htod_returns_caller_value_all` | 未测：读码（2026-11-26“先读后写 info”全库扫描，`scripts/info_init_scan.py`） |
| 参数号（33 个例程，58 处检查） | `info = −k` 的 k 不是被检查参数的位置：多数是从 LAPACK 复制检查，签名里多出块大小、偏移、批量数组等参数后没有重新编号；zhegvx/zhegvr/zhegvdx/zhegvdx_2stage（及 _m）把 JOBZ 和 RANGE 的错误码对调 | `argnum_all_wrong`、`argnum_counts`、`hegv_jobz_range_swapped` | 读码：`scripts/argnum_scan.py` 扫描后逐条对照签名核对（zlarfb_gpu 是误报，zgeqrs3 已登记）；✔ roxy 第 26 轮抽测：zhegvx 非法 jobz 报 −3、非法 range 报 −2（对调）；zgesv_batched 的 lddb 报 −6、zgetrs_nopiv_batched 报 −8（实为第 7 个参数） |
| zcposv_gpu / zchesv_gpu | 不检查 uplo：非法值经 zlat2c 返回 −1，被当作精度转换失败（iter = −2），再由 FP64 回退的 zpotrf_gpu 返回 −1 | `cposv_bad_uplo_reported_as_conversion` | ✔ roxy 第 26 轮（zcposv_gpu）：iter = −2、info = −1，依次打印 zlanhe −2、zlat2c −1、zpotrf_gpu −1 |
| 分治特征值驱动的 lrwmin（14 个驱动：heevd/heevdx/heevdx_2stage/hegvd/hegvdx/hegvdx_2stage 及 _gpu、_m） | `1 + 5n + 2n²` 在 32 位下对 n ∈ [32767, 46339] 为负：自身检查对任何 lrwork 放行，查询返回负数；A 经 zhetrd 改写后由 dstedx 报 −12（zheevd 编号中是 liwork） | `heevd_lrwmin_wraps_all`、`heevd_lrwmin_instance` | 读码（`scripts/wrap32_scan.py`）；第 24 轮已在 _m 路径 n = 38,000 上实测同类行为 ✔ |

## 2. 稠密“状态丢失”审计（15 条全部有 GPU 证据）

| # | 例程 | Lean（`MagmaAudit`） | GPU 0 |
|---|---|---|---|
| 1 | gbtrs_batched_strided 指针别名 | `gbtrs_alias`、`gbtrs_overread` | ✔ `audit_results.txt` |
| 2 | heevdx_2stage Upper | `heevdx_upper_success` | ✔ `audit_results.txt` |
| 3 | hetrf/sytrf_nopiv 最后主元 | `last_pivot_untested`、`nopiv_info_wrong` | ✔ `audit_results.txt` |
| 4 | potf2 0 起下标 | `potf2_col0_success` | ✔ `audit_results.txt` |
| 5 | trtri_gpu 块 info 覆盖 | `trtri_gpu_instances`（精确复现 0/45/256） | ✔ `audit_results.txt` |
| 6 | potrf_vbatched Upper 被忽略 | `vbatched_upper_wrong` | ✔ `audit2_results.txt` |
| 7 | hgetrf 补齐宽度 GEMM | `hgetrf_padded_gemm_overrun`、`hgetrf_1000_padded` | ✔ `audit2_results.txt`（20,903 个元素不同） |
| 8 | 精化接受 Inf/NaN | `MagmaDense.accept_inf`、`accept_nan_*` | ✔ `dense_results.txt` |
| 9 | FP16 转换无范围检查 | `fp16_conversion_unchecked` | ✔ `fgmres24_original.txt`（dhgesv k ≥ 20 全 NaN、info = 0） |
| 10 | gesvj_batched 不报不收敛 | `gesvj_nan_converged` | ✔ `audit3_results.txt` |
| 11 | getrf_m 永不报奇异 | `getrf_m_never_singular`（`getrf_m_offset` 为被掩盖的次要问题） | ✔ `audit3_results.txt`（根因：末尾 getrf_piv 清零） |
| 12 | potrf_vbatched 不清零 info | `vbatched_stale` | ✔ `audit2_results.txt` |
| 13 | getri_outofplace_batched 不写 info | `getri_outplace_info_untouched` | ✔ `audit2_results.txt` |
| 14 | gegqr 秩 0 = 成功 | `gegqr_rank0_success` | ✔ `audit2_results.txt` |
| 15 | hgetrf 奇异面板留转置 | `hgetrf_left_transposed` | ✔ `audit3_results.txt` |

## 3. 稀疏求解器（36 个源文件全部归类；32 个线性求解器全部上 GPU）

| 内容 | Lean（`MagmaSparse`） | GPU 0 |
|---|---|---|
| 收尾逻辑归类：4 个会把 breakdown 覆盖成成功 | `bicg_breakdown_success`、`overwrite_group_size`、`fixed_breakdown_divergence` | ✔ `sweep_results.txt`：b 含 NaN 时报成功的**恰好是这 4 个** |
| CG_MERGE/PCG_MERGE 复数不取共轭 | `cg_merge_false_convergence` | ✔ `sparse_results.txt`、`sweep_results.txt` |
| 早退出查 ‖b‖、绝对阈值 | `early_exit_returns_x0` | ✔ `sweep_results.txt`（21/32 静默失败，其中 5 个直接返回 x0；原记 23/32 有误，已更正） |
| 状态不看真实残差 | `template_ignores_final_res` | ✔ 同上 |
| cg_res 精确初值报非正定 | `cgres_exact_x0_nonspd` | ✔ `sparse_results.txt`、`sweep_results.txt` |
| pcg_merge 以初始残差升级成功 | `pcg_merge_upgrade` | ✔ `sparse_results.txt` |
| 包装层默认分支返回 0 | `wrapper_unhandled_success` | ✔ `sparse_results.txt` |
| IDR 参数改写（s、maxiter 截断） | `idr_params` | ✔ `sweep_idr_restart.txt`（128 = 2n，n = 64）。独立复核指出：这个运行证据只来自 IDRMERGE/PIDRMERGE（IDR、PIDR 10 步内收敛，未达上限）；6 个 IDR 文件中同一截断行都存在，其余 4 个只有读码证据 |
| FGMRES 粘住的成功 | `fgmres_sticky_success` | ✔ `audit3_results.txt`（1152 次扫描中 252 次） |
| LOBPCG 参数检查失效 | `lobpcg_check_dead`、`lobpcg_nan_converged` | ✔ `audit2_results.txt`（非法参数照常运行） |
| Jacobi 存储的 0 对角元 | `jacobi_stored_zero_passes` | ✔ `precond_results.txt`（复数下为 NaN） |
| ILU 数值零主元 | `ilu_numerical_zero_unchecked`、`ilu_zero_pivot_pbicg_success` | ✖ 预测不符：显式存储的零在分析阶段已被检出；**崩溃根因**是 setup 失败被覆盖后对未构建因子做转置，`magma_zmtransposeconjugate` 无限递归（`transpose_unbuilt_diverges`，✔ `round5_ilu_bt.txt`、`round5_ilu_pcg.txt`） |
| ICC 分解失败不报告 | `icc_breakdown_unchecked` | ✖ `audit2_results.txt`：该输入无可见后果 |
| bombard 残差记录错误 | `bombard_wrong_res` | ✖ `round5_results.txt`：未复现（TFQMR 未胜出） |
| bpcg 多右端 | `bpcg_col0` | ✔ `round5_results.txt`：更严重，num_vecs = 1，第 2 列根本不解 |
| IDRMERGE/PIDRMERGE 在所有输入上失败 | —（扫描中发现，非预测） | ✔ `sweep_results.txt`、`sweep_idr_restart.txt` |

## 3b. 第二轮审计：此前未审的部分（2026-11-25；两个子代理读码，主会话核对源码；`round6_*`、`round7_*`）

| 例程 | 发现 | Lean | GPU 0 |
|---|---|---|---|
| magmablas_slag2h(_batched) | 溢出标志通过设备变量的主机影子地址写入；NaN 通过范围检查；批量版只拷回每段第一个 | `MagmaAudit.lag2h_info_always_zero`、`lag2h_nan_passes`、`lag2h_batched_only_first` | ✔ `round6_results.txt`：一个 1e5 ⇒ **非法内存访问且 info = 0**；NaN ⇒ info 0；批量 ⇒ 非法访问、info 全 0 |
| shpotrf_gpu/native | CUDA 路径用 FAST_16F GEMM，无范围检查/缩放；HIP 路径丢弃 s2h_info、分配失败时 goto 越过队列创建 | `shpotrf_ignores_conversion`、`shpotrf_oom_uninit_queues`（HIP） | ✔ SPD 矩阵 ×2^40 ⇒ info = 1025（误报不正定），spotrf 正常；✖ 分配失败崩溃：该路径只在 HIP 编译 |
| dlag2s/zlag2c/zlat2c | NaN 不报 | `lag2s_nan_passes` | ✔ `round7_results.txt`：NaN ⇒ info 0，Inf ⇒ 1 |
| dstedx | 后一块的 dlascl 把前一块 dsteqr 的失败清零 | `stedx_info_erased` | ✔ LAPACK 该块 info 9；magma_dstedx info 0、10 个 NaN 特征值 |
| dgels_gpu（geqrf_gpu 对角块求逆） | trtri 的奇异状态被丢弃，R 块本身被当作逆 | `geqrf_singular_block_used` | ✔ 列 0 为零：info 0、x 有限、‖b−Ax‖/‖b‖ = 95；LAPACK info = 1 |
| dgeqp3_gpu | 固定列的 QR 被注释掉；需要交换时对设备指针调主机 zswap；文档给的 LWORK 被拒 | `geqp3_fixed_unreduced`、`geqp3_doc_lwork_rejected` | ✔ jpvt[0]=1 ⇒ info 0、第 0 列原样返回；jpvt[1]=1 ⇒ **段错误**；文档 LWORK ⇒ −8 |
| dsygvdx（及 hegvdx 全家） | n ≤ 128 时忽略 range | `hegvdx_small_ignores_range` | ✔ RangeI il=iu=50 ⇒ m = 50、w[0] = λ_min；RangeV 同 |
| sync-free ILU（CUDA ≥ 11） | csr2cscEx2 宏的 CSC 输出数组顺序反了，n+1 长的数组写入 nnz 个行号 | `MagmaSparse.syncfree_csc_swapped` | ✔ L.drow = 0 1 1 2 2 3 …（nnz 127 > 65），setup 0，PBiCGSTAB −202；加 guard 分配器后损坏位置变化，变为 info 0、残差 1 |
| magma_z_spmv 维度不符 | 无最终 else，返回 0 不计算 | `spmv_mismatch_success` | ✔ A 4×4、x 3×1 ⇒ 0，y 不变 |
| DENSE SpMV | 行主序存、列主序读 ⇒ Aᵀx | `dense_spmv_transpose` | ✔ y = (1,5)，应为 (3,3) |
| CSC SpMV | 数组布局与 zmconvert 不一致 | `csc_spmv_layout_mismatch` | ✔ y = (1,2)，应为 (3,3) |
| SELL-P（alignment 1） | 网格按列数开 | `sellp_tall_rows_skipped` | ✔ 300×1 矩阵 44 行未写 |
| zmscale UNITDIAG | 只查 s == 0 | `mscale_nonfinite_success` | ✔ 负对角或纯虚对角 ⇒ 全 NaN、返回 0 |
| 复数 ICC | 用 Lᵀ 而非 Lᴴ | `icc_transpose_not_conjugate` | ✔ Hermitian 三对角：PCG+ICC 1000 步 −202；CG 17 步收敛；实数版 1 步 |
| zsymbilu（ILU(k) 符号分解存储不足） | 返回值未检查 | — | ✖ 预测不符：setup 返回 −3009（后续 cuSPARSE 报错被捕获），不是静默 |
| CPU 上的 magma_z_spmv | 释放调用者的数组、结果只写进局部副本 | `spmv_cpu_frees_caller` | ✔ `round8_results.txt`：返回 0，y 未得到结果（读到被重用的内存） |
| CSR→CSRD / ELLD 缺对角元 | 越界写一个槽位 | `csrd_missing_diag_overflow` | ✔ 返回 0；之后 glibc `malloc(): invalid size` 中止；ELLD 列号 −1 |
| zmdiff | 只比共有元素、只比实部 | `mdiff_blind` | ✔ I₂ 对 {(0,1)} 得 0；1+5i 对 1−5i 得 0 |
| zmfrobenius | 不初始化 *norm；行数不等时什么也不算 | `frobenius_uninit` | ✔ 入口 5 ⇒ √5；行数不等 ⇒ −1 不变、返回 0 |
| zmdiagadd | 没有存储对角元的行不加 | `diagadd_missing_row` | ✔ (1,1) 仍为 0 |
| zmtranspose（CPU） | 维度不交换 | `transpose_cpu_dims` | ✔ 1×3 转置后仍为 1×3 |
| magma_z_csr_mtx | 下标不查界 | `mtx_unchecked_index` | ✔ 行号 9 的 3×3 文件 ⇒ 段错误 |
| magma_z_applyprecond（ILU/ICC）、applyprecond_right（ConjTrans） | 分支不写 x | `applyprecond_ilu_noop`、`applyright_conjtrans_noop` | ✔ 返回 0、x 保持 7 |
| CSC SpMV 布局 | drow/dcol 角色与 zmconvert 相反 | `csc_spmv_layout_mismatch` | ✔（见上：y = (1,2)） |
| zsymbilu | 符号分解状态未检查 | `symbilu_status_dropped` | ✖ 后续 cuSPARSE 失败被报告（−3009），不静默 |
| zhegvx | B 不正定时提前返回不释放 | `hegvx_leak_512` | ✔ n = 512、20 次调用显存少 423.6 MB（每次 21.2 MB，多于 dA/dB/dZ 的 12.6 MB） |
| zgels（CPU） | 查询得到的 LWORK 对大 nrhs 不够，zunmqr 状态不查 | `gels_lwork_too_small` | ✔ nb = 64、LWORK 8192：nrhs = 16384 ⇒ DORMQR 参数 12 非法，参考 LAPACK 的 xerbla 直接结束进程；nrhs = 4096 正常 |
| zgeqrs3_gpu | 参数编号错 | `geqrs3_argnums_wrong` | ✔ lddb 错报 −8（geqrs_gpu 正确报 −9） |
| zhegvd 系 | 负的分配错误变成正 info | `hegvd_alloc_error_positive` | 不可达，有 GPU 证据：第十轮 zhegvd k = 3（zpotrf_gpu 的锁页分配失败）⇒ 段错误，zpotrf_gpu 不返回 −113 |
| magma_zgels_batched | 头文件有声明，库里没有定义 | `dense_declared_not_defined` | ✔ `nm -D libmagma.so` 中 0 个符号 |

假警报（已撤回）：gbtrf/gbsv_native “不清零 info”——内核自己写 info。

## 3c. 多 GPU / 核外驱动与最后 3 个未审文件（第九轮；ngpu = 1 在 GPU 0 上运行；`round9_*`）

全部 202 个 `src/z*.cpp` 的分类见 `operators/magma/INVENTORY.md`：此前已审 127，本轮多 GPU/核外 31，新读 3，无状态判断的辅助例程 41。

| 例程 | 发现 | Lean | GPU 0 |
|---|---|---|---|
| zheevdx、zheevdx_m、zheevdx_2stage、zheevdx_2stage_m（range V） | 内部缩放后的特征值与未缩放的 (vl, vu] 比较 | `rangev_unscaled_bounds` | ✔ A = 1e-150·diag(1..1000)：4 个驱动都返回 m = 10（应为 100）、info 0；n = 200 时 1-stage 为 m = 2，2-stage 走 LAPACK 正确 |
| 同上，输入含 NaN | stedx 用 NaN 作 dlascl 的 cfrom，参考 LAPACK 的 xerbla 结束进程 | `dlascl_nan_rejected` | ✔ 4 个驱动都打印 “DLASCL parameter number 4” 后进程退出；LAPACK zheevd 返回 info 0 |
| zheevdx_2stage(_m) | stedx 的失败被 bulge_back、unmqr 清零 | `heevdx_2stage_stedx_lost` | ✔ 第十轮故障注入：dlaex0_m 分配失败 ⇒ info 0、误差 1（NaN 输入则更早结束进程） |
| zunmtr_m | 吞掉 zunmqr_m 的 NOT_IMPLEMENTED | `unmtr_m_not_implemented_lost` | ✔ Right、m = 10、n = 200：info 0，C 完全不变（单 GPU 版正常相乘） |
| zhetrd_he2hb(_mgpu) | 只查 lwork ≥ 1 | `he2hb_small_lwork_underflow` | ✔ lwork = 1056：info 0、带状结果差 2.5e119；lwork = 100：堆损坏中止 |
| zgeqrf_m | ngpu = 0 通过检查 | `geqrf_m_ngpu0_accepted` | ✔ SIGFPE |
| zcgeqrsv_gpu | 不查 R 的对角元；NaN 通过拒绝式判据 | `cgeqrsv_nan_accepted` | ✔ 秩亏：iter 0、info 0、x 非有限；A = [0]：x = (inf, nan)；LAPACK zgels info 1 |
| zgglse | 去掉了 ztrtrs 的奇异检查 | `gglse_singular_unreported` | ✔ info 0、x = (−inf, nan), (inf, nan)；LAPACK info 1 |
| zggrqf | 接受的 lwork 小于内部 geqrf 的要求 | `ggrqf_lwork_inconsistent` | ✔ 返回 −7 时 A、B 已全部被改 |
| zmscale_generate（UNITCOL） | 转置释放了调用者的 rowidx，清理时再释放一次 | `MagmaSparse.mscale_generate_double_free` | ✔ glibc “double free detected” |
| zpotrf_mgpu_right | 函数体全在 `#if 0` 里，头文件仍声明 | `dense_declared_not_defined` | ✔ 导出符号 0 个 |
| zhegvd_m/zhegvdx_m/zgeev_m 等 | 分配失败后的状态被丢弃、后续在未变换的数据上运行 | `early_alloc_failure_hidden` | ✔ 第十轮故障注入（见 §3d） |
| zgeqrf_ooc | lwork 检查弱于文档；NB 可为 0 导致死循环 | `geqrf_ooc_lwork_short` | ✔ 占满显存使其走核外路径：lwork = n ⇒ 段错误；✖ NB = 0 不死循环，返回 −113 |
| zpotrf3_mgpu（z/c，经 zpotrf_mgpu、zpotrf_m、zhegvd(x)(_2stage)_m）| M255：设备对角块缓冲 dlP 每 ngpu 步轮换，第 k 步 stream 2 的 trsm 读它与第 k+ngpu 步 stream 1 的拷贝改写它之间无序（ngpu ≥ 2，任意 h） | `potrf_mgpu_dlp_reuse_at`、`potrf_mgpu_dlp_not_reused_before` | ✔ 第 50 轮单卡模拟多卡（虚拟设备 + 单点 100 ms 延迟）：z 16/60、c 14/20 出错，info 0；加一个事件等待 0/80；d/s 不受影响（0/30） |
| zpotrf_mgpu（四种精度）| M256：写死 h = 2 的主机工作区槽在 h < ngpu 时被下一批块行提前改写（ngpu ≥ 3）；源码注释承认 h = 1 出错、h = 2 是临时修复 | `potrf_mgpu_host_slot_covered_iff`、`potrf_mgpu_h2_uncovered` | ✔ 第 50 轮模拟：z/c 在 h < ngpu 时 37/60 出错（全部报 info > 0），h ≥ ngpu 时 0/48；d/s 在单卡模拟中被 trtri_diag 的 legacy 流 memset 掩盖，由模型 `scripts/potrf_mgpu_hb.py` 给出 |
| magmablas_[zcds]trtri_diag（经 trsm_work、trsm_outofplace）| M257：`cudaMemset` 不带流，走 legacy 默认流；文档称这两个 trsm 是异步的 | `trtri_diag_waits_unrelated`、`trtri_diag_not_async` | ✔ 第 51 轮：另一条流忙 150 ms 时调用要 156–164 ms 才返回（空闲 0.4–0.6 ms），三种 queue 相同，结果正确 |
| zgetrf_m（CPU 接口，多卡 / 核外） | M258：ngpu 在分配前和循环中各算一次，分配时减少卡数后循环仍用满 ngpu0 张卡，访问未建的 queue 和未分配的缓冲区；清理也跑到 ngpu0 | `getrf_m_loop_exceeds_alloc`、`getrf_m_ngr64_instance`、`getrf_m_fix_within_alloc` | ✔ 第 52 轮模拟：恰在 ngpu0 > ceildiv(MAGMA_NGR_NB, 32) 时访问违例；补丁后 12/12 正确 |
| 第 52 轮调度模糊测试：getrf_mgpu、geqrf2_mgpu、potrf_mgpu、hegst_m、trsm_m、getrf_m、potrf_m | 单点延迟 5,445 次（前置点 + 后置点），只发现 M255（potrf_mgpu、potrf_m）；其余全部正确 | — | ✔ 正面结果（z 精度，ngpu 1–4） |
| 第 52 轮调度模糊测试，后续批次：unmqr_m、geqrf_m、ungqr_m、hetrd_mgpu（n = 3072 抽样）、heevdx_2stage_m（he2hb_mgpu、hemm/her2k_mgpu、bulge_back_m、bulge_applyQ_v2_m、unmqr_m、stedx_m）、gehrd_m（lahr2_m、lahru_m） | unmqr 144、geqrf_m 724、ungqr 492、hetrd 595、heevd2s 2,267 次全部正确；gehrd_m 见第 52 轮 TRIAGE | — | ✔ 正面结果（z，ngpu 1–4；主机数组可分页，见锁页复测） |
| zunghr_m / dorghr_m（经 zgeev_m、dgeev_m，Version5）| M259：gehrd_m 把 ilo 向下取整并按全局列号存 T，unghr_m 按子问题列号读 T 且不平移，ilo > 1 时每个块用错 T | `unghr_m_reads_wrong_T`、`unghr_m_inconsistent`、`unghr_m_ilo1_consistent`、`unghr_single_consistent`、`unghr_m_fix_consistent`、`unghr_m_fix_extra` | ✔ 第 53 轮（库版本，1 张 GPU）：49 个 (n, ilo, ihi) 中恰好"ilo > 1 且 nh > nb"的 36 个出错（‖QHQᴴ − A‖/‖A‖ 4–21，info 0），单卡这一对和补丁版 49/49 正确；geev_m 端到端：ilo = 2、4 时特征向量残差 8.6–15，单卡 geev 1e-13 |
| zgehrd_m（经 zgeev_m）| M260：最后一次 zlahr2_m 在每张卡上排了 Y = work → dY 的异步拷贝，循环后只同步持有剩余块的卡，zgehd2 随即覆写 work；R < ngpu 时面板卡读到草稿（锁页 work） | `gehrd_m_panel_unsynced`、`gehrd_m_unsynced_device`、`gehrd_m_fix_syncs_all` | ✔ 单卡模拟 + 锁页数组：ngpu 2、n = 96 必现（误差 2.35e3，info 0）；补丁后 2,534/2,534 正确；其余组合被共用的拷贝引擎掩盖 |
| 锁页主机数组复测：trsm_m、hegst_m、getrf_m（核内 / 核外）、potrf_m、unmqr_m、geqrf_m、ungqr_m、heevdx_2stage_m、hetrd_mgpu | 约 4,100 次，只有 M255（potrf_m）和 M260（gehrd_m） | — | ✔ 正面结果 |
| zunmqr_gpu（dT 取 geqrf_gpu 文档大小）| M261：larfb 工作区放在 dT + 2k·nb，需要 nw·nb，文档大小只剩 roundup(k, 32)·nb；写出界 (nw − roundup(k, 32))·nb 个元素 | `unmqr_gpu_dT_overflow`、`unmqr_gpu_tester_size_suffices` | ✔ 第 55 轮：保护区写入数与预测 72/72 相等；不加保护区时相邻 dC 被改写，64/72 结果错，info 0 |
| zunmqr_2stage_gpu（公开）| M262：side = Right 时仍用 MagmaLeft 调 larfb，工作区 n·nb 按 ldwork = m 使用；M263：文档要求的输入只是 zgeqrf 的输出，实际需要显式 V 和每一块的 T | `unmqr_2stage_right_applied_left`、`unmqr_2stage_reads_unwritten_T` | ✔ 第 55 轮：按文档调用 96/96 错；V、T 都补齐后 Left 48/48 对、Right 48/48 错 |
| 第 55 轮：ungqr_gpu、ungqr、geqrs_gpu、geqrs3_gpu、gels_gpu、gels3_gpu（非方阵、k < n、ldda 填充） | 140 例全部正确 | — | ✔ 正面结果 |
| zhetrf_aasen（Upper）| M264：文档描述了 Upper，参数检查对非 Lower 一律返回 NOT_IMPLEMENTED | `aasen_rejects_documented_upper` | ✔ 第 56 轮：Upper 8/8 返回 −117 |
| 第 56 轮：hesv_nopiv_gpu、sysv_nopiv_gpu、he/sytrf_nopiv_gpu + he/sytrs_nopiv_gpu、hetrf_nopiv（CPU）+ hetrs_nopiv_gpu、hetrf/hetrf_gpu + LAPACK zhetrs、hesv | 512 例全部正确（U/L、nrhs 1/7、ld 填充、正定 / 对角占优不定 / 随机不定） | — | ✔ 正面结果 |
| zunmqr_batched（Right）| M265：文档列出了 Right，代码对非 Left 返回 −1；小规模时内部的 unm2r_reg/sm_batched 也拒绝 Right | `unmqr_batched_rejects_documented_right` | ✔ 第 57 轮：Right 60/60 返回 −1 |
| 第 57 轮：hetrd_gpu、hetrd2_gpu → unmtr_gpu（U/L × L/R × N/C × ld 填充）；geqrf_batched → unmqr_batched（Left） | 192 + 60 例全部正确；wA 与 dA 逐位相同 | — | ✔ 正面结果 |
| 第 58 轮：MAGMA 的 gebrd/geqrf/gelqf(_gpu)/geqlf/hetrd → unmbr/ungbr/unmqr/ungqr2/unmlq/unmql/unmtr（非方阵、ld 填充、所有 side/trans/vect/uplo） | 264 例全部正确；与 LAPACK 兼容的格式的交接没有问题 | — | ✔ 正面结果 |
| 第 59 轮：13 个带文档化设备工作区的例程（geqrf/geqrf3/geqrs/geqrs3_gpu、he2hb、hetrd2_gpu、getri_gpu、zstedx、gegqr_gpu、ztrsm_work、zhemv_work、zlarfb_gpu、geqp3_gpu），工作区取文档大小并加哨兵 | 89 例保护区均未被写（M261 之外没有越界）；geqp3_gpu 的 z 精度拒绝文档 LWORK，补进 M120 | — | ✔ 正面结果 |
| 第 60 轮：expert_gpu_work 系列（geqrf/getrf/potrf/getri/getrs/potrs/trtri/gegqr）、geqrf_batched_work、gbtrf/gbsv_native_work，按查询值分配并加主机和设备保护区 | 95 例保护区均未被写，结果正确；Native 模式被拒与文档一致 | — | ✔ 正面结果 |
| zpotrf_vbatched（max_n 超过交叉点）| M266：大路径调用面板时 gbstep 传 0，非正主元报成 (c mod 128) + 1，整批都如此 | `potrf_vbatched_lg_info_local` | ✔ 第 61 轮：10/10 符合预测（500 → 117）；对照组 potrf_batched 22/22 正确 |
| zpotrf_vbatched（仅 Z，360 ≤ max_n ≤ 448）| M267：lpout 需要的共享内存超过 47,000 字节，返回 −33，整批都没有分解，info_array 仍为 0 | `potrf_vbatched_z_gap_all`、`potrf_vbatched_cds_no_gap` | ✔ 第 61 轮：n = 360/361/384/448 时矩阵逐字节不变；352、359、449 正确 |
| 第 61 轮：getrf/getrf_max_nocheck_work/getrf_nopiv/potrf_vbatched 逐矩阵与 LAPACK 比较 | 非奇异矩阵全部正确；奇异矩阵上重现 M164、M173 | — | ✔ |
| 第 62 轮：资源不足时返回负值的 7 个批量和原生内核的调用者回退；zgetrf_native 在 m 跨过 7168 时 | 只有 vbatched 的 lpout 不回退（M267）；zgetrf_native 22/22 正确 | — | ✔ 正面结果 |
| magmablas_[zc]geam（ConjTrans）| M268：文档写 op = Xᴴ，三个转置内核都不取共轭，按 Trans 计算；MAGMA 自己的测试程序也不取共轭 | `geam_conjtrans_wrong` | ✔ 第 63 轮：含 C 的 50/50 错，N/T 40/40 对 |
| magmablas_[zc]gemm_reduce | M269：文档写 AᵀB，内核累加 conj(A)·B，即 AᴴB | `gemm_reduce_conjugates` | ✔ 第 63 轮：4/4 与 AᴴB 一致 |
| 第 63 轮：geadd2、gemv_conj、laset_band、laswpx、laswp2、transpose_conj(_inplace)、transpose_inplace、trmv、symmetrize、symv_work、larfg、swapdblk、lacpy_conj | 372 例全部正确 | — | ✔ 正面结果 |
| 第 64 轮：magmablas gemm/gemv/trsm/trmm（非批量、batched、vbatched）、trsv 的 ConjTrans（复数数据，N/T/C） | 312 例全部正确；M268 只出在 geam | — | ✔ 正面结果 |
| sparse TFQMR / TFQMR_MERGE / PTFQMR / PTFQMR_MERGE（复数）| M270：影子内积写成 zdotc(r, r̃)，得到共轭值，α、β 被取共轭；非 Hermitian 复系统不收敛 | `tfqmr_dotc_reversed`、`tfqmr_rho_wrong_iff` | ✔ 第 65 轮：原版 −201（残差 0.04），补丁后 39/39/14 步收敛 |
| 第 65 轮：21 个稀疏求解器 × 复 Hermitian 正定 / 不定 / 非 Hermitian | 其余都是已知缺陷（M084、M103、M089）或如实报告的失败 | — | ✔ |
| sparse zlobpcg（`--ev k`）| M271：只要第 0 个 Ritz 对达标，就跳出主循环并返回成功，其余 k − 1 对没有收敛 | `lobpcg_stops_on_first` | ✔ 第 66 轮：24/24 info 0，残差 6e-6–6e-4（要求 1e-8）；补丁后 24/24 达到 1e-8 |
| magma_zprint_csr_mtx | M272：RowMajor 的尺寸行打印成 "n_col n_row"；M273：ColMajor 经由 z_transpose_csr，调用者读的是已释放的数组 | `print_header_swapped`、`z_transpose_csr_returns_freed` | ✔ M272（3×4 输出 "4 3 5"）；M273 只是读码 |
| 第 67 轮：16 种 sparse 格式与 CSR 转置 / 共轭转置，在真正复数数据上 | 与第 46 轮（实数）结论一致，没有复数特有的问题；转置的共轭处理正确 | — | ✔ |
| magma_zgeqr2x{,2,3,4}_gpu（dwork）| M274：文档写 N 个复数或 3N 个 double，实际用量为 4n、5n、6n，最多到 50n；MAGMA 的测试程序分配 130·min_mn | `geqr2x_overruns` | ✔ 第 68 轮：文档大小下 47/56 例保护区被写 |
| 第 69 轮：magmablas_zlange、zlanhe（含对角元虚部、另一侧三角的哨兵）、zgeqr2_gpu，工作区取文档大小加保护区 | 117 例全部正确；zlanhe 的 F 范数文档写明不支持 | — | ✔ 正面结果 |
| magmablas_[zcds]laswp_sym | M275：按 32 个主元分块后，内核按绝对下标读块内的相对 ipiv，越界；M276：同一次启动中多个主元之间存在线程间竞争 | `laswp_sym_second_chunk_out_of_bounds`、`laswp_sym_race_instance` | ✔ 第 70 轮：k2 > 32 时误差约 200；多个主元时误差约 1.2–1.5 且每次不同；一个主元时正确 |
| 第 70 轮：herk_small_reduce、hemv/symv（Upper、incx 2）、symmetrize_tiles | 208 例全部正确 | — | ✔ 正面结果 |
| 第 71 轮：gemm/gemv_batched_strided、geadd/lacpy/laset/transpose_conj/hemv_batched、gbsv_batched(_strided) | 102 例全部正确（复数数据，strided 步长加了余量） | — | ✔ 正面结果 |
| 第 72 轮：gesvj_expert_batched_strided、heevj_batched_strided_small_sm | 与指针数组接口逐位相同；偏差都来自 M240 | — | ✔ |
| magma_z_spmm（稀疏 × 稀疏，magma_zcuspmm） | M277：CUDA ≥ 12 分支把 beta = NULL 传给 cusparseSpGEMM_workEstimation，每次返回 −3003；alpha 从不使用 | `spmm_alpha_ignored` | ✔ 第 73 轮：n = 10、100、500 都是 −3003 |
| magma_zm_27stencil | M278：只清除 x 方向的边界耦合，y 方向回绕的耦合被保留；n = 2 时反而丢失耦合 | `stencil27_wrap_instance` | ✔ 第 73 轮：n = 3 多 84 个，n = 5 多 312 个，与预测一致 |
| 第 73 轮：matrix_tril/triu、mdiagadd、indexsort(val)、zm_5stencil、mtranspose{conj,abs,struct}_cpu（方阵） | 全部正确；_cpu 转置变体在非方阵上的问题归入 M137/M138 | — | ✔ 正面结果 |
| magma_[zcds]_spmv_shift（CSR/ELL/ELLPACKT 内核） | M279：文档 y = α(A − λI)x + βy，实际算 αAx − λx + βy | `spmv_shift_lambda_unscaled`、`spmv_shift_instance` | ✔ 第 74 轮：α ≠ 1 时差 5–9%，α = 1 正确 |
| magma_[zcds]mslice | M280：末尾几片起点越过 n，片大小为负；返回 −112 或不确定地返回 0；出错后的输出被 magma_zmfree 释放时堆损坏 | `mslice_size_negative`、`mslice_instances` | ✔ 第 74 轮 |
| magma_[zcds]mconjugate | M281：不检查格式，ELL 触发非法内存访问仍返回 0 | — | ✔ 第 74 轮 |
| 第 74 轮：mcsrcompressor_gpu（纯虚 / 纯实 / 零值混合）、mshrink、vtranspose（CPU 与设备）、mconjugate（CSR）、mslice（起点在 n 以内） | 全部正确 | — | ✔ 正面结果 |
| magma_[zc]{p,}idr{,_merge,_strms}（直接调用） | M282：角度保护用 \|Re(tᴴr)\|，A = iS 时全部不收敛；strms 变体与 merge 变体一样在所有输入上失败（归入 M103） | `idr_rotation_rotates_dot`、`idr_rho_instance` | ✔ 第 75 轮：120 例；补丁版 20/20 收敛，对相位不变 |
| magma_zqr（稀疏 QR 包装） | M283：设备上的 A 被原地改写成 Q；Q、R 正确 | — | ✔ 第 75 轮 |
| magma_[zcds]_applyprecond_left/right（MagmaTrans，ICC），因而 PBiCG / PQMR + ICC | M284：ICC 的左右求解在转置分支中没有对调，影子系统用了 L⁻¹L⁻ᵀ | `icc_trans_swapped`、`icc_trans_instance` | ✔ 第 76 轮：恒等式误差 0.03–0.22；PBiCG + ICC −201；补丁后与 ILU 相同 |
| magma_zcustomspmv、magma_zapplycustomprecond_l/_r | 给用户替换的模板（spmv 调 zge3pt，precond 为空），不算缺陷 | — | 读码 |
| magma_[zc]_applyprecond_left（MagmaTrans，Jacobi），因而复数矩阵上的 LSQR / PBiCG / PQMR + Jacobi | M285：转置分支的含义是共轭转置，Jacobi 却乘 D⁻¹ 而不是 conj(D⁻¹) | `jacobi_trans_adjoint_iff`、`jacobi_trans_instance` | ✔ 第 77 轮：Hermitian 恒等式误差 0.388；LSQR −201、PBiCG −202；补丁后与无预条件相同 |
| 第 77 轮：实矩阵上的 LSQR（无预条件 / Jacobi / ILU / ICC）、复数矩阵上的 ILU 转置分支 | 全部收敛，ILU 满足 Hermitian 恒等式；ICC 在 LSQR 里受 M284 影响但仍收敛 | — | ✔ |
| 第 78 轮：magmablas_z{gemm,gemv,hemv,hemm,herk,syrk,her2k,syr2k,trmm,trsm}_vbatched（174 例，尺寸含 0，ld 余量与另一侧三角填哨兵） | gemm / gemv / hemv / trmm / trsm 全部正确；herk 系列另一侧三角被写（M246）；hemm 读错元素（M248，补丁后 8/8 正确） | — | ✔ |
| 第 79 轮：magmablas_ztrtri_diag_batched / _vbatched（U/L × N/U，含非 16 倍数与空矩阵）、zget/zsetmatrix_transpose（单卡） | 有效块的逆全部正确，另一侧三角未引用；拷贝全部正确。观察：upper 的 part1 内核在补齐列里留下旧寄存器值（库内不读这些位置）；getmatrix_transpose 返回前不同步（文档未说明，库内调用者自行同步） | — | ✔ |
| magmablas_[zcds]lacpy_sym_in/out | M286：文档的 rows 布局与交换语义和代码（交错二元组、赋值）不一致；库内调用者 Aasen 与代码一致 | — | 读码 |
| 第 80 轮：M279、M280、M284（d）与 M282、M285（c）的精度复核 | 全部复现 | — | ✔ |
| magma_[zcds]geqr2x2/3/4_gpu（n > 128）、magma_[zcds]larfbx_gpu（k > 128） | M287：trmv 内核的 __shared__ sum[128] 以 k 个线程启动，n ≥ 130 时非法内存访问，info = 0 | `geqr2x_tkernel_overflow_iff`、`geqr2x_tkernel_instances` | ✔ 第 81 轮：z、d 都在 129/130 之间出错 |
| magma_[zcds]larfbx / larfx / larfgx_gpu | M288：文档与代码不一致（Hᴴ、读 T 的下三角、tau 约定、xnorm、beta 的位置） | `larfbx_applies_adjoint` | ✔ larfbx；其余读码 |
| magma_[zc]hetrs_gpu / magma_[ds]sytrs_gpu | M289：Lower 的前代只解第 0 列；M290：nrhs > 1024 时 <<<1, nrhs>>> 启动失败 | `hetrs_lower_wrong_cols`、`hetrs_launch_invalid_iff` | ✔ 第 82 轮：Lower 恰好 nrhs − 1 列错；nrhs = 1025 全错；补丁 11/11 |
| zpotrf_mgpu_right.cpp | 整个文件在 `#if 0` 里，不编译 | — | 不需要测 |
| 第 54 轮：heevd/heevdx/heevdx_2stage/hegvd/hegvdx/hegvdx_2stage_m、geev_m、getrf/potrf/geqrf/gehrd_m（lda = n+7 加哨兵、z、Upper、itype 2/3、jobz N、range V、m ≠ n） | 492 例，出错全部归入 M121、M040、M065、M216（M216 新路径：hegvdx(_2stage)_m 在 itype 1、range V 时 74/75 个向量错误）；填充行均未被改写 | — | ✔ 正面结果（ngpu = 1） |

## 3d. 分配失败路径：故障注入（第十轮；`fail_malloc.c`、`round10_*`）

LD_PRELOAD 垫片让驱动内部第 k 次 `cudaMalloc`/`cudaHostAlloc` 失败（k 取遍该驱动的所有分配），结果与 LAPACK 比较；失败点用 backtrace 定位（`round10_backtraces.txt`）。11 个驱动、共 55 次失败注入（6 + 8 + 3 + 38 = 55；早先写的 7/37 有误，经独立复核更正）：

| 结果 | 次数 | 位置 |
|---|---|---|
| **info 0、结果错误** | 6 | zhegvd、zhegvd_m（zhegst 的工作区，误差 599：得到 A 的特征值而非 (A,B) 的）；zheevdx_2stage（zhetrd_he2hb，0.498）；zheevdx_2stage_m（zstedx_m 内 dlaex0_m，误差 1）；zgeev、zgeev_m（zgehrd，0.067） |
| 进程被 dlascl/xerbla 结束 | 8 | zheevd、zheevd_m、zhegvd、zhegvd_m：zhetrd 的 iinfo 不读，在未初始化的三对角上缩放 |
| 段错误 | 3 | zgesv、zposv、zhegvd：zgetrf_gpu/zpotrf_gpu 不检查锁页分配 |
| 报告失败或不受影响 | 38（报告 19、不受影响 19） | — |

辅助例程（第十一轮，`round11_*`）：zunmtr、zunmbr、zunghr 内部唯一一次分配失败 ⇒ info 0，zunmtr/zunmbr 返回原样的 C，zunghr 返回错误的 Q；失败点在 zunmqr / zungqr（`helper_iinfo_dropped`）。

Lean：`early_alloc_failure_hidden`、`hetrd_failure_reaches_dlascl`、`pinned_alloc_unchecked`。

ngpu = 2 而只有 GPU 0 可见：zpotrf_m、zheevd_m 结果正确（magma 退回 1 个 GPU），但留下粘滞的 “invalid device ordinal” CUDA 错误。

## 3e. magmablas 中写状态的 GPU 内核（39 个 z 文件；第十二轮；`round12_*`）

子代理逐文件读码，主会话核对关键行。

| # | 内核（文件） | 发现 | Lean | GPU 0 |
|---|---|---|---|---|
| F1 | zgetf2_fused_kernel_vbatched（zgetf2_kernels_var.cu l.470） | 全局列号与面板内 my_minmn 比较，第一面板外的零主元被清零 | `vbatched_panel_info_erased` | ✔ A(40,40)=0 ⇒ info 0（LAPACK 40）；A(10,10)=0 ⇒ 10 |
| F2 | izamax 内核 + 通用带状 LU 回退（zgetf2_kernels.cu l.39/48，zgbtrf_batched.cpp l.209） | 每列重置 info，只有最后一列算数 | `gb_generic_info_last_column` | ✔ n=200、kl=ku=64：A(5,5)=0 ⇒ 0；只有 A(200,200)=0 ⇒ 1 |
| F3 | zgetf2.cu l.117–132 | 奇异检查被注释掉 | `getf2_gpu_never_singular` | ✔ info 0、因子 30 个非有限元 |
| F4 | zpotf2.cu l.316 `xreal < 0` | 0 与 NaN 不报 | `potf2_zero_nan_pass` | ✔ zpotf2_gpu、zpotrf_native(Upper) 对 0 和 NaN 都 info 0 |
| F5 | zpotf2_devicesfunc.cuh l.38（任意宽度版） | 无 NaN 检查 | `potf2_zero_nan_pass` | ✔ zpotrf_batched n=12 ⇒ 0，n=16 ⇒ 10 |
| F6 | magma_izamax_native（length > 15360 走 cublasIzamax） | 不写 info | `izamax_long_column_no_info` | ✔ m=16384 零列 ⇒ 0；m=4096 ⇒ 3 |
| F7 | zgetf2_native.cpp l.164 | 第二半递归缺 n1 偏移 | `native_recursive_offset` | ✔ 第十三轮：expert 接口 recnb = 64 ⇒ info 18（LAPACK 50），与模型给出的值一致；recnb = 32 ⇒ 50 |
| F8 | 带状 LU 的 km 与列尾（zgbtrf_batched_fused_sm.cu l.80 等） | m ≠ n 时多读一行/少读行 | `gb_km_one_extra_row` | ✔ m=6,n=4 ⇒ 假 info 4（LAPACK 0）；m=4,n=6 ⇒ info 0、ipiv=5 > m（LAPACK 4） |
| F9 | zheevj_batched_small_sm.cu l.190 | NaN 视为收敛；noconv 竞争；max_sweeps 不查 | `heevj_nan_converged` | ✔ NaN 矩阵 info 0，且 8 个特征值全部为有限数（NaN 被静默吞掉，结果不可能正确）；max_sweeps = 0 见第十四轮（✔），noconv 竞争未复现 |
| F10 | zgetf2_nopiv_kernels_var.cu l.52 `*info++` | 指针自增，计数不变 | `nopiv_count_not_incremented` | ✔ 2048×2048 替换 1 个主元 ⇒ info 0 |
| F11 | zgesv_batched_small、zgbsv_kernels | 奇异时仍回代，B 被写成 Inf/NaN | `gesv_small_solves_singular` | ✔ info 3 正确，但 B 有 3 个非有限元 |
| F12 | zgetf2_native_kernel 原子写 info 与 ipiv 之间无 fence | 疑似竞争 | — | ✖ 300 次运行 info 全为 3（未复现） |
| F13 | 融合 getf2 在 m < n 时循环到 n | 假 info | `fused_mlt_n_spurious_info` | ✔ 4×8 随机矩阵 ⇒ info 5（LAPACK 0） |
| F14 | 树形 argmax 遇 NaN（比较为假） | 假奇异、跳过选主元 | `argmax_nan_hides_max` | ✔ zgetrf_batched 16×8/64×8/512×8，列 (0, NaN, 0, …, 5) ⇒ info 1（LAPACK 0）；4×4 顺序路径正确 |
| F15 | zlascl 参数编号 | −3/−7（LAPACK −7/−9） | `zlascl_argnums` | ✔ |

其余文件：QR 系列（geqr2*、geqrf_batched_smallsq）、laswp 系列、hetrs、lascl_diag、getrf_batched_smallsq_noshfl 为干净或无状态逻辑。

## 3g. sparse control/blas 66 个 z 文件的逐文件判定（第十五轮；`operators/magma/SPARSE_INVENTORY.md`、`round15_*`）

判定：已知发现 14、新发现 31、读码干净 13（其中 5 个经两次复核改判：zcsrsplit、zmcsrcompressor、zmlumerge、zvpass、zdiagcheck）、无状态逻辑 8。新发现中 13 条上 GPU 0：

| # | 文件 | 发现 | Lean | GPU 0 |
|---|---|---|---|---|
| 18/19/23 | zgecsrmv / zgeellmv / zgesellcmv | transA 不读 | `csrmv_trans_ignored` | ✔ MagmaTrans ⇒ y = (0,0)，应为 (0,1) |
| 21 | zgeelltmv | 填充元乘 x[0] | — | ✖ 填充行结果正确；看到的 NaN 来自复数 Inf 乘法 |
| 22/27 | zmdotc / zgemvmdot | 不取共轭 | `mdotc_unconjugated` | ✔ 第十六轮（输出改为设备缓冲区）：v = r = (i) ⇒ −1（应为 1），k = 1、2 相同 |
| 26 | zmdot_shfl | 复数 warp 归约是空操作 | `mdot_shfl_reduction_noop` | ✔ 64 个 1 求和得 1（IDR_MERGE 全部失败的可能原因） |
| 28 | zmgecsrmv | 多向量输出步长用列数 | `mgecsrmv_stride` | ✔ Y = 1 2 7 4，应为 1 2 4 5 |
| 31 | zmgesellpmv | 只算一半向量 | `sellp_multi_half` | ✔ 第 2 列不变 |
| 8 | zvread | 读不了自己写出的两列格式，堆溢出 | `vread_two_columns_overflow` | ✔ 值 (1,0)，随后 glibc 报堆损坏 |
| 4 | zcsrget | 返回已释放副本的指针 | `csrget_dangling` | ✔ val[0] 读回 4.9e-310 |
| 15 | zcompact（LOBPCG） | NaN 范数视为收敛 | `compact_nan_converged` | ✔ active 0、cBlock 0 |
| 13 | zmcsrcompressor_gpu | CPU 矩阵被清空 | `compressor_cpu_emptied` | ✔ 0×0、返回 0 |
| 1 | zfrobenius | 只比实部 | `frobenius_real_only` | ✔ [i] 对 [0] 得 0 |
| 6 | zmdiagdom | 空行 ⇒ NaN | `diagdom_empty_row_nan` | ✔ avg = NaN、返回 0 |
| 7 | zsort | 全 NaN 时不终止 | `zsort_nan_no_progress` | ✔ 120 s 超时 |

第十六轮（`round16_*`）再上 GPU 13 条：

| # | 文件 | 发现 | Lean | GPU 0 |
|---|---|---|---|---|
| 2a | zcsr_sort | 只排 col 不排 val | `csr_sort_values_not_permuted` | ✔ A(0,0) = 10（应为 20） |
| 2b | zcsrcoo_transpose | 维度不交换 | `csrcoo_transpose_dims` | ✔ 1×3 仍为 1×3（未见堆损坏） |
| 6 | zmbdiagdom | 块大小之和 > n 越界 | — | ✖ 无可见后果 |
| 7 | zbitonic_sort | 递归错误被丢弃 | `bitonic_error_dropped` | ✔ 打印错误、返回 0、结果未排序 |
| 9 | zvget_dev | 返回已释放的设备指针 | `vget_dev_dangling` | ✔ cudaMemcpy 报 invalid argument |
| 10 | zcuspaxpy | 未初始化的 buf | — | ✖ cuSPARSE 要求的 buf 大小为 0，结果正确 |
| 11 | zcuspmm | CPU 输入时清理未创建的描述符 | `cuspmm_cpu_uninit_destroy` | ✔ 段错误（设备输入时报 −3003） |
| 12 | zcsr_sort_gpu | CPU 矩阵被换上设备指针 | `csr_sort_gpu_mixed_location` | ✔ memory_location CPU、val 为设备指针 |
| 16 | zge3pt | m = 1 读 dx[−1] | `ge3pt_reads_before_start` | ✔ y = 998（随 x 前的哨兵值变化） |
| 20 | zgeellrtmv | 线程数 > 1024 启动失败不查 | `ellrt_launch_unchecked` | ✔ invalid configuration、y 不变、返回 0 |
| 24 | zjaccard_weights | 下标错 | `jaccard_identical_rows` | ✔ 相同行权重 1/3（应为 1） |
| 25 | zlobpcg_shift | 只移位第 0 行 | `lobpcg_shift_row0_only` | ✔ 第 1 行不变 |

按编号计（子代理把同类文件合并编号，共 31 个编号）：**31 个全部已上 GPU**，其中 29 个确认（#21 ELLPACKT 填充、#10 cuspaxpy 未复现；#6 中 zmdiagdom 确认、zmbdiagdom 未复现）。第十七轮通过导出符号直接调用不在公开头文件中的例程：

| # | 例程 | Lean | GPU 0 |
|---|---|---|---|
| 17 | sync-free 三角求解（等待被注释掉），用正确的 CSC 数组直接调用 | `syncfree_wait_disabled` | ✔ 下双对角 n = 2 ⇒ x = (1, 2)；n = 4096 ⇒ 4095 个分量错（应为 1, 0, 1, 0, …），返回 0 |
| 5 | zcsrget_gpu | `csrget_gpu_dangling` | ✔ 返回的设备指针已释放（cudaMemcpy: invalid argument） |
| 7 | zorderstatistics | `orderstat_last_unchecked` | ✔ {1, NaN} ⇒ 返回 0、结果 NaN；{NaN, 1} ⇒ MAGMA_ERR_NAN |
| 3 | zrowentries（zmatrixchar.cpp） | `rowentries_imaginary_missed` | ✔ DENSE 行 (i, 1) ⇒ max_nnz_row = 1 |
| 14 | ztrisolve | `trisolve_singular_success` | ✔ L = [[0,0],[1,1]]（无解）⇒ x = (1, 0)、返回 0 |

## 3h. 独立复核与非 z 文件（第十九至二十四轮；`operators/magma/reviews/1_INDEPENDENT_REVIEW.md`、`operators/magma/NONZ_INVENTORY.md`）

**独立复核**（新子代理，不依赖先前判定）：
- 清点完整：202 个 src/z、41 个辅助例程、39 个 magmablas 写状态文件、sparse/src 38 + 5、sparse control/blas 66，与文件系统一致。
- 漏网：magmablas/zgetf2_nopiv_devicefunc.cuh 也写状态（变量名 linfo，未被先前的 grep 匹配），读码无缺陷。
- 计数错误两处，已更正：§3 微小 b 为 21/32（原 23/32），§3d 为 6 次静默错误（原 7）。
- 判定不同的 4 条，第十九轮上 GPU：

| 例程 | 发现 | Lean | GPU 0 |
|---|---|---|---|
| zungbr | 内部 zungqr2/zunglq 的 iinfo 不读 | `helper_iinfo_dropped`（注释） | ✔ 故障注入：Q、P 都 info 0、相对误差 52 / 8.6 |
| zlascl_2x2 | ldda 报 −4（应 −6）；lddw 不查 | `lascl2x2_args` | ✔ |
| zresidual | 多右端只写 res[0]；A 0 行除零 | `residual_multi_first_only` | ✔ res = (2, −1)，应为 (2, 4)；除零一项第十九轮的输入（b 也是 0 行）没有走到 l.74 的分支；第二十三轮 A 0 行、b 4 行 ⇒ **SIGFPE** ✔（`residual_div_by_zero`） |
| zcsrsplit | 未排序行越界 | `csrsplit_unsorted_overflow` | 第十九轮用 bsize = 1，越界的代码（l.241–245，需 bsize ≥ 2）没有执行到；第二十三轮 bsize = 2 ⇒ D 中出现列号 681594464 的垃圾元素，随后 `free(): invalid pointer` ✔ |

**非 z 文件**（复核指出的范围缺口：写状态、不由 z 生成；18 个中 15 个在 z 调用链上，另 3 个 dlaln2、dlaqtrsd、dsidi 没有 z 调用方；第二次复核后又补入 dsiinertia/ssiinertia、magma_trisolve.cpp、mmio.cpp）：dlaex0/1/3（含 _m）、dlaln2、dlaqtrsd、dsidi、dstedx(_m)、dmove_eig、getrf_setup_pivinfo.cu、gesvj_kernels.cuh、sort.cu、vbatched_check.cu、vbatched_aux.cu、magma_templates.h 已逐文件判定。dlaex0/1/3、dlaln2、dlaqtrsd 与 LAPACK 一致（干净）。新发现 10 条（#1 range V 特征向量按未缩放界选取、#2 dstedx 子块 info 未映射到全局、#3 adjust_ipiv 在 m > 1024 时不做任何事、#4 分块 gesvj 除以 0 奇异值、#5 dsidi Lower 用上三角、#6 32 位工作区大小溢出、#7 range V 区间闭合、#8 gesvj 批量分块的收敛标志未偏移、#9 列范数未缩放、#10 NaN 破坏排序置换）。

第二十轮（`round20_*`）：第一次编译失败（已修正）；随后 GPU 0 被用户的 ParaCOSM 作业占用，探针排队，在 GPU 0 连续 3 分钟空闲后自动运行：

| # | 发现 | Lean | GPU 0 |
|---|---|---|---|
| 1 | range V 按未缩放的界选特征向量 | `stedx_rangev_unscaled` | ✔ zheevdx 与 zheevdx_2stage，n = 300：m = 145 正确，但 max\|AV − VΛ\|/max\|A\| = 13.5、max\|VᴴV − I\| = 296，info 0 |
| 3 | adjust_ipiv 在 m > 1024 时什么也不做 | `adjust_ipiv_noop_large` | ✔ native LU nb = 600：1200 个无效主元、max\|PA − LU\|/max\|A\| = 52.7、info 0；默认 nb 对照正确 |
| 4 | 分块 gesvj 除以 0 奇异值 | `gesvj_blocked_zero_sigma` | ✔ 零列 ⇒ U 非有限、info 0（n = 32…256，均为分块路径）；A = 0 ⇒ U 全部非有限 |
| 5 | dsidi 逆矩阵错误 | `dsidi_lower_reads_upper`、`dsidi_inverse_wrong` | ✔ 第二十一轮：按文档输入（LAPACK dsytrf 的结果），连无 2×2 主元的 SPD 矩阵也错：max\|AX − I\| = 0.131（Upper）/ 1.16（Lower），info 0；惯性正确。第二十轮的“不确定”判断有误。读码给出的原因（Lower 读上三角）不能解释 Upper 也错，Upper 出错的原因未确定 |
| 7 | range V 区间闭合 | `rangev_closed_interval` | ✔ diag(1..300)，(10, 20] ⇒ m = 11（LAPACK 10） |
| 9 | 列范数未缩放 | `gesvj_norm_overflow` | ✔ 元素 ~1e160 ⇒ S = Inf；~1e-170 ⇒ U 非有限；info 0 |
| 2 | dstedx 子块 info 未映射到全局 | `stedx_subblock_info_unmapped`、`stedx_blocks_indistinguishable` | ✔ 第二十一轮（LD_PRELOAD 让某一次 LAPACK dsteqr_ 报失败）：块 1 第一个叶子与块 2 第一个叶子失败都返回 info = 226 |
| 8 | 批量 gesvj 分块后收敛标志未偏移 | `gesvj_chunk_flags_not_offset` | ✔ 第二十一轮：70000 个 64×64，前 65534 个为单位阵；之后抽查的 47 个全部奇异值错（相对误差至 0.49）、info 0 |
| 10 | NaN 破坏排序置换 | `sort_nan_rank_collision` | ✔ 第二十一轮：n = 4096、一个 NaN ⇒ 下标数组一个重复、一个缺失 |
| 6 | 32 位工作区大小溢出 | `dc_sizes_wrap` | ✔ 第二十四轮：直接调 zstedx_m（ngpu = 1，Z 放在主机内存，GPU 上只放分治工作区）。n = 38,000：返回 −12，来自 magma_dstedx_m——zstedx_m 自己的 lrwmin 回绕为负、检查通过，传下去的 lrwork − n² = 703,483,647 小于 dstedx_m 正确的最小值 1,444,152,001；真实需求 2.89e9 超过 INT_MAX，任何 32 位 lrwork 都不合法（这是正确报告的错误，GPU 工作区分配没有执行到）。n = 53,600：进程段错误；这条路径上 dlaex0_m 的工作区、rwork + n²、lrwork − n² 与拷贝循环的 ldz·j 下标都在 32 位下溢出，究竟哪一个导致崩溃未确定。先前“本机无法触发”的判断有误；第四次复核纠正了本条最初的原因解释 |
#2、#8、#10 已在第二十一轮确认，#6 在第二十四轮确认。

**第二十二轮**（`round22_results.txt`，独立复核余下的条目）：zdiinertia 把 NaN 计为零特征值（2/1/1）✔ `diinertia_nan_as_zero`；zmcsrcompressor 对不拥有数组的矩阵什么也不做、返回 0 ✔ `compressor_not_owned_noop`；zmlumerge 第 4 次主机分配失败 ⇒ double free ✔ `lumerge_double_free`（k = 4 这一行的输出因进程中止而丢失，由中止发生在 k = 3 与 k = 5 之间推断）（用 `fail_malloc_cpu.c` 拦截 magma_malloc_cpu）；头文件与库不一致 ✔ `declaration_mismatches`：magma_zdiagcheck_cpu 有声明无定义，magmablas_zdiinertia、magma_zmlumerge 有定义无声明。

**第二次复核**（`operators/magma/reviews/2_SECOND_REVIEW.md`）：核对第一次复核的更正是否落实、各文件中是否还有过时或矛盾的数字。Lean 总数与论文表格完全一致；第十五至二十一轮抽查的数字都与结果文件一致；指出的约 18 处文字问题大部分已改，余下的由第三次复核找出并改正；指出第十九轮两条“未复现”的测试没有走到目标代码（已在第二十三轮重测并确认）；另发现 zheinertia.cu 也有同样的 NaN 计零：第二十三轮（设备上的 ipiv）diag (1, −1, NaN, 2) ⇒ 2/1/1 ✔。

## 3f. 完整性说明（截至第二十四轮；逐项可查）

“覆盖”在这里的含义：读过代码并判断有无状态/接受逻辑；有发现的写成 Lean 模型（`MagmaDense`/`MagmaSparse`/`MagmaAudit`/`MagmaFgmres`/`MagmaFixed`）；能在单 GPU 上构造输入的跑 GPU 0。每条结论都对应本文件某节的一行与一个结果文件。

| 范围 | 文件数 | 读码 | Lean | GPU 0 |
|---|---|---|---|---|
| `src/z*.cpp` | 202 | 全部（逐文件分类见 `operators/magma/INVENTORY.md`：此前 127、多 GPU/核外 31、补读 3、辅助例程 41） | 有发现的都有 | 见下面“未在 GPU 上运行的条目” |
| 41 个辅助例程 | 41 | 全部，逐文件判定见 `operators/magma/INVENTORY.md` 末节 | 丢弃 iinfo 的一类：`helper_iinfo_dropped` | 有分配且有公开入口的 12 个中 11 个单独故障注入，全部正确报告（zhetrd_hb2st 未单独注入，经 2-stage 驱动覆盖）；zunglq 的头文件声明 (…, dT, nb) 与定义 (…, work, lwork) 不一致，按头文件调用段错误（`unglq_header_mismatch`，第十八轮）；本身不分配、但内部调用会分配的 zunmtr、zunmbr、zunghr（第十一轮）与 zungbr（第十九轮）注入后 info 0、结果错误；无分配的只有参数检查；内部例程经 2-stage 驱动注入覆盖。41 个全部逐文件判定（先前漏掉的 zbulge_applyQ_v2.cpp 已补上） |
| `magmablas/z*` | 165 | 写状态的 39 个全部逐文件判定（§3e），另有 zgetf2_nopiv_devicefunc.cuh（无缺陷）、zdiinertia.cu（NaN 计零，✔ 第二十二轮）与 zheinertia.cu（NaN 计零，✔ 第二十三轮）经两次复核补入，共 42 个；其余 123 个无状态输出 | F1–F11、F13–F15 | F1–F6、F7、F8–F11、F13–F15 |
| `sparse/src/z*.cpp` | 38 | 全部 | 有发现的都有 | 32 个线性求解器 × 6 输入 |
| sparse control / blas（z 文件） | 66 | 全部逐文件判定（`operators/magma/SPARSE_INVENTORY.md`，§3g） | 第二、十五、十六轮的发现 | §3b、§3g；31 个编号全部上 GPU（29 个确认、2 个未复现） |

**未在 GPU 上运行的条目（全部列出）**：
- 第十四轮把剩下的单 GPU 条目全部运行（`round14_results.txt`）：
  - ✔ zlag2c 全局标志跨线程/队列竞争：两线程各 2000 次，溢出矩阵 1889 次报 0、干净矩阵 7 次报 1（`lag2c_shared_flag_race`）；
  - ✔ heevj max_sweeps = 0：64 个矩阵全部 info 0、特征值全错（`heevj_zero_sweeps_success`）；
  - ✔ zmtranspose_cpu 1×1000：段错误（`transpose_cpu_heap_overflow`）；
  - ✔ 截断的 .mtx（声明 3 个元素、实有 2 个）：返回 0、nnz 3，第 3 个是第 2 个的复制（`mtx_truncated_phantom`）；
  - ✖ zlascl2 越界读：把 D 放在未映射页之前也不出错（编译器把读取移到了界检查之内）；
  - ✖ zpotf2 warp 归约：1000 次运行结果逐位相同且正确；
  - ✖ heevj noconv 竞争：12800 次求解全部正确；
  - 辅助例程故障注入：zunmqr、zunmql、zunmlq、zungqr2 正确报告 −113/−112；ztrevc3、ztrevc3_mt 不做设备分配。
- zhegvd 系负 info 映射：GPU 证据表明不可达——第十轮故障注入中 zhegvd 的 k = 3 正是 zpotrf_gpu 的锁页分配，结果是段错误而不是返回 −113（`round10_results.txt`、`round10_backtraces.txt`）。
- 运行了但未复现：F12 原子写 info 的竞争、zlascl2 越界读、zpotf2 warp 归约、heevj noconv 竞争、ICC 分解失败、bombard 残差、ILU(k) 存储不足、shpotrf 分配失败（CUDA 构建中不存在该路径）、zgeqrf_ooc 的 NB = 0 死循环。
- 需要另一种环境：ngpu ≥ 2 的多卡通信与并发的实际运行（只允许 GPU 0；其状态簿记已在 Lean 中对任意 ngpu 证明）；HIP 后端，包括 shpotrf 在 HIP 编译下的分配失败路径（无 AMD 卡）。

## 4. 仍未覆盖的部分及原因

- **多 GPU 例程**：第 38 轮已在 ngpu = 1 下与 LAPACK 做了数值比较（`round38_mgpu/`，108 例；没有新缺陷，M236、M049、M156 得到补充证据）。分布的索引算术已在 Lean 中对所有 n、nb、ngpu 证明（`nlocal_eq_cols_on_all`，HANDOFF §29.66）。第 50 轮用单卡模拟多卡（每个虚拟设备有自己的缓冲区和队列，`magma_setdevice` 为空操作）检验了多卡 Cholesky 的并发顺序，登记 M255、M256（HANDOFF §29.67）；这种模拟能检验队列、事件间的顺序，但测不到跨设备的硬件行为，也会被 legacy 默认流上的操作掩盖。第 52 轮把它做成不依赖模型的通用调度模糊测试（`probes/mgpu_fuzz/`），对 7 个多卡例程做了 5,445 次单点延迟，并发现 M258（HANDOFF §29.69）。未覆盖的只有**实际多卡运行时的行为**（ngpu ≥ 2 时的数据分布与通信）：这些例程的状态逻辑已全部读码（§3c），能用 ngpu = 1 触发的都已在 GPU 0 上运行。2026-11-26 获准用 GPU 1 跑第 25 轮（`round25_results.txt`），但 zpotrf_m 和 zgetrf_m 的双卡结果都是错的（误差 63 和 inf），随后节点驱动挂死。读码显示 info 偏移正确，这更像 GPU 1 故障，**不登记为 MAGMA 缺陷**，要换机器复测（HANDOFF §29.28）。
- **HIP/ROCm 后端**：节点只有 NVIDIA 卡，HIP 专用分支只做了读码。2026-11-26 把 src、magmablas、sparse 里带 `MAGMA_HAVE_HIP` 的 126 个文件过了一遍：绝大多数只是换成 hipBLAS/hipSPARSE 的同名调用或调线程数，状态逻辑与 CUDA 相同（已有发现在 HIP 下同样成立）。只有两处是 HIP 独有的：shpotrf 的两条问题，以及 gesv_rbt 静默关闭精化（`rbt_hip_refine_dropped`）。
- **其余精度变体**：s、c 由 z 源文件生成；已对 4 个关键发现用 c/s 复测，结果一致（`round5_results.txt`）。
- **41 个辅助例程**（`operators/magma/INVENTORY.md` 的 (c) 类；第 36 轮起，其中的 Q 生成与作用、larfb、trevc3 已直接与 LAPACK 比较，发现了 M249）：自身不做状态判断，通过调用它们的驱动的 GPU 运行与故障注入间接覆盖；其中 zunmtr/zunghr/zungtr/zunmbr/zungbr 丢弃的 iinfo 只可能是分配错误，属于 §3d 的同一类。
- **不属于接受逻辑的例程**：BLAS 核、特征值与 SVD 的数值精度本身。审计只看了状态报告，其中单阶段特征值、SVD 驱动和 17 个带区间参数的驱动经子代理检查为干净。
