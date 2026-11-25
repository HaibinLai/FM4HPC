# 迭代精化（IR）接受逻辑：跨库源码普查（2026-11-24）

## 范围与方法

- **范围**：所有“低精度分解 + 高精度精化”或带精化的求解器，看它们**接受解的判据**、**内层求解器的停止判据**、**失败如何报告**。
- **方法**：只读源码或文档，不编译、不运行，没有用GPU。3个子代理并行读码，主会话逐行复核了标 ✔ 的条目。
- **状态**：标“已运行证实”的7项已于2026-11-25在节点CPU上确认（见“运行确认”一节）；其余预测仍未运行。
- **源码位置**：`/private/tmp/claude-501/-Users-haibin-Desktop-phd01-kernel-fm/1e29af12-88d9-4f68-b168-3cc9880ba18f/scratchpad/irsurvey/`（临时目录，会被清理）。

| 库 | 版本 / 提交 |
|---|---|
| MAGMA | v2.10.0（752e63c） |
| SLATE | master 878439b（2026-08-26） |
| PLASMA | master 548e0bd |
| hpl-ai（ICL参考实现） | 8c7ab18 |
| rocHPL-MxP | a22a987 |
| SuperLU_DIST | 9.3.2（239280c） |
| SuperLU | a8d98be |
| MUMPS | 5.8.1 |
| STRUMPACK | 8.1.0（edc9a72） |
| PaStiX | 6.4.0（e2ecf99） |
| Ginkgo | 796a388 |
| PETSc | 0bedf3a |
| Trilinos | bc7b016 |
| rocm-libraries | a69f369（hipSOLVER/rocSOLVER） |
| cuDSS、oneMKL | 闭源，只看文档 |

## 缺陷类（在原 D1–D9 基础上新增 D10、D11）

| 类 | 含义 | 对应的Lean结果 |
|---|---|---|
| D1 | 接受判据写成“拒绝式”`R > X·cte`，NaN使比较为假，于是被接受 | `IrsContract` 的 `coarse_unsafe` |
| D2 | 容差在 (A,b)→s(A,b) 下不是一次齐次（多乘了‖b‖、或少了‖x‖） | 缩放协变定理 |
| D3 | 绝对常数阈值（1e-18、EPSILON、atol）影响接受或中断 | 同上 |
| D4 | 内层Krylov结束后，不在外层重算真实残差 | `fixed_post` 正是补这一步 |
| D5 | 达到最大迭代次数仍报成功，或没有回退、没有标志 | — |
| D6 | 阈值常数低于FP64残差本身的舍入误差（√n·eps，而残差误差约n·eps） | `IrsContract` |
| D7 | API参数被忽略（trans、nrhs、ld*、x） | — |
| D8 | 转成低精度时不检查范围 | — |
| D9 | 回退不可达、是空壳，或根本不存在 | — |
| **D10** | **外层封装丢掉了状态**：info/返回码写死为成功，或输出参数不写 | — |
| **D11** | **最大值归约吞掉NaN**：`MAX(a,b)=(a>b)?a:b` 或 `if(tmp>v) v=tmp`，只要后面有有限值，NaN就被丢掉。这是D1的另一条触发路径：即使判据本身写对了，NaN也到不了判据 | 待形式化 |

## 总表

符号：● = 有（代码可见）；◐ = 部分或有条件；○ = 无；? = 未读到或不确定。

| 例程 | D1 | D2 | D3 | D4 | D5 | D7 | D8 | D9 | D10 | D11 | 最关键的预测失败输入（置信度） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MAGMA** `dsgesv/dhgesv_iteref`（GMSTRS）✔（已上GPU） | ● | ● | ● | ● | ● | ● | ● | ● | ○ | ? | 已在V100证实（第24节） |
| MAGMA `zcgesv_gpu`（dsgesv） | ● | ○ | ○ | — | ○ | ◐ | ● | ○ | ○ | ? | A=1e10·I、b=1e39：B在FP32中变为Inf，`Inf>Inf` 为假，返回Inf，iter=0、info=0（高） |
| MAGMA `zcposv_gpu` | ● | ○ | ○ | — | ○ | ○ | ○ | ○ | ○ | ? | b含NaN（高） |
| MAGMA `zcgeqrsv_gpu` | ● | ○ | ○ | — | ○ | ○ | ○ | ○ | ○ | ? | b含NaN（高） |
| **MAGMA `zchesv_gpu`** ✔ | ● | ○ | ● | — | — | ○ | ? | **●** | ○ | ? | FP64回退被注释掉，只做 `zlacpy(B→X)`：IR不收敛时返回 **X=B、info=0**（代码路径高） |
| **MAGMA `dshposv_gmres`**（FP16 Cholesky + FGMRES）✔ | ● | ● | ● | ● | ● | ● | ● | ● | ○ | ? | 2^-60·(A,b) 返回 x=0、iter=0；nrhs=2 时第2列为0（高） |
| SLATE `gesv_mixed/posv_mixed` | ● | ○ | ○ | — | ◐ | ○ | ● | ○ | ○ | ○（用 `mpi_max_nan`） | A=1e10·I、b=1e39 返回Inf、iter=0；b含NaN（高） |
| SLATE `*_mixed_gmres` | ● | ○ | ○ | **○** | ○ | ○（nrhs>1 明确报不支持） | ● | ○ | ○ | ○ | b含NaN（高） |
| **SLATE LAPACK接口 `slate_dsgesv`** ✔ **（已运行证实）** | ● | ○ | ○ | — | — | **●**（x被忽略，X与b共用内存） | ● | ● | **●**（`*info = 0;  // todo`） | ○ | A=2I、b=1：迭代改写了b，回退解的是被改写的b，结果 info=0（高） |
| SLATE ScaLAPACK接口 `pdsgesv` | ● | ○ | ○ | — | — | ● | ● | ● | ● | ○ | A奇异仍报 info=0（高） |
| **PLASMA `zcgesv`（dsgesv）** ✔ **（已运行证实）** | ● | ○ | ○ | — | ○ | ○ | ○ | **●** | ◐ | **●**（`core_dzamax`） | FP32 LU奇异或转换溢出后，后续所有任务被跳过，**FP64回退也不执行**，X没有写回（高） |
| PLASMA `zcposv` | ● | ○ | ○ | — | ○ | ●（转换整个A，读了未引用的三角） | ○ | ● | ◐ | ● | FP64下SPD、FP32下不正定，或未引用的三角含Inf（高） |
| PLASMA `zcgbsv` | ● | ○ | ○ | — | ○ | ? | ? | ? | ? | ● | 未细查 |
| LAPACK `dsgesv`（参考实现） | ● | ○ | ○ | — | ○ | ○ | ○ | ○ | ○ | ◐（IDAMAX） | 已证实；PR #1501 已提出修复 |
| **STRUMPACK `SparseSolverMixedPrecision`** ✔ **（已运行证实）** | ● | ● | ● | ● | ● | ● | ● | ● | **●**（`// TODO check convergence` → 总是SUCCESS） | ? | (A,b)×1e-12：x=0，返回SUCCESS（高） |
| STRUMPACK 经典IR / `SparseSolver` **（已运行证实）** | ● | ● | ●（atol=1e-10） | ◐ | ◐ | ○ | — | ◐（BiCGStab中断时报SUCCESS） | ○ | ? | ‖b‖₂≤1e-10 时返回 x=0（高） |
| **PaStiX GMRES精化** ✔ **（已运行证实）** | ●（跑满后报SUCCESS） | ○ | ●（1e-50） | ○ | ● | ○ | ●（mixed模式直接cast） | ○ | ●（总是 `PASTIX_SUCCESS`） | ? | **新bug**：`normx` 只在循环外算一次，x0=0 时每次重启都用 r=b，x不断累加，约 k·x*（代码高，需运行） |
| PaStiX 简单IR / BiCGSTAB | ◐ | ○ | ○ | ●（BiCGSTAB不复核） | ● | ○ | ● | ○ | ● | ? | mixed模式下RHS溢出：`pastixRhsDoubletoSingle` 释放了调用者的b（疑似use-after-free） |
| SuperLU_DIST `psgsrfs_d2`（单精度LU + 双精度IR） **（已运行证实）** | ● | ○ | ○ | — | ◐（只有误差界） | ◐（只是文档不一致） | — | ○ | ○ | **●**（`SUPERLU_MAX`） | A=I、b=(NaN,1)：CONVERGED，误差界约6e-8，X[0]=NaN（高） |
| SuperLU_DIST `pdgsrfs`（+ 新的 `SLU_GMRES` 内层） | ◐ | ○ | ●（GMRES写死 atol=1e-14） | ○ | ○ | ● | — | ○ | ○ | ● | (A,b)×2^-50：内层GMRES直接返回 d=0，精化不起作用 |
| SuperLU `dgsrfs` **（已在节点CPU上运行确认）** | ◐ | ○ | ○ | — | ○ | ●（`SLU_DOUBLE` 选项是空的） | — | ○ | ○ | ● | **已证实**：两个解耦的三对角块（n=1000）且 b[0]=NaN/Inf 时，x有500个NaN，却返回 info=0、berr=6.94e-17（与有限基线相同）。ferr为NaN，没有被屏蔽。见 `experiments/ir_survey/` |
| MUMPS 5.8.1 IR（ICNTL(10)；没有混合精度功能） **（5.8.2上运行证实）** | ●（经 `idamax`） | ○ | ○ | — | ◐ | ●（nrhs>1 等情况静默关闭精化） | — | — | ◐ | ● | A=I、b=(NaN,1)：OMEGA=0，判为收敛，INFO=0（中高，取决于BLAS） |
| hpl-ai 参考实现（GMRES-IR） ✔ | ●（GMRES内部） | ●（`/‖b‖`） | ◐ | ◐ | ● | — | ● | —（基准程序） | ●（`main` 总返回0） | ? | 驱动程序的最终检查对NaN安全，但FAILED只打印到stdout |
| hipSOLVER `DnXXgesv`（AMD后端） ✔ | — | — | — | — | — | ●（`niters` 从不写） | — | — | ● | — | 从cuSOLVER移植、靠 `niters<0` 判断回退的代码会读到未初始化的值（高） |
| Ginkgo `Ir` | ○ | ○ | ◐（仅absolute基线） | ○ | ●（静默，`apply` 返回void） | ○ | ● | — | ● | ? | b含NaN：只配了ResidualNorm时无限循环（高） |
| IterativeRefinement.jl | ●（警告判据） | ○ | ○ | — | ◐ | ○ | — | — | ○ | ? | A含NaN：不发警告（中） |
| cuDSS（文档） | ? | ○ | ○ | ? | ◐（默认 IR_TOL=0 时不检查） | ◐（批量时只检查一个系统，文档写明） | ? | — | ○（有 `IR_FAILED`） | ? | — |
| oneMKL `dsgesv`（文档） | ? | ○ | ○ | — | ○（iter=−31 时回退） | ○ | ○（iter=−2 时回退） | ○ | ○ | ? | 闭源；若照抄参考LAPACK，就有D1 |

### 安全或基本安全的设计（反例同样重要）

- **rocHPL-MxP**：`if (res < thrsh || iter == maxits-1) break`，用HPL的缩放残差，NaN安全、缩放不变，最终检查有 kpass/kfail 计数。
- **Trilinos Belos FixedPoint（GMRES-IR）**：判据写成 `>tol / <=tol / 否则抛 NaNError`；外层显式重算 R = b − A·Z；`MaxItersReached` 是单独的返回码。唯一的问题：内层求解的返回值被丢弃。
- **PETSc `KSPConvergedDefault`**：显式检查 `PetscIsInfOrNanReal` 并返回 `KSP_DIVERGED_NANORINF`。但默认 abstol=1e-50 是绝对阈值（D3）：(A,b)×2^-200 时直接返回 x=0，报 `CONVERGED_ATOL`。
- **SLATE `*_mixed_gmres`**：每个周期开头重算真实残差，itermax用完就回退。只剩NaN（D1）和转换（D8）两个问题。
- **cuDSS**：有显式的 `CUDSS_STATUS_IR_FAILED`。

## 运行确认（2026-11-25，节点CPU，详见 `experiments/ir_survey/README.md`）

以下 7 项全部证实，其中 3 项比预测的更严重：

| 例程 | 实测结果 |
|---|---|
| SuperLU `dgsrfs` | x有500个NaN，返回 info=0，berr与有限基线相同（6.94e-17）；ferr为NaN |
| SuperLU_DIST `psgssvx_d2` | **比预测更严重**：b[0]=NaN 时，info、两个前向误差界（1.03e-7）和berr（2.96e-16）**全部与干净基线相同**，而x有500个NaN |
| MUMPS 5.8.2 | b[0]=NaN 时 INFOG(1)=0、omega1=1.94e-16（与基线相同）；NaN在其他行时给出警告 INFOG(1)=2 |
| PaStiX GMRES | x0=0（文档允许）时跑满250次迭代，x≈249·x_true，残差为249，返回 `PASTIX_SUCCESS` |
| PLASMA `dsgesv` | b含NaN或FP32主元增长溢出时：返回0、iter=0、X全为NaN；FP32奇异或b溢出时：返回错误、iter=−31（声称已回退），但X从未写入 |
| STRUMPACK | **比预测更严重**：(A,b)×1e-12 时 x≡0、SUCCESS，**默认的双精度 `SparseSolver<double>` 也一样**；b含NaN时 x≡0、SUCCESS |
| SLATE `slate_dsgesv` | **比预测更严重**：良态矩阵也返回 info=0、iter=23，x没有写入，b被改写为0，解在哪个数组里都没有；奇异矩阵同样 info=0 |

## 规律（论文可以用的发现）

1. **D1几乎普遍存在。** 25个左右的实现里，只有rocHPL-MxP、Trilinos、PETSc、Ginkgo、SLATE的GMRES版和hpl-ai的驱动程序检查做对了。根源有两条：
   - 判据写成拒绝式，从LAPACK `dsgesv` 一路继承到MAGMA、SLATE、PLASMA；
   - D11，最大值归约吞掉NaN（SuperLU、PLASMA、MUMPS经idamax、LAPACK IDAMAX）。
   上游 PR #1501 只修了参考LAPACK的第一条路径。
2. **内层用Krylov的混合精度IR，明显比经典IR更不可靠。** MAGMA的两条GMRES路径、STRUMPACK mixed、PaStiX GMRES、cuSOLVER IRS（黑盒）都**信任内层的递推残差**，结束后不复核（D4）。外层复核的只有SLATE的GMRES版和Trilinos。这正是 `fixed_post` 的主张：验证结果，而不是迭代。
3. **封装层丢状态（D10）是一个独立的类别，以前没人归纳过。** 例子：SLATE的LAPACK/ScaLAPACK接口写死 `info=0`；STRUMPACK mixed总是返回SUCCESS；PaStiX总是返回SUCCESS；hipSOLVER不写 `niters`；hpl-ai的 `main` 总返回0。内核里的检查做得再好，到了调用者那里也会丢失。
4. **绝对阈值（D3）随处可见。** 例子：MAGMA的1e-18和EPSILON；STRUMPACK的atol=1e-10；SuperLU_DIST GMRES的1e-14；PETSc的1e-50；PaStiX的1e-50。它们都会让结果依赖输入的绝对尺度，而2^k缩放预言机能统一检出这一类。
5. **转换到低精度时不检查范围（D8）也很常见**（MAGMA的zcgesv路径、SLATE、PaStiX、STRUMPACK、hpl-ai、Ginkgo）。LAPACK和MAGMA的部分例程查了A，却没有查B或残差。
6. **回退本身也会坏（D9）**，表现为三种形式：MAGMA `zchesv` 的回退是空壳；PLASMA的任务失败会连带跳过回退；MAGMA的GMRES路径回退不可达。

## 与契约和证明的对应

- 统一契约：返回 iter ≥ 0（或SUCCESS），就意味着 ‖b − A·x‖ < c·‖A‖·‖x‖ 且 x 有限。
- 三类结果覆盖上面的缺陷类：
  - `coarse_unsafe`、`coarse_safe`：判据形式与NaN，对应D1；
  - 缩放协变：对应D2、D3；
  - `fixed_post` 与 `fixed_nan_fallback`：对应D4、D5、D9。
- **已补上（2026-11-25，HANDOFF §26）**：
  - `MaxFold.lean`：D11的定理，并给出“判据安全 ⇔ `<` 形式且归约传播NaN”（`detects_iff`）；
  - `IrDriver.lean`：契约的充要条件 C1–C4（`contract_iff`）。D10（封装层丢状态）对应C4，D9对应C2/C3，D4对应C1，D1和D11经 `c1_forces_accept_propagate` 也归入C1。

## 各库自带测试能否发现这些缺陷（2026-11-25，只读，没有编译或运行）

**问题**：
1. 库自带的测试（默认列表、CI，以及测试程序已有的选项）会不会触发这些缺陷？
2. 触发后，测试的判定能不能发现？

**方法**：
- 2个子代理分别读ICL库（MAGMA、SLATE、PLASMA）和稀疏库（SuperLU、SuperLU_DIST、MUMPS、PaStiX、STRUMPACK）的测试程序、测试列表和CI脚本；
- 主会话逐条复核了标 ✔ 的证据；
- 源码版本与上面的版本表相同。

| 库 / 缺陷 | 默认测试会触发吗 | 触发后判定能发现吗 | 证据 |
|---|---|---|---|
| MAGMA GMSTRS（`dsgesv/dhgesv_iteref`） | **否**：`testing_dxgesv_gpu` 不在 `run_tests.py` 中；b 固定为 U(0,1)；生成器没有NaN类型 | NaN：能。**‖b‖≤1e-18 → x=0：不能**，因为度量是 ‖r‖/(N‖A‖)，不含 ‖x‖ | `testing_dxgesv_gpu.cpp`、`magma_util.cpp:192` |
| MAGMA `dsgesv_gpu` 接受NaN ✔ | **否**：默认列表只有 `testing_zcgesv_gpu -c`（`run_tests.py:813`）；没有NaN，也没有增长矩阵 | 能：`status += !(error < tol)`（`testing_zcgesv_gpu.cpp:245`）；info、iter只打印 | 同左 |
| MAGMA `zchesv_gpu` 空壳回退 | **没有任何测试调用** | — | grep `testing/` |
| MAGMA dshposv GMRES | 否；手动加 `--nrhs 2` 可以触发第2列为0 | 能 | `testing_dshposv_gpu.cpp` |
| MAGMA FGMRES 忽略 trans/nrhs | 否；手动加 `-T`、`--nrhs` 可以触发 | 能 | `testing_dxgesv_gpu.cpp:131-136` |
| SLATE LAPACK接口 `slate_dsgesv` ✔ | **没有任何测试或示例调用**（只在定义文件里出现） | — | `lapack_api/lapack_gesv_mixed.cc:89-90` |
| SLATE `gesv_mixed/posv_mixed` | 否：默认 rand、n=100；**有选项可触发**：`--matrixB rand_large`、`--matrix gfpp`（未运行验证） | 能：`okay = (error <= tol)`，范数用 `max_nan` | `test_gesv.cc:377` |
| SLATE ScaLAPACK接口 `pdsgesv` | 只有一个没进构建的示例 | **不能**：不读info，只打印残差 | `example_pdsgesv.c` |
| PLASMA `dsgesv` 接受NaN ✔ | **否**：`dsgesv` 不在 `run_tests.py` 中（LU组只有 `gesv` 等）；A、b为 U(0,1) | 能：`residual < tol`；iter不查 | `test_zcgesv.c:152-170` |
| PLASMA 任务失败后跳过回退 | 否：`--zerocol` 让A在FP64下也奇异，不是“只在FP32下奇异” | 间接能：与 `LAPACKE_zcgesv` 的返回值比对 | `test_zcgesv.c:174-186` |
| SuperLU `dgsrfs`（D11） ✔ | 否：x_true 全为1，b=A·x_true，没有NaN选项 | **不能**：判定是 `result[i] >= THRESH`（NaN通过）；**检查函数自身也用 `SUPERLU_MAX`**（`dgst02.c:121`、`dgst07.c:145`） | `ddrive.c:473` |
| SuperLU_DIST `psgssvx_d2`（D11） ✔ | **没有注册的测试覆盖d2路径**：`psdrive` 默认 NOREFINE，且不在ctest里；`pstest_ci` 要 `-B n -i 2` 才走d2，没有注册这种运行 | 示例驱动**不能**：`psinf_norm_error` 用 `SUPERLU_MAX`，并 `return 0`。`ptest_ci` **能**：明写“NaN anywhere fails”，判定 `value <= tol`，但它没有跑d2 | `psutil.c:1467`、`ptest_ci.c:550-567` |
| MUMPS 精化（ICNTL(10)/(11)） ✔ | **没有测试**；示例都不开精化（默认0） | **不能**：只查 `INFOG(1) < 0`；MATLAB示例是 `if norm(...) > sqrt(eps)`（NaN通过） | `examples/` grep 无 ICNTL(10) |
| PaStiX GMRES x0=0 | 否：所有驱动 x0=b（`memcpy(x, b)`），或者从LU解开始 | 部分：`refinement.c` 在 `NBITER >= ITERMAX` 时判失败，能发现；`simple.c` 靠 `spmCheckAxb`（`spm/` 子模块为空，未读到） | `refinement.c:140`、`simple.c:120` |
| STRUMPACK atol（小尺度 → x≡0） | 否：矩阵为O(1)；平衡化测试只缩放 2^±8、2^−10 | 稀疏测试能（缩放残差为1，超过阈值）。**混合精度测试 `test_SPD_mixedPrecision` 只打印，总是 `return 0`** ✔ | `test_SPD_mixedPrecision.cpp:82-90,160` |
| STRUMPACK b含NaN | 否 | 不可靠：`if (res > tol)`，加上 `max_scaled_residual` 用 `std::max` 丢掉NaN行；x≡0 能被有限行抓到，但“部分NaN、其余正确”的x会通过 | `CSRMatrix.cpp:994` |

**结论**（可用于论文RQ3）：
1. **16个缺陷中，没有一个会被任何库的默认测试或CI触发。** 没有一个测试往b里放NaN或Inf、把(A,b)缩放到1e-12以下，或给PaStiX零初值。b几乎总是U(0,1)，或者等于 A·全1。
2. **6个例程完全没有带判定的测试**：
   - MAGMA `zchesv`；
   - SLATE `slate_dsgesv`、`pdsgesv`；
   - SuperLU_DIST d2路径；
   - MUMPS精化；
   - STRUMPACK混合精度（只打印）。
   其中4个（`slate_dsgesv`、SuperLU_DIST d2、MUMPS、STRUMPACK mixed）正是我们已运行确认“报成功、答案错”的例程；`zchesv`、`pdsgesv` 只读码，未运行。
3. **测试的判定和bug犯同样的错**：
   - SuperLU的检查函数、SuperLU_DIST的 `psinf_norm_error` 用的就是丢NaN的 `SUPERLU_MAX`；
   - STRUMPACK用 `std::max`；
   - 判定又写成 `>=`/`>`（拒绝式）。
   所以即使有人加上NaN输入，SuperLU、MUMPS、STRUMPACK 的测试也会判“通过”。这是D1/D11在测试预言机里的再现。
4. **生成器有危险选项，但从未用于IR例程**：MAGMA 的 `_ofl/_large`、SLATE 的 `gfpp/--matrixB`。两个生成器都没有NaN/Inf类型。
5. **正面例子**：
   - SuperLU_DIST 的 `ptest_ci.c` 专门写成对NaN安全，可惜没跑d2；
   - MAGMA、SLATE、PLASMA 的判定都是 `<`/`<=`（NaN安全）。它们漏掉问题，是因为输入覆盖不到。
6. **MAGMA的测试度量不是后向误差**（没有 ‖x‖），所以“小‖b‖返回x=0”在它的测试里会判通过。

**边界**：
- 只读码，没有运行；SLATE `gfpp`/`rand_large` 能否触发没有实测；
- PaStiX 的 `spmCheckAxb` 未读到；
- 判定对NaN安全，依赖所链接BLAS/LAPACK的 `xlange` 传播NaN（参考实现会传播，MKL未查）；
- cuSOLVER、LAPACK参考实现的测试集不在本次范围内。

## 下一步：用CPU就能确认的部分（不需要GPU）

PLASMA、SLATE（Host target）、SuperLU/SuperLU_DIST、MUMPS、STRUMPACK、PaStiX、hpl-ai、Ginkgo（reference executor）、PETSc 都能在CPU上运行。按“影响 × 置信度”排的优先级：

1. **PaStiX GMRES的 `normx` bug**：给出有限的错误解且报SUCCESS，是新bug。
2. **MAGMA `zchesv` 的空壳回退**：属于MAGMA，CPU上无法运行（需要GPU），等GPU空闲。
3. **PLASMA `dsgesv`**：任务失败跳过回退；b含NaN、b=1e39。
4. **SLATE LAPACK接口**：X与b共用内存，info=0。
5. **STRUMPACK mixed**：总是SUCCESS，小尺度时返回x=0。
6. **SuperLU_DIST `psgsrfs_d2`、MUMPS**：NaN屏蔽。
7. hpl-ai、Ginkgo、PETSc 的 atol 问题。

在哪里运行（Mac本地，或节点CPU、数据放 /ssd/haibin）需要用户决定。确认之后的报告属于对外发布，由用户自己提交。
