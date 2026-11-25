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
- 还缺的：
  - D11（归约吞NaN）的形式化。可以写一个 `max_fold` 模型，证明“NaN安全的max ⇔ 判据安全”；
  - D10属于接口层，可以把“返回码 = 内核状态”写成契约的一部分。

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
