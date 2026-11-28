# MAGMA 2.10.0 缺陷挖掘：交接文档（截至 2026-11-27，第 52 轮）

给接手的 agent：本文说明现在做到了哪里、怎么继续、有哪些不能碰。细节见下面引用的文件，每一轮的完整记录在 `HANDOFF.md` §29.28–§29.69。

## 1. 目标与现状

- **目标**（用户的原话）：“持续深挖这个 HPC 问题，尽可能把 MAGMA 剩下的算子都建模去挖 bug”。具体是用 Lean 建模，加上在 GPU 上做差分测试，找出 MAGMA 2.10.0 的缺陷，并研究这些缺陷主要集中在哪里。
- **现状**：
  - `operators/magma/FINDINGS.csv` 共 **289 条**（编号 M001–M290，其中 M244 已撤回，号码空缺不再使用）；
  - Lean 共 **509 个 theorem**，`lean/logs/axioms.txt` 467 行，没有 sorryAx；
  - 论文 `paper/DRAFT.md` 的 Appendix A 总计 5,982 行；
  - **什么都没有 commit**：只有用户要求时才 commit。
- roxy 上能测的部分已经基本测完。第 26–42 轮依次覆盖了：
  - 稠密驱动、结构化病态输入、稀疏迭代求解器、批量 Jacobi 特征值与 SVD、混合精度迭代精化；
  - 跨 queue 竞争、未对齐的 ldda、批量与 vbatched BLAS、41 个辅助例程；
  - 两阶段特征值链路、RBT、多 GPU 驱动在 ngpu = 1 下的数值；
  - 复数精度复核、按已知写法的全库扫描、alpha = 0、多卡分布的索引。

## 2. 环境

### roxy（主要的实验机）

- 访问：`ssh roxy`。这是 Windows 11，RTX 5070 Ti（sm_120，16 GB），是**用户的个人台式机**：桌面程序占用 7–12 GB 显存，跑大任务前先用 `nvidia-smi` 看剩余显存。
- 环境位置：`D:\lean4hpc`。
  - `magma_env`：pip 安装的 nvcc 13.5、CUDA 13 wheels、cmake/ninja；
  - `openblas`：OpenBLAS 0.3.34；
  - `magma\build\lib\magma.lib`（静态库）和 `magma_sparse.lib`；
  - `magma\probes`：probe 源码；`magma\bin`：编译出的可执行文件。
- **不要删除 D:\lean4hpc 下的任何东西，也不要提出清理**（包括 venvs 和 MSVC Build Tools）。**安装任何新的系统软件之前（WSL2、CUDA 安装包、驱动）先问用户。**
- 编译与运行：
  - `ssh roxy 'cmd /c "D:\lean4hpc\magma\build_probe.bat <name>"'`，编译 `probes\<name>.cpp`，输出中出现 `PROBE_BUILD_EXIT 0` 才算成功；
  - `ssh roxy 'cmd /c "D:\lean4hpc\magma\run_probe.bat <name>.exe <args>"'`；
  - 需要设环境变量时写成 `cmd /c "set X=1&& D:\...\run_probe.bat ..."`（`&&` 前不要留空格）；
  - 长任务用 WMI 启动（`Invoke-CimMethod Win32_Process Create`），这样 ssh 断开后任务还在跑。
  - 输出是 CRLF 换行，解析前先 `tr -d '\r'`。
- 补丁验证的做法：把 MAGMA 源文件复制一份并修改，与 probe 一起编译，链接在 magma.lib 之前，用来覆盖库里的同名符号。
  - .cpp 用 `cl`，见 `scripts/roxy/build_fix31.bat`、`build_fix36.bat`；
  - .cu 先用 nvcc `-arch=sm_120` 编成 .obj，见 `build_fix31b.bat`、`build_fix35.bat`、`build_fix37.bat`。
- Windows 是 LLP64，`sizeof(long) = 4`（M240 就是因此而来）。

### s82 / eightsocket（目前不可用）

- 状态：GPU 1 从 11-21 起就有故障；GPU 2 时好时坏；GPU 3–7 被别人卡死的进程占着。在管理员重置之前，**什么都不要在上面运行，包括 GPU 0**。
- 恢复之后的规则：只能用 GPU 0；数据放在 `/ssd/haibin`，不要放 `/home`，用完删掉自己产生的数据；不要碰用户的其他目录和进程。

### 本地

- Lean 构建：`cd lean && PATH=<repo>/.toolchain/elan/toolchains/leanprover--lean4---v4.36.0/bin:$PATH lake build`。
- MAGMA 源码副本（只读，用来读码和 grep）：`/private/tmp/claude-501/-Users-haibin-Desktop-phd01-kernel-fm/1e29af12-88d9-4f68-b168-3cc9880ba18f/scratchpad/magma_full/magma-2.10.0`。这个目录在 scratchpad 里，可能被清理；如果找不到，就从 roxy 的 `D:\lean4hpc\magma\magma-2.10.0` 取。

## 3. 文件结构与“登记一条发现”的流程

| 文件 | 用途 |
|---|---|
| `operators/magma/FINDINGS.csv` | 每行一个独立缺陷，z/c/d/s 的副本合并成一行。列：id, routines, source_file, subsystem, defect_class, trigger, impact, evidence, lean_theorems, where_recorded |
| `operators/magma/COVERAGE_zh.md` | 每个例程的覆盖情况和证据。§4 是“仍未覆盖的部分” |
| `operators/magma/BUG_DISTRIBUTION_zh.md` | 缺陷分布的分析，6b–6f 是各轮的补充。统计用 `python3 operators/magma/findings_stats.py <magma源码根目录>` 计算 |
| `HANDOFF.md` §29.x | 每轮的记录 |
| `experiments/magma_gpu/probes/roundNN_*.cpp` | 每轮的 probe |
| `experiments/magma_gpu/results/roundNN_*/` | 每轮的 CSV 和 `TRIAGE.md` |
| `lean/OptimiKahan/Magma/...` | Lean 模型，模块划分见 `lean/OptimiKahan/README.md`。近几轮的新内容都在 `Magma/Dense/Differential.lean` 的 §34–§36f |
| `paper/DRAFT.md` Appendix A | Lean 文件的行数和 theorem 数 |

登记一条新发现时：
0. **先查重**：写探针之前，按源文件名在 FINDINGS.csv、`SPARSE_INVENTORY.md`、`NONZ_INVENTORY.md`、COVERAGE 里搜一遍。第 51 轮的四个候选中有三个早已登记。
1. 在 FINDINGS.csv 末尾追加一行，编号接着 M290 往下（M291…）。用 Python 的 csv 模块写，因为字段里常有逗号。如果只是已有缺陷在新文件或新例程中的延伸，就扩充原来那一行，不要新开条目。
2. 在 Lean 中加模型和 theorem，同时加 `#print axioms`，然后 `lake build`。
3. 重新生成 `lean/logs/axioms.txt`。格式是 `文件:行:0: '名字' depends on axioms: [...]`。做法是：对每个含 `#print axioms` 的 .lean 文件（**不要包括 README.md**）运行 `lake env lean`，再按 `#print` 所在行配对、加上前缀。注意：在 zsh 里不要写 `for f in $files`（zsh 不做分词），要直接写 `for f in $(grep ...)`。
4. 更新 paper 中 Differential.lean 那一行和总计（行数、theorem 数）。
5. 更新 COVERAGE（加一行）、HANDOFF（新增一节）、BUG_DISTRIBUTION（补充一段，并更新文末的总条数），以及该轮的 TRIAGE.md。
6. 重新跑 `findings_stats.py`，确认总条数。

## 4. 方法与判定规则（这几轮的教训）

- **先怀疑测试程序，再怀疑库**。这几轮出现过的测试程序问题：
  - batched 接口所有矩阵只能用同一个 ldda；
  - 参考 BLAS 的 gemv 在 m 或 n 为 0 时不缩放 y；
  - larfb_gpu 要求 V 中的 0 和 1 显式存储；
  - range V 的区间端点落在重复特征值上，期望的个数就有歧义；
  - CSV 字段里带逗号；
  - 未初始化的 info 数组；
  - 在 Python 里取错了列下标。
- **判定是否为缺陷，要看成文的契约**（MAGMA 自己的文档，或参考 BLAS/LAPACK 写明的语义）。只是与参考实现的行为不同，就记为观察，不登记。例如 trevc3 不支持 lwork = −1 的查询，以及第 41 轮 alpha = 0 时仍读 A。
- **确认根因的手段**：
  - 用补丁链接进去验证；
  - 用 `CUDA_LAUNCH_BLOCKING=1` 对比运行，区分竞争和确定性错误；
  - 用环境变量开关加不同的同步点；
  - 用结构化输入（A = I、单位阵、哨兵值）。
- **检查“不应改动的部分”**：另一侧三角、填充行、ld 之外的元素都填哨兵，然后逐位比较。M246 就是这样发现的。
- **范围要实测**：不要想当然地写 [zcsd]。第 39 轮发现 M242 在 c 精度下不出现、M249 只影响实数版，已据此收窄。

## 5. 主要发现（第 31–50 轮新增，按严重程度）

| 编号 | 内容 | 状态 |
|---|---|---|
| M248 | magmablas 的 [zcsd]hemm/symm（含 batched、vbatched）对任何输入都算错：`fetch()` 传的是基指针，而宏期望 offs_dA | 用补丁确认 |
| M242 | [sdz]getrf_native：getf2_native 融合内核跨线程块读数据时命中过时的 L1，`ldda·sizeof % 32 ≠ 0` 时 LU 错误；dsgesv/zcgesv 内部固定 lddsa = n，因此受影响 | 用 `__ldcg` 补丁确认 |
| M243 | dsgesv/zcgesv、dxgesv_gmres 的 FP64 回退：lacpy 与 getrs 在不同的 queue 上，中间没有同步 | 用同步点和串行运行确认 |
| M246 | herk/syrk（批量、vbatched 和非批量）改写“不引用”的另一侧三角，potrf_batched/vbatched 因此改写了上三角 | GPU 确认 |
| M249 | [ds]trevc3(_mt)：lwork 取文档推荐值时，A/S 模式的特征向量错误（漏掉了 LAPACK 的 OVER 条件） | 用补丁确认 |
| M250 | gesv_rbt 在 n ≤ 3 时出错（gerbt 的 Uᵀb 内核对 n < 2 提前返回） | 用补丁确认 |
| M245 | 批量和非批量的 gemm/herk 在 k = 0 时不乘 beta | GPU 确认 |
| M247 | trsm/trmm（批量，以及非批量的 trmm）在 alpha = 0 时不把 B 置零 | GPU 确认 |
| M252 | sparse：CSR→COO 把行下标写进 row，而使用方读 rowidx，往返后所有元素都进了第 0 行 | 用补丁确认 |
| M253 | ELLRT SpMV 的 warp 内归约没有同步，一行超过 6 个非零元就算错 | 用补丁确认 |
| M254 | CSR↔CSRLIST 对空行越界写（堆损坏） | 用补丁确认 |
| M251 | sparse：SELL-P/ELLRT 参数在 zmconvert 回退路径和 zmtranspose 中丢失，整数除零导致进程崩溃；需要 Aᵀ 的 6 个求解器用 SELLP 必崩 | 用补丁确认 |
| M255 | 多卡 Cholesky（z/c，ngpu ≥ 2）：设备对角块缓冲 dlP 每 ngpu 步轮换，stream 2 的 trsm 读它与 stream 1 的下一次拷贝改写它之间无序，因子静默出错 | 单卡模拟多卡 + 单点延迟确认，加一个事件等待后消失 |
| M275 | magmablas_laswp_sym：超过 32 个主元时，内核按绝对下标读块内的相对 ipiv，越界 | GPU 确认 |
| M276 | magmablas_laswp_sym：同一次启动中多个主元之间存在线程竞争，结果错且不确定 | GPU 确认（两次运行结果不同） |
| M274 | geqr2x{,2,3,4}_gpu：文档写的 dwork 偏小，按文档分配会越界写，最多 50n 个 double | GPU 确认（保护区） |
| M289 | magma_zhetrs_gpu（Lower）：前代的融合内核不带列下标，只解第 0 列，nrhs ≥ 2 时其余列全错 | GPU 确认 + 补丁确认 |
| M290 | magma_zhetrs_gpu：<<<1, nrhs>>> 在 nrhs > 1024 时启动失败且不检查，所有列错、info = 0 | GPU 确认 |
| M287 | geqr2x2/3/4（文档 N ≥ 0）与 larfbx（k > 128）：trmv 内核的 sum[128] 以 k 个线程启动，n ≥ 130 时非法内存访问且 info = 0 | GPU 确认（边界正好在 129/130） |
| M288 | geqr2x 的公开辅助函数：larfbx 算 Hᴴ 并读 T 的下三角，larfx 的 tau 约定相反、不更新 xnorm，larfgx 把 beta 存到 dA | GPU 确认（larfbx）+ 读码 |
| M286 | magmablas_lacpy_sym_in/out：文档写 rows[i] 与 rows[n+i] 交换，代码按交错二元组做赋值（Aasen 用的正是后者），按文档调用会出错 | 读码 |
| M285 | applyprecond_left 的 MagmaTrans 分支（含义是共轭转置）：Jacobi 乘 D⁻¹ 而不是 conj(D⁻¹)；复数矩阵上 LSQR / PBiCG + Jacobi 不收敛 | GPU 确认 + 补丁确认 |
| M284 | applyprecond_left/right 的 MagmaTrans 分支：ICC 的左右求解对调，PBiCG / PQMR + ICC 的影子系统用了 L⁻¹L⁻ᵀ | GPU 确认 + 补丁确认 |
| M282 | 复数 IDR(s) 的 6 个变体：角度保护用 \|Re(tᴴr)\| 而不是 \|tᴴr\|，A = iS（S Hermitian 正定）时全部不收敛 | GPU 确认 + 补丁确认（20/20，对相位不变） |
| M283 | 稀疏 magma_zqr：设备上的输入 A（文档为 [in]）被原地改写成 Q | GPU 确认 |
| M279 | sparse spmv_shift（CSR/ELL/ELLPACKT 三个内核）：文档 y = α(A − λI)x + βy，实际算 αAx − λx + βy | GPU 确认 |
| M280 | magma_zmslice：末尾几片起点越过 n，片大小为负，读 A.row 越界；返回 −112 或不确定地返回 0；出错后输出里留着已释放的指针，magma_zmfree 导致堆损坏 | GPU 确认 |
| M281 | magma_zmconjugate：不检查格式，ELL 矩阵触发非法内存访问仍返回 0 | GPU 确认 |
| M277 | sparse magma_z_spmm（稀疏 × 稀疏）：CUDA ≥ 12 分支把 beta = NULL 传给 cuSPARSE，每次返回 −3003；文档里的 alpha 从不使用 | GPU 确认（CUDA 13） |
| M278 | magma_zm_27stencil：只清除 x 方向的边界耦合，y 方向回绕的耦合被保留（n = 3 多 84 个，n = 5 多 312 个） | CPU 确认（与预测完全一致） |
| M271 | sparse LOBPCG：第 0 个特征对一收敛就停止并报成功，其余未收敛 | GPU 确认 + 补丁确认 |
| M270 | sparse TFQMR 四个变体：影子内积顺序反了，非 Hermitian 复系统不收敛 | GPU 确认 + 补丁确认 |
| M268 | magmablas_[zc]geam：ConjTrans 不取共轭，按 Trans 计算 | GPU 确认 |
| M269 | magmablas_[zc]gemm_reduce：文档写 AᵀB，实际算 AᴴB | GPU 确认 |
| M266 | potrf_vbatched：max_n 超过交叉点时，非正主元的位置报成面板内的局部值 (c mod 128) + 1 | GPU 确认 |
| M267 | zpotrf_vbatched：360 ≤ max_n ≤ 448 时整批都不分解，info_array 为 0，返回 −33 | GPU 确认 |
| M261 | unmqr_gpu：dT 按 geqrf_gpu 文档大小分配时，larfb 工作区写出界 (nw − roundup(k, 32))·nb 个元素，改写相邻分配 | GPU 确认（保护区） |
| M262 | unmqr_2stage_gpu：side = Right 没有实现（仍按 Left 调 larfb），info 0 | GPU 确认 |
| M263 | unmqr_2stage_gpu：文档要求的输入只是 zgeqrf 的输出，实际需要显式 V 和每一块的 T，按文档调用全错 | GPU 确认 |
| M260 | gehrd_m：zgehd2 覆写 work 时，面板卡上 Y = work → dY 的异步拷贝还没同步（锁页 work，R < ngpu），面板上半部分静默出错 | 单卡模拟 + 锁页数组 + 补丁确认 |
| M259 | gehrd_m + unghr_m：ilo > 1 时 T 的块对不上，Q 错误、info 0；zgeev_m/dgeev_m 在 gebal 做了置换时特征向量全错 | GPU 确认（1 张卡）+ 补丁确认 |
| M258 | zgetrf_m：分配时减少了卡数，循环仍用满 ngpu0 张卡，访问未建的 queue，进程崩溃（`MAGMA_NGR_NB` 小于 ngpu0·nb 时必现） | 单卡模拟 + 补丁确认 |
| M257 | trtri_diag 的 `cudaMemset` 不带流：文档称异步的 trsm_work/outofplace 会等待设备上所有无关的流（实测 150 ms） | GPU 确认（Windows） |
| M256 | magma_[zcds]potrf_mgpu 写死 h = 2：主机工作区槽在 h < ngpu（即 ngpu ≥ 3）时被提前改写，SPD 矩阵报 info > 0；源码注释自己承认 h = 1 出错 | z/c 单卡模拟确认；d/s 被 trtri_diag 的 legacy 流 memset 掩盖，由模型给出 |

更早的发现（M001–M241）见 FINDINGS 和 `BUG_DISTRIBUTION_zh.md` 第 5 节的 9 种机制。第 9 种“并发与内存模型”是第 31 轮新增的。

## 6. 还可以继续做的方向

0. （已完成，2026-11-27）round43 证实 M245–M247、M250 在 s、c 精度下也成立；round44 中 gesvd 的各条 lwork 路径都正确，gesdd 的替代路径受 M049 影响而不可达。见 HANDOFF §29.60。
1. **多卡行为**（需要第二块可用的 GPU）：zpotrf_m、zgetrf_m 等在 ngpu ≥ 2 时的数据分布和通信。第 42 轮已经证明分布的索引算术本身是对的；第 25 轮在 s82 上双卡结果出错，更可能是 GPU 1 的故障。要等 s82 修好，或者用户提供别的双卡机器。
   - 第 50 轮已在单卡上模拟多卡检验了多卡 Cholesky 的并发顺序（M255、M256，`probes/potrf_emu/`、`scripts/potrf_mgpu_hb.py`）。同样的办法可以用在其他 *_mgpu / *_m 例程上：先用模型找出无序的冲突访问，再只延迟那一个操作。注意 legacy 默认流上的操作会在单卡上把所有虚拟设备排序。
   - 第 52 轮的通用调度模糊测试（`probes/mgpu_fuzz/gen_fuzz.py`，HANDOFF §29.69）已覆盖 getrf_mgpu、geqrf2_mgpu、potrf_mgpu、hegst_m、trsm_m、getrf_m、potrf_m（z，ngpu 1–4）。后续批次又覆盖了 unmqr_m、geqrf_m、ungqr_m、hetrd_mgpu、heevdx_2stage_m、gehrd_m，都是正面结果。锁页主机数组复测（`R52_PIN=1`）也已完成，找到 M260。剩下的是其他精度。注意单卡模拟中所有虚拟设备共用一个拷贝引擎，会掩盖一部分竞争，模拟得到的"正确"只是下界。注意延迟点必须放在操作之前。
   - 第 55 轮的做法：把"文档承诺的前提"作为差分测试的输入（例如 dT 按生产方文档的大小分配，V/T 用生产方的原样输出），不要照抄 MAGMA 测试程序的写法，因为测试程序常常多分配工作区或者事先补好数据。
   - 第 53 轮的教训：多卡版本若改了中间结果（T、工作区）的存放约定，要检查所有消费者是否跟着改。MAGMA 测试程序只用默认参数（ilo = 1 之类），约定恰好重合时测不出来。
   - 有真实多卡时，先复测 M255、M256（自然时序下的出现频率），再复测第 25 轮 zpotrf_m 的双卡误差（可能与 M255 有关）。
   - 第 51 轮已扫描不带流的 `cudaMemset`/`cudaMemcpy`：新登记 M257（trtri_diag），其余早已登记（HANDOFF §29.68）。非阻塞 queue 下的无序危险在 Windows 上测不出，可以在 Linux 上复测。
2. **HIP 后端**（需要 AMD GPU）：目前只读过代码（126 个文件）。
3. **其余精度的复核**：已全部完成（M245–M247、M250 四种精度都已实测；M242 只在 c 精度下不出现；M249 只影响实数版）。
4. **继续按已知写法扫描**（只读码就能做）：
   - 宏的约定被弄混（类似 M248 的 `fetch`）；
   - 配套内核的提前返回阈值不一致（类似 M250）；
   - 循环末尾的分块代码没有放进条件分支（类似 M249）；
   - 带自旋标志的跨块通信，读数据时没有用 volatile 或 `__ldcg`（类似 M242）。
5. **sparse 库**：第 45 轮测过边界输入（M251），第 46 轮测过全部 16 种格式的往返和 SpMV（M252–M254）。已完成：zmergecg 中的 ELLRT 内核也在 GPU 上证实出错（直接设置 output_format；CG/PCG 等求解器报告成功，真实残差为 5e−3）；M251–M254 在 d 精度下全部复现（第 47 轮）；带 volatile 的 warp 内归约在 sm_120 上实测正确（第 48 轮，没有新缺陷）。s/c 精度的复核也已完成（第 49 轮，M251–M254 四种精度都复现）；setup 失败之后的状态没有新发现。roxy 上能测的部分到此已基本测完，剩下的主要是需要硬件的多卡行为和 HIP 后端。**汇总差分测试结果时，必须核对每个进程的退出状态和每种组合的行数**，第 29 轮就因为没有核对而漏掉了崩溃。

### 第 55–62 轮之后可以继续的方向（2026-11-28）

- **约定不一致**：问题集中在 MAGMA 自定义的中间格式（T、dT 及其工作区部分）。与 LAPACK 兼容的格式（反射向量与 tau、ipiv）在第 56–58 轮约 1,000 例中都没有问题。可以继续检查剩下的自定义格式，例如 he2hb 的 dT 交给 bulge 系列公开函数时、geqrf3_gpu 的 dT 交给 unmqr_gpu 时。
- **按尺寸分派**：M266、M267 都出在 vbatched 的大、小两条路径之间。c/d/s 精度的 potrf_vbatched 已实测：M266 四种精度都有，M267 只有 Z。其他 vbatched 例程的交叉点与写死的资源上限，也可以照第 62 轮的表逐个核对。
- **工作区越界**：第 59、60 轮覆盖了文档写明大小或走查询的设备工作区，当时只有 M261；第 68 轮又在 geqr2x 系列中发现 M274（第 59 轮没有把它列进去）。凡是 MAGMA 测试程序分配得比文档大的地方都值得查。
- **稀疏工具函数**：第 73 轮测了 tril/triu、diagadd、indexsort、5 点 / 27 点模板、稀疏乘稀疏，以及转置模板的 conj/abs/struct 变体。第 74 轮又测了 spmv_shift、mconjugate、mcsrcompressor_gpu、mslice、mshrink、vtranspose。第 75 轮直接调用了 6 个 IDR 变体和 magma_zqr。MAGMA 2.10 的 sparse 里已经没有 ISAI / ParILU。custom spmv / precond 是给用户替换的模板，第 76 轮已读过。第 76 轮还用恒等式 uᵀ(M⁻¹v) = (M⁻ᵀu)ᵀv 检查了转置预条件。第 77 轮在复数矩阵上用 Hermitian 恒等式查了 LSQR / PBiCG 的转置预条件（ILU 正确，Jacobi 为 M285，ICC 为 M284）。mconvert 在主机上只支持 CSR ↔ X（两端都不是 CSR 时明确返回 NOT_SUPPORTED），设备路径第 46 轮已测过。
- **vbatched BLAS**：第 78 轮测了 174 例；gemm / gemv / hemv / trmm / trsm 正确，herk 系列为 M246，hemm 为 M248。注意：按文件名搜记录会漏掉按例程名登记的条目，写探针前也要按例程名搜。
- 第 79 轮测了 trtri_diag_batched / vbatched 和 get/setmatrix_transpose，没有新缺陷（两条观察见其 TRIAGE）。按文件名扫描仍未直接测过的：zlarfgx-v2 / zlarfbx / zlarfx（geqr2x 的内部函数，经 geqr2x 间接覆盖）。zlacpy_sym_in/out 已在第 80 轮核对，登记为 M286。geqr2x 的辅助函数第 81 轮直接测过（M287、M288）。教训：前面的轮次如果只测到某个尺寸上限（这里是 128），值得再看代码里有没有写死的资源上限。第 82 轮按"运行时线程数启动"扫描后找到 M289、M290；剩下的候选（getf2_native_init、swapdblk、potf2）的线程数受调用方的块大小限制。第 80 轮还在 d/c 精度上复现了 M279、M280、M282、M284、M285。
- 测试程序：`scripts/roxy/run_list.ps1` 按行读用例表，一个用例一个进程，结果行须以 `r<数字>` 开头。用例参数不要用单独的 `-`，在 `-File` 模式下会被当成参数名。

## 7. 约束汇总

- 用户说明自己提交 bug 报告，所以不要向外部（GitHub 等）发布任何东西。
- 不要 commit 或 push，除非用户要求。
- 不要在对外请求里带上用户的邮箱等个人信息。
- 用中文向用户汇报。
- 不要碰用户的其他目录和进程。roxy 上的环境保留，不删除。s82 暂停使用。
