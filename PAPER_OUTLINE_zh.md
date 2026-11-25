# 论文提纲（路线C：方法 + 实证 + 带证明的接受门）

2026-11-25 草稿。参照了两份工作的写法：
- TensorLean：头条数字、bug表、“自带测试没抓到”；
- arXiv 2611.04447：每个定理配非空性实例、单列“建模决策”、明确可信基、AI使用声明。

依据的材料：
- `BRIEF_zh.md`；
- `HANDOFF.md` §22–§26；
- `operators/IR_SURVEY_zh.md`、`operators/RELATED_WORK_zh.md`；
- `experiments/ir_survey/README.md`；
- `lean/OptimiKahan/`。

每条论断后面标了状态：
- **[有]**：已有证据；
- **[缺]**：投稿前必须补；
- **[可选]**：能加分，不是必需。

---

## 0. 题目候选

1. *Reported Success, Wrong Answer: Formalizing and Fixing the Acceptance Logic of Mixed-Precision Iterative Refinement*
2. *When Should a Mixed-Precision Solver Say "Converged"? A Machine-Checked Acceptance Contract and an Empirical Study of 10 Libraries*
3. *Trust the Result, Not the Iteration: Certified Acceptance for Mixed-Precision Linear Solvers*

## 1. 摘要（草稿，约150词，中文先写）

- **问题**：混合精度迭代精化先用低精度（FP16/BF16/FP32）分解，再用FP64修正。它向调用者承诺：报告成功，就意味着达到FP64的后向误差。这个承诺全靠一段很短的**接受逻辑**：判据、归约、回退、状态返回。
- **发现**：我们对约25个实现做了源码普查，覆盖16个开源库和2个闭源库的文档。在10个库中实际运行，确认了“报告成功、答案错误”。例子：
  - SLATE 的 LAPACK 接口在良态矩阵上把解丢掉，报 info=0；
  - SuperLU_DIST 在解含 NaN 时返回的误差界与干净输入完全相同；
  - MAGMA 在 FP16 溢出时返回全 NaN，报 info=0；
  - cuSOLVER GMRES 路径在 2^40 缩放下相对误差 0.93–23，报“收敛”。
- **形式化**：用 Lean 4 证明：
  1. 接受契约（报成功 ⇒ 解好；FP64可解 ⇒ 报成功）**当且仅当**四个条件 C1–C4 同时成立；
  2. 判据与最大值归约的6种组合里，只有一种对 NaN 安全；
  3. 文档阈值 √n·eps 低于 FP64 残差的最坏舍入误差，用FP64残差不可能可靠认证。
- **修复**：给出一个与求解器无关的接受门（Dot2 残差 + 外向舍入 + 非有限即回退），并机器证明它对**任意**内层输出都可靠。在 cuSOLVER 上 36/36 通过认证。

## 2. 贡献列表（照 TensorLean/一般 SE 论文的格式，4条）

1. **接受契约及其充要条件**（`IrDriver.contract_iff`）。把十一类缺陷 D1–D11 都归入“违反 C1–C4 中某一条”。**[有]**
2. **数值上 C1 何时可达的三个定理**。**[有]**
   - `detects_iff`：判据形式 × 归约方式；
   - `eval_scale`：缩放协变，可作零误报的蜕变预言机；
   - `fp64_cannot_certify`：FP64残差不足以认证文档阈值。
3. **实证研究**。**[部分有]**
   - 已有：约25个例程的缺陷矩阵；10个库上的运行确认；预测先于运行，7/7 证实。
   - 缺：上游报告状态；自带测试是否覆盖这些输入。
   - 新增（RQ6，待做）：无参考答案的跨硬件判据 |out_A − out_B| > 2B，用于求解器依赖的 GEMM/归约构件。
4. **带证明的接受门及其在 MAGMA、cuSOLVER 上的实现**（`fixed_post`、`wrapper_post`）。
   - 已有：CPU 核对（17/20 → 0/20 违约）、cuSOLVER 包装（36/36 认证）。
   - 缺：MAGMA 补丁上 GPU、开销测量。
   - 要把 `Faithful` 假设推出来（路线B），工作量大。

**不能说的**（见 RELATED_WORK §末）：
- 首次在 Lean 中验证 GPU 内核；
- 缩放不变性是我们发现的；
- 首次发现求解器把 NaN 当成功（LAPACK PR #1501 系列）。

## 3. 章节结构

### §1 引言（1.5页）

- **开头例子**：选最直观的一个。候选：
  - SLATE `slate_dsgesv`：100×100 良态矩阵，info=0，iter=23，解在 x 和 b 里都不存在；
  - SuperLU_DIST：x 有 500 个 NaN，四个状态量与干净输入逐位相同。
- **为什么测试抓不到**：这些输入是 NaN、尺度、x0=0、FP32 主元增长，而库自带测试的 b 几乎总是 U(0,1) 或 A·全1。16个缺陷中0个被默认测试触发。部分库的测试判定本身就会放过NaN。**[有，见 RQ3]**
- **为什么现有形式化不覆盖**：TensorLean 和 2611.04447 都在设备端，浮点按实数（加 ε）处理。2611.04447 明说“不推理数值稳定性”。接受逻辑在主机端，而出错恰恰出在 NaN 和舍入上。
- 贡献列表（上面4条）。
- **头条数字**（TensorLean式）：
  - 约25个例程普查，18个库（16个开源，cuDSS、oneMKL 只看文档）；
  - 10个库运行确认，预测先于运行，7/7 证实；
  - IR相关 Lean：5个文件、1,821行、138条定理/引理，无 sorry，只用3个标准公理；
  - 接受门对任意内层输出可靠。

### §2 背景（1页）

- 混合精度 IR 的通用骨架：低精度分解 → 内层（经典或 GMRES）→ 接受判据 → 回退 → 状态返回。
- **文档承诺原文**：
  - LAPACK `dsgesv`：`RNRM < SQRT(N)·XNRM·ANRM·EPS`；
  - cuSOLVER：“always a good solution”（开启回退时）；
  - MAGMA。
- 图1：骨架图，标出 D1–D11 各自出现的位置。

### §3 接受契约（形式化核心1，1.5页）

- **定义**：
  - 抽象驱动 `Impl`：`check`、`fbReach`、`fb`、`force`；
  - 内层是对抗性的；
  - Sound、Complete。
- **定理**：
  - `contract_of_conditions`：充分性，无需假设；
  - `contract_iff`：必要性，需要四个建模假设，**逐条说明为什么合理**。
- **推论表**（表1）：

  | 推论 | 违反 | 库 |
  |---|---|---|
  | `trust_inner_unsound` | C1 | MAGMA GMSTRS、`dshposv_gmres`、PaStiX、STRUMPACK mixed |
  | `forced_unsound` | C4 | SLATE LAPACK接口、STRUMPACK、PaStiX |
  | `stub_fallback_fails` | C3 | MAGMA `zchesv` |
  | `skipped_fallback_incomplete` | C2 | PLASMA |

- **非空性**：`tOk_contract`（照 2611.04447 的做法明写）。
- **如实写**：这一层的证明逻辑简单。价值在于给出完整的条件清单，并把各库映射到条件上。数值上的难点都在 C1，交给 §4。

### §4 C1 在浮点下怎样才能成立（形式化核心2，2.5页）

- **4.1 判据形式与 NaN**（`MaxFold`）
  - SuperLU 写法：NaN 当且仅当最后一个元素是 NaN；
  - PLASMA/IDAMAX 写法：NaN 当且仅当第一个元素是 NaN；
  - `masked_equals_baseline`：与 SuperLU_DIST 实测逐位吻合；
  - `detects_iff`：6种组合只有1种安全；
  - `c1_forces_accept_propagate`：D1、D11 都是违反 C1。
- **4.2 缩放协变**（`IrsContract`：`rnd_scale`、`eval_scale`、`iterate_scale`、`oracle`）
  - 类型正确的程序与 2^k 缩放交换 ⇒ 结果随 k 改变，就证明实现中有尺度相关常数，或发生了溢出/下溢（D2/D3）；
  - 对闭源的 cuSOLVER 也能用。
- **4.3 FP64 残差不够**（`fp64_cannot_certify`、`fp64_n1024`）：c = √n·u < (n+1)u。
- **4.4 带证明的接受门**（`wrapper_post`、`fixed_post`、`fixed_nan_fallback`、`fgmresFixed_scale`）
  - 对任意 FGMRES 输出（含 NaN）都可靠；
  - `fixed_accepts_example`、`example_faithful` 保证定理不是空真。
- **4.5 建模决策**（照 2611.04447 §3.2 单列一小节）
  - 有理数加 NaN；不建模 Inf；
  - 不建模指数范围，所以不考虑下溢；
  - 手工逐行翻译，不是机器抽取；
  - 每个选择为什么不破坏结论；
  - **加法次序**（多线程、GPU归约）：说明哪些定理对任意次序成立，哪些需要固定次序。`fp64_cannot_certify` 用的是与次序无关的最坏界；`Faithful` 里的 Dot2 界目前取串行版本，路线B应推广到任意加法树。
- **路线B的加强 [缺，最大的一项形式化工作]**：在 Lean 里从 `rnd` 推出 `Faithful`，即 Dot2 误差界和行和误差界。做完后，“接受门可靠”就只依赖 IEEE 舍入模型。

### §5 方法：从契约到反例（1页）

- 契约卡六栏（BRIEF 方法1）。
- 白盒代码级模型：用于 MAGMA GMSTRS（`MagmaFgmres`）。由模型推出的反例：
  - `nan_runs_to_maxiter`；
  - `tol_ge_one_one_step`；
  - `tiny_rhs_returns_zero`。
- **流程**：读码 → 归入 D 类 → 预测输入 → 运行。强调**预测写在运行之前**。注意：普查（11-24）和运行结果（11-25）是在同一次提交 06a3922 里进入 git 的，git 历史证明不了先后。能用的证据只有会话记录和文件内的日期。以后的预测应该先单独提交，再运行。

### §6 实证研究（3页）

- **RQ1 普遍性**：
  - 表2：约25例程 × D1–D11 矩阵（`IR_SURVEY_zh.md` 总表，压缩为一页）；
  - 安全设计作对照：rocHPL-MxP、Trilinos Belos、PETSc、SLATE `*_mixed_gmres`、cuDSS。
- **RQ2 预测是否成立**：
  - 表3，bug表，TensorLean 格式：`库 | 例程 | 版本/提交 | 缺陷类 | 违反条件 | 触发输入 | 现象 | 上游链接/状态`；
  - CPU 7项（SuperLU、SuperLU_DIST、MUMPS、PaStiX、PLASMA、STRUMPACK、SLATE）；
  - GPU：MAGMA（V100）、cuSOLVER（5070 Ti，CUDA 11.8–13 各版本）；
  - LAPACK 3.12 作为已有先例（PR #1501）。
- **RQ3 自带测试为什么没抓到**：**[有，只读码]**，见 `IR_SURVEY_zh.md`“各库自带测试”一节。
  - 8个库、16个缺陷，**0个**会被默认测试或CI触发；
  - 6个例程没有带判定的测试；
  - SuperLU、SuperLU_DIST、STRUMPACK 的**测试判定本身也用丢NaN的max和拒绝式比较**，就算加上NaN输入也会判通过；
  - 正面例子：SuperLU_DIST `ptest_ci.c` 写成NaN安全，但没有跑d2路径。
  - [可选] 实际给测试程序加上NaN输入运行一次，把“会判通过”从读码变成实测。
- **RQ4 接受门的代价与效果**：
  - 已有：cuSOLVER 包装 36/36 认证，误差 ≤ 1.5e-11；同一测试用FP64残差 0/36 通过；MAGMA CPU 移植 17/20 → 0/20；
  - **[缺]**：MAGMA 补丁上 V100（等 GPU）：原 24 次调用重跑、开销计时。
- **RQ5 缩放预言机的检出力**：cuSOLVER 三轮多种子（R1–R3、R3IRS），经典路径 15/15 尺度不变，作为零误报的对照。**[有]**
  - 前提：两次运行的加法次序相同。不确定的归约会让缩放前后的差异变成误报。R3 每次调用新开进程，经典路径仍逐位不变，算是实测排除了这一点。要写成预言机的显式前提，并给出检验方法：同一输入先重复运行，确认逐位一致后，再做缩放比较。
- **RQ6 无参考答案的跨硬件判据能否发现精度违约？** **[缺，全部待做]**
  - **原理**：
    - 算子承诺 |out − exact| ≤ B，其中 B 对任意加法次序成立；
    - 硬件 A、B 次序不同，但都必须满足这个界，所以 |out_A − out_B| > 2B ⇒ 至少一个违约；
    - 不需要参考答案，零误报。
    - 相关工作没找到这个判据（`RELATED_WORK_zh.md` 补充一节）。
  - **适用范围**（如实）：
    - 只适用于**有前向误差界**的算子：GEMM、点积、归约，以及求解器内部用到的这些构件；
    - 求解器本身的契约是后向误差，两边的解可以相差 κ 倍，不适合用 2B，继续用残差核对和缩放预言机。
  - **界从哪来**：
    - 任意树的 γ_d·Σ|xᵢyᵢ| 引用 VSCL 的 `sumTreeDotProduct_error_bound`，或在 core Lean 中重证；
    - 是否引入 Mathlib 待定（见待办7）；
    - 加法次序未知时，深度只能取最坏的 K−1，所以 B 偏宽。
  - **能抓到什么**：只能抓到大的违约。典型的是“实际累加精度低于文档声称”：
    - DeepSeek-V3 报告 H800 FP8 GEMM 累加约14位，K=4096 时相对误差近2%；
    - MMA-Sim 报告 Hopper FP8 累加只有13位小数位。
    - 文档若声称FP32累加，B ≈ K·u₃₂·Σ|xy| ≈ 2.4e-4·Σ|xy|（K=4096），这种违约会被抓到。
  - **实验设计**（用现有硬件）：
    - 平台：eightsocket CPU、V100（s82，等GPU空出）、RTX 5070 Ti（roxy）；
    - 算子：
      - FP64 DGEMM（OpenBLAS vs cuBLAS）；
      - cuBLAS FP64 仿真（5070 Ti）vs OzBLAS/ozIMMU（V100）vs CPU DGEMM；
      - FP16 输入、FP32 累加的 GEMM（V100 vs 5070 Ti 张量核）；
      - PyTorch matmul 的 FP16/BF16 路径。BF16 V100 不支持，只能 5070 Ti vs CPU。
    - 输入：随机、病态点积（GenDot）、行内跨度大的矩阵。
    - 对照：同时用精确有理参考验证每一边是否真的在界内，评估 2B 判据的漏检率。
  - **限制**：
    - FP8 只有 5070 Ti 支持，做不了两张卡的 FP8 比较；
    - 两个平台的库版本也不同，违约只能定位到“平台 + 库”的组合。
  - **可选的形式化加强（§4.6）**：同一棵树里不同节点用不同的 δ（张量核内部 vs 外层累加器），给出比全局 δ 更紧的 B。没有找到机器证明的先例；纸面分析见 Blanchard 等 2020。

### §7 讨论、局限、可信基（1页）

- **可信基**：
  - Lean 内核；
  - 手工翻译的模型与源码一致（证据是 GPU/CPU 结果吻合，不是机器检查）；
  - `Faithful` 假设，若做完路线B则移出可信基；
  - 库到条件的映射靠读码。
- **局限**：
  - 只测方阵；大多 nrhs=1；
  - MAGMA 与 cuSOLVER 的硬件不同；
  - 闭源库的机制只能推断。
- **AI 使用声明（必须）**：说明哪些代码、模型、证明、文字由 AI 协助，作者逐项核对过哪些。

### §8 相关工作（1页）

按 `RELATED_WORK_zh.md` 分四组，每组一句区别：
1. **GPU 内核形式化**（TensorLean、2611.04447、Volta、GPUVerify）：设备端、实数语义；我们做主机端接受逻辑，带 NaN。
2. **数值库验证**（VST/LAProof、Jacobi、Flocq 系）：证明某个实现正确；我们用模型**推出失败**，并覆盖十几个库。
3. **浮点缺陷检测**（KLEE-Float、ESBMC、GPU-NBDetect、Di Franco 2017）：低于契约层的自动反例，没有“报告成功”这个语义。
4. **IR 理论与认证**（Carson–Higham、LAWN 165/175、Ogita–Rump–Oishi、INTLAB verifylss）：理论上的误差界或区间包围；我们针对库的接受门本身。
- 单列一段：LAPACK PR #1499–#1520 系列。比较我们多出的部分：主元增长触发、D11 路径、MAGMA/PLASMA/SLATE/SuperLU 的继承、非 NaN 的错误解。

## 4. 图表清单

| 编号 | 内容 | 状态 |
|---|---|---|
| 图1 | IR 骨架 + D1–D11 位置 | [缺] 待画 |
| 表1 | 推论 ↔ 条件 ↔ 库 | [有] HANDOFF §26.2 |
| 表2 | 普查矩阵 | [有] 需压缩 |
| 表3 | bug 表（含上游链接） | [缺] 链接待用户提交后填 |
| 表4 | 定理汇总：定理名、文件、行数、公理、非空性实例 | [缺] 整理即可 |
| 图2 | MAGMA 2^k 缩放：iter 与后向误差/阈值随 k 变化 | [有] 数据在 BRIEF |
| 图3 | 接受门开销 | [缺] 等GPU |
| 表5 | RQ6：平台对 × 算子 × 输入，|out_A − out_B| / 2B 及精确参考核对 | [缺] |

## 5. 投稿前待办（按优先级）

1. **上游报告**（用户提交）：MAGMA、NVIDIA、SLATE、SuperLU_DIST、PaStiX、STRUMPACK、PLASMA、MUMPS。没有链接，表3就不成立。
2. ~~**自带测试核对**（RQ3）~~：已完成（2026-11-25，只读码）。
3. **定理汇总表**（表4）：整理即可。
4. **MAGMA 补丁上 V100、开销计时**：等 GPU。顺带测 `zchesv` 和 `dshposv_gmres` 两个未运行的预测，以及 trans/nrhs 探针。
5. **路线B：在 Lean 中推出 `Faithful`**：Dot2 误差界、行和误差界、外向舍入。决定这篇的形式化分量能否支撑 PL/FM 会议。
6. [可选] hpl-ai、Ginkgo、PETSc 的运行确认。
7. **RQ6 跨硬件判据**：
   - (a) 决定 B 的来源：引用 VSCL（Lean 4 + Mathlib），或在 core Lean 里重证任意树的点积界；
   - (b) 先在 CPU 与 5070 Ti 上做 FP64 DGEMM 和 FP16/FP32 GEMM（不需要 s82）；
   - (c) GPU 空出后加 V100；
   - (d) [可选] §4.6 分节点 δ。

## 6. 会议方向（只说适配度，不含截稿日期，需另查）

- **软件工程，实证加形式化**（ISSTA、FSE、ICSE）：最贴合现有材料。bug 表、普查和“测试抓不到”是这类会议看重的；形式化部分作为方法和修复的保证。
- **高性能计算**（SC、PPoPP、IPDPS；期刊 TOMS）：读者就是这些库的作者，重视开销数据和上游修复。表3和图3是必需的。
- **形式化方法**（CAV、FMCAD、ITP/CPP）：只有在做完待办5、接受门不再依赖 `Faithful` 之后才合适。现在的 `contract_iff` 逻辑太浅，撑不起这类会议。
