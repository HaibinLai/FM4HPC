# 相关工作与创新点核对（2026-11-24）

检索方式：4个子代理并行检索，约100次搜索。主会话复核了以下各项：
- TensorLean
- arXiv 2611.04447
- lean-ieee754
- GPU-NBDetect（ISSTA 2025）
- Reference-LAPACK PR 列表（GitHub API）
- arXiv 2611.19008
- MMAlgebra / HardwarePrimitives（仓库已404）

其余条目来自子代理的读取，多数只看了摘要或片段；标注“未核实”的不可直接引用。**写论文前应逐篇通读第1–4节标为“强”或“部分”的文献。**

覆盖程度的标记：**强** = 实质已有；**部分** = 有相关成分；**无** = 没找到。

---

## 0. 结论速览

| 我们的说法 | 判定 | 主要对照 |
|---|---|---|
| 白盒代码级Lean模型（手工逐行翻译主机端控制流，精确有理浮点+NaN语义），用来**推出**违约输入，再上GPU确认 | **组合是新的**；每个成分都有先例 | TensorLean（PTX层、SMT、AI内核）；VST/LAProof/Jacobi（代码级求解器模型，但只做证明）；Cedar（Lean模型+差分测试）；KLEE-Float/ESBMC（自动浮点反例，低于契约层） |
| 有证明的蜕变预言机（2^k缩放协变 ⇒ 输出一变即有缺陷，零误报） | **底层事实是经典的**；机械化、推广到GMRES/停止判据、用作闭源预言机是新的 | Bauer 1963 / Higham《Accuracy and Stability》§9.8（未核实）；Higham–Pranesh–Zounon 2019；LAWN 165 把“行缩放不变”作为设计目标 |
| 契约卡（规则 / 文档承诺 / 实际可支持的定理 → 反例族，先预测后测试） | **没找到等价物** | DocTer（ISSTA 2022，从文档抽取参数约束，只管输入合法性） |
| “报告成功、答案错误”的实证 | **部分已有，且有同期工作** | Di Franco 2017；GPU-NBDetect（ISSTA 2025）；**LAPACK PR #1499–#1520 系列（2026-11-22，同一作者，约22个PR）** |
| MAGMA/cuSOLVER 的IR接受逻辑偏离文档判据 | **新**（没找到任何公开报告） | 只有 LAPACK 的NaN情形有先例（PR #1501） |
| 文档阈值 √n·eps 低于FP64残差的最坏舍入误差，FP64下无法可靠检查 | **作为明确陈述和定理是新的**，成分属于常识级 | dgerfs 的 (n+1)·eps·(\|A\|\|x\|+\|b\|) 项；Carson–Higham 的 n·u 极限残差；LAWN 175 是 √n 判据的出处，未论证 √n |
| 带证明的后验检查（Dot2 + 外向舍入 + 非有限即失败 → 回退） | **技术是增量的，作为成果没人占** | Ogita–Rump–Oishi 2005；Rump 2010 / INTLAB verifylss（验证的是包围区间，不是库的接受门）；lean-ieee754（Lean 中的补偿点积，LLM写成、未经评审） |
| cuBLAS ADP：零元素使位数估计膨胀，零感知修正已证安全 | **诊断和修正思路已有（强）**；带零的安全性证明和定量结果是新的 | **arXiv 2611.19008（NVIDIA，2026-11-13）** |
| ozIMMU 自动选片的两个bug + 平均损失准则 | **新** | 无公开报告 |
| GEMMul8：CRT范围的Lean证明、fast模式静默0、Inf/NaN不传播、README忽略k | Lean证明和具体缺陷是**新**的；**“精度与k有关”已有** | arXiv 2602.02549（Ozaki-II误差分析，含k的线性项）；Ozaki-II原文假设无溢出、需“检测并回退” |
| OzBLAS分片引理与重组误差界的Lean证明 | 大概率新 | MMAlgebra（Lean，已404，无法核实） |

---

## 1. 白盒代码级Lean模型（重点）

### 1.1 最接近的竞争者

- **TensorLean**（Columbia + CertiK，2026-11-20发布）。 https://tensorlean.ai/ ， https://github.com/TensorLean/TensorLean
  - 做法：在PTX层把编译后的内核对照Lean写的张量级规格。
  - 证明的性质：无数据竞争、终止、在浮点容差内正确、其余内存不被改动。
  - 结果：149个实例证明了83个，发现38个bug（FlashAttention占23个）。
  - 局限：规格建立在实数上，误差用ε界住；张量核舍入模型属于可信基；**还没有论文**。
  - 不覆盖：主机端控制流、FP64仿真、LU/迭代精化、NaN语义。
  - **区别**：
    - 对象：设备端AI内核 vs 主机端求解器/仿真库的判据与回退逻辑；
    - 方法：自动SMT证明，失败即bug vs 手工模型推出违约输入族再实测；
    - 浮点语义：容差 vs 精确有理数 + IEEE NaN。
- **Formalizing PTX in Lean to Prove GPU Kernel Correctness**（arXiv 2611.04447，2026-11-03）。 https://arxiv.org/abs/2611.04447
  - 内容：PTX内存模型与张量核的Lean形式化，证明了手写的WGMMA GEMM、FlashAttention、MoE流水线。
  - 用Mathlib，约3.4万行；不是浮点误差分析。
  - 判定：(a) 部分。
- **Cedar**（AWS，FSE 2024）。 https://arxiv.org/abs/2407.01688
  - 内容：生产代码的可执行Lean模型，用于差分测试，发现25个bug。
  - 方法论接近，都是“Lean模型当预言机”；但它用随机输入，不是从模型推出输入，也不涉及数值。

### 1.2 代码到证明助手（说明手工翻译的必要性）

- Frama-C/WP + Why3（含浮点；Gappa只能处理直线代码）。
- VST/Clight + VCFloat2（CPP 2024）。 https://dl.acm.org/doi/10.1145/3636501.3636953
- CompCert。
- CLean（Certora）：Clight → Lean，证明了zlib的内存安全，不处理浮点。
- Aeneas（Rust → Lean），浮点支持只在未合并的分支里。
- **结论：没有可用的C/C++ → Lean工具能处理带浮点的数值代码**，手工翻译是现实的选择。这同时也是我们最大的弱点（翻译没有机器检查）。

### 1.3 代码级数值求解器的形式化（只做证明）

- **LAProof**（ARITH 2023，Coq）：点积、矩阵向量积、GEMM的后向/混合误差界，以及用VST验证的C语言CSR矩阵向量积。 https://github.com/VeriNum/LAProof
- **Jacobi 求解器**（CICM 2023，Coq + VST）：真实C代码的正确性、精度、收敛，含溢出。 https://link.springer.com/chapter/10.1007/978-3-031-42753-4_14
- **Boldo 等，波动方程**（JAR 2013）：Frama-C + Gappa + Coq。 https://arxiv.org/abs/1112.1795
- 判定：(a) 部分。它们都是证明导向，不用于推反例；没有GPU、迭代精化或回退逻辑。

### 1.4 自动寻找浮点反例

- KLEE-Float（ASE 2017）、ESBMC/CBMC的浮点后端、JFS、Ariadne（POPL 2013，在GSL上用符号执行找浮点异常）。
- 动态工具：FPChecker（ASE 2019）、GPU-FPX（HPDC 2023，二进制插桩，能看到闭源内核里的异常）。
- 判定：部分。它们是自动的，停在“出现Inf/NaN”这一层，**抓不到有限值的错误答案被报为成功**，也不涉及文档契约。

### 1.5 没找到的

- 对迭代精化、GMRES/FGMRES-IR、浮点LU后向误差，或LAPACK/MAGMA/cuSOLVER例程的机器检查形式化。
  - 最接近的：arXiv 2611.31033，Lean中用AI做的CG收敛自动形式化，**实数**算术，只看了摘要。
- 手写的数值库控制流（停止判据、回退）形式模型，用来预测失败。

### 1.6 写法建议

- 主张一种组合：“主机端判据/回退逻辑的代码级模型 + 精确浮点与NaN语义 + 从模型推出违约输入族 + 实测确认”。
- 必须明确对比TensorLean和2611.04447：设备端 vs 主机端，自动证明 vs 推导反例。
- 用VST/LAProof说明“代码级模型”本身不新，新在用途（预测失败）和对象（仿真库的参数/判据）。

---

## 2. 有证明的缩放预言机

- **经典事实**：
  - 按基数的幂缩放不改变舍入误差。Bauer 1963；Higham《Accuracy and Stability》§9.8（**未核实**，子代理只看到目录）。
  - Higham–Pranesh–Zounon，SISC 2019：2的幂缩放且主元序列不变时，fl(μRAS) = μR·fl(A)·S；LAPACK xyyEQUB 正是因此把缩放因子取成2的幂。 https://epubs.siam.org/doi/pdf/10.1137/18M1229511
- **蜕变测试**：
  - Segura 等，TSE 2016 综述。
  - Rahman–Kanewala，MET 2018：用机器学习预测矩阵程序的蜕变关系。
  - Donaldson 等，MET 2016：指出实数上成立的关系在浮点下会失效，这正是我们定理解决的误报问题。
  - 没找到任何对BLAS/LAPACK/cuSOLVER做精确2^k缩放、并证明零误报的蜕变测试。
- **写法**：引用Bauer/Higham作为底层事实，主张三点：机械化证明；推广到GMRES和停止判据（次数一致性）；用作闭源库的零误报预言机。cuSOLVER GMRES-IR 的错误收敛没有公开先例。

## 3. 混合精度迭代精化与可验证残差

- **LAWN 175**（Langou 等，SC'06）：√n·ε 判据的出处（‖r‖₂ ≤ √n·ε·‖A‖_F·‖x‖₂，最多30次，然后回退DGESV），**没有论证√n**。 https://netlib.org/lapack/lawnspdf/lawn175.pdf
- **Carson–Higham**，SISC 2018：残差最好能达到约 p·u·(‖b‖+‖A‖‖x‖)，稠密时 p = n。没有和任何库的阈值联系起来，不涉及NaN或回退。 https://eprints.maths.manchester.ac.uk/2629/1/cahi18.pdf
- **dgerfs.f**：FERR中加了 (n+1)·eps·(|A||x|+|b|)，用来覆盖工作精度下残差的舍入误差。这是我们“阈值低于残差舍入误差”这条观察的**直接成分先例**。
- **LAWN 165**（Demmel 等，TOMS 2006，xGERFSX）：用双倍精度残差让误差界和终止可靠，终止判据按行缩放不变来设计。判定：部分。 https://www.netlib.org/lapack/lawnspdf/lawn165.pdf
- **Higham–Mary**，SISC 2019，概率误差分析：典型常数约 √(n log n)·u，可能是 √n 的启发式来源（只看了摘要）。
- **Haidar–Tomov–Dongarra–Higham**（SC18）：没有误差分析，没有外层接受检查，没有NaN处理。
- **Higham–Mary**，Acta Numerica 2022：称MAGMA和cuSOLVER会把超出FP16范围的元素映射为 ±θ·xmax。**这与我们在V100上看到的MAGMA 2.10 `dhgesv_iteref` 行为不符**（元素 >65504 时得到全NaN的x，info=0）。很可能是不同例程或研究代码，写报告时要说明这一差异。 https://perso.lip6.fr/Theo.Mary/doc/surveyMixed.pdf
- **Ogita–Rump–Oishi**（SISC 2005）、**Rump**（Acta Numerica 2010）/ INTLAB verifylss：Dot2和补偿残差是现成工具，但用于验证解的包围区间，不是用作库的接受门。
- **cuSOLVER 13.5 文档**：承诺回退“will guarantee that the IRS solver always provide the good solution”，没有提NaN。
- **没找到**：任何关于MAGMA或cuSOLVER停止判据、NaN当成功、1e-18中断的公开报告。唯一相关的是MAGMA 1.3.0论坛里一个未初始化导致NaN的旧帖，与本问题无关。

## 4. 同期工作：Reference-LAPACK PR系列（重要）

- 2026-11-22，同一作者开了约22个PR（#1499–#1520）。大多数修复“INFO=0却返回非有限或错误结果”、缩放溢出、次正规主元等问题，至今多数仍开着。 https://github.com/Reference-LAPACK/lapack/pulls
  - **#1501**：xSGESV/xSPOSV在单精度解非有限时以ITER=0、INFO=0返回，修复为回退双精度并设ITER=−2。
  - 这些PR说明是用Claude Code准备的，没有说明发现方法。
- 这是一次针对LAPACK的、系统性的LLM辅助审计，和我们“报告成功、答案错误”的主题**直接同期**。
- **区别**：
  - 对象：闭源/GPU库（cuSOLVER、cuBLAS）和MAGMA、Ozaki系；
  - 方法：契约、定理、代码级模型推出输入族；
  - 修复：带证明的修复。
- 论文必须引用并比较这个系列。

## 5. Ozaki / FP64仿真

- **arXiv 2611.19008**（Brower、Caday、Gunnels，NVIDIA，2026-11-13）。 https://arxiv.org/abs/2611.19008
  - 指出coarsened ESC中零的指数相当于−∞，“科学应用常乘足够稀疏的矩阵”会触发这条路径。
  - 报告：1%的零使估计到63–64位，10%的零强制回退FP64。
  - 提出binned ESC和p-norm ESC，零映射为零代表，已在cuBLAS开发版中。
  - 它的证明假设输入“有限、正规、非零”，**没有含零时的精度结果**。
  - 对我们第21节：**诊断和修正思路强覆盖**。剩下的新意是含零的安全性（充要条件）机械化证明，以及约4000位膨胀、239→27 ms的实测。
- **Schwarz 等**（NVIDIA，SCA/HPCAsia 2026，arXiv 2511.13778）：ADP/ESC原文；承认ESC界不总是紧，会导致不必要的回退；Inf/NaN触发回退。 https://arxiv.org/abs/2511.13778
- **Zhang**（arXiv 2611.31030）：每个乘积都满足精度界但求逆失败；ADP回退比原生慢；cuBLAS句柄状态bug。判定：部分。
- **Ozaki-II**（arXiv 2504.08009）：明确假设无溢出/下溢，异常值需“检测并回退”。 https://arxiv.org/abs/2504.08009
- **Ozaki-II误差分析**（arXiv 2602.02549）：只覆盖accurate模式，含k的线性项，假设 k ≤ 2^17，排除全零行列；fast模式只做定性讨论。
  - 我们的“README没计入k”只能作为对README的批评，不能说“精度与k有关”是新发现。
- **Kawakami–Takahashi**（arXiv 2606.29129）：现在是GEMMul8 fast模式的默认缩放，提到常数缩放会导致CRT恢复失败。我们用的GEMMul8提交是 `d8d3901`（2026-11-16），晚于这篇论文（2026-06）。如果“已成为默认缩放”属实，我们的 ≳2^512 / ≲2^-537 静默0就是在包含该修复的代码上测到的。**还需要对照源码确认默认缩放就是它。**
- **Świrydowicz 等**（AMD，arXiv 2611.13332）：Ozaki-II“对缩放敏感”，没有报告具体bug。
- **ozIMMU**（Ootomo 等，IJHPCA 2024）、**Uchino 等**（IJHPCA 2025）：没有涉及自动选片的bug。
- GitHub issue：GEMMul8 的 #1、#5 和 ozIMMU 的 #1 都与数值缺陷无关。我们的ozIMMU和GEMMul8缺陷**没有公开报告**。OzBLAS的issue没查。
- **MMAlgebra / HardwarePrimitives**（GitHub用户 pauljiang03，Lean 4）：
  - 搜索引擎的索引描述为“Verified semantics of GPU matrix multipliers in Lean”，含Ozaki-I/II、ADP/ESC在张量核模型上的证明。
  - 2026-11-24复核：两个仓库都返回404，用户本身存在。**无法核实**。
  - 如果它确实做了这些，会与第14、15、17、21节的Lean工作强重叠。声称“首次机械化Ozaki”之前必须查清。

## 6. 浮点形式化基础（背景，不构成重叠）

- Flocq/Gappa（Coq）、Isabelle AFP IEEE_Floating_Point、HOL Light、FPTaylor、Daisy/FloVer、Rosa、Satire、PRECiSA。
- Lean 4（均为2026年，都很新）：FloatLib（lean-dojo）、FloatSpec（Flocq移植）、Flean、lean-ieee754。
- 误差无关变换：Boldo–Graillat–Muller，TOMS 2017（纸面证明）。**没找到经过同行评审的Dot2机器证明**。

---

## 7. 对创新点表述的修正

1. **不要说**：“首次在Lean中验证GPU数值内核”（TensorLean、2611.04447）；“缩放不变性是我们发现的”（Bauer/Higham）；“首次指出ADP对零悲观”（2611.19008）；“精度与k有关”（2602.02549）；“首次发现求解器把NaN当成功”（PR #1501）。
2. **可以说**：
   - 第一个针对**低精度模拟高精度库的参数与接受判据**的“契约—代码级模型—实测—带证明修复”流程；
   - MAGMA、cuSOLVER的IR接受逻辑缺陷（无先例）；
   - 文档阈值在FP64下无法可靠检查的明确定理；
   - 求解器无关的后验检查的机器证明；
   - 含零ESC安全性的证明；
   - ozIMMU、GEMMul8的具体缺陷。
3. **最需要补的**：
   - 通读Higham书 §9.8 和 §12.3（最可能已讨论残差精度与停止判据）；
   - 查清MMAlgebra；
   - 对照 `d8d3901` 源码确认fast模式默认缩放就是 2606.29129 的公式；
   - 在论文中正面比较TensorLean和LAPACK PR系列。

---

## 补充（2026-11-25）：任意次序误差界与跨硬件精度测试

**起因**：考虑把“与加法次序无关的误差界”作为三类库（HPC、仿真、AI算子）的统一精度契约，并用“两种硬件结果之差 > 2B ⇒ 至少一个违约”做无参考答案的跨硬件测试。

**检索**：2个子代理检索；主会话打开了VSCL与lean-ieee754的Lean源码，以及numoracle、Grading the BLAS、VERNA的论文原文核实。

### 已被做过的部分（不能再当贡献）

- **任意加法树 + 参数化 u 的求和/点积误差界，已在 Lean 4 中机器证明**：
  - **VSCL / lean-computational-mathematics**（Georgantzas、Melnichenko 等，MIT，2026-03起）
    - `SumTree.forward_error`：`|t.eval fp v − Σvᵢ| ≤ γ(depth)·Σ|vᵢ|`（`Algorithms/Summation/Tree/Core.lean:569`，已核实）；
    - `sumTreeDotProduct_error_bound`：`γ(depth+1)·Σ|xᵢ||yᵢ|`（`TreeDotProduct.lean:82`，已核实）；
    - 还有任意置换、块点积、扩展精度点积；
    - 也有GEMM `matMul_error_bound`，但基于串行matVec，不是树；
    - 模型是抽象 `FPModel`，全局一个 u，不含下溢项。
    - https://github.com/the-numerical-proving-ground/lean-computational-mathematics
  - **lean-ieee754**（tunnellm/girving，2026-09起，README说主要由LLM写成，未经专家评审）
    - `ReductionTree.mixed_error`：任意树，界为 `((1+u)^height − 1)·mass + ε·roundWeight`，含下溢项ε，舍入函数抽象（`FP/Summation.lean:78`，已核实）；
    - 点积只有串行版本。
  - 纸面结果早已有：Higham §4.2（深度为 h 的树给出 γ_h）；Jeannerod–Rump 2013（n·u，任意次序）；Lange–Rump 2017/2019（忠实/定向舍入，以及 h·v）。
- **证明过的误差界作为测试容差**：
  - **Grading the BLAS**（Demmel、Henry、Kozachenko、Langou、Li、Riedy、Vanover，arXiv 2609.12307，2026-09）
    - 按 |GEMM − AB| ≤ f(n)·ε·|A||B| 给BLAS评级；只给“真阳性”；
    - 测了 Netlib、BLIS、OpenBLAS、oneMKL、cuBLAS（B200）、ozIMMU、GEMMul8；
    - 只做FP64，混合精度列为未来工作；参考答案是80位long double，不是精确有理数；
    - 有同机置换测试（Test 4b），但不是跨硬件的2B判据。
  - **numoracle**（Xinyi Shi，Duke，arXiv 2610.38983 v2，2026-11-09）
    - 从带精度标注的参考程序推出逐元素接受区域，与次序无关、按累加格式参数化；
    - 主要是概率界（零均值假设），另有固定输入模式；证明是纸面的；
    - 13个Triton算子（Liger、Unsloth、vLLM），在A100与RTX PRO 6000 Blackwell上**各自对照FP64参考**，零误报；
    - **不是**跨GPU的无参考比较。
  - **VERNA**（Fountoulakis、Melnichenko、Georgantzas…、Murray、Mahoney，arXiv 2611.21570，2026-11-16）
    - 提出“超出VSCL中证明的界 ⇒ 拒绝该内核”；
    - **只是设计**，“设计描述留给另一篇论文”；没有硬件或累加器建模，没有跨硬件部分。
  - Brower–Caday–Gunnels（arXiv 2611.19008）已对cuBLAS ADP证明经典点积界可保持，并用Grading the BLAS的测试核对。
- **跨硬件差异的解释（逐位级，没有误差界）**：
  - FPRev（ATC 2025，黑盒反推累加树）；
  - MMA-Sim（MICRO 2026，10种NVIDIA/AMD架构的逐位模型，例如Hopper FP8累加只有13位小数位）；
  - Khattak–Mikaitis 2026（张量核模型）；
  - Valpey等 NFM 2025（张量核的SMT形式化）；
  - FTTN（CCGrid 2024）。
- **跨后端、跨硬件的差分测试**：CRADLE、LEMON、Audee、FreeFuzz、DeepREL、TitanFuzz、NNSmith、Mind the Gap（arXiv 2511.11601，5种加速器）。**容差全部是人为设定的**。

### 没有找到的部分

1. **无参考答案的跨硬件判据**：利用“两边都必须满足同一个界”，以2B为阈值、零误报。没找到。
   - 最接近的是 Grading the BLAS 的同机置换测试，以及ABFT的舍入阈值（A-ABFT、V-ABFT）。
2. **同一棵树里不同节点用不同的 δ**（张量核内部累加精度 ≠ 外层累加器，或块FMA）：没有机器证明。
   - VSCL、lean-ieee754 都是每棵树一个u或一个舍入函数；Blanchard等 2020 只有纸面分析。
3. **从逐位硬件模型（MMA-Sim式的对齐、截断、规格化）推出可用的误差界**，并接到真实内核：没有找到。
   - TensorLean把张量核舍入放在可信基里，论文未出，细节未知。
4. **Lean证明的界当作真实库的测试阈值**：VERNA只是提出，没有实现。numoracle、Grading the BLAS做了，但都是纸面证明。**竞争激烈**，VERNA团队正在做。

### 对我们的含义

- “任意次序误差界”本身**不能**作为贡献。要用的话，应在VSCL（MIT，Lean 4 + Mathlib）之上做，或者引用它。我们的代码只用core Lean，二者不能直接合并，需要决定是否引入Mathlib。
- 迭代精化接受逻辑这条主线**不受影响**：上述工作都是GEMM、点积、逐元素算子的精度，没有一项涉及求解器的接受判据、回退或状态报告。
- 可以加进来的新东西：第1点（跨硬件2B判据）和第2点（分节点δ）。它们可以作为论文中的一个RQ或一节，不宜单独成文。
