# 非BLAS算子库补充调研：低精度换高精度

调研日期：2026-11-23。动机：BLAS（GEMM）里的Ozaki类实现已相当成熟，想找**算子级**（求解器、注意力、优化器、FFT等）里“用低精度部件达到更高精度或更稳定精度”的开源库，作为OzBLAS之外的第二个验证对象。

**证据等级：** “已核实”指本次直接查询了GitHub API（仓库元数据、文件名），日期为2026-11-23；“摘要”指仅来自搜索结果摘要，**没有读过代码或论文原文，不能当作事实**。没有克隆或审计任何一个候选库，没有运行任何一个。

## 先看一条硬约束：这台V100能测什么

计算节点是8×Tesla V100（sm_70）。V100只有**FP16** Tensor Core：没有BF16、INT8、FP8 Tensor Core。所以：

| 需要的硬件 | 候选 | 在本节点能否实机测试 |
|---|---|---|
| FP16 Tensor Core 及以下 | OzBLAS（已测）、MAGMA FP16 LU、tcFFT、纯PyTorch优化器 | 能 |
| Ampere（sm80+） | SageAttention、Marlin、FlashAttention-2 | 不能 |
| Hopper/Blackwell（sm90/100） | DeepGEMM、FlashAttention-3/4、TransformerEngine的FP8 | 不能 |

对Lean目标，源码级规格不依赖本机硬件；但“编译结果→硬件执行”这一层的证据在不能实机的库上会缺失。**如果能拿到Ampere或Hopper机器，下面的排序会变。**

## 候选清单

| 库 | 机制：低精度怎样换高精度 | 许可证、体量、最近提交（已核实） | 本节点可测 | 数值核心规模与Lean适合度（本次判断） |
|---|---|---|---|---|
| [optimi](https://github.com/warner-benjamin/optimi)（及 torchdistx 的 AnyPrecisionAdamW） | bf16权重 + **Kahan补偿缓冲**，模拟FP32主权重（摘要：显存省37~45%） | MIT；约0 MB（纯Python）；2026-10-24 | 能（PyTorch；V100上bf16做存储/逐元素运算，未验证） | **极小**：每个优化器一个文件，核心几行；Kahan有清晰的“值=高位+补偿”结构，与Ozaki分解同构。适合快速出第一个定理 |
| [MAGMA](https://github.com/icl-utk-edu/magma) | **FP16/FP32低精度LU + FP64残差迭代精化**（IR/GMRES-IR），达到FP64精度（摘要：约4倍加速） | BSD-3；31 MB；2026-11-17。已核实存在：`xhsgetrf_gpu.cpp`、`xshgetrf_gpu.cpp`、`dxgesv_gmres_gpu.cpp`、`zcgesv_gpu.cpp`、`README_FP16_Iterative_Refinement.txt` | **能**（摘要：原始FP16 Tensor Core实验就在V100上做） | LU本身是分块算法、难形式化；但**迭代精化层很小**，收敛有明确条件（摘要：κ∞(A)<1e8，工作/残差精度FP32/FP64）。可把LU当黑盒规格，只证精化层 |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | FP8 GEMM + 细粒度缩放 + **CUDA core两级累加（promotion）**，补偿FP8 Tensor Core累加精度不足 | MIT；2 MB；2026-11-19。已核实目录含 `csrc`、`deep_gemm`、`tests`；摘要称现支持SM90/SM100 | **不能**（Hopper+） | 代码小、结构清楚；核心是“每隔固定K把部分和提升到FP32”。**“FP8 Tensor Core只累加约14位”只来自第三方博客，README没写，未核实**；这条硬件事实正是需要实测的 |
| [SageAttention](https://github.com/thu-ml/SageAttention) | INT8量化QK^T（**对K做平滑**）+ FP16/FP8的PV | Apache-2.0；63 MB；2026-01-17 | **不能**（摘要：Ampere/Ada/Hopper） | 有一个干净的精确恒等式：softmax对K减去按token的均值不变（每行多一个常数）；其余是量化误差界。适合做“精确不变性 + 量化误差”两层规格 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention)（FA3/FA4） | FA3：FP8分块量化 + **incoherent processing**（随机正交矩阵摊平离群值）；FA4（摘要）：用FP32 FMA上的3次多项式**模拟exp2**并按Cody-Waite范围约化 | BSD-3；24 MB；2026-11-23 | **不能**（Hopper/Blackwell）；online softmax算法本身可用参考实现在V100上测 | online softmax的重缩放有经典代数不变量，适合先证；exp2多项式的误差界（摘要：FP32最大相对误差8.8e-5，BF16下与硬件在1 ULP内的占99%）可做区间算术式证明；但摘要数字来自作者测试、未复核 |
| [tcFFT](https://github.com/rox906/tcFFT)（及若干fork） | FP16 Tensor Core做FFT | **GPL-3.0**；最近提交2021-05（主仓），其余fork到2024-04；摘要称相对误差约1.7% | 能（论文面向V100/A100） | 目标是**快**而不是更高精度（1.7%误差不属于“模拟高精度”）；仓库陈旧。另有摘要提到OSTI论文用按列缩放的动态分裂模拟FP32精度，但未找到对应开源代码 |
| [Marlin](https://github.com/IST-DASLab/marlin) | FP16×INT4核，INT4反量化后做FP16 Tensor Core | Apache-2.0；2024-09最后提交 | 不能（Ampere/Ada） | INT4→FP16反量化本身精确，数值问题只在FP32累加；代码陈旧、价值有限 |
| [Ginkgo](https://github.com/ginkgo-project/ginkgo) | 存储精度与计算精度分离（accessor）、压缩基GMRES、自适应精度预条件子 | BSD-3；168 MB；2026-11-22 | 能（V100） | 体量大；我只在摘要里确认了accessor和CB-GMRES，**没有确认它有FP16 LU+IR** |
| [SLATE](https://github.com/icl-utk-edu/slate) | 含混合精度迭代精化（摘要） | BSD-3；22 MB；2026-08-27 | 需MPI/多卡 | 比MAGMA更重，暂缓 |
| [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | FP8训练配方（amax历史、延迟缩放） | Apache-2.0；29 MB；2026-11-21 | 不能（Hopper+） | 体量大、策略性强，不适合首个目标 |

## 与OzBLAS的结构对比

| 共同结构 | 出现在 |
|---|---|
| “高位部分 + 补偿/残差”表示（值=hi+lo） | OzBLAS分片、optimi的Kahan缓冲、DeepGEMM的两级累加、MAGMA的残差迭代 |
| “低精度做重活，高精度只做修正” | OzBLAS、MAGMA、DeepGEMM |
| “规格 = 精确恒等式 + 舍入误差界”两层 | SageAttention的K平滑、FlashAttention的online softmax、Kahan |

所以OzBLAS里已经证明的**“位宽预算使片积在有限位数内精确”**这类引理，思路可以复用于DeepGEMM（累加位宽）和Kahan（补偿位宽）；MAGMA则对应完全不同的一类定理（迭代收敛）。

## 建议

按“本节点可实机验证 + 数值核心小 + 与现有工作互补”排序：

1. **optimi / Kahan补偿优化器**：最小、最快出第一个可检查定理，纯PyTorch，能在任意节点复现；与Ozaki分解同构，能检验“规格可复用”这个论点。需要先读 `adamw.py`（约5 KB）确认补偿具体写法，**不要凭论文描述建模**。
2. **MAGMA 的 FP16 LU + 迭代精化**：算子级、V100可实机、能展示“信任边界分层”（把LU当黑盒，只证精化层）。需先读 `README_FP16_Iterative_Refinement.txt` 和 `dxgesv_gmres_gpu.cpp`。
3. **仅规格层、等硬件**：DeepGEMM、SageAttention、FlashAttention-3/4。可先做纯数学部分（K平滑不变性、online softmax重缩放、promotion的位宽预算），实机验证需要Ampere/Hopper。
4. 暂不选：tcFFT（目标不对、仓库陈旧、GPL）、Marlin（陈旧、价值低）、Ginkgo/SLATE/TransformerEngine（体量大）。

## 局限

- 所有“摘要”条目的数字（加速比、14位、2.6倍、8.8e-5、1.7%等）都**没有核对原文**，只可作为线索。
- 没有搜索到的方向不代表不存在：例如归一化（LayerNorm的Welford）、稀疏、随机舍入内核没有单独调研。
- 仓库的“最近提交日期”只说明活跃度，不说明数值核心稳定。
