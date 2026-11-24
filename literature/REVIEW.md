# 低精度计算高精度 HPC 算子的数值验证：文献调研

调研日期：2026-11-22（America/Chicago）。范围：低精度乘法/整数矩阵单元模拟 FP32/FP64 GEMM，以及可用于证明其数值性质的工具。本文是阅读原论文、作者项目和官方文档得到的研究判断；没有声称复现所有论文，也没有把工具 README 的声明当作独立验证结果。

## 1. 建议选题与判断

建议首先研究 **ozIMMU / Ozaki scheme I 的 INT8→FP64 GEMM**。它直接对应“低精度拟合高精度”，可拆成：输入按共享指数切片、INT8×INT8 的 INT32 精确点积、缩放并用 FP64 重组。整数乘加部分能绕开一部分 Tensor Core 浮点累加语义的不确定性，适合先在 Lean 证明无溢出和精确性。第二选择是 Ootomo 的 FP16/TF32→FP32 恢复方案。

研究目标应写为：在明确输入域、内积长度、切片数和舍入语义下，证明真实实现满足误差契约，并在条件不满足时找到可复现反例。现有数值误差分析很丰富；“首次对低精度 GEMM 分析误差”不成立。更值得尝试的是把已有分析落到固定版本开源实现，自动检查适用条件，生成 Lean 可检查的证据。

必须区别：

| 结果 | 能说明什么 | 不能说明什么 |
|---|---|---|
| 与 cuBLAS/BLAS 相近 | 给定样本上与某实现接近 | 对所有输入正确、标准库是真实答案 |
| 与精确有理数或 MPFR 参考相近 | 样本上的数值误差 | 全输入保证 |
| 纸面误差定理 | 算法在假设下的误差界 | 源码、编译器、设备符合模型 |
| Lean 中的算术定理 | 形式化模型满足给定性质 | 自动涵盖 CUDA 并行执行和硬件语义 |
| 固定编译结果的 refinement 证明 | 实现和模型之间的对应关系 | 未覆盖架构、编译选项或输入域 |

## 2. 优先阅读的 12 项文献与技术资料

### 1. Henry, Tang, Heinecke — Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations（ARITH 2019）

- 原文：[PDF](https://arxiv.org/pdf/1904.06376)；[摘要](https://arxiv.org/abs/1904.06376)。已打开全文。
- 内容：将 FP32 数拆成多个 BF16 分量，用低精度乘法与 FP32 累加恢复更高精度的点积和矩阵乘。
- 价值：给出 multi-component arithmetic 的基础切入点；适合研究分解精确性、哪些交叉项可以省略、误差如何累积。
- 限制：其准确性和性能结论依赖硬件/运算模型；论文的性能投影不能当作当前 GPU 实测。不是形式化证明项目。

### 2. Ootomo, Yokota — Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance（IJHPCA 2022；arXiv 修订 2023）

- 原文：[PDF](https://arxiv.org/pdf/2203.03341)；[DOI](https://doi.org/10.1177/10943420221090256)。已打开全文。
- 内容：基于 CUTLASS，通过残差修正、缩放和累加设计恢复 FP32 水平精度；区分 FP16 受限指数域与 TF32 路径。
- 价值：明确指出 Tensor Core 内部舍入和修正项下溢会影响方案有效性，是“代码看起来等价，设备数值行为不同”的直接例子。
- 限制：“达到 FP32 精度”不可解释为逐位等于任意 cuBLAS SGEMM。论文对特定 GPU 的发现也不能无条件推广。

### 3. Ootomo, Ozaki, Yokota — DGEMM on Integer Matrix Multiplication Unit（arXiv 2023；IJHPCA 2024）

- 原文：[全文](https://arxiv.org/html/2306.11975v3)；[DOI](https://doi.org/10.1177/10943420241239588)；[ozIMMU](https://github.com/enp1s0/ozIMMU)。已阅读算法 1–4、BPS 推导及实验结构。
- 内容：将输入分解成整数切片，调用整数 GEMM，再恢复尺度并累加为 FP64；研究切片数量和精度/速度的关系。
- 可证明点：给定切片绝对值上界和内积长度，证明 INT32 累加不溢出，随后处理截断和 FP64 重组误差。
- 限制：固定切片数不是对所有指数分布的 FP64 保证；异常输入和共享指数损失须单独处理。HTML 的早期稿含模板元数据，书目信息采用期刊 DOI。

### 4. Fasi, Higham, Lopez, Mary, Mikaitis — Matrix Multiplication in Multiword Arithmetic: Error Analysis and Application to GPU Tensor Cores（SISC 2023）

- 原文：[作者 PDF](https://nickhigham.files.wordpress.com/2023/02/fhlm23.pdf)；[DOI](https://doi.org/10.1137/21M1465032)。已打开全文。
- 内容：建立 multiword GEMM 的误差分析，并结合模拟和 V100/A100 实现研究 double-fp16。
- 关键提醒：两个算法可能满足相同最坏误差界，实际误差分布仍不同。因此只证明一个宽松上界不足以解释“精度是否等同”。
- 对项目意义：报告应同时列理论保证、观察到的误差、条件数/抵消程度；不能只给一次 allclose。

### 5. Abdelfattah, Dongarra, Fasi, Mikaitis, Tisseur — Analysis of Floating-Point Matrix Multiplication Computed via Integer Arithmetic（arXiv 2025；期刊 DOI 已公开）

- 原文：[PDF](https://arxiv.org/pdf/2506.11277)；[DOI](https://doi.org/10.1137/25M1768539)。已打开全文，读取研究范围与结论；链接可能指向持续修订版本。
- 内容：分析整数切片 GEMM 的误差并估计达到指定精度所需的乘法数量。
- 关键结果：A 的行或 B 的列尺度不均匀时，算法可能不准确或不高效。
- 对项目意义：这是最直接的理论基线。把其假设翻译成可执行检查/Lean 前提，比重新发现“大动态范围会失败”更有研究价值。此项是数值分析，不等于已验证 ozIMMU 全部源码。

### 6. Ozaki, Uchino, Imamura — Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique（arXiv 2025；期刊版 2026）

- 原文：[2025 v2 全文](https://arxiv.org/html/2504.08009v2)；[期刊 DOI](https://doi.org/10.1177/10943420261467787)。已读取算法构成、符号约定和实验概述。
- 内容：用中国剩余定理重构整数结果，减少传统切片交叉乘积的成本，并允许通过乘法次数控制精度。
- 对 Lean 的价值：模数互素、重构范围和唯一性有清晰的离散数学证明目标。
- 限制：论文数值运算假设排除溢出/下溢；不能把 CRT 精确重构直接等同于整个浮点 GEMM 正确。第一轮建议先完成 Ozaki I，再扩展 II。

### 7. Boldo, Melquiond — Flocq: A Unified Library for Proving Floating-point Algorithms in Coq（ARITH 2011）

- 原文：[作者 PDF](https://guillaume.melquiond.fr/doc/11-arith20-article.pdf)；[项目与安装文档](https://flocq.gitlabpages.inria.fr/)。已打开论文与项目。
- 内容：多基数、多精度浮点/定点形式化，提供舍入和数值算法证明的基础。
- 用途：作为 Lean 建模的参考，或直接在 Rocq 中开展数值证明。
- 限制：Flocq 是算术基础库，不会自动导入 CUDA 算子。换证明语言也不能消除源码到模型的对应问题。

### 8. Gappa — 数值性质自动证明工具（持续维护的工具资料）

- 原始资料：[官方入口](https://gappa.gitlabpages.inria.fr/)；[用户手册](https://gappa.gitlabpages.inria.fr/gappa/)；[其他工具集成](https://gappa.gitlabpages.inria.fr/gappa/tools.html)。已读取官方入口。
- 能力：表达浮点/定点舍入和区间约束，证明数值性质；可与 Why3、Rocq 集成。
- 用途：先验证缩放、一次切片、一次重组这样的短表达式；支持从自动分析走向证明检查。
- 限制：需要准确输入模型和前提，不能作为整套 GPU GEMM 的开箱即用验证器。Gappa 输出不自动成为 Lean 证明。

### 9. Appel, Kellison — VCFloat2: Floating-Point Error Analysis in Coq（CPP 2024）

- 原始资料：[会议页面](https://popl24.sigplan.org/details/CPP-2024-papers/14/VCFloat2-Floating-point-error-analysis-in-Coq)；[项目](https://verinum.org/vcfloat/)；[代码](https://github.com/VeriNum/vcfloat)；[DOI](https://doi.org/10.1145/3636501.3636953)。项目页和会议摘要已读取；Princeton PDF 访问返回 403，未把它计作已读全文。
- 内容：在 Coq 中自动计算并证明浮点表达式误差界，改进精度、性能、非标准格式和外部函数接口。
- 用途：为项目提供“可组合的误差证明”设计参考。
- 限制：需要表达式及外部函数契约；并发、编译器和 Tensor Core 行为须另外连接。

### 10. Solovyev 等 — Rigorous Estimation of Floating-Point Round-off Errors with Symbolic Taylor Expansions（TOPLAS，2018 在线/2019 卷期）

- 原始资料：[作者页面](https://soarlab.org/publications/2018_toplas_sbbjrg/)；[代码](https://github.com/soarlab/FPTaylor)；[DOI](https://doi.org/10.1145/3230733)。已读取作者发布页，未全面复核实现。
- 内容：FPTaylor 用符号 Taylor 展开和严格全局优化推导误差界，并支持 HOL Light 检查的证明证书。
- 用途：与区间传播比较界的紧致程度；适合小规模算术片段。
- 限制：完整矩阵计算不适合直接逐项展开；需要块级不变量和组合推理。HOL Light 证书与 Lean 证书是不同产物。

### 11. SMT-LIB FloatingPoint theory（2010/2015 理论文献；当前规范 2.7）

- 原始资料：[官方规范](https://smt-lib.org/theories-FloatingPoint.shtml)。已读取。
- 内容：二进制浮点类型、舍入方式和运算的标准 SMT 表达。
- 用途：借助支持此理论的求解器为短表达式、固定长度点积、边界条件搜索反例；反例应在独立执行模型中重放。
- 限制：大 GEMM 展开通常代价高；SAT 反例和 UNSAT 结果都受编码正确性约束。把求解器结论接入 Lean 需要明确证明重建或可信边界。

### 12. Lean 浮点与 GPU 形式化生态（2026 资料核查）

- [FloatLib](https://github.com/lean-dojo/FloatLib)：作者宣称有可执行算术和机器检查证明；其 README 同时列出尚未证明的 IEEE 完整一致性部分，`is754version2019` 仍为 false。应逐一定理审查，不能据名称声称完整 IEEE 已验证。
- [FloatSpec](https://github.com/Beneficial-AI-Foundation/FloatSpec)：README 描述 Flocq 风格移植、误差层和状态审计；无 `sorry` 的声明不等于语义忠实性已经独立复核。此轮只读文档，未构建审计。
- [Lean Kernel Verify](https://github.com/facebookresearch/lean_kernel_verify)：README 明确不验证 numerics；检查覆盖 case 输入，且外部 clone 的完整前端运行存在限制。可辅助安全验证，不能代替本项目的数值误差证明。
- [Formalizing PTX in Lean](https://sf-tensor.com/engineering/formalizing-ptx)：作者报告对若干 GPU kernel 的形式化实践；应逐个审查指令语义和所证明定理，不能从 GEMM 示例推断已支持所需低精度误差性质。

## 3. 可操作的第一组证明目标

建议先把每个点积作为单元，定义低精度切片及重组顺序。

1. **整数累加安全性**：若长度为 K，每个整数输入满足 |a_i|≤M_A、|b_i|≤M_B，则任意部分和绝对值≤K M_A M_B。要求该上界≤2^31−1，得到 INT32 不溢出的充分条件。注意有符号范围不对称，−128 不能按 127 处理。
2. **切片重建及残差界**：证明 a=Σ s_i a_i+r_a；共享指数、截断、符号和零行都必须进入定义。
3. **省略交叉项的误差**：显式列出实际计算的切片对集合，界定剩余项；不要用“很小”替代前提。
4. **浮点重组误差**：从精确整数子结果出发，对 FP64 缩放、加法、顺序和异常情况作条件化分析。
5. **实现对应**：固定仓库 commit、编译选项和设备，逐步连接分解/重组源码；无法证明的指令作为明确假设保留。

第一条的 Lean 证明可完全不依赖 GPU，也不依赖复杂浮点库；但它只证明一个局部引理。第二至四条完成后才能讨论数值契约，第五条完成前必须称为“算法模型证明”。

## 4. 比较实验的设计建议

- 标准实现：GPU 用原生 FP64 cuBLAS DGEMM；CPU 原型用 NumPy 实际链接的 BLAS，并记录后端。两者都是比较对象，不是数学真值。
- 真值：小尺寸使用 Python Fraction 对二进制输入作精确有理数点积；较大尺寸可用 MPFR 并记录精度和舍入设置。
- 用例：普通随机矩阵、行/列大指数跨度、强抵消、全零行/列、非整 tile 尺寸、切片数不足、接近整数累加界。NaN/Inf/subnormal 应作为单独测试组，与定理输入域对应。
- 指标：最大绝对误差、归一化范数误差、以 |A||B| 为尺度的逐元素误差；结果接近零时避免单独使用相对误差。
- 若没有 NVIDIA GPU，运行 CPU 算法模型可检验反例与误差分解，但必须保留“尚未执行原库 GPU kernel”的状态；不得报告 GPU 性能或原库 bug 已确认。

## 5. 研究空缺与继续条件

值得投入的候选贡献是：**给定真实开源实现和误差目标，自动判断所需切片数/输入条件，并生成可检查的局部证明和可复现反例**。这是目前调研后的方向判断，不是已完成的全球新颖性证明。

若最终只得到“随机矩阵接近 cuBLAS”或“写出了经典 GEMM 的一般误差界”，研究贡献较弱。若能发现现有默认配置的适用范围未说明，证明一个可执行 guard 保证数值契约，并在不同输入族和设备上验证 guard 的成本/覆盖范围，则更接近有说服力的系统工作。

下一轮优先精读 3、5、4；复核固定版本源代码之后再选 FloatLib、FloatSpec 或 Flocq 作为数值基础。需要实际构建、审查公理依赖和前提强度后，才将相应工具标为“本项目已验证可用”。
