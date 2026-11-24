# 第三轮调研：整数量化与低精度格式库（可在CPU上跑的）

调研日期：2026-11-23。动机：此前两轮（`REVIEW.md`、`REVIEW2.md`）偏向GPU和Ozaki类模拟高精度；这台V100只有FP16 Tensor Core，Ampere/Hopper以上的库没法实机测。这一轮换一类：**量化推理内核、整数GEMM的重量化、低精度格式仿真库**——它们的数值核心多半是整数运算，规格清楚，**可以在本机（Mac CPU）直接编译和运行**，也最适合Lean。

**证据等级：** “已核实”= 本次直接查了GitHub API（仓库元数据、文件大小，2026-11-23）或直接下载并读了源码；“摘要”= 仅来自搜索结果摘要，**没读过代码或论文原文，不能当事实**。没有克隆、编译或运行任何候选库。

## 候选清单

| 库 | 低精度机制 | 许可证 / 体量 / 最近提交（已核实） | 本机可跑 | 数值核心与Lean适合度（本次判断） |
|---|---|---|---|---|
| [gemmlowp](https://github.com/google/gemmlowp)（`fixedpoint/fixedpoint.h`） | int8量化GEMM的**定点重量化**：`SaturatingRoundingDoublingHighMul`（取 `round(a·b/2^31)` 的饱和定点乘）+ `RoundingDivideByPOT` | Apache-2.0；3 MB；**2024-01最后提交**（维护模式，但被TFLite等广泛嵌入） | 应能（C++头文件库，**未验证编译**） | **极小而精确**：`fixedpoint.h` 共914行，我已下载读过 `SaturatingRoundingDoublingHighMul` 的完整函数体（约10行：64位乘积、按符号选 `nudge = ±2^30`、除以 `2^31`、仅 `(INT32_MIN, INT32_MIN)` 饱和）。纯整数、无浮点语义问题；规格是“按某舍入规则取 `a·b/2^31`”；**非常适合Lean** |
| [tflite-micro](https://github.com/tensorflow/tflite-micro)、[CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) | 同一重量化链的**独立实现**：把实数乘数 `M` 量化成 `(M0, shift)`（`QuantizeMultiplier`），再用定点乘 + 带舍入的右移；CMSIS-NN有C参考实现和DSP/MVE SIMD实现 | 均Apache-2.0；37 MB / 8 MB；2026-11-19 / 2026-11-23 | 应能（C/C++，未验证） | 是gemmlowp规格的**对照伙伴**：同一数学规格，多个后端。摘要提到“在每个后端逐位一致”和“双重舍入路径”是真实的差异来源；**一个Lean规格可同时检验多份实现**，正好贴合“规格复用、优化后重新验证”的主题 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)（`ggml/src/ggml-quants.c`） | 分块量化（Q8_0、Q4_0、Q4_K等）：每块一个fp16尺度 + 整数元素；点积用int8×int8累加到int32，再乘尺度 | MIT；506 MB / 38 MB；2026-11-23。已核实 `ggml-quants.c` 为228 KB、`ggml-quants.h` 10 KB | 能（含ARM NEON路径；未验证编译） | 数值核心分散在一个大文件里，但**每种格式独立**，可从最简单的Q8_0（32元素一块）开始；整数累加的精确性条件与我之前证过的 `K·4^s ≤ 2^24` 同类（int8乘积累加到int32几乎总安全）。摘要提到CPU路径对激活先量化成q8再点积、可能放大离群值误差，**未证实** |
| [microxcaling](https://github.com/microsoft/microxcaling) | OCP Microscaling（MX）格式的PyTorch仿真：每块共享一个2的幂指数，元素用低位宽格式 | MIT（摘要，上游许可证文件未读）；<1 MB；2026-07-17。已核实 `mx/` 下 `mx_ops.py` 10 KB、`elemwise_ops.py` 10 KB、`formats.py` 3 KB、`quantize.py` 1 KB | 能（纯PyTorch，本机已装CPU版） | 规格来自**已发布标准**（OCP MX规格 + 论文 arXiv 2310.10537），库是“符合标准的仿真”；核心是 `floor(log2(max))` 求共享指数 + 元素舍入 + 钳位，与此前 `tau` 的 `log2` 现象同题材；仿真用float32计算，所以证明对象是“量化函数”，不是硬件 |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 8位优化器：分块 absmax 归一化 + 256项非均匀码本查表；更新在fp32里做，状态存8位 | MIT；6 MB；2026-11-02 | 部分（核心是CUDA内核，CPU路径需另核实） | 数值核心清楚（最近邻码本 + 反量化），但规格不像定点乘那样一句话能说清；与optimi（已做）同属优化器状态压缩，可作对照 |
| [QPyTorch](https://github.com/Tiiiger/QPyTorch) | 低精度浮点/块浮点/定点的PyTorch仿真（就近偶数与随机舍入） | MIT；<1 MB；**2024-05最后提交**（陈旧） | 能 | README自己说：矩阵乘等复合运算仍在单精度里算，**不适合研究累加策略**；是仿真工具而非内核库，验证价值低 |
| [pytorch/ao](https://github.com/pytorch/ao)（torchao） | float8训练缩放配方、int8仅权重量化 | **许可证 NOASSERTION**（需自行查文件头）；175 MB；2026-11-22 | 部分（真实float8内核需新硬件） | 体量大、依赖 `torch.compile` 与新硬件，不适合作为下一个目标 |
| [XNNPACK](https://github.com/google/XNNPACK) | 大量CPU低精度算子 | **许可证 NOASSERTION**；206 MB；2026-11-23 | 能但体量过大 | 过大，暂缓；其重量化路径可作为gemmlowp规格的又一个对照对象 |
| [Apex](https://github.com/NVIDIA/apex)（融合LayerNorm） | 半精度融合LayerNorm，Welford在线方差 | （未查元数据） | 需CUDA | 摘要：用Welford避免平方和相消；**未找到半精度误差分析**，需GPU，暂缓 |

## 为什么这一类值得做（相对已做的OzBLAS和optimi）

| 共同点 | 内容 |
|---|---|
| **纯整数、可全程精确** | gemmlowp的核心不依赖浮点语义，不需要“GPU `log2` 现象”或“Tensor Core累加器”这类硬件假设；Lean里的模型就是代码本身 |
| **多实现 + 一个规格** | 同一个重量化规格被gemmlowp、tflite-micro、CMSIS-NN（C与SIMD）、XNNPACK等分别实现；这正是“优化后需要重新验证是否仍满足规格”的现实案例 |
| **本机可测** | 不需要GPU，也不需要新硬件；测试向量可穷举极端值（`INT32_MIN`、平局、负数） |
| **与已有结论同类** | 舍入规则、平局取舍、饱和、位宽预算，与分片引理和Kahan的整数格思路一脉相承 |

## 建议

1. **下一个目标：gemmlowp的定点重量化（`fixedpoint.h`）**，并以tflite-micro / CMSIS-NN为对照实现。
   - Lean目标（均为整数、核心Lean即可）：
     ①`SaturatingRoundingDoublingHighMul` 等于“`round(a·b/2^31)`，平局向远离零方向取舍（或等价的单个下取整式）”，仅 `(INT32_MIN, INT32_MIN)` 饱和；
     ②`RoundingDivideByPOT` 的舍入规则；
     ③两者复合后与理想实数运算 `round(acc·M)` 的误差界（含**双重舍入**可能多出的1个最低位）；
     ④`QuantizeMultiplier` 的 `(M0, shift)` 表示的精确性。
   - 先读源码再建模；先用C++/Python在本机对**穷举的小位宽版本**和随机/极端的int32输入核对，再进Lean。
   - 摘要里第三方声称“gemmlowp的带符号 `nudge` 加截断除法等价于单个 `floor((a·b + 2^30)/2^31)`”：**这正好可以用Lean证明或证伪**，不要先当成事实。
2. **第二个**：ggml的 `Q8_0`（最简单的块量化）及其int8点积，作为“整数累加精确 + fp16尺度”的例子。
3. **第三个**：microxcaling 的量化函数（共享指数 + 元素舍入），对照OCP MX标准。
4. **暂缓**：torchao、XNNPACK（体量与许可证待查）、QPyTorch（陈旧且不模拟累加）、bitsandbytes（内核在CUDA里）、Apex（需GPU）。

## 局限

- 所有“摘要”条目（如“双重舍入路径”“逐位一致”“提升/放大离群值误差”）都**没有核对原文**，只可作为线索。
- 没有克隆、编译或运行任何候选库；“本机可跑”是对依赖的判断，不是实测。
- 只直接读了gemmlowp的 `SaturatingRoundingDoublingHighMul` 函数体和 `RoundingDivideByPOT` 的行位置；`RoundingDivideByPOT` 函数体、tflite-micro、CMSIS-NN、ggml的源码这一轮**没有读**。
- 没有调研的方向不代表不存在：稀疏低精度、posit、随机舍入专用库、编译器级低精度（Triton/MLIR）等没有单独看。
- 仓库“最近提交日期”只说明活跃度，不说明数值核心稳定（gemmlowp 2024-01后无提交）。
