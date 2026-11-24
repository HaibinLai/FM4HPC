# 第四轮调研：大模型（LLM）里的低精度库与算子实现

调研日期：2026-11-23。动机：用户想看 LLM 栈里低精度是怎么实现的。本文按 **LLM 推理/训练栈的层次**组织，并标出每一层在**这两台机器**（Mac arm64 CPU、V100 节点）上能测什么、数值上有哪些“值得形式化的点”。

**证据等级：** “已核实”= 本次直接查了GitHub API（仓库元数据，2026-11-23）或直接下载并读了源码；“摘要”= 仅来自搜索结果摘要，**没读过论文或代码原文，不能当事实**。没有克隆、编译或运行任何库。

## 先看两台机器能测什么

| 硬件 | 能做的低精度实验 | 不能做的 |
|---|---|---|
| **Mac（arm64，CPU）** | llama.cpp / ggml 量化内核（含NEON）、BitNet、T-MAC（ARM）、PyTorch的 `float8` 转换（CPU）、微缩放格式仿真（microxcaling） | 所有CUDA内核 |
| **V100 节点（sm_70）** | FP16 Tensor Core；`DP4A` int8点积（llama.cpp的MMQ，摘要：V100上走原生 `__dp4a`，**主线调度未核实**）；`half2` 的 `hsub2/hfma2` 与 `lop3` 指令；vLLM的GPTQ（摘要：文档标注Volta起支持） | **FP8**（需sm89+）、**FP4**（Blackwell）、int8 Tensor Core（Turing+）、Marlin完整内核（Ampere/Ada）、QServe、DeepGEMM、FlashAttention-3/4 |

摘要还指出：vLLM文档里 **AWQ在Volta上不支持**，FP8 W8A8 需要计算能力8.9以上。`NVIDIA/TensorRT-LLM`、`cutlass`、`pytorch/ao`、`GPTQModel` 的许可证在GitHub API里标为 `NOASSERTION`，使用前要自己看文件头。

## 按层次的库与算子

### A. 权重量化的矩阵乘（W4A16 / W8A16）

| 库 | 要点（摘要，除非另注） | 元数据（已核实） | 机器 |
|---|---|---|---|
| [Marlin](https://github.com/IST-DASLab/marlin) | FP16×INT4核；**反量化用位技巧**，见下文“Marlin反量化” | Apache-2.0；2024-09最后提交 | 完整内核Ampere/Ada；**反量化函数本身只用 `lop3`、`half2`，V100可跑** |
| [llm-awq](https://github.com/mit-han-lab/llm-awq) | AWQ：按激活保护重要通道的权重量化 | MIT；182 MB；2025-07 | vLLM文档：Volta不支持AWQ |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | GPTQ工具链 | NOASSERTION；166 MB；2026-11-23 | 摘要：vLLM的GPTQ从Volta起支持 |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3格式（摘要未涉及细节） | MIT；14 MB；2026-11-21 | 需CUDA，未核实架构要求 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | k-quants（Q4_K等）+ CPU/CUDA/Metal | MIT；506 MB；2026-11-23 | **Mac与V100都能跑**（V100经MMQ/DP4A，摘要） |

### B. W8A8（int8）

| 库 | 要点 | 元数据 | 机器 |
|---|---|---|---|
| [smoothquant](https://github.com/mit-han-lab/smoothquant) | 把激活的离群值通过**数学等价的逐通道缩放**转移到权重上；内核用CUTLASS INT8 GEMM | MIT；6 MB；2024-07最后提交 | INT8 Tensor Core需Turing+，V100不行 |
| bitsandbytes（LLM.int8） | **离群特征维度保留FP16**、其余走int8（摘要，来自二手博客） | 见 `REVIEW3.md` | 部分 |

### C. FP8

| 库 / 现象 | 要点（摘要） | 机器 |
|---|---|---|
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 细粒度缩放；**CUDA core两级累加（promotion）** | Hopper+ |
| DeepSeek-V3训练 | 激活按 1×128 tile、权重按 128×128 block 各自缩放；全部GEMM用E4M3；**H800的FP8 Tensor Core累加约14位**，K=4096时最大相对误差接近2%，所以把部分和提升到CUDA core做FP32累加（“每128个元素”只来自二手总结，**未核对原文**） | Hopper+ |
| vLLM FP8 KV cache | 默认未校准时缩放因子为1.0；Triton的float→fp8 store**不饱和**，所以内核要显式钳位到E4M3有限范围；**vLLM博客（2026-04）称**Hopper上FP8 KV在128k大海捞针任务精度从91%掉到13%，原因是Tensor Core的FP32累加不精确，**两级累加**修复到89%（博客数字，**未复核**） | sm89+ |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | vLLM推荐的校准工具 | Apache-2.0；43 MB；2026-11-21 |

### D. FP4（NVFP4 / MXFP4）

摘要：两者元素都是E2M1（1符号、2指数、1尾数）。**NVFP4**每16个元素一个FP8 E4M3尺度，外加一个张量级FP32尺度，约4.5位/值；**MXFP4**每32个元素一个E8M0（2的幂）尺度，约4.25位/值。NVIDIA称E4M3尺度编码误差（0.08 MSE）远小于E8M0（0.72），独立研究认为MXFP4的2的幂尺度是精度主要问题。硬件原生支持在Blackwell；之前的卡上vLLM/SGLang走Marlin的W4A16回退。**以上数字多为厂商数据或二手博客。**

### E. 注意力与KV cache

| 库 | 要点 | 元数据 | 机器 |
|---|---|---|---|
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | LLM服务的注意力内核库 | Apache-2.0；359 MB；2026-11-23 | 未核实架构要求 |
| [KIVI](https://github.com/jy-yuan/KIVI) | 2位KV；**key按通道量化、value按token量化**，非对称 | MIT；17 MB；2025-11 | CUDA内核，Python部分可在CPU看 |
| [omniserve（QServe）](https://github.com/mit-han-lab/omniserve) | W4A8KV4，“渐进式分组量化”让W4A8在INT8 Tensor Core上跑；CUDA+PTX | Apache-2.0；5 MB；2025-03 | INT8 TC，V100不行 |
| FlashAttention / SageAttention | 见 `REVIEW2.md` | — | Ampere+ |

### F. 1位 / 三值 / 查表

| 库 | 要点（摘要） | 元数据 | 机器 |
|---|---|---|---|
| [BitNet](https://github.com/microsoft/BitNet)（bitnet.cpp） | 三值权重 {-1,0,1}（1.58位）；两套内核：三值查表 `TL` 与 `I2_S`（2位+尺度）；论文称 `I2_S` “无损”推理、查表内核用 **16位中间和**；ARM上1.37~5.07倍、x86上2.37~6.17倍（作者数据） | MIT；8 MB；2026-07；42434星 | **CPU，Mac可跑** |
| [T-MAC](https://github.com/microsoft/T-MAC) | **用查表代替反量化和乘法**的混合精度矩阵乘（int1/2/3/4 × int8/fp16/fp32） | MIT；20 MB；2025-06 | CPU，ARM可跑 |

### G. 推理引擎与训练框架（编排层）

[vLLM](https://github.com/vllm-project/vllm)（Apache-2.0；380 MB；2026-11-23）、[SGLang](https://github.com/sgl-project/sglang)（Apache-2.0；540 MB）、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)（NOASSERTION；2200 MB）：数值行为主要在它们调用的内核里（A~F），自身体量太大，不适合作为单个验证对象。训练侧见 `REVIEW2.md`（TransformerEngine、DeepGEMM）与 `REVIEW3.md`（torchao、bitsandbytes）。

## 值得形式化的数值点（本次最重要的部分）

### ① Marlin / FasterTransformer 的INT4→FP16反量化位技巧（**已读源码**）

`marlin/marlin_cuda_kernel.cu` 第128~152行的 `dequant(int q)`：
- 低半：`lo = (q & 0x000f000f) | 0x64006400`。`0x6400` 是 FP16 的 **1024.0**，而 1024~2047 区间内FP16的最小间隔恰好是1，所以把4位整数 `n` 直接拼进尾数低位，得到的就是**精确的 `1024 + n`**。再 `__hsub2(lo, 0x64086408)` 减去 `1032 = 1024 + 8`，得到**有符号的 `n − 8`**（零点 8 融进了减法）。
- 高半：`hi = (q & 0x00f000f0) | 0x64006400` 得到 `1024 + 16n`；再做 `__hfma2(hi, 0x2c002c00, 0xd480d480)`，即 `(1024+16n)·2^-4 + (−72) = n − 8`（`0x2c00` 是 `2^-4`，`0xd480` 是 `−72`）。
- 整个过程只用了 `lop3` 与两条 `half2` 指令。**我在CPU上用numpy的 `float16` 对全部16个半字节值做了检查**（常数确认为 `0x6400=1024`、`0x6408=1032`、`0x2c00=2^-4`、`0xd480=−72`；融合乘加用精确算术再舍入一次来模拟）：两条路径的结果对16个值**全部精确等于 `n−8`**。这只验证了半字节级的算术，**不是对GPU上完整 `2^32` 个输入的测试，也不是Lean证明**。这是“利用 FP16 的间隔恰好等于 1 的区间做整数到浮点转换”的经典技巧，同样出现在 FasterTransformer 和其他W4A16内核里。
- **为什么适合我们：** 它完全落在已有的 `Rep 11`（FP16 格）机制里——`1024+n`、`1024+16n` 的可表示性、乘 `2^-4` 精确、减法/融合乘加的结果可表示；证明很小。**而且 `lop3` 和 `half2` 在V100上都能执行，所以可以把这个函数单独抽出来，在V100上对全部 `2^32` 个输入做穷举测试**（即使整个Marlin在V100上跑不起来）。我没有做这件事，只是判断它可行。

### ② FP8转换的饱和与NaN语义
E4M3 范围很小（vLLM文档摘要写“±240”，这是某些变体；通常的 `e4m3fn` 最大值是448，**两者不一致，需核实具体变体**）；Triton的转换不饱和所以要手动钳位。`torch.float8_e4m3fn` 在CPU上就有转换，**Mac上可测**。

### ③ 低精度Tensor Core累加器到底做了什么
FP8 KV精度崩溃、DeepSeek的“14位累加”、以及我们在OzBLAS那条线上悬而未决的问题（`REVIEW`/`HANDOFF` 第14.5节第5条：Tensor Core累加器是否满足“结果可表示则精确”）是**同一个问题**。H100/H800 没法在这里测，但 **V100 的 FP16 Tensor Core 可以用受控输入做微基准**（我印象中有文献研究过V100的累加行为，比如乘积精确、累加有对齐后的截断，**这是我的记忆，本轮没有核实**）——**这是一个既有研究价值、又直接在本节点能做的实验**。

### ④ 块缩放因子的计算
MX / NVFP4 的共享尺度本质是 `floor(log2(amax)) − emax_elem` 或 E4M3 编码；与此前发现的“GPU `log2` 对2的幂偏差±1”同题材，**落在2的幂附近最容易出错**。microxcaling（`mx_ops.py` 10 KB）可在Mac上直接测。

### ⑤ 三值/查表内核的中间和位宽
bitnet.cpp 的查表内核用 **16位中间和**：与我们证过的“位宽预算”（`K·4^s ≤ 2^24`）同类问题，这里是**int16溢出界**。CPU可测。

### ⑥ KV量化的仿射量化
KIVI 的“按通道量化 key、按token量化 value”本质是带零点的仿射量化；误差界和零点舍入是小而清楚的整数/浮点混合问题。

## 建议（供用户选择，我没有开始任何一项）

1. **Marlin反量化技巧**：Lean证明 + 在V100上对全部 `2^32` 个输入穷举测试。最小、最干净，用到我们已有的FP16格工具，而且**绕开了“Marlin在V100上跑不起来”的限制**。
2. **V100 Tensor Core累加器微基准**：用受控输入回答“累加器是否对可表示结果精确、对齐截断多少位”，同时解决OzBLAS线上遗留的第14.5节第5条，并为理解FP8累加问题提供类比。
3. **BitNet / T-MAC 的查表内核**（Mac CPU）：整数/查表内核，16位中间和的溢出界，“无损”主张可直接检验。
4. 备选：gemmlowp的定点重量化（`REVIEW3.md`，纯整数、核心约10行）。

## 局限
- 所有“摘要”条目（精度数字、加速比、`-240` vs `448`、“14位累加”“每128个元素”、V100上的支持情况）都**没有核对原文**，只能作为线索。
- 没有克隆、编译或运行任何库；“能在V100/Mac上跑”是对依赖的判断。
- 只直接读了Marlin的 `dequant` 函数；其余库的源码这一轮没有读。
- 没有调研：MoE内核、通信量化（FP8 all-reduce）、投机解码、LoRA/QLoRA、采样与logits的低精度、编译器路线（Triton/MLIR）。
