# lean4HPC 研究总结与交接

更新日期：2026-11-23（America/Chicago）；上一版为2026-11-22。本文件记录截至本轮会话的实际进度；命令模板不表示已经执行。优先阅读本文件，再阅读 `experiments/fp16_fp64/REPORT.md`（该报告尚未纳入第11节的GPU结果）。

**研究方向已由用户重新界定（2026-11-23），以 `PLAN.md` 为准：验证A类“AI时代写出的算子”（产出快、要基本可靠）与B类“低精度模拟高精度的算子”（产出慢、要求高可靠）。第三、四轮调研（`operators/REVIEW3.md`、`REVIEW4.md`）中的建议不属于这两类，已放弃。**

**最新（第16节）：B1——ozIMMU `fp64_int8_auto` 的选片有两处实现缺陷（计数器越界读栈、零元素丢计数→选3片、误差约1e-3）以及准则层面的问题（平均损失不保证逐元素精度），附修复补丁、GPU实测与Lean定理。**

**第17节：B1——GEMMul8（Ozaki II）。**
- CRT不越界：两种模式、N=2..20都成立（Lean整数部分 + GPU上对移位函数的穷举）。
- 精度只按范数保证，约 log2P−log2√k 位；README的推荐表没有计入k。
- fast模式在行/列范数 ≳2^512 或元素 ≲2^-537 时静默输出0。
- 两种模式都不传播Inf/NaN。

本版相对上一版的主要变化：原库D_H路径已在V100上编译并运行，并完成CPU模型的逐元素对齐、cuBLAS kernel选择调查、形状扫描、极端指数输入和1 ULP差异的根因定位，详见第11节。**同日追加一轮约2小时的覆盖与检测探索（快速模型、多种子、大形状、转置、alpha/beta、库模式、特殊值、病态点积、跨卡、模糊测试），详见第12节；第12节的结论覆盖第11.5节中“GPU式模型只含恰好2的幂”的表述。**

## 1. 当前目标与状态

研究目标：针对开源低精度GPU算子，构建可复用的数值规格与Lean证明流程，并明确从源码、编译结果到硬件执行的可信边界。

当前主线是 **FP16分片模拟FP64 GEMM**，选定OzBLAS；INT8路线保留调研与CPU实验，按用户要求暂缓硬件测试。第一阶段关注正确性与误差，不预设V100上的性能收益。

| 工作项 | 实际状态 |
|---|---|
| 文献与算子库调研 | 已分别完成并保存在两个文件夹 |
| FP16→FP64源码选择 | 已固定OzBLAS提交与选定源码快照 |
| FP16 CPU算术模型 | 已运行，4组输入、56条记录 |
| 累加顺序实验 | 已发现7组配置的正序/逆序差异，含1 ULP见证 |
| 原库GPU对比cuBLAS | **已编译并运行**：4组原始输入、33个形状、18个极端指数输入，见第11节 |
| 计算节点连接 | 助手已能经 `ssh s82` 再 `ssh -i <KEY_PATH> haibin@<NODE_IP>` 进入eightsocket；早先的DNS失败是临时的 |
| CPU模型与GPU逐元素对齐 | 已完成：除一类已定位原因的差异外逐位一致，见第11节 |
| Tensor Core是否执行 | 已有kernel名与SASS证据，**无硬件计数器证据**；与形状有关，见第11节 |
| 1 ULP差异 | 已定位到GPU `log2` 对部分2的幂不精确，使分解指数偏大1，见第11节；2的幂±8 ULP范围的完整GPU `tau` 表见第12节 |
| 快速向量化模型 | `fastmodel.py`，对原模型逐位等价，快约1.5万倍，见第12.1节 |
| 覆盖与模糊测试 | 多种子、形状、转置、alpha/beta、库模式、填充、特殊值、病态点积、大K、跨卡、随机模糊，见第12节 |
| Lean形式化检查 | **已运行**：Lean 4.36.0已装在项目 `.toolchain/`。①第13节 `Kahan.lean`（optimi的Kahan补偿更新，564行）；②第14节 `OzSlice.lean`（OzBLAS D_H分片引理，360行）；③**第15节 `OzRecombine.lean`（OzBLAS的FP64重组，173行）**：任意顺序的双精度累加误差界 `2^53·|误差| ≤ 2(N−1)·Σ|T|`、`alpha=1,beta=0` 的最终合成精确；均无 `sorry`、只依赖标准公理。**尾项与分片引理合成的端到端GEMM误差界、一般alpha/beta的fma尚无Lean证明**；INT8草稿 `ComplexBeta.lean` 仍未检查 |
| 第二个库（optimi） | 已选定并固定提交，Python模型对真实optimi逐位验证；Lean中Fast2Sum精确性、N步守恒、**无前提的多步误差界**在具体bf16格上已证；剩余：模型与PyTorch的连接是经验性的，见第13节 |
| 性能、跨GPU架构、多卡实验 | 未执行 |

**不能将当前进度表述为“OzBLAS已被形式化验证”或“V100实机验证通过”：Lean里目前只证了OzBLAS的分片与片积精确性（第14节），重组与整体误差界没有证明；V100实测是有限样本（第11、12节）。**

## 2. 机器与连接信息

| 项目 | 已知信息 | 证据/限制 |
|---|---|---|
| 登录入口 | `ssh s82` | 用户提供的SSH别名 |
| 实际连接尝试中的登录主机 | `login2.mystic.cs.iit.edu` | 来自SSH错误信息，不是计算节点名 |
| 进入计算节点的命令 | 在登录节点执行 `bash login_s8.sh` | 用户提供的终端日志已确认连接成功；助手未检查脚本内容 |
| 计算资源 | 8张Tesla V100-PCIE-32GB（各32768 MiB），驱动580.178.04 | 2026-11-22助手经 `nvidia-smi` 探测；GPU UUID见下 |
| 实际计算节点hostname/IP | `eightsocket`；`<NODE_IP>` | hostname来自用户终端提示符；IP来自登录信息，接口enp71s0f0np0；内网地址 |
| 远端用户名、工作路径、调度器 | 用户`haibin`，home为`/home/haibin`；**无调度器**（`SLURM_JOB_ID`、`PBS_JOBID`、`CUDA_VISIBLE_DEVICES` 均未设置），是直连的共享机器 | 助手探测；本项目工作目录见第11.1节 |
| 计算节点系统 | Ubuntu 24.04.4 LTS；Linux 6.8.0-147-generic x86_64 | 用户提供的登录信息，助手 `uname` 一致 |
| CUDA、驱动、cuBLAS版本 | nvcc 12.0.140（`cuda_12.0.r12.0/compiler.32267302_0`）；驱动580.178.04（驱动API报告13.0）；运行时12.0；cuBLAS/cuBLASLt 12.0.2.224 | 助手探测，`.environment.txt` 与运行日志可复核 |
| 可用工具 | gcc、g++、python3（系统）、cuobjdump、nvdisasm、compute-sanitizer、nvprof | **没有** lean、lake、ncu、nsys；nvprof可抓kernel名，但 `--metrics` 报内部错误 `4190:27` |
| 其他用户任务 | 用户自己的 `qwen-author-probe` 任务长期占用多张卡（约14 GB、约99%利用率） | 使用GPU前先查占用；本项目的运行很小，数值结果与负载无关 |
| GPU UUID | 0: `GPU-7961f273-980c-646b-b98e-7b4b259c6e09`；2: `GPU-a86f28ab-cc75-0ac8-f4d0-641c7024eef1`；6: `GPU-fabcf9ba-da72-4882-4bd0-aaef74989638` | 实际用过的卡；其余UUID见 `nvidia-smi` |

**GPU 2：** 用户后来告知“2号卡出问题了”，原因未说明，助手未再查询或使用它。曾在该卡上跑过一批极端指数输入，**那批输出视为不可信，从未分析**，（该输出已按数据清理规则从节点删除，没有保留），之后已在0号卡重跑。第11节所有结果只来自GPU 6（原始输入与形状扫描）和GPU 0（极端指数输入与分解导出）。

用户日志确认路径为`ssh s82 → haibin@login2 → bash login_s8.sh → haibin@eightsocket`。登录信息时间为2026-11-23 05:35:46 UTC，即芝加哥2026-11-22 23:35:46。登录节点显示Ubuntu 26.04.1 LTS；不要与计算节点系统混淆。

助手此前观察到的错误：

```text
Could not resolve hostname login2.mystic.cs.iit.edu: -65563
exit code: 255
```

**后续更新（2026-11-23）：** 助手端的DNS失败是临时的。`ssh -G s82` 显示目标 `login2.mystic.cs.iit.edu`、端口9999、密钥 `~/.ssh/atpi_relay`；本机 `dscacheutil` 与 `nslookup` 均解析到 `216.47.155.22`（DNS经Tailscale MagicDNS）。macOS错误码 `-65563` 是DNS查询超时，不是域名不存在。用户在终端执行 `ssh -o ConnectTimeout=10 s82 hostname` 返回 `login2` 后，助手的连接恢复。

`login_s8.sh` 的内容是单行 `ssh -i <KEY_PATH> 'haibin@<NODE_IP>'`（登录节点路径 `<KEY_PATH>`）。因此助手可以非交互地一次性执行远端命令：

```sh
ssh s82 "ssh -i <KEY_PATH> haibin@<NODE_IP> 'COMMAND'"
```

文件传输用管道，例如 `tar cf - ... | ssh s82 "ssh -i <KEY_PATH> haibin@<NODE_IP> 'tar xf - -C DIR'"`。不要猜IP或改写用户SSH配置。

没有调度器，GPU靠 `CUDA_VISIBLE_DEVICES` 手工选择。使用前先 `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`，避开用户自己的任务，并在记录里写明实际使用的卡。注意：`nvidia-smi` 不受 `CUDA_VISIBLE_DEVICES` 影响，`run_gpu.sh` 生成的 `.environment.txt` 因此列出全部8张卡，**不记录实际使用的是哪一张**；设备名只在 `.run.log` 里，没有UUID。

## 3. 本地环境与材料入口

本轮项目绝对路径：

```text
/Users/haibin/Documents/Codex/2026-11-22/new-chat/lean4HPC
```

本轮环境为macOS ARM，CPU实验使用Python 3.9.6与NumPy 2.x，精确版本在结果JSON中；没有可用的`nvcc`、`lean`或`lake`。环境限制属于本轮记录，接手后应重新探测。

所有下列路径均相对于项目根目录：

| 路径 | 用途 |
|---|---|
| `literature/REVIEW.md` | 初始文献与形式化工具调研 |
| `literature/TENSORLEAN_AND_FP16.md` | TensorLean与FP16原始论文的补充分析 |
| `operators/REVIEW.md` | 初始算子库比较；初始INT8优先级已被当前FP16主线更新 |
| `operators/REVIEW2.md` | 非BLAS算子库补充调研（optimi/MAGMA/DeepGEMM/SageAttention/FlashAttention等）；optimi已做（第13节） |
| `operators/REVIEW3.md` | 第三轮调研：整数量化与低精度格式库（gemmlowp、tflite-micro、CMSIS-NN、ggml、microxcaling等），推荐下一个目标为gemmlowp的定点重量化；**只做了调研，未克隆或运行任何候选库** |
| `operators/REVIEW4.md` | 第四轮调研：**LLM里的低精度库与算子**（按层：权重量化GEMM、W8A8、FP8、FP4、注意力/KV、三值查表、推理引擎），含两台机器能测什么、六个值得形式化的数值点；其中Marlin的INT4→FP16反量化技巧已读源码并在CPU上对16个半字节值验证；**只做了调研，未克隆或运行任何库** |
| `VALIDATION_SPEC.md` | 总体数值契约与可信边界；选库与任务排序以本交接为准 |
| `experiments/fp16_fp64/REPORT.md` | 当前主线详细实验报告 |
| `experiments/fp16_fp64/compare.py` | CPU模型、精确参考与输入生成 |
| `experiments/fp16_fp64/results.json` | 56条实验记录和求和顺序见证 |
| `experiments/fp16_fp64/*.bin` | 4组固定输入的原始二进制数据 |
| `experiments/fp16_fp64/gpu_compare.cu` | 原库模板入口与独立cuBLAS DGEMM对比 |
| `experiments/fp16_fp64/run_gpu.sh` | 固定源码与sm_70构建、运行入口；尚未实机验证 |
| `experiments/fp16_fp64/analyze_gpu.py` | GPU二进制输出与精确Fraction参考比较 |
| `experiments/fp16_fp64/manifest.json` | 固定提交、代码/输入SHA-256、GPU运行状态、工具链和产物hash |
| `experiments/fp16_fp64/align_gpu.py` | 新增：GPU输出与CPU模型各求和变体的逐元素（bit/ULP）对齐，并比较分片数 |
| `experiments/fp16_fp64/scan_shapes.py` | 新增：按kernel名把半精度GEMM分为Tensor Core/普通kernel，并对可承受的形状调用对齐 |
| `experiments/fp16_fp64/dump_split.cu` | 新增：调用原库 `cuozblasSplit<double,half>` 导出各片和指数的调试程序 |
| `experiments/fp16_fp64/log2probe.cu` | 新增：测GPU `ceil(log2(2^e))` 对全部2098个2的幂是否精确 |
| `experiments/fp16_fp64/verify_gpu_log2.py` | 新增：用实测的GPU式log2复现GPU输出，验证1 ULP根因 |
| `experiments/fp16_fp64/gen_extreme_inputs.py` | 新增：18个极端指数输入的生成脚本（种子见脚本） |
| `experiments/fp16_fp64/mid_64.bin` | 新增：64×64×64均匀随机输入（种子1），Tensor Core路径实验用；不在原4组输入内 |
| `experiments/fp16_fp64/fastmodel.py`、`validate_fastmodel.py` | 新增：快速模型（含表驱动GPU `log2`、`fastMode`、`splitEpsMode=9`）及其对原模型的逐位验证 |
| `experiments/fp16_fp64/gpu_general.cu`、`gpubatch.py` | 新增：通用测试程序（转置、alpha/beta、初始C、非有限输入、大形状、填充、库模式、批处理、重复运行）及批处理驱动 |
| `experiments/fp16_fp64/explore.py`、`analysis.py` | 新增：套件驱动（multiseed、shapes、maxsplit、transposes、alphabeta、modes、nearpow、deepexp、pad）与比较/抽样精确误差工具 |
| `experiments/fp16_fp64/specials.py`、`boundary.py`、`illcond.py`、`bigk_cap.py`、`fuzz.py`、`accuracy_effect.py`、`cross_card_legacy.py`、`tau_event_rates.py`、`log2table.cu` | 新增：第12节各实验的脚本 |
| `experiments/fp16_fp64/gpu_results/explore/` | 新增：第12节全部结果（各套件 `*_summary.json`、`*.log`、`fuzz_seed1.json` 等） |
| `experiments/fp16_fp64/gpu_results/v100_gpu6/` | GPU 6上的结果：4组原始输入、`mid_64`、33形状扫描（`scan/`、`scan_summary.json`）、SASS统计 |
| `experiments/fp16_fp64/gpu_results/v100_gpu0_extreme/` | GPU 0上的极端指数结果、`split_dump/`（GPU导出的分解数据）、`ext_inputs/`、`gpu_log2_bad_powers.txt` |
| `operators/vendor/ozblas/` | 固定版本选定源码、配置及LICENSE；不是完整可构建checkout |
| `experiments/EXPERIMENT.md` | 保留的INT8模型与复数更新问题报告 |
| `experiments/ComplexBeta.lean` | Int层具体反例草稿，尚未通过checker |

此前完整材料包：[lean4HPC-fp16-bundle.zip](https://chatgpt.com/library?file_id=libfile_fde3_f-30ef4c53a6c0819196e7007fd0dc4642_0f2de5afbc90819197c3cd4a18f9f1d0&account_id=personal)。该包生成于本交接之前，包含实验材料，不包含本交接文档。

## 4. 选库与具体配置

库：[mukunoki/ozblas](https://github.com/mukunoki/ozblas)，固定提交：

```text
a6f6198bc7af6cf288607ef867693f991cf48862
```

选用`cuozblasRgemm<double, half, float>`：FP64输入 → FP64残差分解与缩放 → 多个FP16片 → FP32片积 → FP64尺度恢复和重组。它不是全部运算都用FP16，也不是任意输入下逐比特模拟所有FP64计算的承诺。

源码包含D_H显式实例化、CC70配置，wrapper调用FP16输入/FP32输出的`cublasGemmEx`并请求Tensor Op算法。请求该算法不等于已确认实际执行Tensor Core指令，尤其小shape可能涉及不同后端选择。**已用kernel名和SASS核实：小形状（如3×3）确实不走Tensor Core，较大或部分形状才走，见第11.2节。**

已读源码并核对的行为（固定提交）：`sumMode=2` 的最终重组顺序是层级 `ik` 最外、`ia`、`ib` 升序，逐项 `devCtmp += scalbn(c, seA+seB)`（双精度累加），与模型的正序一致；`fastMode=0` 时 `maxlevel=(nSplitA-1)+(nSplitB-1)`，不丢项；A先转置再按列分解（`dim=K`），B直接按列分解；分解指数为 `tau = ceil(log2(fabs(max)))`（`short`），`sigma = 0.75 * scalbn(1, rho+tau) * splitShift`。

当前配置：real NN、alpha=1、beta=0、`fastMode=0`、`sumMode=2`、`useBatchedGemmFlag=0`、`reproMode=1`、`splitEpsMode=0`、`splitShift=1`。CPU测试分片上限3/6/12；待运行GPU入口固定上限12。`sumMode=2`不是原库默认模式0。

模型使用`rho=ceil(53-(24-log2(max(K,2)))/2)`；此处24对应FP32片积累加精度，不能误读为FP16有效位数（源码 `getEpse<half>()` 与 `getEpse<float>()` 都返回24）。**浮点分解表达式与实际CUDA数学函数之间的连接已被验证为不完全一致：GPU的双精度 `log2` 对部分2的幂不精确，见第11.5节。**编译器优化（如FMA收缩）与模型的对应仍未单独验证。

三条模型/库对应关系的备注（第12节验证）：(1) **K=1时 `rho`：** 源码里 `getRho<double,half>` 有一段注释为 “special treatment for half when m=n=k=1” 的特化，`dim_ = (dim == 1) ? 2 : dim`，所有 `splitEpsMode` 公式都用 `dim_`，所以K=1当作2，即 `max(K,2)`（通用模板会给出41，我最初读的是通用模板；`dump_split` 在16×16×1上验证GPU片与 `rho=42` 一致，144个K=1作业含 `splitEpsMode=0/2/9` 全部逐位一致）；(2) **DOT：** `cuozblasRgemm` 对 `m==1 && n==1` 强制 `fastMode=0`，所以快速模型对1×1输出不应用 `fast`；(3) 库总是先执行一次分解，**全零操作数也报告1片**（全零片），模型数0片，比较分片数时应取 `max(1, 模型片数)`。

## 5. FP16实验结论与限制

固定种子20261122，M=N=3，4组输入：普通随机K=16、普通随机K=64、指数跨度样本spread_40、强相消样本。每组比较NumPy FP64、直接FP16/FP32，以及3个分片上限×4种最终求和方式，共56条记录，其中一条为直接FP16转换溢出。

参考值是binary64输入精确解码后的Fraction矩阵乘结果。NumPy只是标准库差分参考，不是精确oracle。报告指标为：

```text
max_ij |Y_ij - exact_ij| / max_ij |exact_ij|
```

这不是逐元素相对误差，也不是诱导无穷范数。

| 输入 | NumPy FP64 | 直接FP16/FP32 | 分片上限3、正序 | 分片上限6、正序 |
|---|---:|---:|---:|---:|
| uniform_k16 | 2.262e-16 | 4.899e-4 | 1.730e-10 | 1.185e-16 |
| uniform_k64 | 9.858e-17 | 2.174e-4 | 1.084e-9 | 7.385e-17 |
| spread_40 | 7.394e-17 | 转换溢出 | 3.401e-9 | 1.385e-16 |
| cancellation | 0 | 1 | 0 | 0 |

普通随机K=16、上限6时实际分成5×5片积；K=64实际为6×6。样本支持分片模型达到binary64舍入量级，不支持“6片对所有输入足够”或“总是比标准库准确”。

相消输入`[1, 2^-45, -1, 2^-45]`与全1向量的精确点积是`2^-44`。直接FP16转换丢失小项；当前分片模型保留它。

### 求和顺序见证

测试中每个FP32片积按正序、逆序、平衡树计算都等于精确点积，但最终FP64重组在7组“输入×片数上限”配置中出现正序/逆序差异。K=16、上限6、输出(0,0)的例子：

```text
正序 -0x1.5cb20c2086c29p+0
逆序 -0x1.5cb20c2086c28p+0
差异 1 ULP
```

它说明“片积精确”不能直接推出“重组结果与顺序无关”。这仍是CPU模型见证，不是原库GPU缺陷，也不是违反库规格的已确认反例。浮点加法的关键性质是非结合性。

### 必须保留的模型边界

- 模型对提前完成分解的行/列补零片；原源码仍可能对零最大残差计算`log2(0)`。spread_40、上限12触发了该模型扩展，A/B各2个row-step。**GPU实测：原库在该输入上运行完成，输出与模型逐位一致（9/9），实际分片9×9，没有触及上限12。**这只说明该输入上没有差异，不证明零残差行路径的一般等价；该扩展仍属模型扩展。
- CPU普通FP32求和并未证明等价于V100 Tensor Core内部累加。**更新：** 在已测的所有形状（含cuBLAS实际选Tensor Core kernel的64×64×64等）上，GPU输出与模型逐位一致（第11节），说明片积在这些输入上与模型一致；但这是有限样本，不是内部累加语义的证明。归约、舍入、FTZ仍需另行分析。
- 逆序、平衡树、`math.fsum`是对照变体；不宣称原库提供这些模式。`math.fsum`不是原库NearSum，当前D_H路径的NearSum不受支持。
- 已覆盖的GPU范围见第11、12节；**仍未覆盖**：其他CUDA/驱动/cuBLAS版本、其他GPU架构、并发内存语义、`sumMode=1`（NearSum，D_H不支持）、复数/单精度等其他实例化、M或N>2048的大矩阵。转置、alpha/beta、非有限输入、大K、多种子、库的其他模式在第12节已测。
- 原始ISC 2020论文已经测试V100，报告其中DP模式未加速cuBLAS DGEMM；本轮没有性能数据。

## 6. 已准备的复现入口

CPU重跑会重新生成结果和输入；如需保留历史结果先复制实验目录：

```sh
cd /path/to/lean4HPC/experiments/fp16_fp64
python3 compare.py
```

原始输入格式：little-endian uint32的M、N、K，随后是column-major binary64 A、B。GPU输出采用同一头部，随后保存标准库和目标库两路column-major binary64结果。

进入计算节点、准备完整源码后运行下列模板（`/path/to/...`需替换为实际路径）。**这条命令已在节点上实际执行过一次**（`CUDA_VISIBLE_DEVICES=6`，输入 `uniform_k16.bin`，退出码0），产物当时在节点 `~/lean4hpc-fp16/results/`（**该目录已迁走，现在在 `/ssd/haibin/lean4hpc-fp16/results/`，见11.1节**）；其余输入因下述“checkout变脏”问题，直接复用了该次构建出的可执行文件：

```sh
git clone https://github.com/mukunoki/ozblas.git ozblas-fp16-validation
git -C ozblas-fp16-validation checkout a6f6198bc7af6cf288607ef867693f991cf48862
bash /path/to/lean4HPC/experiments/fp16_fp64/run_gpu.sh \
  "$PWD/ozblas-fp16-validation" \
  /path/to/lean4HPC/experiments/fp16_fp64/uniform_k16.bin \
  "$PWD/results/uniform_k16"
```

脚本要求支持sm_70的CUDA工具链和干净的固定提交checkout，强制重编译上游自带的对象文件，并拒绝非空LD_PRELOAD，以保持cuBLAS基线独立。不要使用vendor选定文件快照替代完整checkout。

注意：上游跟踪了部分构建产物，**首次构建后checkout确实变脏**，脚本再次运行会被干净状态检查拒绝（已预见，未实际触发，因为后续输入没有再走脚本）。后续应分离“一次构建”和“多输入运行”，或使用独立干净checkout；不要为绕过检查而清除用户文件。节点上的 `/ssd/haibin/lean4hpc-fp16/ozblas`（原 `~/lean4hpc-fp16/ozblas`）现在含构建产物，不再是干净checkout。

实际执行记录：`gpu_compare.cu` 在nvcc 12.0.140、`-arch=sm_70` 下第一次就编译通过；构建日志里有大量上游代码的 `-Wformat` 警告，不影响结果。可执行文件SHA-256：`1f3e62f3fc314c44ea3a45023f9b4d8b35de13c4430996edca56004f21ef82bb`（节点 `results/uniform_k16.exe`）。运行后保存了 `.environment.txt`（仅uniform_k16）、`.build.log`、`.run.log`、`.bin`、`.metrics.json`。**逐元素对齐已由 `align_gpu.py` 完成**（见第11节），`run_gpu.sh` 本身仍未集成该步骤，也未修复 `.environment.txt` 不记录实际使用卡的缺陷。

对齐命令：

```sh
cd experiments/fp16_fp64
python3 align_gpu.py INPUT.bin GPU_OUTPUT.bin [RUN.log] [--limit 12] [--json OUT.json]
python3 scan_shapes.py SCAN_DIR INPUT_DIR [--align-budget N] [--reuse-align OLD.json] [--json OUT.json]
python3 verify_gpu_log2.py INPUT.bin GPU_OUTPUT.bin gpu_results/v100_gpu0_extreme/gpu_log2_bad_powers.txt
```

模型对齐很慢（纯Python）：64×64×64约23秒，扫描里 M·N·K 超过30万的形状默认跳过。

## 7. 可信边界与Lean目标

| 层次 | 当前证据 | 未完成部分 |
|---|---|---|
| 数学参考 | Fraction精确计算给定输入的GEMM | 可复用的正式规格与边界定义 |
| 算术模型 | CPU有限样本、精确分片/片积检查 | Lean浮点语义与全称误差定理 |
| 源码对应 | 固定提交、逐段建模与源码快照；已读源码确认求和顺序、`maxlevel`、分解公式；**发现模型与库在 `log2` 对2的幂的行为上不同** | 模型与源码等价/精化关系；零行扩展的一般等价 |；**分片/片积层已在Lean中证明（第14节）**
| 编译结果 | 已有nvcc 12.0.140、`-arch=sm_70`、`-O2` 构建的可执行文件及hash | 未检查PTX/cubin与源码的对应；FMA收缩等优化是否影响未单独验证 |
| 库与硬件 | V100上实测输出；cuBLAS kernel名和SASS的HMMA统计 | 无硬件计数器证据；cuBLAS选择规则；驱动JIT；其他版本；并发语义 |

**对规格设计的直接启示（含第12节）：** (1) 分解指数 `tau=ceil(log2(rowmax))` 在双精度下对2的幂及其数ULP邻域不精确：略上方时 `ceil(log2)` 偏小（libm也如此，GPU多数同样），恰为2的幂或略下方时GPU有约8%的指数偏大1；规格要么写成“GPU式 `tau`”（实测表，仅对该工具链），要么把 `tau` 偏差±1作为允许偏差并量化其对误差界的影响（偏大1相当于该片少保留一位，偏小1实测未损害精度）；直接把数学上的 `ceil(log2(x))` 写进规格会与这个工具链上的库不逐位一致（高斯数据约70%的矩阵、指数跨度大的数据会出现1 ULP量级差异）。(2) **分片上限是规格的显式参数**：K大或条件数大时所需片数增长，上限12在K≥2^17或条件数≳2^120时精度退化，库默认30可恢复；规格的精度结论必须以所需片数不超过上限为前提。(3) **输入前提**：所有有限元素需满足 `tau < 1024−rho`（约 |a| ≲ 2^(1023−rho)≈1e295，否则整个输出为NaN）；输入含Inf会变NaN而非±Inf；`beta=0` 时初始C必须有限（C含NaN/Inf会传播，与BLAS约定不同）。(4) 求和顺序、片积精确性在已测范围（K至2^22、形状至1024×1024）内与模型一致，但仍是有限样本。

建议把第一个证明分解为：分片恒等式与可表示性 → 片积在位宽/K/范围条件下精确 → 尾项与尺度恢复界 → 指定重组顺序的误差界 → 组合成GEMM数值契约。允许其他归约顺序时，另行量化顺序集合，不能默认逐比特相同。

Lean checker通过、源代码映射和实机证据分别记录。`sorry`、未经审计的axiom、将待证误差界直接写成前提，都不能当作目标定理完成。保存工具链版本及定理依赖的axioms。

## 8. TensorLean报道与研究动机

用户补充的报道强调：生产GPU内核经历布局、精度、并发同步等优化，数学规格与实际执行之间存在多层转换；优化后应重新验证实现是否满足可复用规格。这与本项目动机一致。

报道还提到训练投入、服务规模，以及密歇根大学记录的一次千卡训练数据竞争事故（3000多步后报警、排查五天）。**这些具体数字与事故细节来自用户提供的报道片段，尚未核对原始论文或事故报告，不作为本项目已验证事实。** 写论文时需另补一手引用。

此前查看[TensorLean官方仓库](https://github.com/TensorLean/TensorLean)，README声明在PTX层验证无数据竞争、终止性及满足浮点容差的功能正确性；当时Release仍说明完整论文、语义和证明即将发布。该发布状态是2026-11-22的查询快照，接手时需要重新核实。

不能将TensorLean概括为只查内存安全，也不能仅凭“Lean+GPU+误差容差”宣称创新。可研究的具体方向是低精度模拟高精度的分解/重组误差契约，以及代码优化后复用规格、重新生成并检查证明的流程；是否构成新贡献仍需与完整相关工作比较。

“内核附带证明”可以提供更可审计的局部保证，但可信性不会无条件传递到整个AI系统。组合时还需满足调用前提、接口规格、编译与硬件语义，并考虑上层算法和其他组件。

原始FP16研究：[DGEMM Using Tensor Cores, and Its Accurate and Reproducible Versions](https://pmc.ncbi.nlm.nih.gov/articles/PMC7295351/)，ISC 2020，DOI 10.1007/978-3-030-50743-5_12。此前读取了检索返回的原始论文内容，但直接打开PMC遇到验证码，完整公式和证明仍需通读。

## 9. 暂缓的INT8旁支

ozIMMU固定提交`08eea9231729d54dbfd92955f2cbfc21ec236856`。CPU分片模型已完成42条结果；V100没有Turing新增的INT8 Tensor Core路径，因此暂缓INT8硬件测试，不把API可能的其他执行路径混称为INT8 Tensor Core验证。

独立发现：固定源码复数beta更新先覆盖`c.x`，第二行再使用新`c.x`计算`c.y`。原函数体的单线程host C++重放对`C=1+2i, beta=3+4i`得到`-5-14i`，正确复数乘法为`-5+10i`。这是已重放的局部源码算术问题，不是GPU入口已复现，也未获维护者确认。未提交上游issue、未修改上游。`ComplexBeta.lean`仅表达Int层反例且未运行checker。

## 10. 接手后的优先任务与验收

已完成（见第11、12节）：机器探测、原库最小样本、CPU模型与GPU逐元素对齐、Tensor Core kernel证据、形状扫描、极端指数输入、1 ULP根因，以及第12节的覆盖（多种子、大形状、转置、alpha/beta、库模式、填充、特殊值、病态点积、大K、跨卡复跑、随机模糊测试）。

待办，按建议优先级：

1. **建立第一份可检查证明。** 节点上没有Lean；需决定装在节点用户目录还是本地Mac。选择分片恒等式或受限片积精确性作为小目标，先明确假设。第7节末尾的“对规格设计的直接启示”决定了规格里 `tau`、分片上限和输入前提的写法。INT8草稿可作为工具链冒烟检查，不改变FP16主线。
2. **扩大硬件/软件版本覆盖。** 需要用户授权其他GPU（1、3、4、5、6、7上有用户任务，2号被报告有问题），才能做“用其他卡新跑”的跨卡实验；其他CUDA/驱动/cuBLAS版本上 `log2` 出错集合和cuBLAS kernel选择是否相同；M或N>2048的大矩阵；其他实例化（单精度、复数）。
3. **判断 `log2` 现象是否值得向上游报告。** 目前只能说：在CUDA 12.0.140、V100上，库对2的幂的 `tau` 计算有±1偏差，对精度无系统性影响，使GPU与“数学上精确的 `ceil(log2)` 模型”不逐位一致；不应称其为已确认缺陷。
4. **确认Tensor Core执行的硬件证据。** nvprof计数器不可用，节点无ncu/nsys；可找替代环境，或接受kernel名+SASS作为最强证据。
5. **整理脚本。** 把构建与多输入运行分开（`gpu_general --batch` 已做到运行侧）；让 `.environment.txt` 记录实际使用的卡（UUID）；把对齐步骤并入 `run_gpu.sh`；把 `log2table`、`dump_split` 纳入可复跑流程。
6. **更新 `REPORT.md`。** 目前尚未纳入GPU结果。
7. **（optimi/Lean）** Fast2Sum精确性和违反情形的界都已证明（第13.4节）。剩余：①给Lean模型加上float32中间量的双重舍入、foreach路径和Triton路径（fp32补偿量，算术不同）、fp16（p=11）、权重衰减；②多步界的常数不最优（对抗性轨迹上实测 `125·误差/Σ|c+u|` 至多0.08）；③在V100上实测GPU路径。
9. **（OzBLAS/Lean）下一批引理**：①~~FP64重组~~（已证，第15节）；②**尾项合成**：把 `multi_slice` 的残差界、重组误差界和精确恒等式 `a·b = α·β + (a−ρᴬ)ρᴮ + ρᴬ·b` 合成单个输出元素的端到端界（需处理 `a_j·b_j` 在 u0² 单位、结果在 u0 单位的单位换算，以及 `nSplitMax` 截断）；③一般 alpha/beta 的 `fma`（需要比整数格更细的格表示精确乘积）；④把 `tau = ceil(log2 max)` 的实际行为（第12.2节）接入行最大值假设；⑤溢出边界 `rho+tau ≥ 1024`（第12.5节）与下溢区 `P<54`；⑥Tensor Core累加器是否满足第14节引理用到的“结果可表示则精确”（V100上用受控输入实测）。
8. **（库的选择）** MAGMA 的 FP16 LU + 迭代精化是第二梯队（见 `operators/REVIEW2.md`）；DeepGEMM、SageAttention、FlashAttention-3/4 需要 Ampere/Hopper 硬件。
10. ~~（换库）gemmlowp定点重量化~~：**已放弃**，不属于A/B两类（见 `PLAN.md`）。
11. ~~（LLM低精度）Marlin反量化 / V100累加器微基准 / BitNet查表~~：**已放弃**，不属于A/B两类。
12. **（当前方向）** 见 `PLAN.md`：B1——ozIMMU的 `fp64_int8_auto`（按平均尾数损失阈值1.5选3~18片）与GEMMul8（README原文：推荐模数“不是精度保证”）的选参逻辑是否保证FP64级误差界；A1——用加固测试复查Sakana归档中的AI生成内核。B1已完成：ozIMMU见第16节（两处实现缺陷 + 平均准则不保证逐元素精度 + Lean定理），GEMMul8见第17节（CRT不越界已由Lean + GPU穷举确认；精度按范数；fast模式在平方范数溢出/下溢时静默得0；Inf/NaN不传播）。A1尚未开始。

首轮GPU验收必须具备：固定源码与构建信息、输入原始比特及hash、目标库和独立标准库输出、Fraction误差、模型差异说明、真实设备信息。首轮证明验收必须具备：可复跑Lean命令、固定版本、明确命题和假设、checker成功及axiom记录。

下一轮交接应直接追加实际执行状态与产物路径，不将“准备好脚本”更新为“实验成功”。

## 11. GPU实测记录（2026-11-23）

统一说明：所有结果来自 **V100-PCIE-32GB、nvcc 12.0.140、cuBLAS 12.0.2.224、驱动580.178.04**，原库固定提交 `a6f6198…`，配置同第4节（分片上限12）。指标“逐位一致”指输出的binary64位模式相同；“ULP”指按有序整数映射的距离。**逐位一致不能证明库实现了模型的求和顺序，只说明在这些输入上没有区分出来**；`cancellation`、`subnormal`、`underflow` 这类输入上四种求和变体全部一致，无区分度。

### 11.1 工作目录与产物（节点 `/ssd/haibin/lean4hpc-fp16/`）

**数据存放规则（用户2026-11-23要求）：数据放在节点的 `/ssd/haibin`，尽量不要放在 `/home`（`/home` 所在盘仅256 GB且已用约93%）；用完要删除自己产生的数据。** 本项目当时在 `/home/haibin/lean4hpc-fp16` 下产生了约2.8 GB数据（几乎全是各套件的 `batch/` 工作区），已全部删除；仍有复用价值的部分（44 MB）迁到了 `/ssd/haibin/lean4hpc-fp16/`，迁移后逐文件比对一致，并在该处重跑 `uniform_k16` 与早先输出完全相同。`gpubatch.run_jobs` 现在默认使用 `/ssd/haibin/lean4hpc-fp16` 并在取回结果后自动删除远端的批处理工作区（`keep_remote=True` 可保留）。

| 路径 | 内容 |
|---|---|
| `ozblas/` | 固定提交的完整checkout，已含构建产物（不再干净） |
| `exp/` | 同步过去的实验文件与输入，SHA-256与 `manifest.json` 一致 |
| `results/` | 4组原始输入的输出、日志、`uniform_k16.exe`、SASS统计 `sass_hmma_counts_lt.txt` |
| `dbg/` | `gpu_general`（批处理测试程序）、`dump_split`、`log2probe`、`log2table` 等可执行文件 |
| `run_out/` | 迁移后的验证输出（可删） |

已删除（结果的本地副本都在 `experiments/fp16_fp64/gpu_results/`，输入可由脚本按种子重新生成）：`batch/`（各套件工作区）、`scan_in`、`scan_out`、`ext_in`、`ext_out`、`ext_out_gpu2_suspect`（GPU 2上的不可信输出，从未分析）、`imp_in`、`imp_out`（被中止的早期任务）、`tiny_in`、`tiny_out`、`gen_out`，以及早先nvprof写在节点 `/tmp` 的5个 `tc_*.bin`。

本地副本在 `experiments/fp16_fp64/gpu_results/`；`ext_inputs/` 只保存了与1 ULP分析相关的4个输入，其余极端输入可由 `gen_extreme_inputs.py` 重新生成（生成脚本的输出文件名与路径参数见脚本）。

### 11.2 原始4组输入与Tensor Core

GPU 6上、M=N=3：

| 输入 | 分片A×B | OzBLAS D_H误差 | cuBLAS DGEMM误差 | 对模型forward |
|---|---|---:|---:|---|
| uniform_k16 | 5×5 | 1.185e-16 | 1.629e-16 | 9/9逐位一致 |
| uniform_k64 | 6×6 | 7.385e-17 | 7.264e-17 | 9/9 |
| cancellation | 2×1 | 0 | 0 | 9/9 |
| spread_40 | 9×9 | 1.385e-16 | 7.700e-17 | 9/9 |

误差指标同第5节（归一化最大误差，对Fraction精确值）。运行日志里的 `tensor_core_execution_confirmed` 是**硬编码的false**，不代表检测结果。

**Tensor Core证据：** 3×3形状下，半精度分片GEMM用的是 `gemmSN_TN_kernel<float, …, __half …>`（`nvprof --print-gpu-summary` 抓到）；`libcublasLt.so`（12.0.2.224）的sm_70 SASS里，260个 `gemmSN_TN_kernel` 变体HMMA指令数全为0，而同库6146个函数中有1010个含HMMA（对照有效，如 `cutlass_70_tensorop_f16_s884gemm`）。64×64×64用的是 `cutlass_70_wmma_tensorop_s161616gemm_f16_32x32_64x2_tn_align8`，SASS含64条HMMA。**注意：`gemmSN` 在 `libcublasLt.so` 而非 `libcublas.so`；第一次统计 `libcublas.so` 得到全0是统计对象错误。** nvprof `--metrics` 报内部错误，**没有硬件计数器证据**。

### 11.3 形状扫描

33个形状（方阵2~128、K=1~4096的若干组合、细长形状），全部正常退出且输出有限。**29个做了对齐，全部对模型forward逐位一致（最大0 ULP）**；未对齐的4个是 `128x128x64`、`16x16x4096`、`64x64x4096`、`96x96x64`（模型太慢，超过30万的M·N·K预算，只验证了能运行）。半精度GEMM kernel分类：21个走Tensor Core，12个走普通CUDA core kernel。

选择规律不简单，**不要外推**：偶数/2的幂的M、N倾向Tensor Core，K不是8的倍数时倾向普通kernel，但 `2x128` 用Tensor Core而 `128x2` 不用，`2x2` 用而 `3x3` 不用，`16x16x16` 用而 `16x16x17` 不用。Tensor Core路径出现的kernel：`volta_s884gemm_fp16_64x64_ldg8_tn`、`cutlass_70_wmma_tensorop_s161616gemm_f16_32x32_64x2_tn_align{1,2,8}`。第一版分类曾因没排除FP64 DGEMM基线kernel（`gemmSN_NN_kernel<double,…>`、`gemmk1_kernel`、`transpose_readWrite…`）而全部显示为混合，已改为只统计名字含 `__half` 或 `f16` 的kernel。

### 11.4 极端指数输入

18个输入（9类×16×16×16与3×3×16两个形状），GPU 0上运行，全部退出码0。**16个做了对齐：14个逐位一致，2个有1 ULP差异；2个 overflow 输入无法对齐。**

| 类别 | 内容 | 结果 |
|---|---|---|
| big_400、tiny_500、underflow、subnormal、row_scale、outlier | 乘积约2^800、2^-1000、2^-1120、A为subnormal、行间差2^±200、含2^500离群值 | 两个形状均逐位一致 |
| spread_100（12×12片、144片积） | 指数跨度±100 | 3×3一致；16×16×16为 **253/256，最大1 ULP** |
| spread_300（12×12片、144片积） | 指数跨度±300 | 3×3一致；16×16×16为 **255/256，最大1 ULP** |
| overflow | 乘积约2^1200 | OzBLAS输出全为NaN（3×3：9个；16×16：256个）；cuBLAS 3×3全NaN，16×16为223个NaN、16个+Inf、17个-Inf。**未探究NaN的来源，也未判断这是否是上游缺陷** |

### 11.5 1 ULP差异的根因

1. 排除：零残差扩展（这两个输入里没有任何行在12片前耗尽）；求和顺序（源码读过，与模型一致）；丢项（`maxlevel` 不截断）。
2. `dump_split.cu` 导出GPU的分解后与模型逐片比对：GPU与模型的分解**从某一片开始、某一行的指数 `tau` 差1**，之后各片连锁偏离。5个首次分歧：spread_100的A第4行第7片；spread_300的A第6行第8片、A第14行第11片、B第1行第4片、B第13行第6片。
3. 这5个事件里残差最大值**恰好是2的幂**（2^-3、2^125、2^59、2^236、2^204），数学上 `ceil(log2)` 就是该整数；模型给出正确值，GPU给出整数加1。
4. `log2probe.cu` 实测：对2^-1074~2^1023共2098个2的幂，GPU上 `ceil(log2(x))` 有 **181个出错**，`log2` 返回略大于整数的值（例如e=-1014返回-1013.9999999999999）。出错的e见 `gpu_log2_bad_powers.txt`；常见范围内（|e|≤60）为 -52、-48、-26、-24、-13、-12、-6、-3、51、55、59。上述5个事件的指数全在其中。
5. 闭环：给模型换上“GPU式log2”（在出错的2的幂上让 `tau` 加1）后，spread_100为253/256→**256/256**，spread_300为255/256→**256/256**，outlier、big_400仍为256/256。`verify_gpu_log2.py` 可复跑。

影响与限制：`tau` 偏大1使该片少保留一位，不破坏片的可表示性，误差为1~2 ULP量级，不是灾难性的。出错集合是**这个工具链（CUDA 12.0.140）数学库**的行为，换版本可能不同；模型里的“GPU式log2”是实测查找表，不是对库的建模。`compare.py` 没有改动，该变体只在 `verify_gpu_log2.py` 里。outlier输入的2^500也在出错集合里但输出仍一致，说明 `tau` 错误不一定体现在输出上。

### 11.6 本轮的更正记录

对话中曾误报两个计数，以本文为准：形状扫描是 **29/33 做了对齐**（不是28个、5个未对齐）；极端指数是 **16个对齐、14个一致、2个有1 ULP差异**（不是14个对齐、12个一致）。


## 12. 覆盖与检测探索（2026-11-23，约2小时）

统一说明：除另行标注外，所有GPU运行都在 **GPU 0（Tesla V100-PCIE-32GB，`GPU-7961f273…`）**、同一次构建的库（可执行文件基于固定提交 `a6f6198…`，nvcc 12.0.140，cuBLAS 12.0.2.224）上完成，批处理由 `gpu_general --batch` 一次进程跑完。**GPU 2 不使用**（用户报告有问题）；**其他卡（1、3、4、5、6、7）上有用户自己的任务，用户只授权过GPU 0**，助手尝试在这些卡上重跑被权限检查拒绝，因此没有做“用其他卡新跑”的跨卡实验，只做了第12.7节那种“用0号卡重跑、与6号卡早先保存的输出比对”的检查。以下“逐位一致”均指输出的binary64位模式相同（-0与+0视为相同），模型为第12.1节的快速模型，**“表驱动模型”指带有第12.2节GPU `log2` 表的模型**。所有结论都是有限样本观察，不是证明。

### 12.1 快速向量化模型 `fastmodel.py`

与 `compare.py` 的 `emulate(...)['forward']` 做同样的IEEE运算：逐行分解、FP16可精确表示的片、FP32片积（用float64矩阵乘并断言可表示为FP32，等价于原模型断言的“与精确点积相等”）、按行/列指数 `ldexp`、层级顺序的双精度累加。`validate_fastmodel.py` 在9个已有输入×（libm与GPU式 `log2`）两种变体上与原模型**逐位一致**；速度：64×64×64上0.003秒对44秒，约1.5万倍。

边界上的一个模型偏差已在此改正：源码是 `CONST * scalbn(1., rho+tau) * splitShift`，`rho+tau=1024` 时 `scalbn` 先溢出为Inf；`compare.py` 用 `ldexp(0.75, rho+tau)` 在该处仍是有限数。快速模型改用源码的运算顺序；`compare.py` **未改动**，两者只在这个溢出边界不同（其余范围数值完全相同；改后在4个输入上重新验证仍逐位一致）。快速模型额外支持 `fast=True`（`fastMode=1`：只累加 `ia+ib ≤ min(nSplitA,nSplitB)-1` 的项）和 `rho_mode=9`（`splitEpsMode=9` 的 `rho` 公式；模式0与2对half的 `rho` 相同）。不支持 `sumMode=1`（NearSum，D_H不支持）和非有限输入。

### 12.2 GPU `log2` 现象的完整表征

`log2probe2.cu`（探测，未存入项目）与 `log2table.cu`（导出，已存入）在V100上用库里的原表达式 `short tau = ceil(log2(fabs(x)))` 测试2098个2的幂及其±8 ULP邻近值，另有300万个随机双精度数：

| x相对2^e的位置 | GPU `tau` | 主机libm `tau`（模型原用） |
|---|---|---|
| 恰为2^e及略下方（-4到0 ULP） | 约8%的指数**偏大1**（164~181 / 2093） | 全对 |
| 略上方（+1到+4 ULP） | 1808~1858 / 2093的指数**偏小1** | 2019~2045 / 2093偏小；与GPU不同的有191~219个指数 |
| 300万随机数 | 与精确值零不一致 | 零不一致 |

- **偏小是 `ceil(log2(x))` 在双精度下的固有缺陷**（x略高于2^e时 `log2` 舍入成整数e），libm同样如此，原模型已复现；**偏大是GPU特有**。
- **偏小不损害精度**：24个构造输入（行最大值为 2^e(1+k·2^-52)，e取10、100、300、-40）上GPU与libm模型逐位一致，GPU误差1.6e-16~3.3e-16，对照组（最大值1.3·2^e）为1.0e-16~3.2e-16，cuBLAS同量级；模型未触发“FP16不可精确表示”断言。
- `gpu_log2_table.txt`（35581行，`T e k tau`）是该工具链下的实测表；`make_gpu_log2_table` 用它在2的幂±8 ULP范围内给出GPU的 `tau`，其余用libm。**`nearpow` 套件**：在360个（指数, 偏移）组合（GPU `tau` ≠ libm `tau`）上，GPU与表驱动模型**360/360逐位一致**，而libm模型与“只含恰好2的幂”的模型都是**0/360**（相差最多4~5 ULP）。
- 随机连续数据几乎不触发；但**自然数据会触发**：见12.3（高斯数据）。表只覆盖±8 ULP、normal范围；不在表内的情形（如离2的幂9~数百ULP）在本轮所有套件中**没有出现不一致**，但没有被专门枚举过。

### 12.3 多种子覆盖（`multiseed` 套件，420个作业）

10类输入（uniform、gaussian、f16来源、f32来源、2的幂、1/8倍数、小整数、spread30、spread100、行缩放）× 7个形状（4×4×16到64×64×256，含K=17、100这类非8倍数）× 6个种子。**GPU与表驱动模型420/420逐位一致，最大0 ULP，分片数420/420匹配。**

对libm模型（不含GPU `log2` 表）：uniform、f16、f32、2的幂、1/8倍数、小整数、行缩放全部一致（42/42）；**gaussian仅13/42一致，spread30仅15/42，spread100为41/42**，最大差1~2 ULP。`tau_event_rates.py` 的CPU估计显示，高斯float64数据几乎每个矩阵（300/300）都有“残差行最大值恰为出错的2的幂”的事件（约占行步的10%），而uniform(-1,1)一个都没有；**这个差异的机理只是推测**（行残差最后只剩单个比特时最大值是2的幂，高斯数据最低位是2^-52，而 -52 在出错集合里），未单独验证。

**精度上无系统性影响**（`accuracy_effect.py`，小形状全部元素对精确值）：GPU与libm模型有差异的元素里，谁更接近精确值基本各半——高斯 51:57、spread30 430:461、spread100 25:23（GPU更近:libm更近）；各族最大误差均值在三位有效数字内相同。即该现象影响的是“与模型是否逐位一致”，不是精度。

### 12.4 形状、分片上限、转置、alpha/beta、库模式、填充

| 套件 | 作业数 | GPU与表驱动模型逐位一致 | 内容 |
|---|---:|---:|---|
| `shapes` | 96 | 96/96，分片数全匹配 | 1×1×64到1024×1024×128、128×128×4096、64×64×16384、32×32×65536、4×2048、2048×4、1×64、64×1、255×257×129、K=1等 |
| `maxsplit` | 192 | 192/192 | 分片上限1、2、3、4、5、6、8、12（模型的 `limit` 与GPU的 `nSplitMax` 语义一致） |
| `transposes` | 144 | 144/144 | NN/NT/TN/TT，含非对称形状33×17×100 |
| `alphabeta` | 96 | 96/96 | 12组(alpha,beta)含0、负数、1e-3/1e3、非零初始C；最终运算为 `C = fma(alpha, t, beta*C)`，模型按单次舍入建模 |
| `modes` | 360 | 360/360 | 10种库配置：`sumMode=0/2`、批处理、`fastMode=1`、`splitEpsMode=2/9`、`reproMode=0` 及组合 |
| `pad` | 252 | 252/252 | 前导维度=行数+{0,1,3,5,8,17}，填充区为NaN；**填充区未被改写，输入矩阵未被修改**（若库读了填充区输出会变） |

补充：`sumMode=0` 与 `sumMode=2` 的求和顺序和运算相同（读源码确认），实测逐位一致；`fastMode=1` 对libm模型最大差126 ULP，是不同的算法，快速模型的 `fast` 变体准确复现；**`sumMode=1`（NearSum）对D_H会打印 `OzBLAS error: Nearsum is not supported when TYPE1 != TYPE3.` 并 `exit(1)`，终止宿主进程，而不是返回错误码**，调用方需知道。

### 12.5 特殊值与边界行为（`specials.py`、`boundary.py`）

有限的特殊输入——全零A或B、零行、零列、单个非零项、subnormal输入、乘积下溢、列缩放2^±400、K=1、负零——OzBLAS与模型**逐位一致**，输出有限。

与cuBLAS语义不同的行为（cuBLAS为IEEE逐元素语义，这里只记录不评判）：

| 情形 | OzBLAS | cuBLAS |
|---|---|---|
| A或B含Inf（+Inf、-Inf） | 受影响的整行/整列为NaN | 该行/列为±Inf |
| Inf×0 | 4个NaN（整行为NaN） | 1个NaN、3个Inf |
| A含NaN | 整行NaN，其余元素与cuBLAS一致（相对差约3e-16） | 同样的NaN位置 |
| **beta=0且初始C含NaN或Inf** | **输出NaN**（`beta*C` 仍被计算：`fma(alpha, t, beta*C)`） | 不读C，输出有限 |
| beta=1且C含NaN | NaN | NaN |
| 最大元素很大的有限输入 | **NaN**（见下） | 有限 |
| 输出本身溢出（乘积约2^1200） | 全NaN（16×16上256个） | 部分±Inf、部分NaN |

**大幅值阈值**（`boundary.py`，4个K×84个指数共336个作业）：当某行/列的最大元素使 `tau ≥ 1024 − rho` 时，`scalbn(1., rho+tau)` 溢出为Inf，`(input+sigma)−sigma` 变NaN，整个输出为NaN，即使精确结果有限。首个出错的 `tau`：K=8→981，K=64→980，K=1024→978，K=16384→976；约对应最大元素 ≳ 2^(1024−rho) ≈ 1e295（比DBL_MAX小13个数量级）。改正后的模型在这336个作业上“模型报溢出 ⇔ GPU输出NaN”完全对应。

### 12.6 精度：病态点积与大K（分片上限的作用）

**上限12是本测试程序的设定，不是库默认**（库默认 `NumSplitDefaultMax=30`）。以下结论必须结合上限一起读。

病态点积（`illcond.py`，Ogita-Rump-Oishi GenDot，K=64，2×2，对角元素是病态点积，90个矩阵）：

| log2(条件数) | OzBLAS相对误差（上限12，最大） | OzBLAS（上限30，最大） | cuBLAS（最大） |
|---:|---:|---:|---:|
| 20（2.5e7） | 3.5e-16 | 3.5e-16 | 1.3e-8 |
| 53（4.4e16） | 5.7e-16 | 5.7e-16 | 6.5（完全失效） |
| 100（6.4e30） | 6.1e-16 | 6.1e-16 | 2.2e15 |
| 120（1.3e37） | **2.6e-3** | 2.7e-16 | 4.5e22 |
| 140（1.0e43） | **1.0e8（失效）** | 3.5e-16 | 6.3e28 |

GPU与表驱动模型在两种上限下都是90/90逐位一致，包括上限12下精度崩溃的区间。

大K（`bigk_cap.py`，M=N=2，uniform/gaussian，对Fraction/`fsum`精确参考）：

| K | rho | 上限12：片数、误差 | 上限30：片数、误差 | cuBLAS误差 |
|---:|---:|---|---|---:|
| 65536及以下 | ≤49 | 片数<12（至多11），误差2e-16~4e-16 | 同左 | 3e-16~8e-16 |
| 131072 | 50 | 12片，1e-14 | 14~19片，3e-16 | 2e-15 |
| 1048576 | 51 | 12片，4e-11~9e-11 | 18~26片，2e-16 | 1e-15~4e-15 |
| 4194304 | 52 | 12片，2e-7~5e-7 | 27~30片，6e-16 | 5e-15~9e-15 |

**随K增大 `rho` 增大，每片保留的位数减少，所需片数增长**：上限12在K≥2^17时开始限制精度（误差从1e-15级退化到K=2^22的1e-7），上限30可恢复到约1e-16（K=2^22的gaussian用满了30片）。GPU与模型逐位一致，模型预测了这种退化（`bigk_cap.py`日志里全部True）。

### 12.7 可复现性

`cross_card_legacy.py`：把早先在GPU 6上跑过的38个输入（4个原始、`mid_64`、33形状扫描）用GPU 0重跑，与保存的GPU 6输出逐位比较：**OzBLAS输出0个不同，cuBLAS输出0个不同**；同一进程内每个输入重复20次，**0个差异**。这些输入既含Tensor Core路径也含普通kernel路径。限制：只比了GPU 0与GPU 6两张卡、同一库构建；用其他卡的新跑未做（见本节开头）。

### 12.8 极端指数的补充（`deepexp`，54个作业）

三个输入族：`neg1000`（指数-1000~0，深度subnormal残差）、`pos900`（0~900）、`wide500`（A、B各±500）。**GPU与表驱动模型54/54逐位一致。**对“只含恰好2的幂”的模型：neg1000 18/18，pos900只有14/18，wide500只有5/18，说明**自然数据在大指数处会落入±8 ULP的非恰好2的幂区域**，需要完整表。`wide500` 里元素动态范围极大，很小的元素只是噪声（符号随 `tau` 的1位差异而翻转，ULP距离达2^63量级但无精度含义）；按最大元素归一化的误差，OzBLAS中位数9e-17、最大3e-16，与cuBLAS同量级。

### 12.9 `analysis.compare` 的ULP溢出缺陷（已修）

早期 `compare` 用int64算ULP距离，符号相反的大数相减溢出；已改为无符号单调键。**第11节及之前的结果不受影响**（那些比较里没有符号相反的元素）。`wide500` 之前显示的天文数字ULP来自此缺陷与噪声元素符号翻转。

### 12.10 随机模糊测试（`fuzz.py`）

每个作业随机取：M、N∈[1,128]、K∈[1,600]（对数均匀）、A的12类输入族之一、B的5类之一、转置NN/NT/TN/TT、(alpha,beta)（10组，含非零初始C，仅当 M·N ≤ 4096）、分片上限（1、2、3、4、6、8、12、16、20、30）、`sumMode`∈{0,2}、`fastMode`（25%）、批处理（25%）、`splitEpsMode`∈{0,9}、填充0~5。规格由（种子，序号）完全确定，可重建。

**种子1与种子2各4000个作业，共8000个（模型修正后的结果）：GPU与表驱动模型逐位一致 8000/8000，分片数不匹配0，模型报错0，无输出0，库返回状态非零0。**不含GPU `log2` 表的libm模型只一致 3690/4000 与 3717/4000（约92%）。

首次运行（种子1，用较早的模型）曾得到3995/4000，6个被标记，**全部是模型缺陷而非GPU行为**：5个是M=N=1且 `fastMode=1`（库对DOT强制 `fastMode=0`，模型不应用 `fast`）；1个是B全零时库报告1片而模型数0片（比较约定，输出一致）。这两处改模型后，同一批6个作业全部一致，种子1全量重跑与种子2新跑都是4000/4000。另有一次我把K=1时的 `rho` 按（通用模板的）源码字面改成 `log2(K)`（41），结果模型在所有K=1输入上报“片不可精确表示”，而GPU精度正常；用 `dump_split` 对照后确认GPU实际是42，已**恢复** `max(K,2)`，随后读到 `getRho<double,half>` 的特化（`dim_=(dim==1)?2:dim`）才解释了原因，并据此给 `splitEpsMode=9` 的公式补上同样的钳位（此前模型在K=1且模式9时少钳位，144个K=1作业验证后一致）。教训：改模型前先用 `dump_split` 对照GPU，并找全所有相关的模板特化，不能只按第一处源码字面推。

**更大形状的模糊测试**（`fuzz.py --maxdim 256 --maxk 2000`，种子3，300个作业）：**300/300逐位一致**，分片数0不匹配，无警告；libm模型只一致264/300。首次运行时整个第二批150个作业“无输出”：这是基础设施的瞬时失败（当时有套件重跑并发使用同一节点SSH连接，`gpubatch` 当时还会静默吞掉错误），不是数值差异；已让 `run_jobs` 在远端运行或下载失败时打印警告，重跑无警告。**种子4（3000个作业，最终模型，形状范围同种子1）：3000/3000逐位一致**，分片数0不匹配，模型报错0，无输出0，库返回状态非零0；libm模型只一致2757/3000。至此模糊测试合计 **11300个随机作业全部与表驱动模型逐位一致**（种子1、2、3共8300个用的是 `splitEpsMode=9` 的K=1钳位修复之前的模型——该修复只影响K=1且模式9的组合；种子4的3000个和144个K=1作业用的是最终模型）。

**用最终模型重跑所有套件**（multiseed、shapes、maxsplit、transposes、alphabeta、modes、nearpow、deepexp、pad，日志在 `gpu_results/explore/final/`）：数字与12.3~12.8节完全一致（420、96、192、144、96、360、360、54、252个作业均对表驱动模型逐位一致；`nearpow` 对“仅2的幂”模型0/360；`deepexp` 对该模型37/54）。

### 12.11 本轮的局限、过程记录与更正

**局限：**
- 全部是**一个工具链（CUDA 12.0.140）、一个架构（V100/sm_70）、一个库构建**上的有限样本；“表驱动模型”里的GPU `log2` 表是该工具链数学库的实测，换版本可能不同。表只覆盖2的幂±8 ULP（normal范围）；表外情形本轮没有出现不一致，但没有被专门枚举。
- 形状上限：实际测到最大约1024×1024×128、K最大2^22（M=N=2）；没有测M或N>2048。分片上限12/30之外的组合在模糊测试里取过，但精度结论只系统测了12与30。
- 输入分布：均匀、高斯、FP16/FP32来源、2的幂、1/8倍数、小整数、跨度30/100/500/1000的指数、行缩放；未覆盖真实应用矩阵（如稀疏、带状、真实科学计算数据）。
- 精度对比使用的精确参考对大形状是抽样元素（最多24个），不是全部元素。
- 没有性能数据，也没有硬件计数器级别的Tensor Core证据。
- **未做**用GPU 1、3、4、5、6、7新跑的跨卡实验：这些卡上有用户自己的任务，用户只授权过GPU 0，助手尝试时被权限检查拒绝，未绕过；GPU 2被用户报告有问题，未使用。

**过程记录与更正（以本文为准）：**
- 第11.5节写的“模型给出正确值”只对**恰为2的幂**成立；对略高于2的幂的值，libm模型和GPU同样 `tau` 偏小（见12.2）。
- 第11.5节的“GPU式模型”只含恰好2的幂，不足以解释自然数据（见12.8的 `pos900`、12.2的 `nearpow`），已被12.2的±8 ULP表取代。
- 快速模型改动记录：①`sigma` 表达式改为源码顺序（溢出边界，12.1）；②DOT不应用 `fast`；③分片数比较取 `max(1,·)`；④`rho` 对K=1曾误改后恢复，并给 `splitEpsMode=9` 补上 `dim_` 钳位。`compare.py` 始终未改（其 `max(K,2)` 恰好是对的）。
- `gpubatch.run_jobs` 现在在远端运行或下载失败时打印警告（此前静默，造成过一次整批“无输出”被当作不一致）。
- `analysis.compare` 的ULP溢出缺陷已修（12.9）。
- 节点数据：已按用户要求把数据迁到 `/ssd/haibin/lean4hpc-fp16`（44 MB）并删除 `/home` 下我产生的全部数据，见11.1节。本地 `experiments/fp16_fp64/` 里删除了从未运行的 `impact_study.py` 和 `cross_card.py`。
- 误写过一个空文件到 `/tmp/_unused`，已删除。
- 大K实验曾因Fraction精确求和过慢（K=4M下24个元素需上亿次Fraction运算）而中止，改用TwoProduct+`math.fsum`的快速路径；该路径与Fraction路径在K=3000上给出完全相同的数值。


## 13. 第二个库：optimi 的Kahan补偿更新 + 第一批Lean定理（2026-11-23）

选库依据见 `operators/REVIEW2.md`。选 [optimi](https://github.com/warner-benjamin/optimi)：数值核心极小、纯PyTorch、MIT，结构是“权重 = 高位 + 补偿”，与OzBLAS的“高位+残差”同构。**固定提交 `17a3d5422f3e41bb8973e1e4c9b39206c0d93b6d`**（2025-12-29，“add foreach backend test selection”）；存档在 `operators/vendor/optimi/`（`optimi/` 包、LICENSE、pyproject，不是完整checkout）。

### 13.1 被建模的代码

`optimi/adam.py` 的 `_single_param_adam`（`kahan_sum=True` 且参数dtype为fp16/bf16）；以下四个操作都在参数自己的dtype（bf16）里做：

```
kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr)   # c ← rn(c + (−lr·m)/(√v+eps))
grad.copy_(param.detach())                                             # t ← w
param.add_(kahan_comp)                                                 # w ← rn(w + c)
kahan_comp.add_(grad.sub_(param))                                      # c ← rn(c + rn(t − w))
```

**未建模、未验证的路径：** `_foreach_adam`（torch foreach）、**Triton路径**（`adam.py` 约508~521行，补偿量在fp32里计算再转回参数dtype，算术与上面不同）、fp16、`weight_decay≠0`、其他优化器（adan、lion、sgd等也有类似写法，未读）、GPU执行。

### 13.2 Python模型与验证（`experiments/optimi/`）

`kahan_model.py` 用numpy模拟“float32中间量、一次舍入到bf16（就近偶数）”。`validate_kahan_model.py` 的结果（CPU，PyTorch 2.15.0）：
- 逐元素bf16运算（加、减、开方、`addcdiv`）：各20000个元素，**与torch 0个不一致**。
- **真实 `optimi.AdamW(kahan_sum=True, foreach=False)` 的一步更新**：60组随机状态、共122194个元素，**权重与补偿缓冲0个不一致**。注意：这只验证了CPU上的单张量路径。
- `addcdiv` 的两种运算顺序（`(v·m)/d` 与 `v·(m/d)`）在这些数据上都是0不一致，无法区分，模型取前者。

### 13.3 数值实验（`kahan_experiments.py`、`kahan_violation.py`，bf16，精确算术用Fraction）

| 实验 | 结果 |
|---|---|
| E1 重归一化精确性：`w1 + c2 == w + c1` | 200000个随机状态（`|c|`、`|u|` 至多约0.3|w|）全部精确 |
| E2 守恒式：`V_{k+1} = V_k + u_k − δ_k`，`δ_k = (c_k+u_k) − rn(c_k+u_k)` | 20000步每步精确成立；`|δ_k| ≤ 2^-8·|c_k+u_k|` 无一例外；`Σ|δ|=3.0e-2`，`Σ2^-8|c+u|=9.7e-2` |
| E3 长期累加（w₀=1，20000步）常数更新 −1e-4 | 精确和 −1.00272；**朴素bf16停滞在1.0（误差2.00）**；Kahan −0.99212（误差1.06e-2） |
| E3 常数更新 +3e-5 | 精确和 1.60081；朴素停滞（误差0.601）；Kahan误差7.2e-3 |
| E3 随机N(0,1e-4) / 带漂移 | Kahan误差1.1e-3 / 1.8e-5；朴素8.1e-3 / 1.008 |
| 违反前提 `|c₁|>|w|`（`kahan_violation.py`） | `|c₁|≤|w|`：60000/60000精确；`|c₁|/|w|∈[1,3]`：14943/20000精确，`∈[3,100]`：2088/20000，`∈[100,1e6]`：22/20000；不精确时缺陷最大约 `3.9e-3·|c₁|`（=2^-8），优雅失败 |

**Kahan不是精确的**：常数更新下 `c+u` 的舍入误差有偏、线性累积（E3常数更新的Kahan误差仍有1e-2量级）；但它与朴素累加的差别是量级的（朴素直接停滞）。**optimi的代码不检查 `|c₁| ≤ |w|`。**

**违反情形界的交叉检查**（`violation_bound_check.py`，精确Fraction，bf16模型）：40万个随机状态（其中22.5万个 `|c₁|>|w|`）上 `251·|η|/|c₁|` 最大 **0.9805**，没有越过Lean证明的界，离界只差约2%（界几乎是紧的）；300条400步、违反频繁的轨迹上多步界 `125·误差/Σ|c+u|` 最大 **0.0755**，0次违反（多步界比逐步界松，因各步误差符号随机会抵消）。

**Lean证明后的交叉检查**（`fast2sum_crosscheck.py`，p=3~6、±300内所有可表示数对，`|b|≤|a|`）：就近偶数、远离零、趋向零、随机四种平局规则下Fast2Sum **100%精确**，与Lean定理（对任意最近舍入成立）一致；作为负例，**截断舍入**（非最近）在p=3时失败96/1681、p=4时失败16/4999，但p=5、6在该范围内没有暴露反例——范围太小，这个负例对较高位宽并不充分。

穷举（`fast2sum_exhaustive.py`，p有效位整数格，就近偶数，±600内全部可表示数对）：p=2~6时 **`|b|≤|a|` 的 721、2177、6727、19601、55111对全部精确**；`|b|>|a|` 时只有约一半精确；同时 `a−s` 在 `|b|≤|a|` 时全部精确算出（0个违反）。这说明Fast2Sum的前提就是 `|b|≤|a|`。

### 13.4 Lean（`lean/OptimiKahan/Kahan.lean`，核心Lean，无Mathlib，564行）

**检查：** `lean Kahan.lean`（见同目录 `README.md`），Lean 4.36.0，输出 `check_log.txt`，退出码0，无 `sorryAx`。工具链在项目 `.toolchain/elan`（约3 GB，**归档项目时要排除**）。

建模：所有量是整数（格点=最小量化单位的倍数）；舍入是抽象函数 `rd`，只要求“结果可表示”和“没有更近的可表示点”（`structure Rounding`）。

| 定理 | 内容 | 依赖的公理 |
|---|---|---|
| `step_value` | 一步守恒式 `V' = V + u − δ + η`（纯代数，η为重归一化缺陷） | propext, Quot.sound |
| `eta_zero` | 在 `Fast2SumExact` 与 `|c₁|≤|w|` 下 η=0 | propext, Quot.sound |
| `run_value` | N步守恒 `w_N + c_N = w + c + Σu − Σδ`（**依赖假设 `Fast2SumExact` 和每步 `|c₁|≤|w|`**） | propext, Quot.sound |
| `run_bound` | 在 `RelErr` 假设下 `256·|Σδ| ≤ Σ|c_k+u_k|` | propext, Quot.sound |
| `naive_stagnates`、`naive_run_stagnates` | 朴素累加：更新小于半个间隔时舍入原样返回w，整个列表都被丢弃 | propext, Quot.sound |
| `exists_nearest`、`lattice` | 8位有效数字整数格上“最近舍入”存在并构成 `Rounding`（避免定理空洞） | + Classical.choice |
| `close_nat`、`exists_close`、**`relErr8`** | **`RelErr` 对具体bf16格成立**（相对误差 ≤ 2^-8，含次正规） | + Classical.choice |
| **`run_bound_bf16`** | 具体bf16格上：补偿舍入误差总和 `256·|Σδ| ≤ Σ|c_k+u_k|`（**无条件**） | + Classical.choice |
| `gap_2pow20`、**`bf16_stagnation`** | 具体bf16格上：w=2^20（相当于1.0）时，任意长度、每个小于2^11的更新序列，朴素累加全部丢弃 | + Classical.choice |
| **`fast2sum_bf16`** | **Fast2Sum在8位有效数字格上精确**：对任意可表示的 a、b 且 |b|≤|a|，`s=rd(a+b); d=rd(a−s); e=rd(b+d)` 返回精确误差 `(a+b)−s`；**对任意最近舍入（平局任意取舍）成立** | + Classical.choice |
| **`run_value_bf16`**、**`kahan_error_bf16`** | 具体bf16格上：若每步满足 `|c₁|≤|w|`，则 `w_N+c_N = w+c+Σu−Σδ`，且 `256·|(w_N+c_N)−(w+c+Σu)| ≤ Σ|c_k+u_k|`（**不再依赖 `Fast2SumExact` 假设**） | + Classical.choice |
| **`eta_bound_violation`** | **违反情形**：对**任意**相对误差为2^-8的舍入，只要 `|w|≤|c₁|`，重归一化缺陷 `251·|η| ≤ |c₁|`（约 `1.02·2^-8·|c₁|`）；证明只用 `RelErr`，不需要Fast2Sum | propext, Quot.sound |
| **`eta_bound_bf16`** | 具体bf16格、`w` 可表示：**对所有状态** `251·|η| ≤ |c₁|`（`|c₁|≤|w|` 时 η=0，否则用上一条） | + Classical.choice |
| **`kahan_error_bf16_unconditional`** | **无逐步前提**的多步界：`125·|(w_N+c_N) − (w+c+Σu)| ≤ Σ|c_k+u_k|` | + Classical.choice |

**没有证明的（诚实边界）：**
1. （已解决）每步前提 `|c₁|≤|w|` 不再是假设：违反情形有界（`eta_bound_violation`），合并成对所有状态的无条件多步界（`kahan_error_bf16_unconditional`）。
2. Lean里 `rd` 是单次舍入；PyTorch在CPU上是“float32中间量再舍入到bf16”（双重舍入）。两位bf16数相加时若指数差很大，float32先舍入一次可能与直接舍入到bf16不同；13.2的逐位验证未发现差异，但**Lean模型没有刻画这点**。
3. 格模型没有指数上界（无溢出），次正规用格点最小量化单位近似。
4. 模型与PyTorch的连接是经验性的（13.2），不是证明；GPU路径（尤其Triton）未涉及。
5. `RelErr` 的单位舍入取 2^-8（8位有效数字）；fp16（11位）需另证。

### 13.5 环境与数据

- Python验证用的是会话临时目录里的虚拟环境（torch 2.15.0、numpy 2.5.5），**未持久化**；重跑需要 `pip install torch numpy packaging`。
- optimi这部分没有使用计算节点，节点上没有产生数据。
- 清单：`experiments/optimi/manifest.json`（11个文件哈希、Lean版本、状态）。


## 14. OzBLAS 分片引理（Lean，2026-11-23）

文件：`lean/OptimiKahan/OzSlice.lean`（360行，`import Kahan`，核心Lean，无Mathlib）；检查日志 `check_log_ozslice.txt`（Lean 4.36.0，退出码0）；检查命令见同目录 `README.md`（需先 `lean -o Kahan.olean Kahan.lean`）。清单 `experiments/ozslice/manifest.json`。

### 14.1 建模

对应固定提交 `a6f6198…` 的 `cuozblasSplitCKernel`：`tau = ceil(log2 max)`；`sigma = CONST·scalbn(1., rho+tau)·splitShift`（`CONST=0.75`，`splitShift=1`）；`split = (input+sigma) − sigma`；`input = input − split`；`devSplit = scalbn(split, −tau)`（隐式 double→half）；片积在 FP32 里累加。
- binary64 = 53位有效数字的整数格（`lattice 53`：就近舍入，**平局任意取舍**，单位是最小量化单位，**无指数上界**）；FP16 = `Rep 11`；FP32 = `rdNear 24`。
- 参数 `P = rho + tau`，`q = 2^(P−53) = ulp(sigma)`，`s = 53 − rho`（每片位数）。

### 14.2 已证明的定理

| 定理 | 内容 | 公理 |
|---|---|---|
| `slice_lemma` | 前提 `54 ≤ P`、`3 ≤ rho ≤ 53`、`r` 可表示、**`|r| < 2^tau + q/2`**（因此 `|r| ≤ 2^tau` 的常规情形，以及 `tau` 偏小1的情形都包含在内）。结论：计算出的 `piece = (r+σ)−σ` 恰好等于 `q·n`；`2·|r − q·n| ≤ q`（最近倍数）；残差 `r − q·n` 的FP减法**精确**；`|n| ≤ 2^(53−rho)` | propext, Classical.choice, Quot.sound |
| `slice_fp16` | `rho ≥ 42`（即 `s ≤ 11`）时片整数 `n` 是FP16数 | 同上 |
| `multi_slice` | T片分解（第k片 `P_k = P − k(54−rho)`）：每步计算出的piece等于 `q_k·n_k`、残差精确、`|n_k| ≤ 2^s`，最终残差 `≤ 2^(P − T(54−rho) − rho)`（每片残差缩小 `2^(s+1)` 倍） | 同上 |
| `product_exact` | 两个片整数（各 `≤ 2^s`，`s ≤ 11`）的乘积在FP32中精确可表示 | 同上 |
| `fl32_foldl_exact`、`tree_exact` | FP32累加：只要运行和不超过 `2^24`，**任意顺序（任意列表）和任意分组（任意二叉树）**的结果都等于精确和 | 同上 |
| **`ozblas_partial_dot_exact`** | 片整数 `≤ 2^s`（`s ≤ 11`）、`K·4^s ≤ 2^24`、至多K项：每个FP16×FP16乘积精确，FP32累加等于精确整数点积 | 同上 |
| `IsMaxS`、`isMaxS_unique`、`isMaxS_examples` | 库选的片宽 `s = 53 − rho` 是满足 `K·4^s ≤ 2^24` 的**最大** `s`（唯一）；`K=2,16,64,1024,16384,65536` 对应 `s=11,10,9,7,5,4`（即 `rho=42,43,44,46,48,49`，与第12.6节GPU表一致）；`isMaxS_examples` 不依赖任何公理 | propext, Quot.sound（examples：无） |
| `s_le_11` | `K ≥ 2` 时精确性条件强制 `s ≤ 11`，所以片整数总能装进FP16 | propext, Quot.sound |
| **`k1_needs_special_case`** | `K=1` 时条件允许 `s=12`，但 `2049 ≤ 2^12` 不是FP16数（`¬Rep 11 2049`）——这正是库里 `dim_ = (dim == 1) ? 2 : dim` 特殊处理必须存在的原因 | propext, Quot.sound |

### 14.3 这些结果说明什么

- **每片位数预算：** `rho` 的公式 `ceil(53 − (24 − log2 K)/2)` 等价于“取最大的 `s` 使 `K·4^s ≤ 2^24`”（整数刻画由 `IsMaxS` 表达；两者等价的推导是纸面的：`ceil(53−t)=53−floor(t)`，`floor((24−log2 K)/2)=max{j: K ≤ 2^(24−2j)}`；Python对104万个K值验证两者一致，见14.4）。
- **FP32片积精确且与求和顺序无关**：这解释了第12节“CPU模型与GPU逐位一致、而Tensor Core内部累加顺序未知”并不矛盾——只要累加是“结果可表示就精确”的加法，顺序和分组都不影响片积。引理只用到“可表示的值经舍入不变”，没有用到最近性；对截断型累加器同样成立（**这一点我没有形式化，是对证明用到的性质的观察**）。
- **`tau` 偏小无害有了证明：** 第12.2节发现GPU的 `ceil(log2)` 在2的幂略上方会使 `tau` 偏小1，此时 `|r|` 略超 `2^tau`（相对约 `2^-45` 量级）。`slice_lemma` 的假设放宽到 `|r| < 2^tau + q/2`，而 `q/2 = 2^(tau−s−1)` 远大于这个越界量，所以所有结论不变。`tau` 偏大1（GPU在2的幂处的情形）则落在原假设内（每片少一位，同样满足引理）。
- FP16指数范围：`s ≤ 11` 时非零片值落在 `[2^-11, 1]`，在FP16的正规范围内（这是一个事实性观察，**没有形式化**）。

### 14.4 交叉检查（`experiments/ozslice/slice_crosscheck.py`，真实binary64运算）

1. **22万个样本**（rho 42~49，tau −5~30，含精确平局、`|r| = 2^tau`、很小的r）：`piece` 是 `q` 的倍数、`|r−piece| ≤ q/2`、残差在double里精确（用Fraction判断）、`|n| ≤ 2^(53−rho)`、FP16可表示——**0个失败**。
2. **K=1、rho=41**：`n=2049` 满足 `|n| ≤ 2^12`，但缩放后的片 **不是FP16数**（印证 `k1_needs_special_case`）。
3. **库的 `rho` 公式 vs `53 − max{s: K·4^s ≤ 2^24}`**：`K=1..2^20` 全部加上 2 的幂 ±1 直到 `2^24`，共1048591个值（K=1按2处理），**0个不一致**。
4. **放宽的假设**：`|r|` 略超 `2^tau`（`2^tau·(1+ε)`，ε=2^-45 到 2^-20）及其正负号、以及 `|r|` 贴近 `2^tau + q/2` 的下方，共220个样本，**0个失败**。边界 `|r| = 2^tau + q/2` 恰是 `n=2^s`（偶）与 `2^s+1`（奇）的平局：IEEE就近偶数下20/20仍满足 `|n| ≤ 2^s`；若平局向远离零方向取舍，则20/20得到 `n = 2^s+1`。**因此Lean引理的严格不等号是为“任意平局取舍”而设，不能说它对就近偶数也是最优的。**

### 14.5 没有证明的（边界）

1. **FP64重组**：已在第15节证明（任意顺序的累加误差界；`alpha=1,beta=0` 精确）；**一般alpha/beta的fma和端到端GEMM误差界仍没有证明**。
2. **尾项合成**：`multi_slice` 给出T片后残差界，但没有和重组误差合成整体界；`nSplitMax` 上限和 `check==0` 终止测试、全零行（`log2(0)`）未涉及。
3. **溢出与下溢**：格模型没有指数上界，所以 `rho+tau ≥ 1024` 时 `sigma=inf` 导致NaN（第12.5节）没有建模；要求 `P ≥ 54`（`q ≥ 2` 个最小量化单位），不覆盖片量化低于最小次正规的下溢区。
4. **`tau` 的计算**：`ceil(log2(max))` 与GPU `log2` 的行为只通过行最大值假设进入引理，没有证明；引理对任何满足 `|r| < 2^tau + q/2` 的 `tau` 都成立。
5. **硬件语义**：Tensor Core的累加器是否满足“结果可表示则精确”、FP16转换的舍入细节、cuBLAS选择的kernel——没有证明，也没有在本节验证。
6. 只证了**单个元素**的分解性质；行内最大值、行与行之间的关系、矩阵形状/转置/blocking 没有进入模型。
7. 模型与真实库的连接是经验性的：第12节的逐位一致测试 + 本节14.4的交叉检查，不是证明。


## 15. OzBLAS 的 FP64 重组（Lean，2026-11-23）

文件：`lean/OptimiKahan/OzRecombine.lean`（173行，`import Kahan`，核心Lean，无Mathlib）；检查日志 `check_log_ozrecombine.txt`（退出码0）；命令见 `lean/OptimiKahan/README.md`。清单 `experiments/ozslice/manifest.json`。

### 15.1 建模

对应 `cuozblasLocalFsumKernel`（`sumMode=2`，固定提交 `a6f6198…`）：对每个输出元素，项 `T_k = scalbn(c_k, seA+seB)` 在**一个线程内**按固定顺序累加，**第一项直接赋值**（`if (ic == 0) devCtmp = c`），其余 `devCtmp += c`（双精度加法，一次舍入）；最后 `C = fma(alpha, t, beta*C)`。模型：binary64 = `lattice 53`（就近舍入、平局任意取舍、无指数上界）；项取为可表示的格点（`scalbn` 精确，假设没有上溢/下溢）。

### 15.2 已证明的定理

| 定理 | 内容 | 公理 |
|---|---|---|
| `relErr53` | binary64就近舍入的相对误差：`2^53·|z − rd z| ≤ |z|` | propext, Classical.choice, Quot.sound |
| `recombine_gen`、**`recombine_error`** | **对任意顺序**的N≤2^52个项，第一项赋值、其余逐个 `+=` 的双精度累加满足 `2^53·|结果 − 精确和| ≤ 2(N−1)·Σ|T_k|`（归纳不变量 `2^53·|E_k| ≤ 2(k−1)·A_k`，`A_k` 为前k项绝对值和；长度限制保证 `|E_k| ≤ A_k`） | 同上 |
| `accum_rep`、`final_axpby_exact` | 累加结果可表示；`alpha=1` 且 `beta·C=0`（`beta=0`、C有限）时最终 `fma` 不再引入舍入 | 同上 |
| **`recombine_alpha1_beta0`** | `alpha=1,beta=0`：存入的结果就是累加值，且满足同一个界 | 同上 |
| `rep_mul_pow2` | 把可表示的项乘以 `2^e`（`e ≥ 0`）精确（`scalbn` 上移） | propext |

### 15.3 这些结果说明什么

- 界对**所有顺序**成立：库按层级（大项在前）累加，这个选择**不影响界成立**，只影响 `Σ|T_k|` 的大小；第12.4节“`sumMode=0` 与 `2` 逐位一致”以及模型 `forward` 与GPU逐位一致，都与此相容。
- 相对界 `2(N−1)·2^-53·Σ|T_k|` 是**最坏情形**界；N 至多约 144（12×12片）时，量级为 `2^-53·300·Σ|T|`。
- `alpha=1, beta=0`（本项目所有GPU实验的配置）下，最终合成**不引入任何额外舍入误差**——误差全部来自累加。
- 与第12.5节呼应：`beta=0` 时 `fma(alpha,t,beta*C)` 仍计算 `beta*C`，C含NaN/Inf就会传播；定理的前提是 `beta*C = 0`（C有限），**非有限C的行为不在模型内**。

### 15.4 交叉检查（`experiments/ozslice/recombine_crosscheck.py`，真实binary64加法，精确Fraction）

- **库模型的真实项列表**（层级顺序，D_H，上限12，每族216个元素，每元素至多144项）：`2^53·误差/(2(N−1)Σ|T|)` 的最大值——uniform 0.110、gaussian 0.049、spread100 0.016、rowscale 0.039，**均 ≤ 1**（界要求 ≤ 1）。
- **对抗性项列表**（4000个，随机排列、按绝对值升序/降序、强相消、跨度±300位的指数范围）：最大 0.301（降序）、0.225（升序）、0.192（随机）、0.116（范围）、0.043（相消），**0次越界**。
- 说明：界偏保守（最坏因子 `2(N−1)`，实际误差像随机游走）。

### 15.5 没有证明的（边界）

1. **一般 alpha、beta**：`fma(alpha, t, beta*C)` 的精确乘积 `alpha·t` 不在整数格里（需要更细的格），**没有证明**；只证了 `alpha=1`、`beta·C=0`。符号为零、非有限C未建模。
2. **向下缩放**：`scalbn` 向下（`seA+seB<0`）要求无下溢，项被当作可表示的格点（假设）；只证了向上缩放精确。FP32→FP64转换和次正规项的行为没有建模。
3. **层级顺序没有被利用**：界对所有顺序成立，用 `Σ|T_k|` 表示，**没有**把它与输入量级（`Σ|a||b|` 或行/列最大值）联系起来；`Σ|T_pq|` 与输入量级的关系没有证明。
4. **端到端GEMM误差界**：尾项（`nSplitMax` 截断后的残差项）与第14节的分片引理、本节重组界合成整体误差界——**没有证明**（这是下一步）。
5. 编译后的内核是否每次加法恰好一次binary64舍入（没有FMA收缩、没有扩展精度）：没有在SASS层验证；第12节的逐位一致测试是间接证据。
6. 模型与真实库的连接是经验性的（第12节逐位一致 + 15.4）。


## 16. B1：ozIMMU `fp64_int8_auto` 的自动选片（2026-11-23）

对象：[ozIMMU](https://github.com/enp1s0/ozIMMU)（MIT），固定提交 `08eea9231729d54dbfd92955f2cbfc21ec236856`（2025-12-02，即当时HEAD）。INT8 Ozaki I：每行（A）/每列（B）按 `max_exp = 2·行内最大指数` 对齐，切成 s 片、每片 `b = get_bits_per_int8(k) = min(7, (31−⌈log2 k⌉)/2)` 位，int8 GEMM（`cublasGemmEx`，int32累加），再在FP64里合成。`fp64_int8_auto` 在 s=3..18 中按“**平均**尾数损失 ≤ 阈值”选第一个s，否则回退DGEMM；阈值来自 `set_auto_mantissa_loss_threashold` 或环境变量 `OZIMMU_AUTO_AVG_MANTISSA_LOSS_THRESHOLD`（README推荐1.5；句柄默认值为0）。

文件（`experiments/ozimmu/`）：`auto_probe.cu`（库选择 vs 主机端精确复刻的计数与选择）、`gemm_probe.cu`（端到端：auto/各固定片数/DGEMM 的逐元素误差，单位 `u·(|A||B|)_ij`，`u=2^-53`，参考值用long double累加）、`gemm_probe_results.txt`（原库，阈值1.5与0）、`ozimmu_auto_fix.patch`（最小修复）、`gemm_probe_fix_results.txt`（修复后）。Lean：`lean/OptimiKahan/OzimmuAuto.lean`（208行，独立文件，核心Lean），日志 `check_log_ozimmuauto.txt`（退出码0）。

运行环境：V100（sm_70，CUDA 12.0），只用GPU 0；ozIMMU的CMake要求CUDA≥12.1与sm_80~90，绕过CMake直接用nvcc为sm_70编译全部源文件（`-lcublas -lcublasLt -ldl`）。V100没有INT8 Tensor Core，`cublasGemmEx` 的int8路径在V100上照常运行（非Tensor Core实现）；**选片逻辑与硬件无关**，只有下面缺陷2中warp内未定义行为的具体表现可能因架构不同。节点数据在 `/ssd/haibin/lean4hpc-ozimmu/`。

### 16.1 缺陷1：计数器只分配8个，内核写16个，主机读16个（越界）

- `handle.hpp:22`：`mantissa_loss_counter_length = 13 − 6 + 1 = 8`；`handle.cu` 按8个 `unsigned long long` 分配，`init_mantissa_loss_counter` 只清零8个。
- `calculate_mantissa_loss_kernel` 以 `min=3, max=18` 被调用，`atomicAdd(buffer + (num_split − 3))` 写下标0..15 → **设备端越界写8个**（落在分配粒度的空隙里，没有报错）。
- `get_mantissa_loss_total` 声明 `host_buffer[8]`、只拷回8个，却把 `host_buffer[0..15]` 映射到 `fp64_int8_3..18` → **主机端读栈上8个未初始化的值**（未定义行为）。
- 证据：
  1. 我们尝试从设备拷回16个计数器，`cudaMemcpy` 报 `invalid argument`（越过分配边界）。
  2. 诊断副本（只在拷回后打印 `host_buffer[0..15]`，用volatile读）显示，`[8..15]` 是栈上残留：栈地址、栈保护canary（每次运行不同）、`1`、`8`、`7`/`5`（=每片位数b）、`k`、堆地址。
  3. `[0..7]` 与主机复刻一致（在打印精度内）。
- 后果：`fp64_int8_13` 对应的槽恰好是 `1` → **只要3..10片都不够，库就选13片**，与真实需要无关。
  - 本构建、本机下确定，换编译器/调用栈会变。
  - 阈值1.5：`bug_dgemm`（正确选择是DGEMM）被选成13片，结果 `C=0`，相对误差100%。
  - 阈值0：同样的 `1` 被拒绝，于是多数宽动态范围输入**回退DGEMM**（`rand_s10..s60` 本该11~18片）——安全但白白丢掉性能。

### 16.2 缺陷2：零元素提前返回，warp归约只由lane 0提交（漏计）

`calculate_mantissa_loss_core` 里 `if (in == 0 || max_exp == 0) return;` 发生在全掩码 `__shfl_xor_sync(~0u, …)` 归约**之前**，而归约结果只由 `threadIdx.x & 0x1f == 0` 的线程 `atomicAdd`。于是：
- 只要一个warp负责的32个元素中**第一个是0**，这个warp整组的损失都不会被计入；
- 其余线程以缺员的掩码做 `__shfl_xor_sync(~0u)`，按CUDA规范这是未定义行为；
- `k` 不是32的倍数时，最后一个warp也缺员（`k33` 在V100上没有表现出差异，但同属未定义行为）。
- 证据（`auto_probe`，30%零元素）：设备计数在 s=3 时平均损失为8.58，主机复刻为23.91。
- 对抗输入 `zero_lane0`（沿k每第32个元素置0，其余为指数跨度30的随机数，64×64×256）：
  - 库选 **`fp64_int8_3`**，正确是11片（阈值1.5）/14片（阈值0）；
  - 最坏误差 **8.45e12·u·|A||B|（约1e-3相对）**，即只剩约10位有效位；DGEMM为9.25u。
  - **阈值0也一样选3片**：零元素把计数清空，任何阈值都挡不住。
- `sparse30`（随机30%零）：库选10片，正确是11片；这次误差没有变坏（6.99u），但已经选少。
- 结构化零（三角矩阵、按32对齐的分块/补零）是常见输入。`tri_s30/s60`（A上三角、B下三角，256³）中每行只丢一个warp，库仍因缺陷1选13片。

### 16.3 修复与验证

`ozimmu_auto_fix.patch`（3处）：
1. 计数器长度改为 `18 − 3 + 1 = 16`（分配、清零、拷回、主机数组一起变）；
2. 零元素不再提前返回，以损失0参加归约；
3. 内核循环上界取到32的倍数，越界线程以0参加。

修复后在13个输入×2个阈值上，**库的选择与主机复刻完全一致**（`gemm_probe_fix_results.txt`）；`zero_lane0` 的auto误差从8.45e12u回到5.94u，`bug_dgemm` 回退DGEMM。补丁只在本地副本验证过，**没有提交上游**。

### 16.4 语义问题：即使修好，“平均损失 ≤ 1.5”也不保证逐元素精度

| 输入（修复后，阈值1.5） | 选择 | auto最坏误差（u·\|A\|\|B\|） | DGEMM |
|---|---|---|---|
| `avg_hides`：64³随机，只有A第0行=[1, 2^-100…]、B第0列=[2^-100, 1…] | 8片（与复刻一致） | **9.0e15（C00=0，相对误差100%）**；13片仍为100%，15片才对 | 2.23 |
| `tri_s30`（三角，指数跨度30） | 11片 | **5541**（约6e-13相对） | 14.6 |
| `tri_s60`（三角，跨度60） | 14片 | **2.07e6**（约2e-10相对） | 9.0 |
| `rand_s0..s80`（稠密随机） | 8~18片/DGEMM | 1.3~9.7 | 1.1~10.2 |

- 随机稠密输入上平均准则工作正常（DGEMM量级）。
- 但平均是对全部 `m·k + k·n` 个元素取的：一行/一列的大损失会被其余元素摊薄，输出矩阵中对应的元素就失准。
- **阈值0（修复后）** 等价于“每个元素损失都为0”，所有用例的误差都回到DGEMM量级（`tri_s60` 选18片，9.5u）。
- 这正是下面 `dot_bound` 在 `La=Lb=0` 时给出的条件。

### 16.5 Lean（`OzimmuAuto.lean`，均只用 propext / Quot.sound）

建模：
- 元素幅值 `v = M·2^q`，`2^52 ≤ M`；行最大位 `E ≥ q+52`；`S = s·b`。
- `cut_int8_core` 的s片等价于在格 `2^(E+1−S)` 上向零截断：`kept v g = v / 2^g · 2^g`（已读源码确认：尾数对齐到128位字的bit 127为 `max_exp` 位，逐片取高 b 位）。
- 代码的 `required = (E − (q+52)) + 53`，`loss = required − S`。

| 定理 | 内容 |
|---|---|
| `cut_exact` | `loss = 0` ⇒ 该元素被s片**精确**表示（代码的计数条件恰好是精确性条件） |
| `cut_rel_err` | 一般情形：截断误差 `· 2^52 < 2^loss · v`，即相对误差 `< 2^(loss−52)` |
| `cut_total_loss` | `v < 2^g` ⇒ 全部丢失（`kept = 0`） |
| `term_bound`、**`dot_bound`** | 一个输出元素：A侧各元素损失 ≤ La、B侧 ≤ Lb（且截断向零）⇒ `|Σab − Σâb̂|·2^52 ≤ (2^La + 2^Lb)·Σ|a||b|`（逐元素、DGEMM式的界） |
| `good_side` | 带符号的截断元素满足 `dot_bound` 的A侧前提（把 `cut_rel_err` 接到 `dot_bound`） |
| `avg_not_max` | 对任意阈值θ和元素数N，`Σloss ≤ θN` 允许单个元素损失 `θN`（其余为0） |
| `int32_exact`、`slice_prod_lt` | `b = min 7 ((31−c)/2)`、`k ≤ 2^c` ⇒ `k·2^b·2^b ≤ 2^31`：int8片点积在int32里不溢出（精确） |

结论：
- 保证由行/列上的**最大**损失决定（`dot_bound`），而库用的是**全矩阵平均**（`avg_not_max` + `avg_hides` 实测）。
- 选片要有逐元素保证，正确的准则是“该输出元素涉及的行与列上损失都为0”。阈值0（修复后）满足这一点，阈值1.5不满足。

**没有证明的（边界）：**
1. int32片积在FP64中的合成（`gemm.cu` 的累加与舍入、哪些片对被求和）——端到端误差界还缺这一段；
2. 次正规数（`implicit_one_bit=0`）与 `max_exp` 上溢为Inf（行最大值指数为1023时 `x2` 溢出）没有建模；
3. 模型与CUDA代码的连接靠读源码与上面的实测（复刻与修复后的库逐例一致），不是机器检查的翻译。

### 16.6 还没做

- GEMMul8（Ozaki II）的模数/缩放选择还没读；
- 未在A100/H100（真实INT8 Tensor Core）上复现；
- 没有向上游报告（需要用户决定）。


## 17. B1：GEMMul8（Ozaki II）的缩放是否保证CRT不越界、以及它保证什么精度（2026-11-23）

对象：[GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)（MIT），固定提交 `d8d390125964c7f1c60d5f1e0e5d9656e5614852`（2026-11-16）。只看 **INT8后端、实数FP64 GEMM**。

**做法**：对A的每一行、B的每一列分别缩放，向零截断成整数，用多个模数的INT8 GEMM算余数，再用CRT（对称剩余）重构 `c = Σ x_l y_l`。精确的充要条件是 `|c| ≤ (P−1)/2`（P为模数之积，模数两两互素）。GEMMul8**没有自动选参**：`num_moduli` 由用户指定，README写明推荐值“不是精度保证”。所以B1在这里问的是：**对给定模数个数，缩放是否保证 |c| ≤ (P−1)/2；保证了之后，精度又是什么意义下的。**

文件（`experiments/gemmul8/`）：

| 文件 | 内容 |
|---|---|
| `gemmul8_probe.cu` / `gemmul8_probe_results.txt` | 端到端：25组输入 × {fast, accurate} × N∈{8,12,14,16,20}，对照DGEMM |
| `sft_exhaustive.cu` / `sft_exhaustive_results.txt` | 移位计算的GPU穷举检查 |
| `gemmul8_nvcc120_compat.patch` | 为CUDA 12.0编译所需的两处兼容补丁（不改语义） |

Lean：`lean/OptimiKahan/Gemmul8Range.lean`（200行，独立文件，核心Lean），日志 `check_log_gemmul8range.txt`（退出码0）。

**构建**（V100 sm_70，只用GPU 0）：
- GEMMul8要求CUDA≥12.9，节点只有12.0（驱动580支持13）。从PyPI取NVIDIA官方的cuBLAS / cudart / CCCL 12.9 wheel，解压到 `/ssd/haibin/lean4hpc-gemmul8/cuda129`（约0.9 GB）。
- nvcc仍用系统12.0，只换头文件和库。
- 两处兼容补丁：
  1. herk专用分支改成重载函数（nvcc 12.0前端会对 `if constexpr` 的弃用分支做类型检查）；
  2. 把 `__maxnreg__`（CUDA≥12.4，寄存器上限提示，只影响性能）定义为空。
- 用库自己的Makefile，以 `OPS=gemm OZ2_BACKENDS=INT8 GPU_ARCH=70` 构建静态库（3.5分钟）。
- **构建方式与官方要求不同，结论只对这个构建成立。**

### 17.1 读源码得到的设计（都已逐行核对）

- **fast模式**（README推荐）：
  - 行/列平方2-范数 `v` 全程向上舍入累加（`__fma_ru`、`__dadd_ru`、warp归约也是ru）；
  - `sft = ⌊log2P − log2(v)/2⌋`，其中 `log2P = RD(log2(P−1)/2 − 0.5)`；
  - 中间量全部有向舍入，`__log2f` 的误差（CUDA文档：[0.5,2]上绝对误差≤2^-22）在除以2后用 `+2^-23` 抵消；
  - 截断 `trunc(2^sft·a)` 用位运算实现，向零取整。
- **accurate模式**：
  - 先把 `ceil(|a|·2^pre)` 存为int8，做一次INT8 GEMM得到 `H ≥ 2^(preA+preB)·|A||B|`；
  - 再用H的行最大值修正A的移位、列最大值修正B的移位：`cor = ⌊log2P − log2(max)/(2(1−4u))⌋`，系数补偿 `__log2f` 的2 ulp相对误差。
- **模数表**（已用Python逐项核对）：
  - 20个模数两两互素；
  - `log2P` 表项与公式一致且向下取整；
  - 所有N都满足 `4^log2P ≤ (P−1)/2`。

### 17.2 CRT不越界：两种模式都成立（Lean + GPU穷举 + 读源码）

| 环节 | 方法 | 结论 |
|---|---|---|
| fast：`Σx² ≤ S ∧ Σy² ≤ S ⇒ |Σxy| ≤ S` | Lean `fast_range`（用 `2|ab| ≤ a²+b²`，不需要开方） | 已证 |
| accurate：`|x|≤2^ca·hA`、`|y|≤2^cb·hB`、`H≤R`、`H≤Q`、`4^ca·R≤S`、`4^cb·Q≤S` ⇒ `|Σxy| ≤ S` | Lean `accu_range` | 已证（Lean中 `ca, cb` 为自然数；穷举显示N≥5时修正移位≥3，N≤4会出现负移位，不在Lean覆盖内） |
| CRT：每个模数整除 `c−c'`、两者都在对称范围 ⇒ `c=c'` | Lean `crt_unique`、`dvd_prod_of_pairwise`、`crt_exact`、`fast_crt_exact` | 已证（两两互素作为前提；具体20个模数的互素性用Python验证） |
| 浮点移位的正确性：fast为 `2^ex·vf·4^sft < T+1`，accurate为 `amax·4^cor < T+1`（`T = ⌊(P−1)/2⌋`） | **GPU穷举**，调用库自己的 `calc_sft` 设备函数：fast每个N有 1.76×10^10 个输入（全部指数 × 全部float有效数），accurate有 2^31−1 个 | **N=2..20，两种模式，0个违例** |

- N=2 时实数界偶尔超过T（fast 171336次、accurate 21943次，最大比值1.0000145），但从未达到T+1。整数平方和/整数积 ≤ ⌊界⌋ ≤ T，所以不越界。
- 其余N甚至满足更严的 `≤ T`。
- 实测也一致：近乎取等的 `gram_pos`（B=Aᵀ，全正，k=4096）与 `gram_ones` 在所有N下都没有越界（全1时结果逐位精确）。
- 边界：
  - 范数累加的“全程向上舍入”与截断“向零”来自读源码；
  - `__log2f` 的行为来自真实GPU上的穷举，**不是对文档界的证明**；
  - 穷举只在V100/sm_70上做过。

### 17.3 精度保证的意义：按范数，不是逐元素

`bits = −log2(max|C−ref| / (‖a_i‖₂‖b_j‖₂))`，`wrong` = 相对误差≥100%的元素数：

| 输入 | 形状 | DGEMM | fast N14 | fast N16 | accu N14 | accu N16 |
|---|---|---|---|---|---|---|
| `rand_s0` | 128×128×512 | 53.8 / 0 | 51.9 / 0 | 56.4 / 0 | 51.9 / 0 | 56.4 / 0 |
| `rand_s60` | 128×128×512 | 52.9 / 0 | 52.2 / 0 | 54.9 / 0 | 53.2 / 0 | 54.9 / 0 |
| `gram_pos` | 128×128×4096 | 50.7 / 0 | **47.9** / 0 | 53.8 / 0 | 47.9 / 0 | 53.8 / 0 |
| `avg_hides`（一行一列极端） | 64×64×64 | 52.1 / 0 | 53.6 / **1** | 54.4 / **1** | 53.6 / **1** | 54.4 / **1** |

- **按范数**：有效位约为 `log2P − log2√k − O(1)`（截断误差 `<1`/元素 → `‖δx‖ < √k`）。
  - README的“14个模数≈55位”没有计入k：k=4096时实测只有47.9位，比README少约7位。
- **逐元素**：没有保证。`avg_hides` 的 `C00` 两种模式都是0（真值 64·2^-100），与ozIMMU第16.4节同一现象。
  - 这是定点式设计本身的性质，README以“effective bits”表述，没有声称逐元素精度。
- 端到端误差界（截断 + 最终FP64舍入）**没有在Lean里证明**。

### 17.4 fast模式的缺陷：平方范数溢出/下溢，结果静默为0

fast模式先算 `Σa²`，FP64指数范围因此被减半。

| 输入（所有N相同） | fast | accurate | DGEMM |
|---|---|---|---|
| `big511`：A ≈ 2^511·U，B ≈ 2^-511·U（k=256，`Σa²` 上溢为inf） | **4096/4096个元素为0** | 正常（52.8位） | 正常 |
| `big508`（`Σa² ≈ 2^1022`，未溢出） | 正常 | 正常 | 正常 |
| `big520_row0`：只有A第0行≈2^520 | **第0行64个元素为0** | 正常 | 正常 |
| `bigB515`：B ≈ 2^515，A ≈ 1 | **全为0** | 正常 | 正常 |
| `tinyA545`：A ≈ 2^-545，B ≈ 1（`a²` 下溢，ru取到 2^-1074） | N14：44.0位（掉约9位） | 52.8位 | 53.0位 |
| `tinyA600/800/1000` | **全为0** | 正常 | 正常 |

机制：
- **上溢**：`v=inf` → `ilogb(inf)=INT_MAX` → `sft = INT_MIN` 量级 → 缩放后全为0。存回 `int16(−sft)` 时又截成0。没有任何报错。
- **下溢**：每个 `a²` 被向上舍入到 2^-1074，范数被高估，移位变小，元素被截掉。
  - N=14时，元素 `|a| ≲ 2^-591` 全部丢失；`|a| ≲ 2^-537` 开始掉精度。
- **触发范围**：行/列2-范数 ≳ 2^512 ≈ 1.3e154，或元素 ≲ 2^-537 ≈ 4e-162。这对普通应用少见，但都是合法的FP64输入，DGEMM和accurate模式都能正确处理。

### 17.5 两种模式共有的缺陷：Inf/NaN不传播

- 输入：A[0,0] 为 inf 或 NaN。
- DGEMM：第0行为 ±inf/NaN。
- GEMMul8：第0行64个元素都是**有限值**。fast给0，accurate给如2.37这样的数。
- 原因：`trunc_scalbn_to_i32/i64_bit` 把指数域0x7ff当普通数处理，移位超过63位，是未定义行为；范数/最大值也会变成inf/NaN。
- README没有说明非有限输入的行为。

### 17.6 结论与边界

1. **好消息**：两种模式都保证CRT不越界（N=2..20），即不会出现“回绕”式的灾难性错误。这一点由Lean（整数部分）+ GPU穷举（浮点部分）+ 读源码（舍入方向）共同支撑。
2. 精度是**按范数**的，约 `log2P − log2√k` 位。README的推荐表没有计入k，k大时会高估。
3. **fast模式（默认推荐）在 FP64 指数范围的两端会静默得0**：行/列范数 ≳ 2^512、元素 ≲ 2^-537。accurate模式没有这个问题。
4. 两种模式都不传播Inf/NaN。
5. 没做：
   - FP8后端、复数、symm/syrk等其他例程；
   - A100/H100（真实INT8 Tensor Core）复现；
   - 向上游报告。


## 18. 新机器 roxy（2026-11-23，只做了环境探查）

- 访问：`ssh roxy`（Windows 11 Build 26200，账户 `roxy\11062`，**有管理员权限**）。这是用户的个人台式机（运行着Steam、Wallpaper Engine等），所以**安装系统软件前要先问用户**。
- GPU：RTX 5070 Ti（Blackwell 消费级，sm_120，16 GB，WDDM），驱动617.42，CUDA UMD 13.4。有INT8/FP8/FP4 Tensor Core，可以测FP8、FP4等低精度库，也可以测原生库。
- 软件：
  - 有 Python 3.11 + pip 24.0；
  - **没有** CUDA Toolkit、MSVC、git、WSL；
  - 能访问PyPI。
- 磁盘：C: 剩余359 GB，D: 剩余926 GB。
- 状态：
  - 用户要求先调研可在这台机器上测的其他低精度库（A/B两类），环境搭建（WSL2 / Windows原生CUDA+MSVC / 仅Python）暂缓、待用户选择；
  - 数据目录尚未确定（建议 `D:\lean4hpc`，用完删除）；
  - **还没有在roxy上安装或写入任何东西。**
- 调研结果见 `operators/REVIEW5.md`（2026-11-23）：用户指定B类五项为主线。建议顺序：
  1. cuBLAS FP64仿真（ADP）：sm_120支持，文档承诺“equal or better than FP64”、覆盖全部有限值；可能只需Python + CUDA 13 wheel；
  2. GEMMul8 FP8后端；
  3. cuSOLVER IRS / MAGMA（MAGMA有NaN被当作收敛的结构性问题，未测）；
  4. cuMpSGEMM/TCEC（陈旧）；
  5. BF16x9：sm_120不支持。


## 19. B类：cuBLAS FP64仿真（Fixed-Point / ADP）在 RTX 5070 Ti（sm_120）上的黑盒测试（2026-11-23）

**环境**：
- roxy，`D:\lean4hpc\venv`（Python 3.11；numpy 2.4.6；PyPI上的 `nvidia-cublas==13.10.0.20`、`nvidia-cuda-runtime==13.5.37`）；
- 用ctypes调用 `cublas64_13.dll`，`cublasGetVersion` 返回 131000；
- **没有安装任何系统软件，也没有编译**。

文件（`experiments/cublas_emu/`）：

| 文件 | 内容 |
|---|---|
| `cublas_emu_probe.py` / `cublas_emu_results.txt` | 20组对抗输入 × 5种模式，用精确整数参考 |
| `cublas_emu_large.py`、`cublas_emu_E1.txt`、`cublas_emu_E2.txt` | 大规模：零块病态（E1）；n=2048的对抗输入，对抽样元素精确核对（E2） |
| `cublas_emu_zcheck.txt` | 零块情形是否与原生逐位相同 |
| `cublas_emu_perf4096.txt` | n=4096 的计时 |

**5种模式**：
- `native`；
- `emu_perf`（FP64_EMULATED_FIXEDPOINT_MATH + PERFORMANT，即默认策略）；
- `emu_eager`；
- `emu_eager_sv0`（关闭特殊值支持）；
- `emu_fixed55`（固定55位）。

**判据**：`comp = |C−ref| / (u·(|A||B|)_ij)`；`>k` 计逐元素超过 FP64 GEMM 先验界（≈k）的元素数。

**观测ADP的决定**：`cublasSetFixedPointEmulationMantissaBitCountPointer` 指向一个设备整数，Lt文档说它存放“保留的尾数位数，未使用仿真时为−1”。

### 19.1 精度：ADP在全部对抗输入上都达到FP64级

- 20组小规模输入（含ozIMMU/GEMMul8的反例 `avg_hides`、`zero_lane0`、`tri_s60`、`wide_row0`，以及2^±511/515/1000、2^-600/-1000、非规格化数、Inf、NaN、−0、相消），加上7组 n=2048 的抽样核对：**动态模式（PERFORMANT/EAGER）没有任何元素超过 k**。
  - 最坏为 13.8 u·|A||B|，多数情形比原生FP64更准；
  - `avg_hides` 选154位，结果正确（ozIMMU/GEMMul8在这里100%错误）。
- 原因可以从ADP论文（arXiv 2511.13778 §4）的ESC公式读出：
  - `ESC = exp(x_p)+exp(y_q)−exp(z_r)+1`，按**每个点积**的最大乘积指数 `z_r` 留位，是逐元素意义的保证；
  - 不是ozIMMU那样的全矩阵平均，也不是GEMMul8那样的按范数。
- Inf/NaN：EAGER下（即使把特殊值掩码设为NONE）也会检测到并回退原生，结果正确（bits=−1）。
- 固定55位（用户自选）在宽动态范围下失准（`avg_hides` 100%、`tri_s60` 29个元素100%），并把Inf输入变成NaN。这是文档说明的固定模式语义，不算缺陷。

### 19.2 性能缺陷1：零块使ESC暴涨，“仿真”变成更慢的原生FP64

**机制**：
- 论文的粗化ESC对长为 b 的块用 `max(Max(x_b)+Min(y_b), Min(x_b)+Max(y_b))` 估计 `z_r`；
- 零元素的指数按最小值计；
- 若每个块在x与y中都含零，估计值下降约一千，ESC报出**4248位**（三角情形4308位），超过上限后回退原生。

**实测（E1/zcheck/perf4096）**：
- 沿k每隔 ≤64 个位置有一个零（A的列、B的行）时触发；每隔128个时正常（推断 **b=64**）；只在A中放零或极小数（2^-900）不触发。
- n=1024 时结果与原生**逐位相同**（确实回退了）。指针报告4248，而不是文档说的−1。
- 时间：

  | n | 正常仿真 | 零块/三角（4248位） | 原生FP64 | 结论 |
  |---|---|---|---|---|
  | 2048 | 10 ms | 34 ms | 27 ms | 慢于原生 |
  | 4096 | 39 ms | 230–235 ms | 208 ms | 慢于原生约11% |

  4096³ 的三角×三角（`tri_s0`，**均匀随机值，没有任何动态范围**）同样如此。
- 同一输入用固定55位仿真，精度足够（抽样 0.16 u·|A||B|）。所以真实需求约58位，**ESC因零元素多估了约4190位**。
- **现实中会触发的输入**：三角×三角（L·U、TRMM类）、按块稀疏或结构化零的矩阵。
- 精度不受影响，损失的是性能：既拿不到约5倍加速，还比原生更慢。论文称ADP保证“performance never drops below the baseline”，**此处不成立**。

### 19.3 性能缺陷2：PERFORMANT也会选出比原生更慢的路径

- n=4096 时，所需位数94以内（指数跨度≤40）仿真快约5倍；**114位以上（跨度≥60）则约230 ms，比原生208 ms慢约11%**。
- n=2048 的 `rand_s60`（115位）结果与原生不同，确实在仿真，用时34.6 ms，原生26.1 ms。
- 文档说PERFORMANT“只在有性能收益时才用仿真”，实测在这些点上不成立。
- 小规模（n≤512）时PERFORMANT不仿真，且**不写指针**（保持我们填入的−2），与文档描述的−1不同。

### 19.4 结论、边界与下一步

**结论**：
1. 对cuBLAS 13.10在sm_120上，文档“accuracy equal or better than FP64”在我们的全部对抗输入上成立（黑盒；抽样或小规模的精确核对）。
2. 问题在**性能保证**：
   - 零块/三角输入使粗化ESC高估数千位，变成“回退 + 额外开销”；
   - 需要100位以上时同样回退，且比原生慢约11%。
3. 文档小差异：指针在回退时报告的是所需位数（或不写），而不是−1。

**边界**：
- 只测了 `cublasDgemm`、列主序、alpha=1、beta=0、方阵；
- 计时是单卡WDDM，有桌面负载；
- n≥2048 时只做抽样精确核对；
- 粗化块长 b=64 是由行为推断的；
- 没有测Ozaki-II路径是否被选用，也没有测复数。

**下一步候选**：
- Lean：在论文模型下证明“ESC（含粗化）⇒ 逐元素误差界”，以及“粗化估计 ≤ 真实 z_r”（论文给了口头证明）；
- 一个零感知的ESC修正（计算 Min 时跳过零），并证明它仍然安全；
- 向NVIDIA报告性能问题（需用户决定）。


## 20. 参数契约方法（2026-11-23）

用户要求把“参数到底保证了什么”精确写下来，并让它反过来指导反例构造。方法与四张契约卡见 `CONTRACTS.md`。

**卡片格式**：① 规则（逐字取自代码或论文）→ ② 声称 → ③ 可证（显式列出前提H）→ ④ 缺口 → ⑤ 由缺口推出的反例族 → ⑥ 实测。

### 20.1 cuBLAS ESC 契约的Lean化（`lean/OptimiKahan/CublasEsc.lean`，171行，日志 `check_log_cublasesc.txt`，退出码0）

**建模**：
- 论文（arXiv 2511.13778 §4）的块粗化估计 `est = max(Max(x_b)+Min(y_b), Min(x_b)+Max(y_b))`；
- 零元素没有指数，记为编码 Z。

| 定理 | 内容 |
|---|---|
| `coarse_safe` | 实指数都在 `[lo, hi]`、`Z ≤ lo`、`Z + hi < 2·lo` ⇒ 块内存在实乘积 `(x, y)` 使 `est ≤ x + y`（估计不越过真 z_r，ESC不会被低估） |
| `coarse_unsafe` | 若 `2·lo < Z + hi`，块 `x=[hi, lo]`、`y=[0, lo]` 的估计超过其唯一的实乘积指数，即契约条件**也是必要的** |
| `zero_block_gap` | 块内 x、y 都含零 ⇒ `est ≤ hi + Z`，于是粗化ESC至少多出 `z_r − (hi + Z)` 位 |
| `binary64_contract` | binary64（实指数 ∈ [−1074, 1023]）下契约等价于 `Z ≤ −3172` |

**意义**：
- 论文的安全性论证默认没有零；这个契约把“零怎么编码”变成了显式前提；
- 违反它会导致**精度**风险，满足它的代价是**性能**风险。

### 20.2 由契约推出的预测与实测（`experiments/cublas_emu/esc_predict.py` / `esc_predict_results.txt`，roxy，cuBLAS 13.10）

- **预测**：每块两侧都有零时，`报告位数 = min(exp x_p, exp y_q) − Z + 常数`。
- **P1**：零每隔32个、n=1024，对A、B做7种缩放，位数变化**全部与预测逐位一致**（0, 0, −100, +100, −100, −300, 0）。无零的稠密矩阵始终是59位。
- **P2**：边界块 `x=[2^300, 1, 0…]`、`y=[0, 2^-1035, 0…]`（真值 2^-1035，非规格化数）。
  - 预测报出 4248−1034 = 3214 位并回退；实测**3214位**，4096个元素都精确；
  - 固定55位时**全部为0**，ESC的保守是必要的。
- **推论**：库内零的指数码 |Z| ≈ 4190，满足 `Z ≤ −3172`，余量约1000位。这解释了第19节的4248位，以及三角×三角的回退与性能退化。

### 20.3 方法上的结论

这是本项目第一个**先由形式化契约推出、再被测试精确证实**的定量结果（不是事后解释）。

前瞻对象（契约已写好，尚未测试）：
- MAGMA混合精度迭代精化：接受判据 `Rnrm > Xnrm·cte` 对NaN不安全；预测“低精度分解溢出 ⇒ 返回NaN且报告收敛”；
- cuSOLVER IRS：文档判据为 `<`，预测NaN安全（已证实，见第23节）；但它的GMRES路径不执行文档判据（第22节）。

## 21. cuBLAS ADP：位数模型的完整拟合、逐元素误差界、安全的零感知修正（2026-11-23，`AdpEsc.lean`）

与第20节（另一会话，`CublasEsc.lean`、`esc_predict.py`）并行完成，二者在“零指数码的安全条件”上重叠、结论一致。本节多出的是：位数公式的**全部常数**、把估计接到**逐元素误差界**的完整链条、条件的**紧**形式，以及一个**证明安全且实测有效**的修正。

### 21.1 位数模型：38/38 精确复现（`experiments/cublas_emu/esc_model.py`、`esc_ident.py`、`esc_model_results.txt`）

用与探针相同的随机种子重建输入，在CPU上计算：
- `e(v) = floor(log2|v|)`，零记为 `L`；`x_i`、`y_j` 为行/列最大指数；
- `est_ij = max_b max(MaxA_ib + MinB_bj, MinA_ib + MaxB_bj)`，块长64；
- **`bits = max_ij (x_i + y_j − est_ij) + 54`**，`L = −4195`（= 2·(−1074) − 2047）。

结果：
- 第19节全部38个测得位数（20组小规模EAGER，E1的14个，E2的7个；含4248/4278/4308、154、115、73、62）**逐个相等**。
- 参数是唯一确定的：块长32或128会和E1不符；把非规格化数的指数取成指数域（−1023）会把 `denormA` 预测成54位，实测为62位，所以库用的是真实指数。
- **样本外预测**（`sentinel_probe.py`、`sentinel_probe_results.txt`）：
  - S1：零 + 2^-1074 + 2^1000 同块。预测1054位（若L不安全则为69位），实测**1054位**，回退原生，C00精确。
  - S2：预测114位，实测**114位**，确实仿真且精确。
  - S1里抽样的另一个元素在原生FP64下也错，原因是结果低于2^-1074，与仿真无关。

### 21.2 Lean（`lean/OptimiKahan/AdpEsc.lean`，约720行，日志 `check_log_adpesc.txt`，退出码0，公理只有 propext / Quot.sound）

| 定理 | 内容 |
|---|---|
| `paperEst_le` | 实指数 ∈ [Emin, Emax] 且 `L + Emax ≤ 2·Emin` ⇒ 粗化估计 ≤ 真实最大乘积指数 z（对全部块） |
| `sentinel_iff` | 该条件**充要**（`≤` 为紧；第20节的 `Z + hi < 2·lo` 是充分条件）。FP64：安全 ⟺ `L ≤ −3171`；`fp64_sentinel_ok`：−4195 安全 |
| `skip_zero_unsafe` | 第19.4节设想的“求Min时跳过零”**不安全**：块 A=(2^0, 2^-1000)、B=(0, 2^0) 中 MaxA+MinNZ_B = 0，而唯一非零乘积指数为 −1000 |
| `pigeon`、`cand_sound`、`fixedEst_le` | 安全的零感知候选：(a) 块内任一实乘积（如首、末位置）；(b) `nnzA + nnzB > 块长` 时取 `MinNZ_A + MinNZ_B`（鸽巢保证存在双非零位置）。加入后估计仍 ≤ z |
| `adp_sound` | 整数定点模型：每项 `|a| < 2^XA`、`|b| < 2^XB`，向零截断到 `2^{XA−m}`、`2^{XB−m}` 的倍数，整数点积精确。若 `est ≤ z` 且 `XA + XB + t ≤ m + est`，则 `|Σab − Σãb̃|·2^t ≤ 2k·Σ|a||b|` |
| `adp_paper_sound` / `adp_fixed_sound` | 两种估计分别接入 `adp_sound`（值→指数用 `Nat.log2`，按块） |
| `cublas_rule` | 拟合的规则 `m = x + y − est + 54` 满足 t = 52 ⇒ 截断误差 ≤ 4k·u·(|A||B|)_ij（再加最后一次舍入） |

**模型假设**：
- INT8切片的乘加精确；
- 向零截断；
- “保留位数”指相对行/列最大值以下的位数。

常数4取决于这些未公开的内部细节。逐元素的形式本身不依赖它们。实测最坏为 13.8 u（k=2048），远低于该界。

### 21.3 修正的实际收益（`fix_benefit.py`、`fix_benefit_results.txt`，roxy）

cuBLAS无法打补丁。做法是先用模型算出修正后的位数，再用固定尾数模式按该位数运行，并做抽样精确核对：

| 输入 | 默认ADP | 修正后的位数 → 固定模式 | 原生FP64 |
|---|---|---|---|
| 零每隔32，n=4096 | 4248位，239 ms | **54位，26.7 ms**，comp 0.03 | 208 ms，comp 1.0 |
| 三角×三角（均匀），n=4096 | 4248位，229 ms | **68位，36.8 ms**，comp 0.96 | 211 ms，comp 1.4 |
| 三角×三角（s60），n=2048 | 4308位，32.6 ms | 177位，24.8 ms，comp 8.7 | 27.3 ms，comp 8.5 |

- 在模型里，修正把E1零块情形从4248位降到54–56位，把 `zero_lane0` 从4278位降到100位。
- 也有它降不下来的：三角×s60 仍要170多位，因为粗化本身就有代价；对 `rand_s60`，精确 z 需要67位，粗化估计给出115位。
- 计时不含修正本身多出的ESC计算开销（每块多两次加法和两个计数）。

### 21.4 边界

- 位数模型是黑盒拟合（38个点加2个样本外点），**不是源码**。
- `adp_sound` 是论文层面的模型，没有覆盖Ozaki-II路径，也没有覆盖切片的具体实现。
- 修正只在“固定模式按模型位数运行”的意义上验证过。

**可以对外报告的内容**（需用户决定）：零块/三角输入回退原生且更慢；附上安全修正及其证明。

## 22. B类：cuSOLVER混合精度迭代精化（IRS）——GMRES路径报“收敛”却返回错误解（2026-11-23，roxy）

**环境**：
- roxy，`D:\lean4hpc\venv` 里加装了 `nvidia-cusolver==12.4.1.2`（`--no-deps`；`cusolverGetVersion` 返回 12401）；
- 驱动 617.42，RTX 5070 Ti；
- 只用Python + ctypes，没有编译。

文件在 `experiments/cusolver_irs/`。

### 22.1 契约（文档 cuSOLVER 13.5，§2.4.2.6/2.4.2.7/2.4.1.45–50，原文由子代理摘录）

| 栏 | 内容 |
|---|---|
| 规则 | 当 `RNRM < SQRT(N)·XNRM·ANRM·EPS·BWDMAX` 时停止；BWDMAX=1，均为∞范数，ITERMAX=50。失败时niters<0：−1 不值得、−2 降精度转换溢出、−3 分解失败、−5 计算中溢出、−50/−maxiter 达到上限；随后回退主精度（**默认开启**）。|
| 声称 | “same normwise backward error as the main precision”；开启回退时“always a good solution”；niters>0 表示“reach convergence criteria”。|
| 推论（据此设计的测试） | 文档规则在 (A, b) → (2^k A, 2^k b) 下不变；在没有溢出和非规格化数时，IEEE运算与2^k缩放严格交换。所以在对应格式的规范数范围内，**x 应逐位不变、niters 应不变**。|

### 22.2 结果

**(1) 经典精化（CLASSICAL）符合文档**（`irs_rule.py`）：
- 32F/CL 在 k=−20…20 下，每次都在“首个满足文档规则的迭代”停止；
- niters 恒为5，RNRM/阈值恒为0.0346，满足尺度不变。

**(2) GMRES精化不符合文档**（`irs_scale.py`、`irs_rule.py`）：
- 尺度不变性被破坏。16F/GM 在 k=−12…8（全部位于FP16规范数范围内）：
  - x 随 k 改变；
  - niters 从17降到11；
  - 返回解的后向误差从0.008倍升到601倍文档阈值。
- 停止时机与任何简单规则都不一致：
  - 32F/GM k=−20：第5次迭代已满足文档规则，却继续到第45次；
  - 16F/GM k=20：在 RNRM 为阈值 2.5×10¹³ 倍时报告“收敛”（niters=37）。
- 精确有理数残差复核（n=512，16F/GM，k=12）：后向误差为阈值的 1.277×10⁵ 倍，与FP64残差一致，**不是残差舍入的假象**。

**(3) 默认的一行式API会静默返回错误解**（`irs_legacy_scale.py`、`repro_dbgesv.py`）：
- 设置：`cusolverDnD{H,B,S,X}gesv` 默认设置、默认开启回退；A 的元素为 ±[0.5,1)，x 为均匀分布；
- 结果 iter ≥ 0（因而**不回退**），但解是错的。

| API（下限精度） | 缩放 | iter | 后向误差/阈值 | x 的最大相对误差 |
|---|---|---|---|---|
| DBgesv（BF16） | 2^40，n=1024 / 4096 | 1 | 4–8×10¹¹ | **0.93–3.3 / 23** |
| DXgesv（TF32） | 2^40 | 1 | 4×10¹⁰ | **0.04–2.1** |
| DSgesv（FP32） | 2^40 | 1 | 1.2–4.8×10⁵ | 5×10⁻⁸–1.9×10⁻⁶ |
| DSgesv（FP32） | 2^−60 | 0 | 3–6×10⁸ | **1–3×10⁻³** |
| DBgesv / DXgesv / DSgesv | 2^−100 | 0 | 10⁸–10¹¹ | 0.001–22 |
| DBgesv | 2^0（未缩放），n=1024 / 2048 | 23–43 | 1.4–60 | 3×10⁻¹²–5×10⁻⁹ |
| DDgesv（对照，FP64） | 2^0 / 2^40 | 1 | 0.0109 / 0.0109 | 5×10⁻¹⁴，逐位相同 |

- 2^40 ≈ 10¹²，2^−60 ≈ 10⁻¹⁸：这些量级都在 BF16/TF32/FP32 的范围以内，物理单位下的矩阵完全可能取到。
- 规律：k 增大时，停止阈值看起来随缩放超线性增长，越来越容易“收敛”；k 很负时，残差范数下溢为0，于是在第0次迭代就被接受。机制属于推断，未证实。

**(4) 文档承诺的其他偏差**：
- 元素超过65504（缩放 10⁶）时，16F/GM 没有返回 −2，而是报“收敛”（45次），相对误差 8.5×10⁵；16F/CL 返回的是 −5，不是 −2。
- 尺度 10⁻³⁰ 时 16F/GM、16BF/GM 在第0次迭代就被接受，误差为 21 / 84。
- Wilkinson 增长矩阵（n=512、1000）：FP64 回退本身后向误差约为阈值的 3×10¹² 倍，niters=−50。这是 GEPP 的固有不稳定，但与“回退总能得到好解”的说法不符。同一矩阵上低精度路径反而收敛，推测选主元的平局规则不同。

**(5) NaN安全性（第20节契约预测cuSOLVER安全）——成立**：
- b 或 A 含 NaN/Inf 时一律返回 −5；关闭回退时 info 被置为 256/2048（文档里 info=i 的含义是 U(i,i)=0，属于小偏差）；
- b=0 时 niters=0，x=0（严格的 `<` 对 0<0 为假，说明有特殊处理）。

### 22.3 结论与边界

**结论**：
- cuSOLVER IRS 的 GMRES 精化路径，以及使用它的一行式 `DHgesv/DBgesv/DXgesv/DSgesv`（至少在当前设置下其迭代数与 IRS 的 GM 设置一致），**不执行文档所说的收敛判据**；
- 它会以 niters ≥ 0 返回后向误差超出阈值多达 10¹¹ 倍、相对误差 >1 的解，**默认回退因此不触发**；
- 经典精化符合文档。

这属于B类“产出慢、要求高可靠”的核心失效：通过选参/接受测试声称精度，而接受测试并未被执行。

**边界**：
- 只测了 cuSOLVER 12.4.1.2（CUDA 13.x wheel）与 sm_120，没有测更早的版本和其他架构；
- 只测了 nrhs=1、方阵、随机矩阵；
- 一行式API内部用的是哪种精化求解器，只是由迭代数吻合推断的；
- 停止规则的具体形式没有反推出来。

**MAGMA**：
- 已由并行会话在s82上测过，见本文件第23节“前瞻测试：迭代精化接受判据”（`experiments/ir_accept/`）；
- 该节把cuSOLVER列为“安全的对照”，只对NaN与Wilkinson输入成立；**按本节的缩放测试，cuSOLVER的GMRES路径同样会报告收敛却给出错误解**。

### 22.4 旧版本复核（`repro_dbgesv.py` 已改为自动查找任意 cu11/cu12/cu13 wheel 的DLL；`repro_versions_results.txt`）

**做法**：
- 在 roxy 上为每个版本建独立的 venv（`D:\lean4hpc\v129`、`v124`、`v121`、`v118`），cuBLAS、cuSPARSE、runtime 都固定为同一CUDA发行版；
- 用同一个复现脚本，n=1024，缩放 2^0、2^20、2^40、2^−60。

| cuSOLVER（CUDA） | DBgesv 2^40 | DXgesv 2^40 | DSgesv 2^40 | DSgesv 2^−60 |
|---|---|---|---|---|
| 11.4.1（11.8，2022） | iter=1，误差 **3.1** | 1，1.1e-7 | 1，1.3e-7（后向误差/阈值 8.9e4） | iter=0，误差 1.0e-3 |
| 11.6.1（12.4） | 1，**3.1** | 1，1.1e-7 | 1，8.2e-8（5.2e4） | 0，8.6e-4 |
| 11.7.5（12.9） | 1，**1.24** | 1，**0.033** | 1，2.5e-8（2.6e4） | 0，3.0e-4 |
| 12.4.1（13.x） | 1，**0.93** | 1，**0.041** | 1，5.3e-8（1.2e5） | 0，1.1e-3 |
| DDgesv（对照，各版本） | 尺度不变，误差约1e-13 | | | |

- 所有可运行的版本都在报告 iter ≥ 0 的同时给出错误解，**问题至少从 CUDA 11.8 起就一直存在**。
- 11.7.5 起，DXgesv 变得更差（0.033–0.041），推测与 TF32 路径改动有关。
- 11.4.5（CUDA 12.1）在 roxy 上加载失败（WinError 127，找不到入口点），推测是wheel之间的依赖不匹配，未深究。
- 边界：所有版本都运行在 sm_120 上；旧版本没有 Blackwell 的 SASS，走的是 PTX JIT，所以无法区分“版本”与“架构”两个因素。

**roxy 占用**：
- 各 venv 约1.3–2.0 GB，`D:\lean4hpc` 合计约 9 GB；
- **用户要求保留，不要删除**（2026-11-23）。

### 22.5 多种子复核三轮（`irs_verify.py`、`verify_R1/R2/R3/R3IRS.txt`，cuSOLVER 12.4.1）

**判定**（niters ≥ 0 即“报告收敛”，此时默认回退不会触发）：
- **WRONG**：x 的相对误差 > 1e-6（FP64 在这些系统上约为 1e-13）；
- **VIOLATES**：后向误差超过文档阈值 100 倍以上（远超 FP64 残差舍入可能造成的偏差）。

| 轮次 | 设置 | 未缩放 | 缩放后 |
|---|---|---|---|
| R1 | ±[0.5,1)，n=1024，种子1–10；DB/DX/DS × 2^20、2^40、2^−60；DH 只用 2^0、2^4 | 30/30 ok | 90次中 **WRONG 42、VIOLATES 19**；其余29次中20次是返回−50后回退（结果正确）。DH 40/40 ok |
| R2 | 4种矩阵（均匀、高斯、对角占优、κ=1e3）× n∈{512, 2048} × 种子{101, 202, 303}；十进制缩放 1e6、1e12、1e−18 | 96次中 VIOLATES 3 次（DBgesv，n=2048，最高 1.8e3 倍），WRONG 0 | 216次中 **WRONG 98、VIOLATES 70**；每种矩阵都受影响（WRONG/VIOLATES：均匀 27/14，高斯 27/11，对角占优 17/30，κ=1e3 27/15）；最大相对误差 **16.2** |
| R3 | 每次调用单独开新进程，n=512，种子11–15，并计算精确有理数残差 | — | DB 2^40 5/5 WRONG（0.12–3.5）；DX 2^40 5/5 WRONG；DS 2^40 5/5 VIOLATES；DS 2^−60 5/5 WRONG；DH 2^4 5/5 ok。**全部被接受的解，精确残差与FP64残差一致**（相对差在1%以内） |
| R3IRS | IRSXgesv，n=2048，种子7–9，k∈{−10, 0, 10, 20, 40} | — | **32F/CL：15/15 ok，且同一种子下各缩放的 niters 与误差完全相同**；32F/GM 5/15、16BF/GM 10/15、TF32/GM 9/15 被标记 |

**结论**：
- 现象在所有种子、4种矩阵、2种规模上都出现，与是否新开进程无关；
- 十进制缩放（1e6、1e12、1e−18）同样触发，不依赖2的幂；
- 经典精化在全部情形下尺度不变且符合文档，进一步确认问题只在 GMRES 路径；
- 新增一点：即使不缩放，DBgesv 在 n=2048 时也有 3/24 次超阈值 250–1800 倍（误差约 1e-9）。

### 22.6 形式化（`lean/OptimiKahan/IrsContract.lean`，611行，日志 `check_log_irscontract.txt`，退出码0，无sorry）

用 core Lean 的 `Rat`（有理数）建模，不用Mathlib。公理为 propext、Classical.choice、Quot.sound：Classical.choice 来自 core 的 `Rat` 库本身（如 `Rat.le_total`、`Rat.zpow_add`），无法避免。

**第1部分：契约**

| 定理 | 内容 |
|---|---|
| `Accept`、`Post` | 文档的后置条件：`iter ≥ 0 → ‖b−Ax‖∞ < c·‖x‖∞·‖A‖∞`，精确有理数运算 |
| `accept_scale`、`post_scale` | 文档规则在 (A, b) → (sA, sb)（s > 0）下不变 |
| `absolute_rule_flips` | 绝对容差 `‖r‖ < tol` 不具有这个性质（具体反例） |

**第2部分：为什么缩放测试是可靠的判据**

| 定理 | 内容 |
|---|---|
| `rnd` | p 位尾数、就近舍入、平局取偶，指数不设上下界（即不上溢、不下溢） |
| `rnd_scale` | 对任意整数 k，`rnd(2^k q) = 2^k·rnd q` |
| `rnd_rel_err` | `|rnd q − q| ≤ 2^-p·|q|` |
| `expOf`、`expOf_spec` | 构造出满足 `2^E ≤ |q| < 2^(E+1)` 的指数函数，保证前面的定理不是空真 |
| `Ex`、`HasDeg` | 带缩放次数（degree）的程序语言：舍入的 + − × ÷，精确的 neg、abs、max，比较，常数（次数为0）；次数不同的值相加或比较属于类型错误 |
| `eval_scale` | 类型正确的程序与缩放交换：分支相同，舍入决定相同 |
| `iterate_scale` | 推广到迭代：停止判据和迭代计数作为次数0的状态变量，经 `ite` 更新 |
| `oracle` | 次数为0的输出（例如 A、b 次数为1时的解 x）在缩放下逐位不变 |
| `absolute_tol_breaks` | 比较中出现绝对常数时，分支随缩放翻转 |

**含义**：
- 只用这些运算、并且没有上溢或下溢的实现（LU、三角解、∞范数下的GMRES、文档的停止判据），必须对 2^k 缩放给出逐位相同的 x 和 iter。
- 实测中 x 随缩放改变，**证明实现里存在与尺度有关的常数，或者发生了上溢/下溢**。
- 但这本身不等于违反契约。违反契约是由第1部分的后置条件直接检验得出的（22.5节）。

**模型边界**：
- 没有建模 sqrt（`Rat` 没有平方根），所以2-范数不在模型内；正确舍入的 sqrt 同样与缩放交换，但未形式化；
- 没有建模指数范围的上下界：定理的前提是没有上溢和下溢，而这一点对闭源库无法检查。

**第3部分：可靠的后检**

| 定理 | 内容 |
|---|---|
| `wrapper_sound` / `wrapper_post` | 给定残差范数与矩阵范数的计算误差界（作为前提），若测试 `(1+ε)R̂ + γR(B + X·Â/(1−γA)) < c·X·Â/(1+γA)` 通过，则精确的文档条件成立，与求解器返回什么无关 |
| `fp64_cannot_certify` | 只要 `γR ≥ c`，这个测试就永远不能通过。FP64 残差的 `γR ≈ (n+1)u`，大于 `c = √n·u`：**文档阈值低于FP64残差的最坏舍入误差**，所以严格认证需要补偿算法（Dot2，`γR ≈ n²u²`） |
| `fp64_n1024` | n=1024 的具体数值：c = 32u ≤ 1025u |

残差的计算误差界作为**前提**，没有在Lean里从 `rnd` 推导出来。它是标准结果：Dot2 的误差界见 Ogita–Rump–Oishi 2005，FP64 残差的误差界见 Higham。

**后检的实测**（`experiments/cusolver_irs/wrapper_demo.py`、`wrapper_demo_results.txt`）：
- 设置：Dot2 残差，测试用 Fraction 精确求值；n=1024，种子1–3，DB/DX/DS × {2^0, 2^20, 2^40, 2^−60}，共36次。
- 结果：
  - cuSOLVER“报告收敛却错误”的13次全部被拒绝，改用 DDgesv，并通过认证；
  - 另有11次误差较小但超出阈值的也被拒绝；
  - 12次直接接受；
  - 36次全部通过认证，最终误差 ≤ 1.5e-11。
- **用FP64残差的同一测试 0/36 通过**，与 `fp64_cannot_certify` 一致。
- Wilkinson n=1024：DSgesv 的结果通过认证，误差 1.2e-15。这次没有触发 FP64 回退失败的情形。

**下一步候选**：
- 向NVIDIA报告（需用户决定；最小复现为 `repro_dbgesv.py`）。


## 23. 前瞻测试：迭代精化接受判据的契约预测（2026-11-23；原编号21，因与第21节重号而改为23）

**目的**：用第20节的方法**先写契约、推出反例，再测试**。契约见 `CONTRACTS.md` 第6节。

**文件**（`experiments/ir_accept/`）：

| 文件 | 内容 |
|---|---|
| `cusolver_irs_probe.py` / `cusolver_irs_results.txt` | cuSOLVER，Wilkinson与NaN |
| `cusolver_irs_round2.py` / `cusolver_irs_round2_results.txt` | cuSOLVER，n=1024 块对角 |
| `magma_ir_probe.cpp` / `magma_ir_results.txt` | MAGMA |
| `lapack_dsgesv_probe.c` / `lapack_dsgesv_results.txt` | 系统 LAPACK 3.12.0 |

**环境**：
- cuSOLVER 12.4.1.2（PyPI wheel）在roxy上跑，5070 Ti，ctypes调用，没有编译；
- MAGMA 2.10.0 在节点s82上用Makefile编译（CUDA 12.0，参考BLAS/LAPACK，sm_70，不含Fortran，3分钟），只用GPU 0；
- LAPACK 3.12.0 是节点系统库。

**测试输入**：
- 缩放的Wilkinson增长矩阵 `s·W(n)`（`W_ii=1`、`W_ij=−1 (i>j)`、`W_in=1`）。部分选主元时U的最后一列增长到 `s·2^(n−1)`，可以超过FP16/FP32的上限；n≤40 时FP64精确。
- `x_true = 1`，`b = A·x_true` 精确。
- 另有 n=1024 的块对角 `diag(s·W(m), R)`、随机对角占优对照、b含NaN。

### 23.1 预测与实测

| 对象 | 契约预测 | 实测 |
|---|---|---|
| MAGMA `dsgesv_gpu` | 低精度分解溢出 ⇒ NaN被接受，ITER=0、INFO=0 | **证实**：W(30)·2^120 返回 iter=0、info=0，X全为NaN（`dgesv` 精确）。W(40)·2^100 时它解对了（GPU上的FP32分解在此例没有溢出） |
| LAPACK 3.12.0 `dsgesv` | 同上（判据相同） | **证实**：W(40)·2^100、W(30)·2^120 都返回 ITER=0、INFO=0，全为NaN；`dgesv` 精确 |
| MAGMA `dsgesv_iteref` / `dhgesv_iteref` | 同上 | **证实，且有预测之外的更坏情形**（见下） |
| cuSOLVER `DSgesv/DHgesv/IRSXgesv` | 判据是 `<`，NaN安全 | NaN方面**证实**：b含NaN时 niters=−5（显式失败）；增长矩阵在FP16/FP32路径上也解对。**但它不是安全的对照**：第22节的缩放测试表明，GMRES精化路径与一行式API并不执行文档判据，会以 iter≥0 返回错误解（本会话已在roxy上逐位复现 `repro_dbgesv.py`）。我第二轮的 diag(W(40)·2^100, R) 在 iter=1 被接受、误差1.1e-12，很可能也是这个缺陷 |

**MAGMA FGMRES-IR 的额外缺陷**：
- GMSTRS分支调用FGMRES后直接 `goto cleanup`，**没有外层残差复核，也没有回退FP64**；
- FGMRES在 `iters ≥ maxiter` 时停止，返回正的iter，不带失败标志。
- 结果：
  - NaN解被当作收敛：iter=40/30/500，info=0；
  - **W(40)·2^80 返回有限的错误解 x ≡ −0.1196（误差14.9），iter=1、info=0**。推测是低精度LU给出的预条件向量是垃圾，误触发了Arnoldi中断判据，未逐行验证；
  - b含NaN时，MAGMA的三个混合精度例程都报告成功（iter为0或256），cuSOLVER报 −5。

### 23.2 优先权与新颖性（如实）

- LAPACK层面的问题（`xSGESV/xSPOSV` 在非有限单精度解上以 ITER=0 返回）已经有**上游 PR #1501**（Reference-LAPACK，2026-11-22，作者 rmlarsen，PR注明由 Claude Code 准备）。它以“解上溢”为例，提出用 `DLANGE('M')` 检测并回退，至本次查看时仍未合并。上游 `dsgesv/dsposv/zcgesv/zcposv` 的判据依然是 `RNRM.GT.XNRM*CTE`。
- 我们是由契约**独立预测**出这一点的，并给出了：
  - 另一种触发方式（分解中的主元增长溢出）；
  - MAGMA `dsgesv_gpu` 也继承了它（PR未涉及）；
  - MAGMA FGMRES-IR 缺少外层复核、会返回**有限的错误解**（与PR无关的另一个问题）；
  - ~~cuSOLVER 是安全的对照~~（已更正：只在NaN/Inf上安全；它的GMRES路径会报告收敛却返回错误解，见第22节）。

### 23.3 边界

- MAGMA在V100（sm_70）上，cuSOLVER在5070 Ti（sm_120）上，两者硬件不同；
- 只测了 nrhs=1、方阵、`MagmaNoTrans`；
- FGMRES给出有限错误解的机制是推测，未逐行验证；
- cuSOLVER的FP16路径为何不溢出，没有查明（闭源）。

**下一步候选**：
- 用Lean证明“接受判据（`<` 形式）⇒ 后向误差界”，以及“`>` 形式在NaN时的反例”（与 `coarse_safe`/`coarse_unsafe` 同构）；
- 向MAGMA报告FGMRES-IR与 `dsgesv_gpu` 的问题，需要用户决定。

## 24. 白盒：MAGMA FGMRES-IR（GMSTRS）的代码级Lean模型（2026-11-23，`MagmaFgmres.lean`）

**方法（用户要求）**：
- 对开源代码逐行建模，在模型里推出“哪一步、在什么前提下会偏离文档承诺”，再由此构造输入去确认；
- 不是黑盒地比较输出。cuSOLVER闭源，只能做规约（第22节），所以白盒选MAGMA。

**对象**：MAGMA v2.10.0（tag 752e63c）：
- `src/dxgesv_gmres_gpu.cpp`：`magma_dxgesv_gmres_gpu`。公开接口 `magma_dsgesv_iteref_gpu`（FP32 LU）和 `magma_dhgesv_iteref_gpu`（FP16 张量核LU）都走 `Magma_REFINE_GMSTRS` 分支；
- `src/dgmres_plu_gpu.cpp`：`magma_dfgmres_plu_gpu`。

**从代码读出的事实**（行号即上述文件中的行号）：
- l.490–493：`cte = ‖A‖∞·eps·√n`，并令 `gmtol = cte`。
- l.570–600：`x := 0`，调用FGMRES，`maxiter = restrt = min(n, 500)`；之后**直接 `goto cleanup`**：不复核残差、不回退，`info` 保持0。
- FGMRES：
  - l.154：`if (fabs(ro) <= 1e-18) break`，是**绝对阈值**；
  - l.164：`eps1 = tol·ro₀ = √n·eps·‖A‖∞·‖b‖₂`；
  - l.272：`ro <= eps1` 用的是 Givens 递推的估计值，不是真残差；
  - l.297：`*niters = iters`，没有任何负值路径。

**Lean**（`lean/OptimiKahan/MagmaFgmres.lean`，364行，导入 `IrsContract`；日志 `check_log_magmafgmres.txt`，退出码0，无sorry）：

| 定理 | 对应代码 | 内容 |
|---|---|---|
| `single_cycle` | l.272/293，`restrt = maxiter` | 外层循环只执行一次 |
| `gmstrs_always_success` | l.565–600 | 低精度分解成功之后，**每一次运行**都返回 `info = 0`、`iter ≥ 0`：没有任何报告失败或回退的路径 |
| `nan_runs_to_maxiter` | l.154/164/272（NaN比较为假） | 残差估计为NaN（例如低精度LU溢出）⇒ 跑满 `iter = maxiter` 并报告成功 |
| `enforced_rule` | l.164/272 | 实际执行的停止条件是 `|rs| ≤ √n·eps·‖A‖∞·‖b‖₂`，而文档是 `‖b−Ax‖∞ < √n·eps·‖A‖∞·‖x‖∞`；在 (A, b) → s(A, b) 下，前者随 s² 缩放，后者随 s |
| `givens_nonincreasing` | l.250–261 | Givens 估计值不增（只要 `gam² ≥ Hii1²`，EPSMAC 分支也满足） |
| `tol_ge_one_one_step` | 上两条合起来 | `‖A‖∞·eps·√n ≥ 1` ⇒ 第1步后停止，`iter = 1`，与低精度因子的好坏无关 |
| `oneStep`、`oneStep_not_accept`、`one_step_violates`、`one_step_returned` | l.160–290 | 用代码自己的 Arnoldi/Givens 公式（精确算术，平方根相消）构造 2×2 实例：A = 2^60·I，b = 2^60·(3,4)，FP16 级别（2^-11）的预条件误差。返回 `(iter, info) = (1, 0)`，x₁ = (2185216, 2915044)/728633，残差 ≈ 9.4e-4，违反文档条件 |
| `tiny_rhs_returns_zero`、`tiny_violates` | l.154、l.570 | `0 < ‖b‖₂ ≤ 1e-18` ⇒ 返回 x = 0、iter = 0；实例 A = [[2^-70]] 的文档后置条件不成立 |

**模型边界**：
- 比较用精确有理数，FP64 对 eps1 和估计值本身的舍入未建模；
- NaN 已建模，Inf 未建模；
- 向量核（gemv、dot、getrs）作为抽象输入，只在2×2实例里具体计算。

**数值核对：忠实移植**（`experiments/magma_fgmres/fgmres_port.py`、`fgmres_port_results.txt`）：
- 设置：n=256；常数与停止判据逐字照搬；FP32 LU 用 numpy 实现；A 的元素为 ±[0.5,1)，按 2^k 缩放。
- 结果：
  - 2^-80（‖b‖₂ = 9e-23）：iter = 0，x = 0，误差 1；
  - 2^-60 到 2^-20：iter = 256 = maxiter，“成功”，解本身是好的；
  - 2^0：iter = 3，后向误差/阈值 = 0.09；
  - 2^10、2^20：iter = 2，4.7e3 倍；
  - 2^30 以上：iter = 1，后向误差 1.1e8 倍，误差 9e-5；
  - 相对残差始终 ≤ tol，与 `enforced_rule` 一致。
- **与已有观测吻合（事后）**：第23节中 W(40)·2^80 返回有限错误解、iter = 1，该节当时推测是 Arnoldi 中断。按本模型，‖A‖∞·eps·√n ≈ 40·2^80·2^-53·6.3 ≈ 3e10 ≥ 1，所以必然一步停止。NaN 解报告 iter = 40/30/500，正是 maxiter = min(n, 500)。

**GPU确认**（eightsocket，V100，只用GPU 0，2026-11-23；`experiments/magma_fgmres/magma_fgmres_gpu.cpp`、`magma_fgmres_gpu_results.txt`）：

**环境**：
- MAGMA 2.10.0 官方 tarball，其中 `dxgesv_gmres_gpu.cpp`、`dgmres_plu_gpu.cpp` 与建模所用的 GitHub v2.10.0 只差版本注释；
- CUDA 12.0、参考 BLAS/LAPACK、`GPU_TARGET = Volta`，`make lib` 用时2.5分钟（只用静态库）；
- 用完已删除 `/ssd/haibin/magma_fgmres`（930 MB）。
- 注：此前说“s82中继不可用”是我的误判。`ssh s82` 本来就落在 login2，计算节点需要再跳一步：`ssh -i <KEY_PATH> haibin@<NODE_IP>`（见第1节）。

**测试**：
- 设置：n=1024，A ∝ 2^k，A0 的元素为 ±U[0.5,1)；k ∈ {−80, −70, −60, −40, −20, 0, 10, 20, 30, 40, 50, 60}；两个公开例程各测一遍。
- 后向误差在主机端用 long double 计算：FP64 残差太粗，不足以检验这个阈值（见 `fp64_cannot_certify`）。

| 预测（定理） | dsgesv_iteref（FP32 LU） | dhgesv_iteref（FP16 张量核LU） |
|---|---|---|
| ‖b‖₂ ≤ 1e-18 ⇒ x = 0、iter = 0（`tiny_rhs_returns_zero`） | k = −80、−70：iter = 0，x = 0，误差 1 ✓ | 同左 ✓ |
| 达到 maxiter = 500 时也报告成功（`gmstrs_always_success`） | k = −60…−20：iter = 500，info = 0，解是好的（误差4e-13） | k = −60、−40：iter = 500，info = 0，**误差 3.7e-3**，后向误差 3.8e8 倍 |
| 阈值随尺度变化（`enforced_rule`） | k = 0、10–20、30：iter = 3、2、1；后向误差/阈值 = 2.8、2.2e4、3.1e8；相对残差始终 ≤ tol ✓ | k = 0、10：iter = 9、7；21 倍、1.7e4 倍 ✓ |
| tol ≥ 1 ⇒ iter = 1（`tol_ge_one_one_step`） | k = 40、50、60：iter = 1，误差 1.9e-4，info = 0 ✓（k = 30 时 tol = 0.003 也已一步停止） | 见下一行 |
| NaN ⇒ iter = maxiter 并报告成功（`nan_runs_to_maxiter`） | — | k = 20…60：A 的元素超过 FP16 上限 65504，FP16 LU 溢出成 NaN，代码未检查；**iter = 500，info = 0，x 全为NaN** |

**探针判定**：21/24 MATCH。

3个 MISMATCH 都是 dhgesv 在 k ≥ 40 的情形：
- 探针的预测规则先判断 tol ≥ 1，再判断 NaN，顺序错了；
- `tol_ge_one_one_step` 的前提是“第一个估计值有限”，这里不成立，适用的是 `nan_runs_to_maxiter`，它预测的 iter = 500 与实测一致；
- 所以这3个是判定规则的问题，定理没有被违反。这是看到数据之后的解释，如实记录。
- 另：探针在 x 为 NaN 时把 fwd_err 打印成 0，这是打印的缺陷，应以 `finite=0` 为准。

**结论**：模型从代码推出的偏差在GPU上全部出现：
- 小右端返回零解；
- 阈值与尺度有关，一步停止；
- 达到 maxiter 时报告成功，且解可能是错的；
- NaN 报告为成功。

最严重的一条：**A、b 有限，FP64 可解，dhgesv_iteref 却返回全 NaN 的 x，同时报告 info = 0、iter = 500**。
- A 的元素超出了 FP16 的范围，FP16 分解本来就会失败；
- 问题在于失败没有被检测出来：文档承诺失败时回退FP64，而且 `iter = -2` 正是为“降精度溢出”准备的，但代码只检查了 FP32 的范围。

**bug报告草稿**：`operators/MAGMA_REPORT_fgmres_ir.md`（英文，未提交）。

**轨迹验证：模型与代码是否一致**（2026-11-23；借鉴 Specular 的 trace validation；`experiments/magma_fgmres/trace/`）

**做法**：
- 给 MAGMA 的两个文件插入13个只读的打印点（`trace.patch`，仅在设置 `MAGMA_TRACE` 时输出，用 `%a` 十六进制精确打印）：
  - `cte`/`gmtol`；
  - x₀ = 0；
  - 每次外层开始时的 `ro`；
  - 提前 break；
  - `eps1`；
  - 每一步的 `Hii`、`Hii1`、`gam`、`s`、`ro`；
  - 内层退出、外层测试；
  - 返回的 `niters`、`iter`/`info`；
  - 是否回退。
- 重新编译后在 GPU 0 上跑原探针，n = 512、1024、2048，共72次调用。插桩前后 n=1024 的输出逐位相同。
- `validate.py` 把每次调用转成一行 `#eval checkCall …`（`TraceReplay.lean`）。它用**模型本身的 `fgmres`/`gmstrs` 定义**，在轨迹记录的初始残差范数和每步 Givens 估计（精确有理数）上运行，再与库实际返回的 (niters, 是否提前退出, iter, info) 比较。niters 相同，意味着每一步的退出决定都与代码一致。

**结果**：
- **回放：72/72 OK**（`replay_results.txt`）。
- **模型前提**：在全部72次调用上逐条检查，全部成立（`validate_report.txt`）：
  - A1：restrt = maxiter = min(n, 500)；
  - A2：x₀ = 0；
  - A3：tol = cte = fl(fl(‖A‖·eps)·√n)，逐位相同；
  - A4：代码比较用的是舍入后的 eps1，模型用精确的 tol·ro₀，任何一步的判定都没有因此不同；
  - A5：外层只跑一次（`single_cycle`）；
  - A6：Givens 估计不增，实测最大比值恰为1，|s| 最大恰为1（`givens_nonincreasing`）；
  - A7：没有回退，iter = niters（`gmstrs_always_success`）。
- **对照（检验这个验证本身有没有检测力）**：三个故意改错的模型变体都被抓出（`TraceMutants.lean`、`mutant_results.txt`）：
  - 去掉 1e-18 提前退出：12处不一致，恰好是 12 次小右端调用；
  - 改用绝对容差 `eps1 = tol`：45处；
  - 迭代上限差一：29处。

**意义与边界**：
- 这回答了“手工翻译的模型是否忠实于代码”：在这些运行上，模型的控制流与停止判据和代码逐步一致，前提也都被轨迹满足。
- 验证的范围是标量控制逻辑。向量核（gemv、dot、getrs）在模型中是抽象输入，用的是轨迹记录的值，没有被验证；
- 72 次运行是覆盖，不是证明；
- 节点上的 `/ssd/haibin/magma_trace`（931 MB）已删除。

**修正（代码层面，建议）**：
- l.493 的 `gmtol` 不应乘 ‖A‖：应为 `tol = √n·eps`，并以 `‖A‖·‖x‖` 而不是 `ro₀` 作为尺度；
- l.154 的绝对阈值应改为相对阈值；
- l.600 应在返回前用文档判据复核真残差，不通过则 `goto fallback`；
- FGMRES 达到 maxiter 时应返回负值。

### 24.x 修复与证明（2026-11-24，`MagmaFixed.lean`，`experiments/magma_fgmres/fix/`）

**设计**：
- 只验结果，不信迭代。FGMRES 当作黑盒，返回前用可证明的后检决定"接受"还是"回退"；
- 后检：Dot2 计算残差；按 `wrapper_sound` 的放大形式写成 `L < R`，两边各自向外取整 (1 ± 16u)；x 中有非有限值、或出现 NaN，都判为失败；失败则 `goto fallback`，`iter = -(niters+1)`。

**其余改动**（只影响"多常通过"，与正确性无关）：
- `gmtol = eps·√n`，去掉 ‖A‖；
- `ro = dnrm2`，只在 `ro == 0` 时 break；
- 半精度 LU 之前检查 FP16 范围；
- 逐列调用 FGMRES；`trans ≠ NoTrans` 时直接走 FP64。

**Lean**（`lean/OptimiKahan/MagmaFixed.lean`，311行，导入 `MagmaFgmres`；日志 `check_log_magmafixed.txt`，退出码0，无sorry）：

| 定理 | 内容 |
|---|---|
| `fixed_post` | 对**任意** FGMRES 输出（任意 niters、任意向量，含 NaN），只要计算值满足 `Faithful`，返回值就满足文档后置条件 |
| `fixed_nan_fallback` | x、L 或 R 中有 NaN ⇒ iter < 0（对照 `nan_runs_to_maxiter`） |
| `fixed_rejects`、`fixed_rejects_tiny`、`fixed_rejects_one_step` | 违反文档判据的 x 不会以 iter ≥ 0 返回；原代码的两个反例都被拒绝 |
| `fixed_accepts_example`、`example_faithful`、`example_post` | 接受分支可达，且 `Faithful` 可满足（n = 1024 的常数），定理不是空真 |
| `fgmresFixed_scale` | 改后的 FGMRES 退出决定在 (A, b) → s(A, b) 下不变；原代码的阈值随 s² 变化 |

`Faithful` 是假设，没有在 Lean 里证明，包括：
- Dot2 的已发表误差界；
- FP64 行和的误差界；
- 构成 L、R 的几次运算的舍入在 (1 ± 16u) 之内。

另外不考虑下溢。

**补丁**：`fix/gmstrs_fix.patch`，163行，对 v2.10.0 dry-run 可干净应用，**尚未针对 MAGMA 编译，也未上 GPU**（用户要求先空出 GPU）。
- `certify_cpu.cpp` 是从补丁中原样摘出的后检函数，加主机端桩，已在 Mac 上编译。

**CPU核对**（`fix/port_fixed.py`、`port_fixed_results.txt`；忠实移植 + 上述编译好的 C++ 后检；文档判据用 Fraction 精确判定；n = 256，k ∈ {−80…60}）：
- 原代码：20 次中有 17 次报告成功却违反判据；修复后：0 次。
- FP32 LU：各个尺度都是 iter = 3，比值都是 0.0815，尺度不变。
- FP16 LU（用普通 FP16 LU 代替 MAGMA 的 XSH）：
  - k ≥ 20：范围检查 ⇒ 回退；
  - k ≤ −20：FP16 下溢 ⇒ 后检失败 ⇒ 回退；
  - k = 0、10：13 步后通过认证。

**读代码新发现（未经 GPU 确认）**：
- `magma_dfgmres_plu_gpu` 完全不使用 `trans`、`nrhs`、`lddb`、`lddx`：所有 gemv、dsgetrs 都是 `MagmaNoTrans`，只处理单列。预测：
  - `trans = MagmaTrans` 时，返回的是 A x = b 的解；
  - `nrhs > 1` 时，第 2 列起都是 0；
  - 两种情况都报告成功。
- IR 分支 l.712–716：`*iter = -5; goto cleanup; goto fallback;`，回退不可达。
- `dgmres_plu_gpu.cpp` 头注释写着 "for testing purpose"。

**MAGMA 报告**：已把 "Suggested fixes" 换成 "Proposed patch"，并新增 "Other issues visible in the code"。

**待做（等 GPU 空出）**：
- 在 V100 上编译补丁，重跑 `magma_fgmres_gpu.cpp`（原 24 次调用）；
- 加一个 trans/nrhs 探针；
- 计时，看后检的开销；
- 数据仍放 /ssd/haibin，用完删除。

## 25. 研究重心收窄 + 迭代精化接受逻辑的跨库普查（2026-11-24）

**重心（用户决定）**：相关工作核对（`operators/RELATED_WORK_zh.md`）之后，主线定为**混合精度迭代精化的接受逻辑缺陷**。原因：
- 这一方向没有公开先例；
- 相比之下，ADP的零元素问题已有 arXiv 2611.19008，Ozaki精度与k的关系已有 2602.02549，LAPACK把NaN当成功已有 PR #1501 系列。

Ozaki/ADP/ozIMMU/GEMMul8 的结果改为辅助材料。

**普查**（`operators/IR_SURVEY_zh.md`）：
- 只读源码、没有运行，覆盖约25个实现：MAGMA、SLATE、PLASMA、LAPACK、SuperLU(_DIST)、MUMPS、STRUMPACK、PaStiX、hpl-ai、rocHPL-MxP、Ginkgo、PETSc、Trilinos、hipSOLVER、cuDSS/oneMKL（只看文档）。
- 在原有的 D1–D9 之外，新增两类：D10（封装层丢状态）、D11（最大值归约吞掉NaN）。
- 主会话逐行复核过的：MAGMA `zchesv` 的空壳回退、MAGMA `dshposv_gmres` 的 `goto cleanup` 与 EPSILON中断、SLATE LAPACK接口（X与b共用内存、`info=0` 写死）、PLASMA任务失败后跳过回退、STRUMPACK mixed总是返回SUCCESS、PaStiX GMRES的 `normx` 只算一次、hipSOLVER不写 `niters`、hpl-ai的 `/‖b‖` 判据。
- **以上全部还没有运行确认**。

**下一步**：其中大部分能在CPU上确认，不需要GPU（优先级见普查文档末尾）。在哪里运行由用户决定。

### 25.1 运行确认（2026-11-25，节点CPU，用户批准在 /ssd/haibin/lean4hpc-ir 建conda环境，用完已删除）

在 eightsocket 的CPU上（不用GPU）运行了普查的预测，**7项全部证实**：SuperLU、SuperLU_DIST `psgssvx_d2`、MUMPS 5.8.2、PaStiX GMRES（x0=0 → x≈249·x_true 且报SUCCESS）、PLASMA `dsgesv`（NaN被接受；失败时跳过FP64回退）、STRUMPACK（小尺度 → x≡0 且报SUCCESS，默认的双精度求解器也一样）、SLATE LAPACK接口（良态矩阵也丢失解，并报info=0）。

- 详细结果、命令和运行中做的更正：`experiments/ir_survey/README.md`；
- 汇总：`operators/IR_SURVEY_zh.md` 中“运行确认”一节；
- 还没有运行：MAGMA `zchesv`、`dshposv_gmres`（需要GPU）；hpl-ai、Ginkgo、PETSc。
- 对外报告由用户自己提交。

## 26. 形式化：D11定理与迭代精化接受契约的充要条件（2026-11-25）

两个新文件，都只用核心Lean、没有 `sorry`，公理只有 propext / Classical.choice / Quot.sound。

### 26.1 `lean/OptimiKahan/MaxFold.lean`（D11，日志 `check_log_maxfold.txt`）

把三种求最大值的写法建模，数值是精确有理数加NaN，与NaN比较一律为假，不建模无穷：
- SuperLU 的 `SUPERLU_MAX(s,v)=(s>v)?s:v`；
- PLASMA / IDAMAX 的 `if (v>s) s=v`；
- 会传播NaN的写法。

| 定理 | 内容 |
|---|---|
| `smax_fold_isNaN` / `smax_fold_nan_iff_last` | SuperLU写法：结果是NaN **当且仅当最后一个元素是NaN** |
| `smax_fold_after_nan` / `smax_fold_value` | 最后一个NaN之前的元素全部被丢掉；它后面若都是有限值，结果恰好等于这些值的最大值 |
| `masked_equals_baseline` | SuperLU_DIST 实验的形状（第一块全NaN，第二块与干净值相同）返回的值**与干净基线完全相等**，与实测一致（berr 都是2.96e-16） |
| `pstep_fold_isNaN` | PLASMA/IDAMAX写法：结果是NaN 当且仅当**第一个**元素是NaN |
| `nmax_fold_isNaN` | 传播写法：结果是NaN 当且仅当某个元素是NaN |
| `detects_iff` | 2种判据写法 × 3种归约中，“任何含NaN的输入都被拒绝”成立，**当且仅当**判据写成 `<` 形式且归约会传播NaN；其余5种组合各有一个两元素反例 |

### 26.2 `lean/OptimiKahan/IrDriver.lean`（导入 MaxFold，日志 `check_log_irdriver.txt`）

**抽象驱动**：
- 内层求解器是对抗性的，可以返回任何值；
- 外层检查 `check`；
- 回退能否到达 `fbReach`；回退写出什么、自己报不报成功 `fb`；
- 封装层是否强制报成功 `force`。

**契约**：
- Sound：报成功 ⇒ 调用者读到的答案是好的；
- Complete：FP64可解的问题 ⇒ 一定报成功。

**主定理 `contract_iff`**：在建模假设下，Sound ∧ Complete **当且仅当** C1–C4 同时成立。建模假设是：内层可能返回坏值；好答案只在可解问题上存在；可解和不可解的问题都会出现。

| 条件 | 含义 |
|---|---|
| C1 | 外层检查可靠 |
| C2 | 回退能到达 |
| C3 | 回退正确：报成功就给好答案，可解时一定成功 |
| C4 | 状态没有被强制改成成功 |

- 充分性 `contract_of_conditions` 不需要任何假设。

**推论**：每种实测到的失败，都对应违反了一条条件。

| 推论 | 违反 | 对应的库 |
|---|---|---|
| `trust_inner_unsound` | C1（没有外层检查） | MAGMA GMSTRS、`dshposv_gmres`、PaStiX、STRUMPACK mixed |
| `forced_unsound` | C4 | SLATE LAPACK接口、STRUMPACK、PaStiX |
| `stub_fallback_fails` | C3 | MAGMA `zchesv` |
| `skipped_fallback_incomplete` | C2 | PLASMA |
| `c1_forces_accept_propagate` | 与26.1衔接 | 若检查是“判据(归约(逐行值))”，且每种NaN分布都能由某个坏输入产生，则 C1 迫使判据为 `<` 形式、归约会传播NaN；D1和D11因此都是违反C1的特例 |

- **非空性**：`tOk_contract` 给出一个所有假设都成立、满足契约的玩具实例；它的“强制成功”版本和“去掉检查”版本分别被证明不可靠（`tForced_unsound`、`tTrust_unsound`）。
- **C1在数值上怎样才能做到**：见已有的 `IrsContract.wrapper_post`、`MagmaFixed.fixed_post`（Dot2残差加外向舍入的 `<` 检查，对任意精化输出都可靠），以及 `IrsContract.fp64_cannot_certify`（普通FP64残差无法认证文档中的 √n·eps 阈值）。

### 26.3 边界（如实）

- `contract_iff` 的层次是调用者能观察到的行为，证明本身是简单的逻辑。它的价值在于给出完整的条件清单，并把各库映射到具体条件上；数值上的难点在C1，由26.1和 IrsContract/MagmaFixed 承担。
- 各库被映射到哪条条件，依据的是读码和第25节的运行结果，并不是从C代码机器生成的模型。
- 没有建模无穷大，也没有建模MUMPS所依赖的具体BLAS中 `idamax` 的NaN行为。

## 27. 论文提纲 + 各库自带测试核对（2026-11-25）

- **论文提纲**：`PAPER_OUTLINE_zh.md`，路线C（方法 + 实证 + 带证明的接受门）。参照了 TensorLean（bug表、头条数字）和 arXiv 2611.04447（非空性、建模决策、可信基、AI使用声明）。每条论断标了 [有]/[缺]/[可选]，末尾有投稿前待办和会议方向。
- **注意**：普查与运行结果在同一次提交 06a3922 里进入 git，git 历史证明不了“先预测后运行”。以后的预测应先单独提交。
- **自带测试核对**：`operators/IR_SURVEY_zh.md`“各库自带测试能否发现这些缺陷”一节。只读码，2个子代理加主会话复核。
  - 8个库、16个缺陷，0个被默认测试或CI触发；
  - 6个例程没有带判定的测试；
  - SuperLU 的检查函数自身用 `SUPERLU_MAX`，判定写成 `>= THRESH`；STRUMPACK 用 `std::max` 和 `>`；MUMPS示例只查 `INFOG(1)<0`。所以加上NaN输入也会判通过。
  - 边界：没有运行；PaStiX `spmCheckAxb` 未读到（`spm/` 子模块为空）。
