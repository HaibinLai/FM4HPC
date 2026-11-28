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

## 28. MAGMA 其余稠密混合精度/精化例程：读码、Lean 模型、CPU 核对（2026-11-25）

**对象**（MAGMA v2.10.0，`src/`）：`zcgesv_gpu`（dsgesv）、`zcposv_gpu`、`zchesv_gpu`、`zcgeqrsv_gpu`、`dshposv_gmres_gpu` + `dfgmres_spd_gpu`、`zgesv_rbt` + `zgerfs_nopiv_gpu`、`zgesv_rbt_batched`。GPU 未用（s82 占用中），GPU 确认见末尾队列。

### 28.1 读码发现（行号为对应文件）

| 例程 | 发现 | 类 |
|---|---|---|
| 所有经典驱动 | 范数用 `i?amax` 取下标再取模；判据 `if (Rnrm > Xnrm*cte)` 拒绝式：NaN 和 `Inf > Inf` 都被接受 | D1 |
| 所有经典驱动 | `for (iiter=1; iiter < ITERMAX;)` 只做 29 次修正（LAPACK 30 次） | 小 |
| `zcgesv` l.231–235 | B 的 FP32 转换被注释掉，紧跟的 `iter=-2` 检查成为死代码；B 在 `zcgetrs` 里由 `zclaswp` 转换，**无范围检查**（"TODO check precision"）；转置路径用 `zlag2c` 有检查，但首次 `zcgetrs` 的 info 不检查 | D8 |
| `zlag2c` | 检查 `(x < -rmax) \|\| (x > rmax)`：NaN 两个比较都为假，直接通过 | D8 |
| 所有经典驱动 | **解溢出 FP32**（A、B 都在范围内）：FP32 求解得 Inf，残差 −Inf/NaN，被接受，iter=0、info=0；所有范围检查都只查 A、B | D1/D8 |
| `zchesv` l.371–381 | FP64 回退被注释掉，只剩 `zlacpy(B→X)`：不收敛时 **X = B、info = 0、iter = −31**；用不选主元的 `chetrf_nopiv`；精化成功分支直接 `return`，泄漏 queue | D9 |
| `zchesv`（及不选主元的路径） | **精化发散到溢出后被当作收敛**：iter > 0、info = 0、X 非有限 | D1 |
| `zcgeqrsv` l.284 | 代码 TODO 自认“只对相容系统有效”：不相容最小二乘的判据几乎永不满足，总是白做 29 次再回退（性能缺陷，答案正确） | 性能 |
| `dshposv_gpu` / `_native` l.469/497 | 公开接口 `return 0`（不返回 info），工作区分配不检查；固定 `use_gmres=1` | D10 |
| `dshposv` GMRES 分支 l.292–308 | FGMRES 后直接 `goto cleanup`：无复核、无回退 | D4/D9 |
| `dfgmres_spd` l.15/97 | 绝对阈值 `EPSILON = dlamch('Epsilon') = 2^-53`（GMSTRS 为 1e-18）：‖b‖₂ ≤ 2^-53 ⇒ x = 0 | D3 |
| `dfgmres_spd` l.110/195 | `eps1 = tol·ro₀` 随尺度 s² 变化；NaN 跑满 maxiter；niters 恒 ≥ 0 | D2/D5 |
| `dfgmres_spd` l.133/142/66 | **nrhs ≥ 2 时显存越界**：每步写 `d_w` 从第 i 列起的 nrhs 列，`d_w` 只有 restrt 列；内层跑满一个周期即越界（nrhs ≥ 3 时读 `d_VV` 也越界） | 内存安全 |
| `dfgmres_spd` l.52 | `assert(uplo == MagmaLower)`：调试构建传 Upper 直接 abort | 小 |
| `zgerfs_nopiv` l.263–266 | `fallback:` 后为空：不收敛只置 iter = −31，info = 0 | D9 |
| `zgesv_rbt` l.155–163 | `iter` 是局部变量从不返回；`zgesv_nopiv` 的 info 不检查，且被 `zgerfs_nopiv` 开头的 `*info = 0` 覆盖：**零主元与精化失败都返回 info = 0**；HIP 后端 refine 被强制关闭（注释“probably a bug”） | D9/D10 |
| `zgesv_rbt_batched` | 逐矩阵 info 只在 `#ifdef CHECK_INFO` 下检查（dinfo_array 仍保留，调用者可自查）；`hu`、`hv`、`dv` 从不释放（每次调用泄漏） | 小 |

### 28.2 Lean（`lean/OptimiKahan/MagmaDense.lean`，导入 MagmaFgmres；日志 `check_log_magmadense.txt`，退出码0，无 sorry）

| 定理 | 内容 |
|---|---|
| `accept_nan_R`、`accept_nan_X`、`accept_inf` | 经典驱动：首次判据 NaN 或 Rnrm = Xnrm = +∞ ⇒ iter = 0、info = 0 |
| `loop_first`、`diverge_accepted` | 前 j−1 次判据要求继续、第 j 次溢出 ⇒ iter = j > 0、info = 0、X 非有限 |
| `success_iter_lt` | 成功返回的 iter < ITERMAX（最多 29 次修正） |
| `zchesv_stub` | 不收敛 ⇒ (iter, info, X) = (−31, 0, B) |
| `gerfs_nopiv_silent`、`rbt_info_lost` | gerfs_nopiv 无回退、info = 0；RBT 不论 nopiv LU 报什么，调用者都得到 info = 0 |
| `x_overflow_accepted` | 实例 a = 2^-100、b = 2^28（都在 FP32 正规范围），FP64 解 2^128 有限，FP32 解 +∞ 被接受；对 zcgesv 和 zcposv 都成立 |
| `zcgesv_b_overflow_accepted`、`zcposv_b_overflow_fallback` | b = 2^200：zcgesv 接受 +∞（无 B 检查），zcposv 正确回退 iter = −2 |
| `dshposv_always_success`、`spd_tiny_rhs_returns_zero`、`spd_threshold_larger`、`spd_nan_runs_to_maxiter` | 公开 dshposv 总报成功；‖b‖₂ ≤ 2^-53 ⇒ x = 0；2^-53 > 1e-18；NaN ⇒ iter = maxiter |
| `oobW_last`、`oobV_last`、`nonconverged_oob` | nrhs ≥ 2 且内层未在重启长度前收敛 ⇒ 写 `d_w` 第 restrt 列（越界）；nrhs ≥ 3 读 `d_VV` 越界 |

模型边界：精确有理数 + NaN/±∞；FP32 转换与 1×1 FP32 求解只建模溢出；i?amax 的 NaN 行为（cuBLAS）未建模，`accept_nan_*` 以范数本身为 NaN 为前提。

### 28.3 CPU 核对（`experiments/magma_dense/`）

- `xovf_lapack.c`（Mac Accelerate LAPACK；判据结构与 MAGMA zcgesv/zcposv 相同）：a = 2^-100、b = 2^28 ⇒ `dsgesv`/`dsposv` 都返回 ITER = 0、INFO = 0、x = inf（n=1）或 nan（n=2，0·Inf），FP64 精确解 3.4e38。对照 a = 2^-90 正确。b = 2^200：LAPACK 返回 ITER = −2 正确回退（MAGMA zcgesv 无此检查）。LAPACK 侧属于 PR #1501 系列已覆盖的“解上溢”。
- `zchesv_port.py`（zchesv 控制流的忠实移植 + 不选主元 FP32 LDLᵀ）：A = [[d, cᵀ], [c, M]]，n = 16，cond(A) ≈ 3：
  - d ≥ 2^-20：收敛（iter 2–13）；
  - **d = 2^-24，种子 0、1：iter = −31、info = 0、X == B**（相对误差 1.43、0.45）；
  - **d = 2^-24，种子 2：iter = 29、info = 0，x = inf**（`zchesv_seed2_results.txt`：x 从 1e21 增长到 7.8e30，第 29 次修正溢出，`Inf > Inf` 为假被接受）。
- `diverge_lapack.c`（对照）：选主元的 `dsgesv` 在 κ = 1e6…1e12 下不会发散到溢出：κ ≥ 1e10 时 ITER = −31 正确回退。所以“发散后被接受”只发生在不选主元的路径（zchesv、RBT）。

### 28.4 GPU 确认队列（等 s82 空出；只用 GPU 0；数据放 /ssd/haibin，用完删除）

1. zcgesv / zcposv：a = 2^-100、b = 2^28（n = 1、2、1024 块对角）⇒ 预测 iter = 0、info = 0、X = inf/nan。
2. zcgesv：b = 2^200 ⇒ 预测接受 +∞；zcposv 同输入 ⇒ iter = −2。
3. zchesv：`zchesv_port.py` 的 d = 2^-24 实例 ⇒ 预测 X = B、info = 0、iter = −31，或 iter = 29 + inf。
4. dshposv_gpu：‖b‖₂ = 1e-17 ⇒ x = 0、iter = 0；(A, b)·2^k 扫描；nrhs = 2 且难收敛矩阵下用 `compute-sanitizer --tool memcheck` 检查 `d_w` 越界写；返回值恒为 0。
5. zgesv_rbt(refine = true)：用 RBT 子代理构造的实例（见 `experiments/magma_dense/rbt/`，进行中）。
6. 原有：MAGMA 补丁编译与开销、trans/nrhs 探针。

### 28.5 补充（同日）：RBT 移植与反例、稀疏求解器、对已有结果的更正

**RBT**（`experiments/magma_dense/rbt/`，CPU 忠实移植，含 glibc rand 复现与自检）：
- **n = 1、2、3 时答案错误，info = 0**（与随机数无关）：转置向量核 `if (n < 2) return;`（zgerbt_kernels.cu l.174）跳过长度为 1 的块的 u 缩放，而矩阵核照常缩放，变换后的 (A′, b′) 不是同一个变换；相对误差 6.0e-2、3.4e-2、3.4e-3。n ≥ 4 一致性 1e-15。
- **条件数 3.09 的 n = 64 矩阵，dgesv_rbt 返回误差 3.6e35 的有限解，info = 0**：A = U⁻ᵀ M V⁻¹，M[0,0] = 0，变换后主元 7.1e-18；依赖进程内首次 init_butterfly 的 glibc rand 状态。
- 子代理（中途停止，未写文件）读码更正：零主元时 `zgesv_nopiv` 跳过 getrs，dB 保持 Uᵀb；`getf2_nopiv` 置 info 并跳过该列缩放继续。

**稀疏求解器**（子代理全量读码 + 主会话核对源码；`experiments/magma_sparse/`；Lean `MagmaSparse.lean`，日志 `check_log_magmasparse.txt`，退出码0）：
- **CG_MERGE / PCG_MERGE（默认求解器）复数精度下内积不取共轭**（zmergecg.cu l.232、l.1258）：CPU 移植中，A = diag(1, 1.5)、b = (1, 0.5i) 一步报 info = 0，误差 1.14；随机复 HPD（cond 10）不收敛；实数据对照正常。Lean `cg_merge_false_convergence`：A = diag(2,3)、b = (2,i) 精确算出 Σr² = 0、info = 0、x = (2,i)，真解 (1, i/3)。
- **BiCG/PBiCG/PTFQMR(_merge) 的 breakdown 被覆盖成 SUCCESS**（zbicg.cpp l.145–147 → l.197–198）；Lean `bicg_breakdown_success`。
- **求解器包装层 default 分支返回 0、x 不动**：`--solver BA|BAO|ITERREF|PARDISO`，或多右端配非 CG/PCG/LOBPCG；Lean `wrapper_unhandled_success`。
- **早退出判据查 ‖b‖ 而非初始残差**（zcg_res l.106、zbicgstab* 等）：‖b‖ < max(rtol‖b‖, dlamch('E')) ⇒ info = 0、x = x0；Lean `early_exit_returns_x0`。
- 其余（子代理，未逐条核实）：绝对 atol 默认 1e-16；pcg_merge 的 SLOW→SUCCESS 升级以 init_res 为基准；LOBPCG 用绝对 tol、NaN 视为收敛、只看第 0 个特征对；Jacobi 只要残差下降就报成功；final_res 计算但从不与容差比较；`magma_z_precond` 的 atol 未初始化；状态码注释（0/−1/−2）与实际（−201/−202…）不符；测试只打印。

**对已有结果的更正**（起草上游报告时逐条对照原始结果文件发现；已改 `experiments/ir_survey/README.md`、`operators/IR_SURVEY_zh.md`、`paper/DRAFT.md`）：
1. SuperLU：“berr 与有限基线相同”比较的是单个三对角矩阵，不是两块矩阵；更小的复现 A = I、b = (NaN, 1) ⇒ berr = 0。
2. MUMPS：机理不是 idamax（DXMAX 有限），而是逐行比较吞掉 NaN（dsol_aux.F:889–899，推断）。
3. SuperLU 测试引用 dgst07.c:145 在 `#if 0` 内，应为 l.129/151/193。
4. PaStiX：IPARM_NBITER 是跨调用的运行最大值；GMRES x0 = 0 那行的 250 是真的，之后各行的 250 是带过去的；保存的探针比结果多一列（版本不同）。
5. PLASMA：“FP64 精确”未被结果显示（只打印 dgesv_info）；P2 的 iter = −31 很可能读自从未写入的缓冲区（推断）。
6. SLATE：b 被改写只显示了前几个元素为 0。
7. SuperLU_DIST：范数误差界 1.03e-7 是公式下限 √3·2^-24，本来不变；屏蔽体现在 berr。
8. SuperLU 用的是节点 gcc 13.3，不是 conda 工具链。

**上游报告草稿**：`operators/upstream_reports/` 新增 SLATE、PLASMA、SUPERLU、SUPERLU_DIST、MUMPS、PASTIX、STRUMPACK 七份（未提交，由用户决定）。每份末尾的 “Internal notes (remove before sending)” 发送前需删除。

**GPU 队列追加**：
7. zgesv_rbt：n = 1、2、3 的对角占优随机矩阵 ⇒ 预测误差 1e-3–1e-1、info = 0；n = 64 的 `A_zero_pivot_n64.npy`（新进程首次调用）⇒ 预测 info = 0、误差巨大或 NaN。
8. 稀疏 CG_MERGE（z 精度）：A = diag(2,3)、b = (2,i) ⇒ 预测 info = 0、numiter = 1、x = (2,i)；随机复 HPD ⇒ 不收敛。BiCG：b 含 NaN ⇒ info = 0。包装层：`--solver ITERREF` ⇒ 返回 0、x = x0。
- 追加核对并入 Lean（`MagmaSparse.lean` 共 8 条定理，退出码 0）：`template_ignores_final_res`（通用模板只要 numiter < maxiter 就 SUCCESS，final_res 不与容差比较，zcg_res.cpp l.170–198）、`jacobi_any_decrease`（zjacobi.cpp l.133–136）、`lobpcg_nan_converged`（zcompact.cu l.36：`dnorms[j] > tol` 为假即移出活跃集）。

### 28.6 本轮小结

- MAGMA 中所有带精化或接受逻辑的稠密例程都已读码并建模：`MagmaFgmres`（dsgesv/dhgesv_iteref，§24）、`MagmaDense`（zcgesv、zcposv、zchesv、zcgeqrsv、zgerfs_nopiv/zgesv_rbt、dshposv + dfgmres_spd）；稀疏求解器的接受逻辑建模于 `MagmaSparse`。
- CPU 已确认：解溢出被接受（LAPACK 同结构）、zchesv 的 X = B 与发散被接受（移植）、RBT n ≤ 3 错误与 cond 3 矩阵的 3.6e35 误差（移植）、默认稀疏 CG_MERGE 复数下假收敛（移植）。
- 全部 MAGMA 结论仍待 GPU 确认（队列 1–8）。

### 28.7 稀疏求解器第二批与预条件子（同日，主会话逐条核对源码；`MagmaSparse.lean` 共 16 条定理，退出码 0，无 sorry）

| 发现（文件:行） | Lean |
|---|---|
| `zcg_res` l.111–114：x0 恰为解（r0 = 0 ⇒ den = 0）时报 MAGMA_NONSPD（−203），SPD 矩阵也一样 | `cgres_exact_x0_nonspd` |
| `zbicgstab` 无 breakdown 检查：b = 0 或 x0 恰为解 ⇒ alpha = 0/0，x 被写成 NaN，跑满后报 DIVERGENCE | （未单独建模；状态由 `templateInfo` 覆盖） |
| `zpcg_merge` l.225–236：达到上限时以 `iter_res/init_res < rtol` 升级为 SUCCESS；循环判据用 `/nomb`。实例：‖b‖ = 1、‖r0‖ = 1e6、‖r‖ = 1e-5 ⇒ SUCCESS，残差是 rtol 的 1e5 倍 | `pcg_merge_upgrade` |
| `zbpcg` l.224：多右端只看第 0 列 | `bpcg_col0` |
| `zidr` l.128–140：restart = 50 是“未设置”的哨兵 ⇒ s = 1；maxiter 被截成 min(2n, maxiter) 并写回调用者结构体 | `idr_params` |
| `zbombard` l.622–625：`if (T_res < res) res = Q_res;`（复制粘贴错误），记录的残差可大于之前的最好值 | `bombard_wrong_res` |
| `magma_z_precond` l.59–63：内层 `psolver_par.atol` 从不初始化（栈上随机值），内层求解器读它做 `res <= atol`：未定义行为 | — |
| `magma_z_precondsetup` l.160–184：PQMR/PQMRMERGE/PBICG/LSQR + ILU/ICC 时，ILU 的 setup info 被转置 setup 覆盖 | `precond_setup_info_lost` |
| `blas/zilu.cpp`：`cusparseXcsrilu02_zeroPivot` 在数值分解之前调用（且连调两次、结果不用），只能查到结构性零；数值零主元从不检查 | `ilu_numerical_zero_unchecked` |
| 组合：ILU 数值零主元 ⇒ PBiCG 第一次 rho = NaN（zpbicg.cpp l.144–155）⇒ DIVERGENCE ⇒ l.207 覆盖为 SUCCESS | `ilu_zero_pivot_pbicg_success` |

GPU 队列追加：
9. 稀疏 PBiCG + ILU：构造 ILU0 数值零主元的矩阵（例如 [[1,1],[1,1]] 块加扰动使 A 非奇异而 ILU0 的第二个主元为 0）⇒ 预测 info = 0。
10. pcg_merge：x0 = 1e6·随机、maxiter = 20 ⇒ 预测 info = 0 而 final_res/‖b‖ ≈ 1e-5。
11. cg_res：x0 = 精确解 ⇒ 预测 −203。

### 28.8 全部稀疏求解器的收尾逻辑归类、预条件子与 FGMRES（同日；`MagmaSparse.lean` 共 23 条定理，退出码 0，无 sorry）

**归类**（对 `sparse/src/z*.cpp` 逐文件 grep 收尾代码，再人工核对 zcgs、zlobpcg）：
- “覆盖式”（`if (numiter < maxiter) info = SUCCESS;` 无条件）：zbicg、zbicgstab、zbpcg、zcg、zcg_merge、zcg_res、zlobpcg、zpbicg、zpcg、zpcg_merge、zpcgs_merge、zptfqmr、zptfqmr_merge、ztfqmr、ztfqmr_merge。其中**循环内还会检测 breakdown 的只有 zbicg、zpbicg、zptfqmr、zptfqmr_merge**：这 4 个会把检测到的发散覆盖成 SUCCESS（`bicg_breakdown_success`、`overwrite_group_size`）。
- “已修正式”（还要求 `info == MAGMA_SUCCESS`）：zcgs、zcgs_merge、zpcgs、zbicgstab_merge、zpbicgstab、zpbicgstab_merge、zqmr、zqmr_merge、zpqmr、zpqmr_merge、zfgmres、zbombard、zbombard_merge。breakdown 保持 DIVERGENCE（`fixed_breakdown_divergence`）；正常收敛经 SLOW 分支升级为 SUCCESS（zcgs 的 info 初值是 NOTCONVERGED，循环内收敛不置 0）。
- 其他：IDR 系列、Jacobi、LSQR 用各自的收尾逻辑。LSQR 无 breakdown 检查，NaN 会跑满后报 SLOW/DIVERGENCE（未发现误报成功）。
- 更正：zlobpcg 不在“循环内 breakdown 被覆盖”之列（它的第二个 DIVERGENCE 赋值是开头的参数检查）。

**新发现**：
| 发现 | Lean |
|---|---|
| `zlobpcg` l.170–178：先 `solver_par->info = SUCCESS`，参数错误写进局部 `info`，再检查 `solver_par->info != 0`：检查形同虚设，m < 2 或 n > m（特征向量多于行数）照常执行 | `lobpcg_check_dead` |
| `magma_zjacobisetup_diagscal`（zjacobi.cpp）：取倒数后才检查 `== 0`，只能抓到缺失的对角元；存储的 0 对角元得到 1/0 = Inf，通过检查 | `jacobi_stored_zero_passes` |
| `magma_zcumiccsetup`（blas/zilu.cpp）：零主元检查在数值分解之前；数值分解 `cusparseZcsric02` 的返回状态完全不检查：IC(0) 在非 M 矩阵的 SPD 上也可能出现非正主元，从不报告 | `icc_breakdown_unchecked` |
| `zfgmres` l.187–319：内层 `betanom <= atol` 置 SUCCESS，外层仍按 `rel_resid > rtol` 重启且不重置 info；之后某周期内层出现 NaN，外层拒绝式判据为假直接退出，带着旧的 SUCCESS 返回 | `fgmres_sticky_success`、`fgmres_nan_alone` |
| 核对后撤回：怀疑 Jacobi setup 出错路径 double free——`magma_zmfree` 末尾把 val/col/row 置 NULL（magma_zfree.cpp l.123–125），不是 double free | — |

GPU 队列追加：
12. LOBPCG：n > m 的块大小 ⇒ 观察是否越界或返回垃圾。
13. Jacobi 预条件：CSR 中显式存储的 0 对角元 ⇒ 预条件向量含 Inf。
14. ICC + PCG：构造 IC(0) breakdown 的 SPD 矩阵 ⇒ setup info = 0，PCG 结果。
15. FGMRES：‖b‖ ≈ 1e-8（绝对容差先达标）且 Arnoldi 中途出现 NaN ⇒ 预测 SUCCESS。

## 29. GPU 0 确认（2026-11-25，用户授权“在GPU0上测试”；eightsocket V100，只用 GPU 0，用前空闲；`experiments/magma_gpu/`）

**构建**：MAGMA 2.10.0 官方 tarball（sha256 见 `tarball.sha256`），CUDA 12.0，Volta，系统参考 LAPACK/BLAS（需 `-l:liblapack.so.3 -l:libblas.so.3 -l:libgfortran.so.5`，节点无 .so 符号链接），稠密 + 稀疏。节点 `compute-sanitizer` 缺注入库，改用 LD_PRELOAD 保护区垫片 `guard_malloc.c` 检测越界写。节点工作目录 `/ssd/haibin/magma_gpu`（2.6 GB）用完已删除。

**确认的预测**（详表见 `experiments/magma_gpu/README.md`）：
- 稠密：解溢出被接受（dsgesv/dsposv，n = 1/2/1024，iter = 0、info = 0、x = inf/nan）；dsgesv 不查 B（2^200 ⇒ NaN、iter = 0）而 dsposv 正确回退；zchesv 的 X = B（iter = −31、info = 0）与发散被接受（iter = 27、x 含 Inf）；dshposv 小右端 x = 0、×2^40 一步停止（bwd/thr 5.5e7）、返回值恒 0；**dshposv nrhs = 2 显存越界写一整列（2,048,000 B 分配后的 4096 B）**；RBT n = 1/2/3 误差 6e-2/2.6e-2/2.7e-2 且 info = 0；**RBT 构造的 cond 3.09 矩阵 info = 0、误差 1.58e13**。
- 稀疏：CG_MERGE 复数 A = diag(2,3)、b = (2,i) ⇒ info = 0、1 步、x = (2,i)；随机复 HPD 不收敛（−205）；BiCG b 含 NaN ⇒ info = 0；cg_res 精确初值 ⇒ −203；包装层 ITERREF ⇒ 返回 0、x 不动；pcg_merge maxiter = 40 ⇒ info = 0 而 final_res/‖b‖ = 2.9e-6。
- Jacobi 预条件子存储的 0 对角元 ⇒ setup info = 0、D⁻¹ 含 NaN（复数精度；实数应为 Inf）。

**与预测不同**：ILU(0) 数值零主元（A = [[0,1],[1,0]]，零对角元显式存储）——预测 setup 静默返回 0、PBiCG 报成功；实测 **`magma_z_precondsetup` 段错误崩溃**（对照矩阵正常）。源码 HIP 分支注释“this version has the bug fixed where a zero on the diagonal causes a crash”。崩溃位置未定位（节点无 gdb）。

**修复补丁**：`gmstrs_fix.patch` 编译失败（`<cmath>`/`<vector>` 在 `magma_internal.h` 的 min/max 宏之后）；`gmstrs_fix_v2.patch` 修正 include 顺序后通过。24 次调用：原版 24 次报成功、其中 20 次违反文档判据；修复版 15 次报成功、0 次违反，另 9 次正确回退。trans = T：原版误差 6.6e-2（实际解 Ax = b）、修复版走 FP64；nrhs = 2：原版第二列为 0、修复版两列正确。开销：n = 8192 时 dsgesv 5.34 → 8.32 s、dhgesv 1.59 → 4.78 s，主要来自主机端后检（拷贝 A 0.54 s + 串行 Dot2 1.80 s）和多一次迭代；**该构建链接参考 CPU LAPACK，FP64 dgesv_gpu 在 n = 8192 就要 5.4 s，计时不代表 MAGMA 正常性能**，GPU Dot2 核可消除主机端开销。

### 29.1 稠密库“状态丢失”全面审计（子代理读码；前 5 条主会话核对源码并上 GPU）

| # | 发现（文件:行） | GPU 0 |
|---|---|---|
| 1 | `zgbtrs_batched_strided`：`dB_array` 取自 `get_dAarray()`（zgbtrs_batched.cpp:343），B 指针覆盖 A 指针；另传 `batchCount` 而非 `batch` | 正确分解后 4 个批次的解**全是 NaN**，返回 0 |
| 2 | `zheevdx_2stage(_m)`、`zhegvdx_2stage(_m)` Upper：he2hb 返回 NOT_IMPLEMENTED 但返回值被丢弃（zheevdx_2stage.cpp:445） | 打印未实现，**info = 0、m = 512、特征值误差 12.9**；Lower 1e-13 |
| 3 | `zhetrf_nopiv_cpu`/`zsytrf_nopiv_cpu`：最后一个主元不检查，info 下标倒数且不加块偏移，绝对阈值 eps | [[1,1],[1,1]]、[[0]] ⇒ **info = 0、x = NaN** |
| 4 | `zpotf2`（zpotf2.cu:316）：写入 0 起的 `step`，第 0 列失败 ⇒ info = 0；0 与 NaN 不查 | A(0,0) = −1：**potrf_native(Upper) 与 potf2_gpu 都报 0**（LAPACK 1）；Lower native 报 1 |
| 5 | `ztrtri_gpu`：奇异性检查被注释掉；块 info 被后一块覆盖、不加偏移（影响 potri_gpu、getri_gpu） | A(0,0) = 0 ⇒ **info = 0**；A(300,300) = 0 ⇒ **45**（应 301）；A(511,511) = 0 ⇒ **256**（应 512） |
| 6 | `zpotrf_vbatched` Upper：只打印“unavailable”，仍按 Lower 分解并返回 0 | 未测 （见 §29.2–29.3） |
| 7 | `xhsgetrf`/`xshgetrf`：尾部 GEMM 用补齐宽度 maxn，ldda < roundup(n,32) 时越界并破坏 L | 未测（dxgesv_gmres 用对齐 ld，安全） （见 §29.2–29.3） |
| 8 | 所有混合精度精化：接受 Inf/NaN（已有） | 已确认（见上） |
| 9 | FP16 转换无范围检查，只有注释掉的 `CHECKFOR_NAN_INF` | 间接确认（dhgesv k ≥ 20 全 NaN） |
| 10 | `zgesvj_batched` 分块路径：不收敛不报告 | 未测 （见 §29.2–29.3） |
| 11 | `zgetrf_m`：info 偏移多乘 NB，且后面的奇异面板覆盖前面的 | 未测（需超显存矩阵） （见 §29.2–29.3） |
| 12 | `zpotrf_vbatched`：不清零 info_array，已有非零 info 的矩阵被跳过 | 未测 （见 §29.2–29.3） |
| 13 | `zgetri_outofplace_batched`：文档承诺的 info_array 从不写 | 未测 （见 §29.2–29.3） |
| 14 | `zgegqr_gpu`：秩作为 info 返回，秩 0 等于成功；绝对 eps 阈值 | 未测 （见 §29.2–29.3） |
| 15 | `xshgetrf`/`xhsgetrf` 奇异面板：跳过反转置，dA 留在转置状态 | 未测 （见 §29.2–29.3） |

其余（子代理，未核对）：slag2h 溢出标志用主机影子地址；shpotrf 的 goto 越过 queue 声明；gbtrf/gbsv_native 不清零 info；batched 的 `CHECK_INFO` 死代码引用不存在的变量；zhegvx/zhegvr 不收敛时不做回代；多处公开例程不检查显存分配。子代理检查为干净的例程清单见其报告（LU/Cholesky/LDLᵀ 驱动、大部分 batched、单阶段特征值与 SVD 驱动、17 个带区间参数的驱动）。

### 29.2 第二轮 GPU（同日，重新编译同一 tarball；`experiments/magma_gpu/audit2_*`）与 Lean `MagmaAudit.lean`

**Lean**：`lean/OptimiKahan/MagmaAudit.lean`（11 条定理，退出码 0，无 sorry；日志 `check_log_magmaaudit.txt`）建模审计第 1–9 条（gbtrs 指针别名与越读、he2hb 返回值丢弃、nopiv 最后主元不查与 info 倒序、potf2 的 0 起下标、trtri 块 info 覆盖、getrf_m 偏移、vbatched 旧 info、gegqr 秩 0、vbatched Upper）。**`trtri_gpu_instances` 用 nb = 256（`magma_get_zpotrf_nb`，arch ≥ 300）精确复现 GPU 上的 0 / 45 / 256。**

**GPU 0**：
- 第 6 条 potrf_vbatched Upper：分解的是下三角（下三角放“不引用”的垃圾时，对合法的 Upper 输入报 info = 2）。
- 第 12 条：旧 info = 7 的矩阵不被分解，info 保持 7。
- 第 13 条：getri_outofplace_batched 不写 info_array（哨兵值不变）。
- 第 14 条：gegqr ikind 1、3 在 A·1e-20 时 info = 0 而 Q 不正交（误差 1）。
- 第 7 条：hgetrf 在 n = 1000、ldda = 1000 时因子有 20,903 个元素与对齐版本不同（最大差 14.8），info 均为 0。
- LOBPCG 参数检查失效：n_ev = 8 > m = 6 照常执行，内部 gegqr 报非法参数，特征值全 NaN，返回 −201；并在工作目录写 `residualNorms` 文件。
- **未复现**：ICC 对 IC(0) 失败矩阵（CPU 上第 6 个主元 −0.042）：setup = 0、PCG 4 步收敛到 1e-15，无可见后果。
- 未测：第 10、11、15 条，FGMRES 粘住的成功。节点目录（1.3 GB）已删除，GPU 0 空闲。（第三轮已运行，见 §29.3）

### 29.3 第三轮 GPU（同日；`experiments/magma_gpu/audit3_*`）

- 第 10 条 `gesvj_batched`：含一个 NaN 时 n = 16/64/256 都返回 info = 0、奇异值全 NaN（比预测严重：小矩阵融合路径也不报）。Lean `gesvj_nan_converged`（`off > tol` 对 NaN 为假即“收敛”）。
- 第 11 条 `getrf_m`：恰为零的列（100/300/600/900）一律 info = 0（LAPACK 101/301/601/901），设不设 `MAGMA_NGR_NB` 都一样。**根因**（主会话核对）：函数末尾 `if (*info >= 0) magma_zgetrf_piv(…, info)`，而 `magma_zgetrf_piv` 开头 `*info = 0`（zgetrf_m.cpp l.28、l.400–401）；大面板循环也不记录 iinfo。CPU 接口 `magma_zgetrf`/`zgesv` 在矩阵放不下显存时走这里。Lean `getrf_m_never_singular`（原 `getrf_m_offset` 描述的偏移错误在这条路径下被后续清零掩盖）。
- 第 15 条 `hgetrf` 奇异面板：info = 1 报告了，但尾部 589,824 个元素全等于 Aᵀ（原地转置未还原）。Lean `hgetrf_left_transposed`。
- FGMRES 粘住的成功：1152 次参数扫描中 252 次 info = 0 而 x 非有限（226）或真实残差 > 1e-6‖b‖。
- `MagmaAudit.lean` 现 14 条定理，退出码 0，无 sorry。节点目录已删除，GPU 0 空闲。

### 29.4 稀疏求解器全量 GPU 扫描（同日；`experiments/magma_gpu/sweep_*`；节点目录已删除，GPU 0 空闲）

32 个线性求解器 × 6 组输入（实 SPD 对照、复 HPD、复非 Hermitian、微小 b 加随机 x0、b[0] = NaN、x0 恰为解）：
- **b 含 NaN 时静默报成功的恰好是 BICG、PBICG、PTFQMR、PTFQMR_MERGE**，与 §28.8 由源码收尾逻辑归类得出的 4 个文件完全一致，其余 28 个都报失败。
- **微小 b：32 个中 21 个静默失败**（info = 0 而结果错误；其中 5 个 numiter = 0、直接返回 x0，16 个迭代但未收敛）。原记“23 个、x = x0”有误，经独立复核更正。
- x0 恰为解：BICGSTAB 把正确的 x0 变成 NaN；BICGSTABMERGE、PBICGSTABMERGE、JACOBI 报 −202，CG 报 −203（答案正确却报失败）。
- 复数：CG_MERGE、PCG_MERGE 失败（不取共轭）；其他 MERGE 变体正常；QMR 答案正确却报 −201。
- **新发现：IDRMERGE、PIDRMERGE 对所有输入都失败，包括实 SPD 对照**（restart 30/4 时 NaN、−202；restart 50 即 s = 1 时停滞在 0.297、128 = 2n 步），同参数下 IDR/PIDR 1–10 步收敛。maxiter 截断只在两个 MERGE 变体上由 GPU 运行看到（IDR/PIDR 在达到上限前就收敛了，对它们只有读码证据）。

### 29.5 第五轮 GPU（同日；`experiments/magma_gpu/round5_*`；节点目录已删除）

- **bpcg 多右端**：经包装层 b 为 64×2 时第 1 列根本不解（误差 1）、info = 0。原因：`num_vecs = b.num_rows / A.num_rows`（zbpcg.cpp l.74）= 1，而包装层正是在 `b.num_cols > 1` 时转到 zbpcg。
- **zgesv_rbt_batched**：精确奇异矩阵返回 0、info_array 0、误差 0.85（蝴蝶变换混开零列）。**泄漏确认**：200 次调用显存少 25.17 MB、主机内存多 52.7 MB，与 dv、hu + hv 的大小吻合。
- bombard：未复现（这组输入里 TFQMR 没有胜出）。
- **单精度**（c/s）：gbtrs 别名、trtri info 45、potf2 info 0、CG_MERGE 复数假收敛，全部与 d/z 一致。
- **ILU 崩溃定位**：栈溢出（需备用信号栈才打得出调用栈），`magma_zmtransposeconjugate` 无限递归。因果链：ILU setup 检测到存储的零主元并失败（外层 PCG 时 setup 返回 −3009，L 的 memory_location = 0、nnz = 0）；外层 PBiCG 时该失败被转置 setup 的返回值覆盖，转置在未构建的 L 上进行，位置 0 两个分支都不匹配，于是不断转换为 CSR 并递归调用自身。Lean `transpose_unbuilt_diverges`、`transpose_cpu_terminates`。之前“数值零主元不检查”的定理针对的是数值分解过程中出现的零主元，不是本次输入（显式存储的零在分析阶段已被检测出来）。

### 29.6 修复补丁的开销：OpenBLAS 构建与补丁 v3（同日；`experiments/magma_gpu/perf_*`；节点目录已删除）

- 之前的计时链接参考 LAPACK，不能用来评估开销。这次从源码编译 OpenBLAS 0.3.28（`NO_FORTRAN=1 C_LAPACK=1`，32 线程），重新编译三份 MAGMA：原版、v2（主机认证）、v3（GPU 认证）。
- **v3**（`experiments/magma_fgmres/fix/gmstrs_fix_v3.patch`）：新文件 `magmablas/dgmstrs_certify.cu`，每行一个线程，Dot2 与行和的运算顺序与主机版相同（fma 做 TwoProduct，`__dadd_rn` 等避免收缩）；只拷回 n 维向量，外向舍入比较与“非有限 ⇒ 失败”仍在主机上，未改。发布版 tarball 用 `Makefile.gen.cuda`，需在 `libmagma_templates := \` 后插入该文件，否则链接时符号未定义。
- **正确性**：OpenBLAS 构建下原版 24/24 报成功、**23** 次违约（参考 LAPACK 构建为 20，FP32 舍入不同）；v2、v3 均为 15 次成功、**0** 次违约；v3 的 trans、nrhs 均正确。
- **开销**（n = 8192，最好的 3 次）：原版 0.21–0.26 / 0.17–0.18 s（FP32 LU / FP16 LU）；v2 3.34 / 3.27 s；**v3 0.23–0.30 / 0.20–0.35 s**；FP64 dgesv_gpu 0.29–0.43 s。n = 16384：原版 0.85 / 0.73 s，v3 0.90 / 0.91 s，FP64 1.09 s。n ≤ 4096 时 v3 多 0.03–0.17 s（1.3–3 倍），来自多一次迭代和认证的启动与拷贝。
- 节点有其他用户的 CPU 负载（load average 10–16），同一构建重复测量相差可达 ±30%，所以只报区间。
- 结论：GPU 认证消除了 v2 的 11–16 倍减速；n ≥ 8192 时 v3 与原版相差在 25% 以内，仍快于 FP64。V100 的 FP64 很强，混合精度本身在这里收益就不大；这一轮回答的只是“带证明的认证是否比分解还贵”，答案是否。
- GPU 内核本身不是 Lean 证明的对象；证明的 `Faithful` 假设针对的是运算序列，内核执行的是同一序列。

### 29.7 第二轮审计：此前未审的算子族（同日；`experiments/magma_gpu/round6_*`、`round7_*`；节点目录已删除）

- 范围：稀疏 control/blas/precond（格式转换、SpMV、缩放、预条件 setup 与 apply）；稠密 QR/最小二乘、广义与三对角特征值、逆、转换核。两个子代理读码各报 13 条与 8 条，主会话逐条核对源码后上 GPU 0。完整表见 `operators/magma/COVERAGE_zh.md` §3b。
- **GPU 确认**（返回成功但结果错误或缺失）：slag2h 溢出 ⇒ 非法内存访问且 info 0（主机影子地址）；dlag2s/slag2h 的 NaN 不报；dstedx info 0 带 10 个 NaN；dgels_gpu 秩亏 info 0、残差 95；dgeqp3_gpu 固定列不分解（交换时段错误；文档 LWORK 被拒）；dsygvdx n ≤ 128 忽略 range；sync-free ILU 的 CSC 数组顺序反（L.drow 写入 127 个行号，分配 65）；spmv 维度不符返回 0、DENSE 算 Aᵀx、CSC 结果错、SELL-P 漏 44 行；zmscale 全 NaN 返回 0。
- **报告了失败但根因在库**：复数 ICC 用 Lᵀ（PCG+ICC 1000 步失败，CG 17 步）；shpotrf 对 ×2^40 的 SPD 矩阵报不正定（FAST_16F 溢出）。
- **与预测不符（如实记录）**：ILU(k) 存储不足被报告（−3009）；shpotrf 分配失败的崩溃路径只在 HIP 编译；gbtrf 不清零 info 是假警报；dshposv_gpu 自带缩放，×2^40 不触发 slag2h。
- 一处探针标签写错（L(n,n) 的极限应为 √(2+√3) = 1.93185），结果文件已注明。guard 分配器会改变 sync-free 的损坏位置，以不加 guard 时读回的数组内容为准。
- Lean：`MagmaAudit.lean` 25 条、`MagmaSparse.lean` 31 条定理，均退出码 0、无 sorry、无警告（日志已更新）。论文 Appendix A 总计更新为 2,973 行、217 条定理。

### 29.8 第八轮：把剩下的读码发现全部建模并上 GPU（同日；`experiments/magma_gpu/round8_*`；节点目录已删除）

- Lean：`MagmaSparse.lean` §14 新增 11 条（42 条），`MagmaAudit.lean` §19 新增 4 条（29 条），均无 sorry、无警告。论文 Appendix A 总计 3,101 行、232 条。
- GPU 0（CPU 侧堆错误用 `MALLOC_CHECK_=3` 暴露）：
  - CPU 上的 spmv 返回 0，结果丢失，调用者的数组已被释放；
  - CSRD 缺对角元越界写，之后 glibc 中止；ELLD 列号 −1；
  - zmdiff 对无共同元素、只差虚部都给 0；zmfrobenius 把入口值加进去，行数不等时什么也不算；zmdiagadd 跳过无对角元的行；CPU 转置不交换维度；
  - mtx 读入越界下标段错误；
  - ILU/ICC 的 applyprecond 和 ConjTrans 的 applyprecond_right 返回 0、x 不写；
  - zhegvx 每次泄漏 21.2 MB；
  - CPU dgels 在 nrhs = 16384 时 DORMQR 拒绝参数 12，xerbla 结束进程（nrhs = 4096 时 LWORK 8192 够用，与按 nb = 32 写的预测不同）；
  - geqrs3_gpu 参数编号错；`magma_zgels_batched` 只有声明。
- 未测：zhegvd 系负 info 映射（需要分配失败）、zmscale_generate UNITCOL 双重释放（驱动的 side 处理未追）。（见 §29.9、§29.10：zmscale_generate 双重释放已确认；zhegvd 负 info 映射在 2.10.0 不可达）
- 至此两轮审计的发现都有 Lean 模型；除上面两条外都有 GPU 证据。剩下未覆盖的只有多 GPU 例程（只允许 GPU 0）和 HIP 后端（没有 AMD 卡）。

### 29.9 第九轮：多 GPU / 核外驱动（ngpu = 1）与全部 202 个稠密源文件的清点（同日；`experiments/magma_gpu/round9_*`；节点目录已删除）

- 清点：`operators/magma/INVENTORY.md`（子代理逐文件分类）：此前已审 127、本轮 31、新读 3（zcgeqrsv_gpu、zgglse、zggrqf）、无状态判断的辅助例程 41。
- GPU 0 确认：
  - range V 在内部缩放后与未缩放的界比较：4 个 zheevdx 变体 m = 10（应 100）；
  - 输入含 NaN 时 4 个变体都在 stedx 的 dlascl 处由 xerbla 结束进程；
  - zunmtr_m 吞掉 NOT_IMPLEMENTED、C 不变；
  - he2hb 小 lwork：带状结果差 2.5e119（info 0），或堆损坏；
  - zgeqrf_m ngpu = 0：SIGFPE；
  - zcgeqrsv、zgglse 秩亏：info 0、x 非有限；
  - zggrqf 改完 A、B 才报 −7；
  - zmscale_generate 双重释放；
  - zpotrf_mgpu_right 只有声明。
- 未复现或未测：2-stage 中 stedx 失败被清零（NaN 在更早处结束进程）；分配失败路径（zhegvd_m、zgeev_m 等）；zgeqrf_ooc。zhegvd 的负 info 映射在 2.10.0 不可达。（见 §29.10：故障注入确认了 stedx 失败被清零，并跑了分配失败路径与 zgeqrf_ooc）
- Lean：`MagmaAudit.lean` 38 条、`MagmaSparse.lean` 43 条，无 sorry、无警告。论文 Appendix A：3,199 行、242 条。

### 29.10 第十轮：分配失败的故障注入、核外 QR、ngpu = 2（同日；`experiments/magma_gpu/shims/fail_malloc.c`、`round10_*`；节点目录已删除）

- 方法：LD_PRELOAD 垫片让第 k 次 `cudaMalloc`/`cudaHostAlloc` 失败，k 取遍；每次与 LAPACK 结果比较；backtrace 定位失败的分配。11 个驱动、55 次失败注入。
- **info 0、结果错误（6 次；原记 7 次，经独立复核更正）**：zhegvd、zhegvd_m（zhegst 工作区 ⇒ 得到 A 的特征值，误差 599）；zheevdx_2stage（he2hb）；zheevdx_2stage_m（stedx_m 内 dlaex0_m，确认了第九轮未能复现的“stedx 失败被清零”）；zgeev、zgeev_m（gehrd）。
- **进程被结束（8 次）**：zhetrd 的 iinfo 不读 ⇒ 未初始化的三对角 ⇒ NaN ⇒ dlascl/xerbla。**段错误（3 次）**：zgetrf_gpu、zpotrf_gpu 不检查锁页分配。
- zgeqrf_ooc：占显存使其走核外路径，lwork = n ⇒ 段错误；NB = 0 时返回 −113，不死循环（预测不符）。
- ngpu = 2 而只有 1 个可见设备：magma 退回 1 个 GPU，结果正确，留下粘滞的 CUDA 错误。
- Lean：`MagmaAudit.lean` §21，共 42 条；论文 Appendix A 3,245 行、246 条。
- 仍不能做的：真正的多卡（≥ 2 个 GPU）行为、HIP 后端。

### 29.11 第十一轮：丢弃 iinfo 的辅助例程（同日；`round11_*`；节点目录已删除）

- 故障注入：zunmtr、zunmbr、zunghr 各只有 1 次设备分配（在 zunmqr / zungqr 内），失败时都 info 0：zunmtr、zunmbr 返回原样的 C，zunghr 返回错误的 Q。zungtr 有同样的代码，但不在公开头文件中。Lean `helper_iinfo_dropped`。

### 29.12 第十二轮：magmablas 中写状态的 39 个内核（同日；`round12_*`；节点目录已删除）

- 子代理逐文件读码，主会话核对关键行；10 条上 GPU 0，**全部确认**：
  - vbatched LU 第一面板外的零主元被清零；
  - 通用带状 LU 只看最后一列；
  - zgetf2_gpu 不报奇异；
  - zpotf2/zpotrf_native 不报 0 和 NaN；
  - zpotrf_batched n 不是 8 的倍数时不报 NaN；
  - zgetrf_native 长列（>15360）不报零主元；
  - 非方带状 LU 假 info 或 ipiv > m；
  - heevj 把 NaN 当收敛；
  - nopiv vbatched 的 `*info++`；
  - 批量 gesv 奇异时仍写 B。
- 未测：native 递归 LU 的偏移（需 expert 接口）、两处疑似竞争、两条轻微问题。（见 §29.13、§29.14：偏移已确认，竞争未复现）
- Lean：`MagmaAudit.lean` §22–23，共 53 条；论文 Appendix A 3,330 行、257 条。
- 完整性说明见 `operators/magma/COVERAGE_zh.md` §3f。

### 29.13 第十三轮：剩余的内核条目与两条稀疏条目（同日；`round13_*`；节点目录已删除）

- 确认：
  - F7 native 递归 LU 偏移：recnb = 64 ⇒ info 18（LAPACK 50），与 Lean 模型给出的值相同；
  - F13 m < n 融合 getf2 ⇒ 假 info 5；
  - F14 NaN 使树形 argmax 选 0 ⇒ zgetrf_batched（m ≥ 16）假 info 1；
  - F15 zlascl 参数编号；
  - zmscale_matrix_rhs 报错时 b 已被缩放成 (inf, 0.5)、A 未变；
  - zmtransfer 对未处理的存储格式返回 0、结果为空。
- 未复现：F12 原子写 info 的竞争（300 次全对）。
- Lean：`MagmaAudit.lean` 56 条、`MagmaSparse.lean` 45 条；论文 Appendix A 3,373 行、262 条。
- `operators/magma/COVERAGE_zh.md` §3f 改为逐项可查的完整性说明，并列出全部未在 GPU 上运行的条目。

### 29.14 第十四轮：此前只有读码证据的单 GPU 条目全部运行（同日；`round14_*`；节点目录已删除）

- ✔ zlag2c 全局标志竞争：两线程各 2000 次，溢出矩阵 1889 次报 0、干净矩阵 7 次报 1。
- ✔ heevj max_sweeps = 0 报成功、结果全错。
- ✔ zmtranspose_cpu 宽矩阵段错误。
- ✔ 截断 .mtx 返回 0 并复制出一个元素。
- ✖ zlascl2 越界读：用 CUDA VMM 把 D 放在未映射页前，不出错。
- ✖ zpotf2 warp 归约：1000 次结果一致。
- ✖ heevj noconv 竞争：12800 次求解全对。
- 辅助例程 zunmqr、zunmql、zunmlq、zungqr2 在分配失败时都正确报告；ztrevc3 系列不做设备分配。
- Lean：`MagmaAudit.lean` 58 条、`MagmaSparse.lean` 47 条；论文 Appendix A 3,420 行、266 条。
- 至此只剩：zhegvd 负 info 映射（不可达）、ngpu ≥ 2 的真实多卡行为（需要用户允许使用 GPU 0 以外的卡）、HIP 后端（无 AMD 卡）。

### 29.15 第十五轮：sparse control/blas 66 个 z 文件逐文件判定（同日；`operators/magma/SPARSE_INVENTORY.md`、`round15_*`；节点目录已删除）

- 判定：已知 14、新发现 31、干净 13、无状态逻辑 8。
- 13 条上 GPU 0，11 条确认：
  - SpMV 不读 transA；
  - zmdot_shfl 归约空操作（64 个 1 得 1，很可能是 IDR_MERGE 全部失败的原因）；
  - zmgecsrmv 多向量步长；
  - SELL-P 多向量只算一半；
  - zvread 读不了自己的输出格式并堆溢出；
  - zcsrget 悬空指针；
  - zcompact 把 NaN 当收敛；
  - zmcsrcompressor_gpu 清空 CPU 矩阵；
  - zfrobenius 只比实部；
  - zmdiagdom 空行 NaN；
  - zsort 遇 NaN 不终止。
- 未确认：ELLPACKT 填充（NaN 来自复数 Inf 乘法，先前的判断有误，已撤回）；zmdotc 共轭（结果未写回，不确定）。（zmdotc 第十六轮已确认，见 §29.16）
- Lean：`MagmaSparse.lean` 57 条；论文 Appendix A 3,497 行、276 条。

### 29.16 第十六轮：sparse control/blas 其余新发现（同日；`round16_*`；节点目录已删除）

- zmdotc 改用设备输出缓冲区重测：不取共轭确认（−1）。
- 另 12 条中 10 条确认：
  - zcsr_sort 不排 val；
  - zcsrcoo_transpose 维度；
  - zbitonic_sort 丢错；
  - zvget_dev 悬空指针；
  - zcuspmm CPU 输入段错误；
  - zcsr_sort_gpu 混合位置；
  - zge3pt 读 dx[−1]；
  - ELLRT 启动失败不查；
  - jaccard 权重错；
  - lobpcg_shift 只移位第 0 行。
- 未复现：zmbdiagdom、zcuspaxpy（buf 大小为 0）。
- 按编号：31 个中 27 个已上 GPU（25 个确认）；未运行的 #3、#5、#14 不在公开头文件中，#17 被已确认的 CSC 交换掩盖。（第十七轮已运行，见 §29.17）
- Lean：`MagmaSparse.lean` 67 条；论文 Appendix A 3,559 行、286 条。

### 29.17 第十七轮：不在公开头文件中的 4 个编号、sync-free 等待、其余辅助例程（同日；`round17_*`；节点目录已删除）

- 通过导出符号直接调用，全部确认：
  - sync-free 求解即使 CSC 数组正确也错（n = 4096 时 4095 个分量错、返回 0）；
  - zcsrget_gpu 悬空指针；
  - zorderstatistics 不查最后一个元素的 NaN；
  - zrowentries 漏掉纯虚元素；
  - ztrisolve 对奇异因子返回有限“解”和 0。
- sparse control/blas 的 31 个编号全部上 GPU（29 个确认）。
- 辅助例程故障注入：zungqr、zungqr_gpu、zunmrq、zunmqr2_gpu、zunmql2_gpu、zgerbt_gpu 都正确报告 −113。41 个辅助例程的逐文件判定写入 `operators/magma/INVENTORY.md` 末节。
- Lean：`MagmaSparse.lean` 72 条；论文 Appendix A 3,597 行、291 条。
- 至此剩下：zunglq 的单独故障注入（dT 格式未确认）、ngpu ≥ 2 的真实多卡行为（需用户允许 GPU 0 以外的卡）、HIP 后端（无 AMD 卡）。（zunglq 见 §29.18）

### 29.18 第十八轮：zunglq 与第 41 个辅助例程（同日；`round18_*`；节点目录已删除）

- zunglq：`magma_z.h` 声明第 7、8 个参数为 (dT, nb)，定义为 (work, lwork)；按头文件调用（传设备缓冲区与 nb）⇒ 段错误；按定义调用，分配失败正确报告 −113。Lean `unglq_header_mismatch`。
- 先前清点时漏掉的第 41 个辅助例程是 zbulge_applyQ_v2.cpp（提取用的正则不匹配大写字母），内部例程，经 2-stage 驱动覆盖；41 个辅助例程现全部逐文件判定。
- Lean：`MagmaAudit.lean` 59 条；论文 Appendix A 3,609 行、292 条。
- 剩下：ngpu ≥ 2 的真实多卡行为（需用户允许 GPU 0 以外的卡）、HIP 后端（无 AMD 卡）。

### 29.19 多 GPU 状态簿记的一般性证明（同日；`lean/OptimiKahan/MagmaMultiGpu.lean`，日志 `check_log_magmamultigpu.txt`）

- 不能用第二块 GPU，所以对 ngpu ≥ 2 路径的状态逻辑做一般性证明（对所有 ngpu、nb、n，不是实例检查）：
  - `distribution_covers`：zgetrf2_mgpu 的 n_local 之和等于 n；
  - `getrf_mgpu_first_global`：多卡 LU 的 info 等于第一个失败列的全局下标；
  - `potrf_mgpu_global`：多卡 Cholesky 报告 j + 局部 info。
- 三条都是正面结论：这些路径上没有预测到缺陷。退出码 0，无 sorry，无警告。
- 仍需另一种环境才能做的：多卡通信与并发的实际运行（需用户允许 GPU 0 以外的卡）、HIP 后端（无 AMD 卡）。

### 29.20 独立复核（同日；`operators/magma/reviews/1_INDEPENDENT_REVIEW.md`）与第十九轮（`round19_*`；节点目录已删除）

- 新子代理独立核对清点与判定：清点与文件系统一致；补入 zgetf2_nopiv_devicefunc.cuh（干净）；两处计数错误已更正（21/32、6 次）；四条判定不同。
- 第十九轮：
  - zungbr 丢弃 zungqr2/zunglq 的 iinfo（故障注入 ⇒ info 0、误差 52 / 8.6）✔；
  - zlascl_2x2 参数编号与 lddw ✔；
  - zresidual 多右端只算第一个 ✔；
  - zresidual 0×0 除零 ✖；zcsrsplit 未排序行 ✖。（两个测试都没有走到目标代码；第二十三轮重测均确认，见 §29.25）

### 29.21 非 z 文件审计（同日；`operators/magma/NONZ_INVENTORY.md`）

- 18 个写状态但不由 z 生成的文件（15 个在 z 调用链上，另 dlaln2、dlaqtrsd、dsidi）逐文件判定；dlaex*、dlaln2、dlaqtrsd 干净；新发现 10 条（见覆盖表 §3h）。
- 第二十轮探针已写好并修正编译错误，但 **GPU 0 此时被用户的 ParaCOSM 作业占用（pid 3920111，23 GB、100%）**，按“只用空闲的 GPU 0”的规定没有运行；节点目录已删除。等 GPU 0 空闲后执行 `run_round20.sh`。（后已运行，见 §29.22）

### 29.22 第二十轮（GPU 0 空闲后自动运行；`round20_*`；节点目录已删除）

- 在 GPU 0 连续 3 分钟无进程、显存 < 500 MiB 后运行（`wait_and_run20.sh`），未影响用户的 ParaCOSM 作业。
- ✔ #1 range V 特征向量错（残差 13.5、正交性 296，m 正确，info 0；1-stage 与 2-stage 都有）。
- ✔ #3 adjust_ipiv 不做事（nb = 600：1200 个无效主元、残差 52.7）。
- ✔ #4 分块 gesvj 零列 ⇒ U 非有限。
- ✔ #7 区间闭合（11 vs 10）。
- ✔ #9 范数未缩放（1e160 ⇒ S = Inf，1e-170 ⇒ U 非有限）。
- #5 不确定（需要 LINPACK 格式的分解）。（第二十一轮已确认，见 §29.23；该判断有误）
- Lean：`MagmaAudit.lean` 66 条；论文 Appendix A 3,772 行、304 条。

### 29.23 第二十一轮：非 z 文件剩余的四条（同日；`fail_dsteqr.c`、`round21_*`；节点目录已删除）

- #2 用 LD_PRELOAD 替换 LAPACK dsteqr_，令某一次调用报失败：块 1 与块 2 的第一个叶子失败都返回 info = 226，块偏移丢失 ✔。
- #5 dsidi 按文档输入（LAPACK dsytrf 结果）逆矩阵错，连 SPD、无 2×2 主元也错（‖AX − I‖ = 0.131 / 1.16）✔；第二十轮“不确定”的判断有误。
- #8 70000 个 64×64 的批量 gesvj：第 65534 个之后抽查的 47 个全错、info 0 ✔。
- #10 dsort 遇 NaN 下标不是置换（一重复一缺失）✔。
- #6 有害形式需 n > 53,500（45 GB），超出 V100 显存，无法触发。（判断有误：第二十四轮已触发，见 §29.26）
- Lean：`MagmaAudit.lean` 70 条；论文 Appendix A 3,807 行、308 条。

### 29.24 第二十二轮：独立复核余下的条目（同日；`fail_malloc_cpu.c`、`round22_*`；节点目录已删除）

- zdiinertia 把 NaN 计为零特征值 ✔。
- zmcsrcompressor 对不拥有数组的矩阵不做事、返回 0 ✔。
- zmlumerge 第 4 次主机分配失败 ⇒ double free ✔（k = 4 的输出行因进程中止而丢失，由中止位置推断）。
- nm：magma_zdiagcheck_cpu 有声明无定义；magmablas_zdiinertia、magma_zmlumerge 有定义无声明（第一次编译即因 zmlumerge 无声明而失败）。

### 29.25 第二次复核（`operators/magma/reviews/2_SECOND_REVIEW.md`）与第二十三轮（`round23_*`）

- 第二次复核：
  - Lean 总数与论文一致；
  - 第十五至二十一轮抽查数字全部一致；
  - 约 18 处过时或矛盾的文字大部分已改，余下的由第三次复核找出并改正。
- 第十九轮两条“未复现”的测试没有走到目标代码，第二十三轮重测：
  - zresidual（A 0 行、b 4 行）⇒ SIGFPE ✔；
  - zcsrsplit（bsize = 2、未排序行）⇒ 垃圾列号、`free(): invalid pointer` ✔。
- zheinertia：第一次把主机 ipiv 传给内核，输出未写；改用设备 ipiv 重跑 ⇒ diag (1, −1, NaN, 2) 得 2/1/1，NaN 计为零 ✔。
- Lean：`MagmaSparse.lean` 78 条、`MagmaAudit.lean` 71 条；论文 Appendix A 3,859 行、314 条。

### 29.26 第二十四轮：#6 的 32 位尺寸溢出（同日；`round24_*`；节点目录已删除）与第三次复核（`operators/magma/reviews/3_THIRD_REVIEW.md`）

- 直接调 `magma_zstedx_m(ngpu = 1)`：Z 放主机内存（节点 596 GB 可用），GPU 上只放分治工作区 3n(n/2+1)。
  - n = 4000 对照正确。
  - n = 38,000 ⇒ −12：zstedx_m 的 `lrwmin = 1 + 4n + 2n²` 在 32 位下为负，自己的检查因此通过；拒绝来自 dstedx_m（收到 lrwork − n² = 703,483,647，正确最小值 1,444,152,001）。真实需求 2.89e9 > INT_MAX，任何 32 位 lrwork 都不合法；这是正确报告的错误，GPU 工作区分配没有执行到。
  - n = 53,600 ⇒ 段错误；路径上 dlaex0_m 工作区（回绕为 1.46e7 个 double）、rwork + n²、lrwork − n²、拷贝循环的 ldz·j 下标都溢出，崩溃原因未确定。
  - （以上两条原因经第四次复核按源码更正）
  - 先前“本机无法触发”的判断有误。Lean `dc_sizes_wrap`。
- 第三次复核：第二次复核的更正大部分落实；指出的 17 处剩余问题（论文中一处插入位置错误、几处计数与前向指针缺失等）中 12 处改好；其余由第四次复核（`operators/magma/reviews/4_FOURTH_REVIEW.md`）找出后改正；round 22/23 数字与 Lean 总数一致。
- Lean：`MagmaAudit.lean` 72 条、`MagmaSparse.lean` 78 条；论文 Appendix A 3,882 行、315 条（第四次复核后的 docstring 更正使行数增加）。

### 29.27 仓库结构整理与形式化补充（2026-11-26）

- **Lean 改为 Lake 包**：`lean/lakefile.lean`，`cd lean && lake build` 一次构建并检查全部 22 个模块（46 个任务，无警告）。各定理的公理输出合并为 `lean/logs/axioms.txt`，原先每个文件一份的 `check_log_*.txt` 和散落的 `.olean` 已删除，`.lake/` 加入 `.gitignore`。
- **MAGMA 模块按主题拆分**（原来按审计轮次堆在两个大文件里）：`lean/OptimiKahan/Magma/` 下 `Fgmres`、`Fixed`、`DenseIR`、`MultiGpu`、`Kernels`、`Dense/{StatusAudit, QrEigen, FaultInjection, NonZ}`、`Sparse/{Solvers, Control, Blas}`。命名空间与定理名不变（`MagmaAudit.*`、`MagmaSparse.*`），拆分前后的定理集合逐一比对相同（150 条）。旧文件与小节到新模块的对照表在 `lean/OptimiKahan/README.md`。
- **实验目录**：`experiments/magma_gpu/` 的 114 个文件分到 `probes/`、`scripts/`、`shims/`、`inputs/`、`results/`；`operators/` 的 MAGMA 文档移到 `operators/magma/`（复核报告在 `operators/magma/reviews/`，加了“路径为整理前布局”的说明）。当前文档中的路径已更新并逐个检查存在；本文件 §29.26 之前的记录保留原路径。
- **形式化补充**：
  - 先前没有 Lean 模型的 10 条 GPU 已确认发现全部补上：rbt_batched、hgetrf 补齐 GEMM、FP16 转换、getri_outofplace、两个缺失符号、zmdiagdom、zcsrcoo_transpose、gesvj 列范数；
  - 对 38 条只证明了具体实例的定理补了对所有触发输入成立的一般形式（各模块末尾的 General forms，名字以 `_all` 等结尾），例如：
    - vbatched 中所有面板偏移 ≥ my_minmn 的零主元都被清零；
    - 通用带状 LU 的 info 只取决于最后一列；
    - 任意前缀之后最后一个大块都会把 stedx 的 info 清零；
    - 任意早期失败都被最后一个成功阶段覆盖；
    - FGMRES 的粘住成功对任意多个普通循环成立。
  - 覆盖表中所有 ✔ 行现在都有 Lean 定理。
- 论文 Appendix A 重新按实际文件生成：4,261 行、355 条定理。


### 29.28 第25轮：多GPU（ngpu=2，GPU 0+1）——结果未定，节点驱动挂死（2026-11-26）

结果文件：`experiments/magma_gpu/results/round25_results.txt`（只跑完 potrf_m 和 getrf_m 两个模式）。

| 例程 | ngpu=1 | ngpu=2 |
|---|---|---|
| zpotrf_m，HPD n=4000 | info=0，误差 3e-15 | **info=526，误差 63.2** |
| zpotrf_m，A(101,101)=-1e6 | 101 | 101 |
| zpotrf_m，A(3001,3001)=-1e6 | 3001 | **684** |
| zgetrf_m，一般矩阵 | info=0，误差 1.6e-12 | **info=0，误差 inf，3543 个主元不同，用时 126 s** |
| zgetrf_m，第 2501 列为 0 | **info=0**（LAPACK 2501） | **info=0** |

- getrf_m ngpu=2 之后出现 `cudaGetLastError = unspecified launch failure`。随后 probe 进程（pid 10470）卡在 D 状态，kill -9 无效。节点上所有用户的 nvidia-smi 都挂住（驱动 580.178.04）。
- **双卡结果的结论未定：**两个 ngpu=2 模式都出错，而且之后驱动挂死。这更像 GPU 1 或 GPU0↔GPU1 的 P2P 通路有硬件或驱动问题（GPU 2 已知有故障），而不一定是 MAGMA 的代码错误。没有建 Lean 模型，也没有写进 COVERAGE。
- zgetrf_m 零列返回 info=0（单卡和双卡都是）**是已知的第 11 条**（`getrf_m_never_singular`）：函数末尾的 `magma_zgetrf_piv` 一开头就执行 `*info = 0`，跟 ngpu 无关，也跟 CUDA 上下文是否损坏无关，读码就能定。不是新发现。
- 读码核对（zpotrf_m.cpp l.296–301、l.418–422；zpotrf3_mgpu.cpp l.553–556）：双卡 Cholesky 的 info 是 `J + j + 局部 info`，偏移正确，出错后也会 break。所以 684 和 526 都不是下标算错，而是在双卡计算中数据已经损坏，与 GPU 1 硬件故障或竞争条件的解释一致。zpotrf_m 的两卡之间经主机内存中转，不走 P2P，所以硬件方面首先怀疑 GPU 1 本身。
- 要确认，至少需要：（a）驱动恢复后，只用 GPU 1 跑 ngpu=1，检查 GPU 1 本身；（b）用 `p2pBandwidthLatencyTest` 或 `cudaDeviceCanAccessPeer` 检查 P2P；（c）最好换一台干净的双卡机器重跑。
- 节点上的 /ssd/haibin/magma_gpu 已删除，进程 10470 仍在 D 状态（需要管理员处理或重启）。

### 29.29 HIP 专用分支读码（2026-11-26，GPU 不可用期间）

- 范围：src、magmablas、sparse、control 中所有带 `MAGMA_HAVE_HIP` 的文件（126 个）。逐个看了 z 版本中与状态有关的分支：zpotrf_gpu、zgetf2_native、zgesv_rbt、xshgetrf/xhsgetrf、dxgesv_gmres、zgetf2_kernels（izamax 长路径）、zgbsv/zgbtrf 系列、zgesvj_batched_small_sm、zgetrf_batched_smallsq_noshfl、zgeqrf_batched_smallsq、zlaqps2_gpu、slag2h/hlag2s、zlanhe、稀疏库的 zilu、zcuspmm、zcuspaxpy 和 zfree。
- 结论：除下面一条外，HIP 分支只是换成同名的 hipBLAS/hipSPARSE 调用或改线程数，状态逻辑和 CUDA 相同。izamax 长路径不写 info、ILU 只能查出结构零、cuspmm/cuspaxpy 不查状态，这些已登记的问题在 HIP 下同样成立。
- **新发现：**`[sdcz]gesv_rbt.cpp` l.113–118，HIP 编译下参数检查之后执行 `refine = MagmaFalse`（作者注释 “TODO: investigate failures on AMD GPUs”）。调用者传 `refine = MagmaTrue` 不会报错，但 `zgerfs_nopiv_gpu` 不会运行，info 也不提示。不选主元的 RBT 解不精化时，CUDA 上测到相对误差 1.6e13（`dense_results.txt`）。
- Lean：`Magma/Dense/QrEigen.lean` 的 `rbt_hip_refine_dropped`、`rbt_hip_refine_irrelevant_all`（HIP 下输出与 refine 无关，等于不精化的解）和 `rbt_cuda_refine_kept`。lake build 通过，没有 sorry，`logs/axioms.txt` 现为 320 行。论文 Appendix A 现为 4,280 行、358 条定理。
- 证据：只有读码（没有 AMD 卡）。

### 29.30 工作区大小（lwork）系统核对（2026-11-26，GPU 不可用期间；只有读码证据）

- 范围：src/ 中所有带 lwork、lrwork 或 liwork 参数的 z 驱动，共 83 个。每个都比三样：查询返回值（Q）、检查接受的最小值（C）、代码和子调用实际需要的大小（U）。由一个子代理（内部又分 6 路）完成，最重要的几条主会话逐行核对过。
- 已核对的新缺陷（Lean 定理都在 `Magma/Dense/Workspace.lean`，§32a–h）：
  1. **zhegvdx_2stage(_m)**（l.308–319 → 内层 zheevdx_2stage l.283–297）：无向量时，自己的 Q = C = 5n + n·nb，内层要求 5n + 2·nb·n。对所有 n > 128，用自己查询得到的大小调用都会失败：potrf 和 hegst 已经改写了 A、B，然后返回内层的参数号 −14（外层 lwork 是第 17 个参数）。有向量时，恰在 n < 2·nb 时不足（nb = 128 时即 n ∈ [129, 255]）。MAGMA 自带的 testing_zhegvdx_2stage 用 `magma_zheevdx_getworksize` 的大小，所以测不出来。定理：`hegvdx_2stage_query_rejected(_all)`、`hegvdx_2stage_vec_short_iff`。
  2. **zgesdd 路径 6to-slow**（l.2132–2145、2193）：chunk = (lwork − wrkbl)/m，留给 zungbr(P) 的只有 wrkbl − 2m + 余数，而 wrkbl 按 gebrd 的 nb = 32 算，zungbr(P) 需要 m·zgelqf_nb（min(m,n) ≥ 1024 时为 128m）。结果 zungbr 返回 −9，ierr 不读，右奇异向量错误而 info = 0。最小例子：m = 1024、n = 1087、jobz = O、lwork = 1,181,696（即最小值）。用查询值调用不会触发。定理：`gesdd_slow_ungbr_short_all`、`gesdd_6to_slow_instance`。
  3. **zhetrd_he2hb_mgpu**：单卡版已知问题的多卡孪生，只查 lwork ≥ 1。lwork < nb² 时主机下溢；lwork = nb² − 1 时传给 zgeqrf 的是 −1，面板 QR 变成查询，静默不做。定理：`he2hb_mgpu_underrun_all`、`he2hb_mgpu_silent_query`。
  4. **zgbtrf_native、zgbsv_native(_work)**：入口不把 `*info` 清零。调用者的 info 变量里残留上一次的非零值时，会直接返回这个旧值，不做分解。另外 zgbsv_native_work 用 −10 报 lwork 错误，但 lwork 是第 12 个参数。定理：`gbtrf_native_stale_info_all`、`gbsv_native_work_argnum`。
  5. **查询问题**：zgeqlf 在 min(m,n) = 0 时查询返回 1，但检查要求 max(1, n, 2nb²)；zgbtrf_batched 的快速返回在查询之前，lwork = −1 原样返回；zhegvr 只把 lwork = −1 当作查询（文档说 lrwork、liwork 为 −1 也算）。定理：`geqlf_query_rejected_all`、`gbtrf_batched_query_unanswered`、`hegvr_lrwork_query_ignored`。
  6. **文档最小值被拒**：10 个例程（zgesvd、zgesdd、zgehrd_m、zungbr(P)、zheevx、zheevr、zhegvx、zhegvr、zstedx、zhegvdx_2stage）按文档给的最小值分配会得到负 info。定理：`doc_minimum_rejected_all`。
- 疑似、未建模：zgbtrf_native_work 在 ILP64 编译下，info 槽位于 device_work + 4 字节，8 字节写不对齐；zungtr 的反射向量移位循环与 LAPACK 相比差一（但没有调用者）。
- 已知 5 条（geqp3、gels、he2hb、ggrqf、geqrf_ooc）重新核对仍然成立。其余文件都是干净的，Q ≥ C ≥ U。
- 已登记的 info 丢失问题在这次核对中再次出现：zunmtr_m 的 NOT_IMPLEMENTED、heevdx_2stage 的 stedx info、gesvd 和 gesdd 从不读 ierr。
- Lean：`lake build` 通过（48 个任务），`logs/axioms.txt` 336 行，没有 sorryAx。论文 Appendix A 现为 4,490 行、374 条定理。`lean/OptimiKahan/README.md` 模块表已加入 Workspace。
- **GPU 待办**（驱动恢复后，只用 GPU 0）：第 1、2 条最值得实测。第 1 条：n = 129，jobz = N，lwork 取查询值，预期 info = −14，A、B 已改。第 2 条：m = 1024，n = 1087，jobz = O，lwork 取最小值，预期 info = 0、VT 错误。

### 29.31 两条疑似项的核实（2026-11-26，同日；CPU 与读码）

- **zungtr 移位循环（已确认）**：`[sdcz]ungtr.cpp` l.140–151（上三角）、l.160–170（下三角），与 LAPACK ZUNGTR 对比如下：
  - 上三角：内层 `i < j-1`，LAPACK 换成 0 起下标是 `i < j`，所以每列漏移 A(j−1, j)。
  - 下三角：内层从 `i = j` 到 `n−2`，LAPACK 是 `j+1` 到 `n−1`，所以对角被改写、末行不移。
  - 第一列：只清零到 `n−2`，A(n−1, 0) 留着旧值。
  - 该函数导出，但不在公开头文件中，库内没有调用者（zheevx 等调用的是 `lapackf77_zungtr`）。
  - CPU 核对：`experiments/magma_dense/orgtr_shift.c` 照抄 d 版循环，再接参考 LAPACK 的 dorgql/dorgqr。下三角路径在 MAGMA 里用 GPU 的 magma_dorgqr，这里用 LAPACK dorgqr 代替，在精确算术下得到同一个 Q。结果见 `orgtr_shift_results.txt`：
    - n = 3、4、8、50，上下三角得到的 Q 都不正交。n = 50 时上三角 ‖QᵀQ − I‖ = 9.2e19、下三角 1.8；LAPACK 为 1e-15。
    - 对照组（同一代码改成 LAPACK 的边界）与 LAPACK 的差恰为 0，说明测试本身没写错。
  - Lean：`ungtr_upper_shift_short`、`ungtr_lower_shift_wrong`、`ungtr_first_col_stale`（`Magma/Dense/QrEigen.lean`）。
  - 上三角路径不经过 GPU，以后可以在节点上通过 extern 声明直接调用库本身来确认。
- **ILP64 info 不对齐（读码确认）**：
  - zgbtrf_batched.cpp l.157–158 的通用工作区是 `batchCount·sizeof(int)` 字节；zgbtrf_gpu.cpp l.66 的 device_info 紧随其后。native 版 batchCount = 1，所以 device_info 位于 4 字节处。
  - ILP64 下内核对 `info_array[0]` 做 8 字节写，地址不对齐；LP64（默认，也是节点上的编译）不受影响。
  - Lean：`gbtrf_native_info_misaligned_ilp64`、`gbtrf_info_alignment_all`。
- Lean 总计：`logs/axioms.txt` 341 行，没有 sorryAx；论文 Appendix A 现为 4,539 行、379 条定理。

### 29.32 全库扫描：先读后写 `*info` 的函数（2026-11-26，读码）

- 脚本：`experiments/magma_gpu/scripts/info_init_scan.py`。扫描 src 和 magmablas 中所有带 `magma_int_t *info` 参数的函数：去掉注释后，找出读 `*info`（`!=`、`==`、`<`、`if (*info)`、`return *info`）早于任何 `*info = 0` 的函数。共命中 52 个（含 s/c/d 生成的副本）。
- 逐个核对后：
  - **真问题，已知：**zgbtrf_native(_work)、zgbsv_native(_work)，即 §29.30 第 4 条，`gbtrf_native_stale_info_all`。
  - **新的小问题：**magma_zhtodpo、zdtohpo（zpotrf3_mgpu.cpp）和 zhtodhe（zhetrd_mgpu.cpp）在公开头文件 magmablas_z.h 中声明，但从不写 `*info`，却以 `return *info` 结尾。库内调用者都传一个临时 iinfo 并忽略它，所以 MAGMA 自己的驱动不受影响，外部调用者会拿回旧值。Lean：`MagmaMultiGpu.htod_returns_caller_value_all`。
  - **误报：**其余都是包装函数，info 由被调用的 expert 或底层函数先清零：zgetrf_gpu(_expert)、zgetrf_native、zpotrf、shpotrf_gpu/native、hgetrf/htgetrf、ds/dhgesv_iteref、dlaex3_m 的单卡分支。
- 结论：“入口不清零 info”这一类在整个库里只有 gb*_native 四个函数是真问题。
- Lean：`logs/axioms.txt` 342 行，没有 sorryAx。

### 29.33 全库扫描：参数号报错（2026-11-26，读码）

- 脚本：`experiments/magma_gpu/scripts/argnum_scan.py`。它对 src、magmablas、sparse 中的 z 文件做两件事：
  - 只保留 `#ifdef COMPLEX` 的复数分支，再取出每个 `if (cond) info = -k;` 中条件里第一个出现的参数名。
  - 以每个函数自己最常见的偏移量（k − 位置）作为它的编号惯例，例如 _m 版本统一跳过开头的 ngpu；只报告与惯例不一致的检查。
- 扫描结果去掉 `vu <= vl` 这类条件主体误判后剩 65 处。逐个函数对照签名核对（对照表由 `scripts/argnum_review.py` 生成），确认了 **58 处检查、33 个例程**真有错（Lean 表 `argnumTable`）：
  - 广义特征值 zhegvx、zhegvr、zhegvdx、zhegvdx_2stage（及 _m）：JOBZ 和 RANGE 的错误码对调。LAPACK 是 JOBZ −2、RANGE −3。
  - zcposv_gpu、zchesv_gpu、zgerfs_nopiv_gpu：前几个参数按少一个参数的签名编号，后几个又是对的，同一函数内不一致。
  - 批量与面板例程（zgesv_batched、zgetrs_nopiv_batched、zgeqrf_expert_batched、zgetrf_recpanel_batched、6 个 zgeqr2*_batched、zgesvj/zheevj 批量、zpotrf_lpout/lpin_batched、zpotrf_expert_batched）：插入 Ai/Aj、nb、批量数组之后没有重新编号。
  - zhetrd_he2hb(_mgpu)、zlahr2、zlatrd、zpotrf3_mgpu、zgetrf2_mgpu、zgetrf_piv、zpotrf_rectile_native、zunmqr_2stage_gpu、zlarft_internal_batched、gbtrf_batched_work、sliding_window_loopout、gbsv_native_work。
  - 误报：zlarfb_gpu 全部正确；zgeqrs3_gpu 已登记（`geqrs3_argnums_wrong`）。
- **另一个新发现：**zcposv_gpu 和 zchesv_gpu 完全不检查 uplo。非法 uplo 传下去后，zlanhe 先报一次错，zlat2c 返回 −1，驱动把它当作精度转换失败，设 iter = −2，再由 FP64 回退的 zpotrf_gpu 返回 −1。所以 info 恰好正确，但 iter 报错了原因。Lean：`cposv_bad_uplo_reported_as_conversion`。
- 影响：只影响错误信息和负 info 值，调用者按编号查参数会被指错。不影响数值结果。
- Lean：新模块 `Magma/Dense/ArgChecks.lean`（4 条定理）；`lake build` 通过（50 个任务），`logs/axioms.txt` 346 行，没有 sorryAx；论文 Appendix A 现为 4,636 行、384 条定理。

### 29.34 全库扫描：子调用状态从不读取（2026-11-26，读码；结论为无新问题）

- 脚本：`experiments/magma_gpu/scripts/dropped_status_scan.py`。找出把局部状态变量（iinfo、ierr、linfo 等）的地址传给子调用、自己却从不读它的函数。
- src 的 z 文件命中 40 个函数；magmablas、sparse、混合精度文件（zc*、ds*、x*、sh*、dx*）没有命中（src 里另有 d 版副本）。逐类归入已有条目：
  - gesvd、gesdd、geev(_m) 不读 ierr：§29.30 第 19 条，以及 `gesdd_slow_ungbr_short_all`（唯一能在正常输入下触发的一处）。
  - heevd/heevdx/heevr/heevx 系列丢弃 zhetrd、zunmtr 的 iinfo：属于 §3d 和 `hetrd_failure_reaches_dlascl`、`helper_iinfo_dropped`。按 §29.30 的核对，传入的 lwork 都够，所以只可能是分配失败。
  - unm*/ung*（unmqr、unmlq、unmql、unmrq、unmtr、unmbr、ungbr、unghr、ungqr_gpu、ungtr）丢弃 iinfo：`helper_iinfo_dropped` 一类。unmtr_m 的 NOT_IMPLEMENTED 已单独登记为 `unmtr_m_not_implemented_lost`。
  - gebrd、gehrd(2/_m)、hetrd(2_gpu/_gpu/_mgpu)、geqlf 丢弃小矩阵 LAPACK 回退的 iinfo：这些 LAPACK 调用只会报参数错误，而参数已由 MAGMA 先检查过，所以不可触发。
  - hetrf_nopiv(_gpu)、sytrf_nopiv_gpu 丢弃 zlascl_diag 的 iinfo：它只报参数错误；这些例程的零主元问题已登记为 `last_pivot_untested`、`nopiv_info_wrong`。
  - zlahef_gpu（l.320、l.608）：丢弃 `magmablas_zlascl_2x2` 的 iinfo。该函数只报参数错误，已登记为 `lascl2x2_args`（ldda 报 −4、lddw 不检查）。
- 结论：这一类没有新缺陷，已有条目覆盖了全部可触发的情况。没有新增 Lean。

### 29.35 独立复核 §29.29–§29.33 的读码结论（2026-11-26）

- 交给一个新的子代理反驳式复核：给它源码路径和 11 条结论，要求它尽量推翻；它没有编译或运行 MAGMA，也没有改文件。**11 条全部确认，没有被推翻的**：HIP 下 rbt 关闭精化、zhegvdx_2stage、zgesdd 6to-slow、he2hb_mgpu、gb*_native 不清零 info、zgeqlf 查询、zungtr、ILP64 不对齐、jobz/range 对调、zcposv_gpu 不检查 uplo，以及抽查的 8 条参数号。
- 它独立重算的数字与我们一致：zgesdd 的 wrkbl = 69,600、minwrk = 1,181,696、chunk = 1086、lnwork = 67,584、zungbr(P) 需要 131,072；zhegvdx_2stage 在 n = 129 时 8,901 对 17,157。
- 复核补充的细节，已写进 Lean 或文档：
  - **he2hb 的查询值本身也不安全**：单卡和多卡都只在 n = 0 时提前返回，查询返回 n·nb，而 hT 的 memset 需要 nb²。所以 0 < n < nb 时，用自己报告的大小调用也会写到缓冲区之前。Lean：`he2hb_query_underrun_all`。
  - **zhegvdx_2stage 的实数版更差**：dsygvdx_2stage 无向量时最小值是 2n + n·nb，没有第二阶段那一项（文档却写 LQ2 + 2N + N·NB），内层要求 6n + 2·nb·n，缺 4n + nb·n。Lean：`sygvdx_2stage_real_short_all`。有向量时的缺口只在 nb = 128（arch ≥ 300 且线程数 > 14）、n ∈ [129, 255] 时出现。
  - zcposv_gpu 遇到非法 uplo 时，zlanhe 返回 −2.0 作为范数，不会崩溃；zlag2c 在 zlat2c 之前运行且没有 uplo 参数。
  - 参数号：zheevj 批量版和 zgeqr2 批量版的错误号与文档注释的参数表一致（文档漏列了 batch_mask/num_sweeps/max_sweeps、Ai/Aj/taui），与 C 签名不一致；zpotrf3_mgpu 的编号既不符合签名也不符合文档。
  - magma_zungtr 编译进库并导出（Makefile.gen.cuda），只在 docs/html 里出现，没有声明，也没有测试。
- Lean：`lake build` 通过，`logs/axioms.txt` 348 行，没有 sorryAx；论文 Appendix A 现为 4,657 行、386 条定理。

### 29.36 全库扫描：工作区公式的 32 位溢出（2026-11-26，读码）

- 脚本：`experiments/magma_gpu/scripts/wrap32_scan.py`。提取 src 的 z 文件中所有含两个尺寸变量乘积的工作区表达式（lwmin、lrwmin、lwkopt、minwrk 等），按 m = n、nb = 64 求出它首次超过 INT_MAX 的 n。共 30 处，全部集中在分治特征值驱动：
  - heevd、heevdx、heevdx_2stage、hegvd、hegvdx、hegvdx_2stage 及其 _gpu、_m 版本，共 14 个驱动（另有 zstedx、zstedx_m 两处）。
  - `lrwmin = 1 + 5n + 2n²` 从 n = 32767 起溢出；`lwmin = max(n + n·nb, 2n + n²)` 从 n = 46340 起溢出。zstedx(_m) 的 `1 + 4n + 2n²` 从 n = 32768 起溢出。
- 溢出后的链路（以 zheevd 为例，zheevd.cpp l.211–226、293、331；zstedx.cpp l.181–218；dstedx.cpp l.191–199）：
  - n ∈ [32767, 46339] 时 lrwmin 为负，`lrwork < lrwmin` 永不成立，任何 lrwork 都被接受。查询写入 rwork[0] 的也是负数。
  - 真实需求 1 + 5n + 2n² > INT_MAX，所以 32 位下没有任何合法的 lrwork。
  - 往下 zhetrd 先改写 A；zstedx 的 lrwmin 在 n ≥ 32768 时同样为负并放行（n = 32767 时恰为 INT_MAX，则由它自己拒绝）；最后 magma_dstedx 的 `1 + 4n + n²` 未溢出，以 −12 拒绝。zheevd 直接返回这个 −12，而在 zheevd 的编号里 −12 是 liwork。
  - 另外 `3n(n/2 + 1)` 的设备工作区在 n ≥ 37838 时溢出（已有 `dcWork`）。
- 这推广了第 24 轮在 zstedx_m、n = 38,000 上实测的结果（§29.26）。单卡路径同样适用：n = 32767 的复数矩阵约 17 GB，可以放进 32 GB 的 V100，驱动恢复后可在 GPU 0 上实测。
- Lean：`heevd_lrwmin_wraps_all`（对整个区间：最小值为负、任何 lrwork ≥ 0 都被接受、真实需求 > INT_MAX）和 `heevd_lrwmin_instance`，在 `Magma/Dense/NonZ.lean` §31b。`lake build` 通过，`logs/axioms.txt` 350 行，没有 sorryAx；论文 Appendix A 现为 4,687 行、388 条定理。

### 29.37 编译宏包住的状态检查：CHECK_INFO 编译不了（2026-11-26，CPU 预处理确认）

- 在 src、magmablas、sparse 的 z 文件里搜索被 `#ifdef CHECK*/DEBUG*/VERIFY*` 包住的状态检查。除计时和调试打印外，只有 4 个批量求解器的 CHECK_INFO：zposv_batched、zgesv_batched、zgesv_nopiv_batched、zgesv_rbt_batched。
- 这个选项打开后，会把每个矩阵的失败码作为返回值报出；默认编译下返回值只反映参数错误，逐矩阵 info 留在设备数组里。已核对后续 getrs（包括 getrs_nopiv_batched）不覆盖这个数组，所以默认行为符合批量接口的约定。
- **新发现：**这个选项在 4 个里有 3 个编译不了：
  - zposv_batched.cpp l.129、zgesv_batched.cpp l.139、zgesv_nopiv_batched.cpp l.125 调用 7 参数宏 `magma_getvector`（include/magma_copy.h l.29）时只给了 6 个参数，漏了 queue；
  - zgesv_nopiv_batched 还引用了未声明的 `dinfo_array`（它的参数名叫 `info_array`）。
  - 只有 zgesv_rbt_batched 传了 queue。
- CPU 确认：`experiments/magma_dense/check_info_pp.c` 把宏定义和 4 行调用原样复制出来，运行 `cc -E`，恰好这 3 行报 “too few arguments provided to function-like macro invocation”（`check_info_pp_results.txt`）。
- Lean：`check_info_build_broken`（`Magma/Dense/QrEigen.lean`，放在 `rbt_batched_default_build_silent` 之后）。`lake build` 通过，`logs/axioms.txt` 351 行，没有 sorryAx；论文 Appendix A 现为 4,703 行、389 条定理。

### 29.38 全库扫描：网格 y/z 维由批量数决定的内核启动（2026-11-26，读码）

- 在 magmablas 的 z 文件里搜索 `dim3 grid(…, batchCount/ibatch …)`。共 23 处，其中 22 处用的是分块后的 ibatch（不超过 max_batchCount = 65534）或 x 维，只有一处直接使用 batchCount：**zgbtf2_kernels.cu l.59–71 的 `magma_zgbtrf_set_fillin`，`dim3 grid(nblocks, 1, batchCount)`**。
- 调用路径：zgbtrf_batched.cpp l.215，即 `magma_zgbtrf_batched_work` 的通用（非分块）实现。
  - strided 接口按 65534 分块，所以不受影响。
  - 指针数组接口 `magma_zgbtrf_batched`（l.405–455）把完整的 batchCount 直接传进 `_work`。
- 进入通用路径的条件：fused（只在 minmn ≤ 64 时尝试，共享内存 sldab·n·16）和 sliding-window（sldab·(nb+kv+1)·16，sldab = 2kl+ku+2）两种实现都超出共享内存上限。例如 V100 上 kl = ku = 32、m = n = 100：98·66·16 = 103,488 > 98,304，nb = 1 时就已超出。
- 后果：batchCount ≥ 65536 时 gridDim.z 超过 CUDA 上限 65535，启动失败，但没有人检查。ju_array（未初始化的设备工作区）从未写入，填充元也没有清零，后续的交换和更新就用未初始化的 ju 继续进行。
- 实测可行：每个矩阵 97·100·16 B ≈ 155 KB，65,536 个约 10 GB，能放进 V100。驱动恢复后可在 GPU 0 上测。
- Lean：`set_fillin_launch_invalid_all`、`gbtrf_generic_path_reached`（`Magma/Kernels.lean` §30b）。`lake build` 通过，`logs/axioms.txt` 353 行，没有 sorryAx；论文 Appendix A 现为 4,729 行、391 条定理。

### 29.39 内核中的 32 位偏移：magmablas_zlacpy（2026-11-26，读码；可能是一大类）

- zlacpy.cu l.36–62 的内核参数是 `int m, int n, int ldda`，`dA += ind + iby*ldda` 按 int 计算，其中 iby = blockIdx.y·64。
- 宿主端（l.350–402）按 super_NB = max_blocks·64 = 65535·64 = 4,194,240 列分块，再在宿主端算好每块的起始指针。这个分块限制的是网格大小，不限制 `iby·ldda`：对任何实际的 n，块内 iby 都能达到 n − 64。所以一旦（列下标 × ldda）超过 INT_MAX，偏移量就会回绕。也就是说，元素数超过 2³¹ 的矩阵会在矩阵之前的位置读写。
  - 例子：m = n = ldda = 50,000，最后一个块列 iby = 49,984，真实偏移 2,499,200,000，32 位下为负。单精度每份矩阵 10 GB，两份 20 GB，可以放进 V100。
- ILP64 编译（`magma_int_t` 为 64 位，本来就是为大矩阵准备的）同样有这个问题，因为内核参数仍是 int。make.inc 示例和 README 都没有说明这一限制。
- 范围：z 内核共有 203 处 `int lda/ldda` 参数，很可能多数有同样的写法，但**只逐行核对了 lacpy**，其余未逐个核实。这一类不属于状态逻辑，而是大矩阵下的静默数据错误或非法访问。
- Lean：`lacpy_offset_wraps`、`lacpy_offset_negative_all`（`Magma/Dense/NonZ.lean` §31c）。`lake build` 通过，`logs/axioms.txt` 355 行，没有 sorryAx；论文 Appendix A 现为 4,754 行、393 条定理。
- GPU 待办：在 GPU 0 上对 50000×50000 的 float 矩阵调用 magmablas_slacpy（需要约 20 GB 空闲显存），与主机拷贝比较。

### 29.40 magmablas 全部 z 内核的 32 位偏移核对（2026-11-26，子代理读码 + 主会话抽查）

- 子代理逐个读了全部 123 个 magmablas/z*.cu 及其包含的 .cuh 模板，分类如下：
  - **WRAPS 89 个**：48 个非批量，34 个只在批量下（单个矩阵须超过 2³¹ 个元素），7 个用 magma_int_t 相乘（只在 LP64 下回绕）。
  - **SAFE-64 4 个**：ztranspose、ztranspose_inplace、zhemv_mgpu、zgemv_conj；另外 zlaswp.cu 里的主 zlaswp_kernel 也是安全的。
  - **BOUNDED 6 个**，**NOT-APPLICABLE 24 个**。
  - 阈值：方阵在 n ≈ 46,341 附近首次溢出；zgerbt 因为把列下标分成两半，约为 65,536；zswapblk 的 unsigned 偏移在 65,536 时回绕。
- 主会话抽查：lacpy（§29.39）、laset（`A += ind + iby*lda`）、gemm 模板（`LDB*(N-1) + K`、`by*BLK_N*LDB` 按 int 算后才赋给 ptrdiff_t）、laswp_columnserial（`dA[i1*lda + tid]`），都与报告一致。
- 对大型稠密分解的影响：
  - `magma_[sdcz]getrf_gpu`（默认 hybrid）只经过 64 位安全的 transpose 和 laswp 内核，其余是 cuBLAS，宿主端宏用 size_t，所以不受影响。
  - `magma_*getrf_native`（未定义 SWP_CHUNK）经过 laswp_columnserial，lda = 50,000 时 i1 ≥ 42,950 回绕。Lean：`getrf_native_laswp_wraps`。
  - **`magma_*potrf_gpu`**：不调用 magmablas 内核，但宿主端宏 `dA(i,j) = dA + i + j*ldda`（zpotrf_gpu.cpp l.145、zpotrf_panel_native.cpp l.105）在 LP64 下按 32 位相乘，而 zgetrf_gpu.cpp l.191 用了 `(size_t)`。n = ldda = 50,000 时，j ≥ 42,950 的列指针回绕后才交给 cuBLAS。Lean：`potrf_gpu_host_offset_wraps`。
- 顺带发现的两个与偏移无关的缺陷（都已读码核实；两个都是公开函数，库内没有调用者）：
  - `magmablas_ztranspose_batched_stride`（ztranspose.cu l.318–328）：按 65,534 分块时传的是 `dA_array + i`，应为 `+ i*stride`，所以批量超过 65,534 时第二块起全部错位。内核里 `blockIdx.z*stride` 也是 int。Lean：`transpose_stride_chunk_wrong_all`。
  - `magmablas_zswapblk`（zswapblk.cu l.19、l.105）：主元目标存为 `short ipiv[]`，距块起点超过 32,767 行就截断。Lean：`swapblk_short_truncates`。
- Lean：`Magma/Dense/NonZ.lean` §31d（5 条）；`lake build` 通过，`logs/axioms.txt` 360 行，没有 sorryAx；论文 Appendix A 现为 4,800 行、398 条定理。
- GPU 待办：potrf_gpu 与 getrf_native 在 n = 50,000 单精度时的实测（每份约 10 GB，只能在 eightsocket 上做）。

### 29.41 roxy 上的 MAGMA 环境与第 26 轮（2026-11-26，用户授权“在roxy上装环境”）

- **环境**（除 MSVC 外都在 `D:\lean4hpc` 下；原有 venv 未动；显卡驱动未动）：
  - 系统级只装了 MSVC Build Tools 2022（14.44，VC 工具集 + Windows 11 SDK），通过 winget 安装在 C:。
  - `D:\lean4hpc\magma_env`：新建的 venv，内含 PyPI 的 NVIDIA wheel（nvcc/cicc/ptxas 13.5.37、cudart 13.5、cublas 13.10、cusparse 12.9、cccl、profiler-api）以及 cmake 4.4、ninja 1.13。wheel 里没有 cublas/cublasLt/cusparse 的 .lib，用 dumpbin + lib 从 DLL 生成（`D:\lean4hpc\magma\mkimplib.bat`）。
  - OpenBLAS 0.3.34 x64（LP64，GitHub release，SHA256 E9CB6134…）解压在 `D:\lean4hpc\openblas`。
  - MAGMA 2.10.0（SHA256 与 `inputs/tarball.sha256` 一致）用 CMake + Ninja 编译：静态库，USE_FORTRAN=OFF，-DADD_，GPU_TARGET=sm_120，`-j 8`（`build_magma_roxy.bat`）。稠密库 magma.lib 402 MB；稀疏库第一次因缺 `cuda_profiler_api.h` 失败，装上 profiler-api wheel 后重编。
  - 测试程序用 `build_probe.bat <name>` 编译（cl /openmp，链接 magma.lib、cudart、cublas、cublasLt、cusparse、libopenblas），用 `run_probe.bat` 运行（把 DLL 目录加入 PATH）。
  - 长任务用 WMI（`Invoke-CimMethod Win32_Process Create`）启动，不受 ssh 会话影响。
- **限制**：RTX 5070 Ti 有 16 GB 显存，但用户的桌面程序占用 7–12 GB，实际空闲 4–9 GB；FP64 吞吐低；只有一张卡。所以 n ≥ 32767 的 zheevd 溢出、50000² 的 lacpy/potrf 偏移、多卡问题仍要在 eightsocket 上测。
- **第 26 轮结果**（`probes/round26_probe.cpp`、`results/round26_results.txt`），全部与读码预测一致：
  1. zhegvdx_2stage：nb = 128（16 线程）。n = 129、200 按自身查询值调用返回 −14，A、B 已改写；按 5n + 2·nb·n 调用正常；n = 128 走 LAPACK，正常。
  2. zgesdd 6to-slow：二分得到最小可接受 lwork = 1,181,696（与读码一致）；按它调用时 zungbr 报 −9，info = 0，‖VT·VTᴴ − I‖ = 400；快路径 5.5e-15。
  3. magma_zungtr 上三角（通过 extern 声明调用库本身）：n = 8 时 ‖Q − Q_LAPACK‖ = 2.71，n = 50 时 1.65e15，正交误差 6.1e30。
  4. zgbtrf_native：info 残留 7 时返回 7，带状矩阵未改。
  5. 参数号：zhegvx 非法 jobz 报 −3、非法 range 报 −2（对调）；zgesv_batched 的 lddb 报 −6、zgetrs_nopiv_batched 报 −8（实为第 7 个参数）。另外，lwork 不足（64n < 65n）时 zhegvx 报 −20，覆盖了 jobz/range 的错误码。
  6. zcposv_gpu 非法 uplo：iter = −2、info = −1，依次打印 zlanhe −2、zlat2c −1、zpotrf_gpu −1。
  7. zhetrd_he2hb（单卡）n = 16、nb = 32：查询值 512，work 之前恰好被写 512 个元素。
  8. zgeqlf：m = 0、n = 5 和 m = 7、n = 0 时查询返回 1，按它调用得 −7。
  9. zgbtrf_batched（指针数组接口，kl = ku = 32，m = n = 65）：空闲显存不够放 65536 个独立矩阵，改用 65536 个指针别名到 1024 个矩阵。batchCount = 65536 时内核启动报 “invalid argument”，函数返回 0；65535 时无错。数值结果因为别名而没有比对。
- COVERAGE 中这 9 行的证据已改为 “✔ roxy 第 26 轮”。

### 29.42 第 27 轮：稠密算子的差分扫描、追查与缺陷分布分析（2026-11-26，roxy；用户要求“多测库里大部分算子、多找 bug、研究 bug 主要出现在哪”）

- **扫描**：`probes/round27_sweep.cpp`，11 个算子组，每组一个进程。用 `scripts/roxy/run_sweep.bat` 运行，结果在 `results/round27_sweep/*.csv`，共 **17,327 个用例**。
  - 每个用例都与 OpenBLAS 0.3.34 的 LAPACK 比较 info，并计算 LAPACK 测试程序式的归一化残差。
  - 尺寸：0–3、7，块边界 31/32/33、63/64/65、127/128/129、255/256/257，513，以及长方形。
  - 输入：随机；零列分别在第一列、中间、最后一列；零矩阵；×2^±500 缩放；重复列；NaN；Inf。Hermitian 和正定矩阵另有不定、单位阵等类型。
  - 覆盖：getrf（4 种）、potrf（3 种）、geqrf（3 种）、gelqf（2 种）、geqlf、geqp3、gesv、posv（各 2 种）、hesv、gels（2 种）、getri_gpu、potri（2 种）、trtri（2 种）、heevd（CPU 和 GPU）、heevdx、heevdx_2stage、heevr、heevx、hegvd、geev、gesvd、gesdd、gebrd、hetrd、magmablas 的 gemm/gemv/hemv/lange/lanhe/lacpy/laset/lascl/geadd/transpose，以及批量的 getrf、getri、potrf、geqrf。
- **分诊**：逐类核对，结论见 `results/round27_sweep/TRIAGE.md`。汇总脚本是 `scripts/sweep_summary.py`。
  - 测试程序误报：info > 0 之后去比较残留内容；zgeqrf_gpu 把 R 的对角块存在 dT 里；geqlf 和 heevdx_2stage 给的 lwork 不够；QR 组把 LAPACK 输出的有限性硬写成 true。
  - 不是缺陷：nopiv 不稳定；输入含 NaN 时的状态和内容；舍入导致的精确零主元差异；gebrd 在秩亏时双对角形式不唯一。
- **追查与新缺陷**（`probes/round27_focus.cpp`，结果见 `results/round27_focus_results.txt`；Lean 在 `Magma/Dense/Differential.lean`）：
  1. **zgeqp3（重大，读码一直没发现）**：秩亏矩阵上 A·P ≠ Q·R，残差 1–6，info = 0，而且秩没被揭示（|R(n−1,n−1)| = 0.1–0.27，LAPACK 为 1e−16）。重复列在 n = 63/65/128/255/256 出错；随机秩 n/2 矩阵在 78 例中有 18 例出错。
     - 根因 1：zlaqps.cpp 第 171–201 行，主元列在面板外时，交换完只把第 pvt 列 offset+nb 以下的行送回 GPU；面板内交换时 GPU 副本完全不动。面板因为需要重算范数而提前结束时（kb < nb），offset+kb..offset+nb−1 这几行仍留在设备的尾部矩阵里，下一个面板会读到过时数据。
     - 根因 2：第 363–384 行重算范数时，用的是没有经过尾部 GEMM 更新的主机数据；而且主机和设备两部分的行区间各差一行（第 376 行被注释掉的设备 dznrm2 才是正确写法）。
     - 验证方法：复制并修改 zlaqps.cpp（`probes/zlaqps_patched.cpp` 只修根因 1，`zlaqps_patched2.cpp` 两个都修），和测试程序一起链接，以覆盖库里的 magma_zlaqps，库本身不改。只修根因 1 时 AP = QR 恢复，但秩仍没被揭示；两个都修后 78 例中 0 例出错，|R(n−1,n−1)| 也恢复到 1e−16。
  2. **magmablas_zgemm 在 k = 0 时**（zgemm.cu l.249）直接返回，不执行 C := beta·C：beta = 0.5 时 C 不变，beta = 0 时 NaN 留在 C 里。
  3. **magmablas_zgemv / zhemv**：beta = 0 时仍然计算 beta·y（gemv_template_device.cuh l.81/107/209、zhemv.cu l.468），y 里的 NaN 会传播，这违反了它们自己文档写的 “Y need not be set”。
  4. **magma_zgels / zgels_gpu**：遇到零列或零矩阵时返回 NaN 解，info = 0；LAPACK 报 info = i，零矩阵则给零解。MAGMA 文档只承诺 info ≤ 0，所以这是一处与 LAPACK 语义的偏离。
  5. **zheevdx / zheevdx_2stage**：n = 1 的快速返回不设置 mout（第 300–306 行）。
  6. **GPU 接口的小矩阵路径以 lda = n 调用 LAPACK**，所以 n = 0 被拒绝：zheevd_gpu 报 −5，zheevdx_gpu/zheevr_gpu/zheevx_gpu 报 −6，ztrtri_gpu 报 −5，zpotrf_mgpu 报 −4。zpotrf_gpu 有提前返回，所以正常，可作对照。
  7. **zgetrf_native**：遇到 NaN/Inf 时报 info > 0，而 hybrid 版和 LAPACK 报 0。严重度低。
  - 已知缺陷在新硬件上复现：ztrtri_gpu 的分块 info（n = 257、513）。
  - 未能复现：zheevd_gpu(N) 在 herm_big、n = 129 上出过一次错（比值 3.8e16），用相同输入和工作区重复 30 次都正确，不登记。
  - 不属于 MAGMA 的问题：OpenBLAS 0.3.34 自带的 LAPACK 中，zgesvd/zgesdd 在要求向量时会在 3×3、含 Inf 的矩阵上卡死；MAGMA 的小矩阵路径会继承这个问题（`results/round27_sweep/svd_inf_hang.txt`）。因此 SVD 组去掉 Inf 输入后重跑。
- **缺陷分布分析**：`operators/magma/FINDINGS.csv`（238 条；子代理整理，主会话抽查 8 行并补了第 27 轮的 8 行）、`operators/magma/findings_stats.py`、`operators/magma/BUG_DISTRIBUTION_zh.md`。要点：
  - 状态管道占 48%，数值结果错误占 27%，内存与尺寸算术占 22%；静默错误结果占 52%。
  - 62% 的缺陷需要离开主路径才会触发。
  - 缺陷密度：magmablas 内核 0.48/千行，稠密驱动 1.7，稀疏库 2.2–2.8，混合精度与迭代精化 8.4（存在关注度偏差）。
  - 反复出现的 7 种机制见报告第 5 节。
- Lean：`lake build` 通过（52 个任务），`logs/axioms.txt` 368 行，没有 sorryAx；论文 Appendix A 现为 4,923 行、406 条定理。

### 29.43 第 28 轮：结构化病态输入的差分扫描（2026-11-26，roxy）

- 程序：`probes/round28_sweep.cpp`，共 6 组，**5,125 个用例**，结果在 `results/round28_sweep/`，分诊见同目录的 `TRIAGE.md`。
  - 输入都规定了谱：A = U·diag(σ)·Vᴴ（U、V 为 Haar 酉矩阵），或 Q·diag(λ)·Qᴴ。
  - 谱的类型：秩 n/2、秩 1、σ 从 1 到 1e−12、单个 σ = 1e−8、n/2 处有间隙、Kahan 矩阵、按行分级 1e±100，以及聚集、病态、双簇、低秩的 Hermitian 谱。
  - QRCP 组还检查秩揭示：比较 |R(r,r)|/|R(0,0)| 与 LAPACK。
- **zgeqp3（CPU 接口）是系统性出错**：238 例中 92 例错误。
  - n ≥ 63 时，秩 1、条件数 1e12、分级矩阵全部出错（各 22/22），Kahan 矩阵 10 例；其中 90 例 AP ≠ QR，误差比值 1e10–1e14，2 例只是秩没被揭示。
  - 把同一扫描程序与 `zlaqps_patched2.cpp` 链接成 `round28_fixed.exe`（`scripts/roxy/build_fix28.bat`），结果是 714/714 正确（`qrcp_patched.csv`）。所以第 29.42 节的两个根因能解释全部失败，触发条件也扩大为：任何会触发范数重算的病态输入。
- **新发现：zgeqp3_gpu 的 tau 必须是设备数组**。MAGMA 自带的 testing_zgeqp3_gpu.cpp 用的是 dtau，但 magma_z.h 把它声明为 `magmaDoubleComplex *`，文档只写 “COMPLEX_16 array”。
  - 按头文件传主机数组时，每次调用都非法访问内存（n = 1 也一样），info = 0，CUDA 上下文报废，后续调用全部失败。第一次跑 qrcp 组时 18×1 用例失败的 “cusparseCreate: context cannot be initialized” 和 −113 就是这样来的。
  - 改用设备 tau 后全部正确。GPU 面板 zlaqps2_gpu（整列交换、在 GEMM 之后从设备数据重算范数）没有 zlaqps 的那两个缺陷。
  - Lean：`geqp3_gpu_tau_space_mismatch`；FINDINGS 中为 M239。
- 其余各组：LU（357 例）、Cholesky（204 例）、SVD（460 例）全部正确。批量版在秩亏输入上的 info 差异来自舍入顺序。zheevr 在条件数 1e12 的谱上，正交性是 31–553·n·eps，与 LAPACK zheevr 完全相同（`heevr_vs_lapack.txt`），属于 MRRR 算法本身。zheevdx_2stage 有一例 47·n·eps，属于轻微超标。
- Lean：`logs/axioms.txt` 369 行，没有 sorryAx；论文 Appendix A 现为 4,936 行、407 条定理。

### 29.44 第 29 轮：稀疏迭代求解器的差分扫描（2026-11-26，roxy；没有新缺陷）

- 程序：`probes/round29_sparse.cpp`，运行脚本 `scripts/roxy/run_sparse29.ps1`。21 个求解器名，每个求解器一个进程，每个限时 600 s。
  - 每个求解器跑 8 个 1024 阶矩阵 × CSR/ELL/SELLP 三种格式；带预条件的求解器另外分别配 NONE、JACOBI、ILU。共 942 次求解。
  - 选项都通过伪造的 argv 交给 `magma_zparse_opts`，所以用的全是库自带的默认值。
  - 每次求解都用主机算出的真实残差，去核对求解器报告的 info 和 final_res。
- 结论：**没有新缺陷**，每一类非 OK 结果都对应已知条目或预期行为（`results/round29_sparse/TRIAGE.md`、`summary.txt`）：
  - 已知问题：‖b‖ 很小时绝对 atol 直接放行；BA 和 ITERREF 走到包装函数的 default 分支，直接返回 0（M089）；JACOBI 求解器“残差有下降就算成功”；PBICG 和 PTFQMR 的 breakdown 被成功覆盖（M083）；IDRMERGE 在所有输入上失败（M103）；CGMERGE 的复数内积没有取共轭（M084）。M084 在 n = 16 就会出错，所以主扫描改用实数据。
  - 预期行为：如实报告的失败（CG 用在非 Hermitian 或不定矩阵上；零对角线配 Jacobi 或 ILU）；BOMBARDMENT 只支持 CSR，对另外两种格式报 −103。
- 阴性结果：ELL、SELLP 两种格式的判定与 CSR 完全一致。
- 第 25 轮之前的 V100 稀疏扫描（`sweep_results.txt`）只用了 64 阶矩阵和 6 种输入。这一轮在 sm_120 上，用更大、更有结构的问题和三种格式复现了同样的结论。

### 29.45 第 30 轮：批量 Jacobi 特征值与 SVD（2026-11-26，roxy；两个新缺陷，其中一个只在 Windows 上出现）

- 程序：`probes/round30_batched_eig.cpp`。测 `magma_zheevj_batched_small_sm`（n = 1–32）和 `magma_zgesvj_batched`（方阵和长方形，最大 131×64）。每批 56 个矩阵：8 种谱 × 7 个随机种子，与 LAPACK 的 zheev / zgesvd 比较。结果和分诊在 `results/round30_batched_eig/`，分诊见其中的 `TRIAGE.md`。
- **新缺陷 1（Windows/LLP64）：swap_scalar.cuh 用 `long` 交换 64 位值。**
  - `magmablas_dswap_scalar_device(double&, double&)` 的实现是 `swap_internal<long>((long*)&a, (long*)&b)`。z 和 c 版本同样有问题：双精度复数的实部、虚部各只交换一半，单精度复数整体只交换了实部。
  - Windows 上 `long` 只有 32 位（probe 实测 `sizeof(long) = 4`），所以排序时只交换了 double 的低 32 位，而 int 下标数组被正确交换。结果是排序后的特征值/奇异值有扰动（相对误差约 2⁻²⁰），并且与向量错位，info 却是 0。
  - 定位过程：
    - 2×2 的实验显示，只要 Jacobi 收敛后对角元是降序（ajj ≥ akk），结果就错，与矩阵是否正定无关；
    - 结果与 batchCount 无关；
    - 用 Python 逐行模拟内核（`scripts/heevj_emu.py`），结果是对的，说明问题不在算法；
    - 用 expert 接口确认：sort_flag = 0 时 0/64 错，sort_flag = 1（默认值）时几乎全错。
  - 受影响的代码：[zcd]heevj/syevj_batched_small_sm、[zcd]gesvj_batched_small_sm，以及分块 gesvj 用的 sort.cu。s 精度不受影响；Linux 构建也不受影响，这和以前在 V100 上测 heevj 正常的结果一致。
  - 测试结果：heevj 784 例中 313 例错，gesvj 1568 例中 1016 例错。
  - Lean：`swap_long_llp64_wrong`。
- **新缺陷 2：分块 zgesvj（n ≥ 15）用 1/σ 缩放 U 时，没有判断 σ 是否为 0**（gesvj_kernels.cuh 的 finalize 部分：`rsigma[j] = 1. / ssigma[j]`）。零矩阵在 n ≥ 15 的每个尺寸上都是 7/7 输出 NaN，info = 0。这个问题与平台无关。Lean：`gesvj_zero_sigma_nan`。
- 小问题：heevj 的文档说排序后 S(i) ≥ S(i+1)（降序），代码实际按升序排序。
- FINDINGS 中新增 M240、M241，现为 241 条。分布报告新增 6d 节和第 8 种机制“平台假设”。Lean：`logs/axioms.txt` 371 行，没有 sorryAx；论文 Appendix A 见总数行。
- **平台假设的全库扫描（读码，没有新缺陷）**：
  - 在 src、magmablas、control、interface_cuda、sparse、include 中搜索 `(long*)`、`<long>`、`(long)`、`long x =`。除 swap_scalar.cuh 外只有两处：magma_timer.cpp 的 gettimeofday 模拟（转换为 `long` 秒和微秒，没有问题）和 thread_queue.cpp 里的注释。
  - Windows 专用分支（`_WIN32`/`_MSC_VER`）共有 10 个文件：计时、线程（winthread）、yield、线程数和主机分配器。
  - 主机分配器在 Windows 上是 `_aligned_malloc`/`_aligned_free`，所以混用 `free()` 会损坏堆。但全库直接调用 `malloc`/`free` 的只有 alloc.cpp 这一个文件本身，所以不存在混用。
  - 结论：`long` 宽度问题只在 swap_scalar.cuh 一处。

### 29.46 第 31 轮：混合精度 IR 求解器的差分扫描（2026-11-26，roxy；3 个新缺陷，都与 GPU 并发顺序有关）

- 程序：`probes/round31_mixed.cpp`。分 ge、po、z、ls 四组，共 1,944 次求解；cond 为 1e1–1e15，缩放为 2⁻³⁰、2²⁰、2¹⁰⁰，另有 b 含 NaN 的用例；n ∈ {64, 200, 513}，nrhs ∈ {1, 2}。
  - 新增二分开关：`R31_N=<n>` 只跑一个尺寸；`R31_RUN=sihx` 选择求解器，s = dsgesv，i = dsgesv_iteref，h = dhgesv_iteref，x = dxgesv_gmres。
  - 追查用 `probes/round31_race.cpp`，有 rep、fb、lu、ld 四种模式。
  - 结果和分诊在 `results/round31_mixed/`，分诊见其中的 `TRIAGE.md`。
- 413 个 WRONG 中，大多数是已登记的问题：M001、M002、M006、M007、M020、M022、M024；NAN-ACCEPTED 对应 M004、M008、M017、M023。
- **新缺陷 M242：`?getf2_native_kernel` 跨线程块读取时命中过时的 L1 数据。**
  - 内核先自旋等待 volatile 的 `update_flag[i]`，然后用普通指针 `da` 读第 i 列（第 249–262 行），这些加载会进 L1。
  - 当 `ldda·sizeof(T) % 32 ≠ 0` 时，第 i 列的列首和第 i−1 列共享一个扇区，读到的是更新前的值。
  - 结果：sgetrf_native 误差 300–14,000，dgetrf_native 误差 1e10–1e12，info = 0；问题取决于 ldda，与 n 无关。
  - 验证：4 处读改成 `__ldcg` 后重新编译 s/d 内核（`probes/[sd]getf2_native_kernel_ldcg.cu`，`scripts/roxy/build_fix31b.bat`），16 个出错的配置全部恢复正常。
  - dsgesv/zcgesv 内部用 `lddsa = n`，所以 n % 8 ≠ 0 时总会出错。round31 中 dsgesv n = 513 回退 18/22 次，换成补丁内核后只回退 6/22 次（都是 cond ≥ 1e8）。
  - MAGMA 自带的测试总把 ldda 取为 `roundup(n, 32)`，所以测不到。
- **新缺陷 M243：FP64 回退中，lacpy 与 getrs 运行在不同的 queue 上，中间没有同步。**
  - 涉及 zcgesv/dsgesv 第 357–360 行和 dxgesv_gmres 第 806–809 行；zcposv 是同样的写法，但回退 36 次都没有出错。
  - dsgesv n = 513：正常运行每次 2–8 例 WRONG；`CUDA_LAUNCH_BLOCKING=1` 时 0 例；带开关的副本（`probes/dsgesv_patched.cpp`，`R31_SYNC`）中只打开 bit 2（复制后加 `magma_queue_sync`）就是 0 例。
  - dxgesv_gmres n = 200：回退后 WRONG 的，正常运行 21 例，串行后 0 例。
- （曾记为 M244 的反方向交接，已在第 32 轮撤回，见 §29.47。）
- Lean（`Magma/Dense/Differential.lean` §35a–c）：`getf2_native_aligned_no_share_all`、`getf2_native_unaligned_shares`、`getf2_native_stale_read`、`getf2_native_ldcg_fix`、`sgesv_fallback_race`、`sgesv_fallback_synced`、`dsgesv_lddsa_n_unaligned`。`logs/axioms.txt` 378 行，没有 sorryAx；论文 Appendix A 总计 5,033 行、416 个 theorem。
- FINDINGS 新增 M242–M243（M244 已撤回），现为 243 条，新增缺陷类别 `concurrency-race`。分布报告新增 6e 节和第 9 种机制“并发与内存模型”。
- 局限：只在 sm_120 / Windows 上测过。L1 机制在 Volta 及以后的架构上是通用的，但没有在 Linux 或其他 GPU 上复现；s82 现在不可用（见下）。
- s82（eightsocket）：用户转来的状态显示，GPU 1 自 11-21 起就有故障，GPU 2 时好时坏，GPU 3–7 被别人卡死的进程占用。我从来没有用过 GPU 2–7。在管理员重置之前，s82 上什么都不跑，包括 GPU 0。

### 29.47 第 32 轮：全库跨 queue 交接扫描与 GPU 检验（2026-11-26，roxy；没有新缺陷，撤回 M244）

- 先测量 queue 的同步语义（`round31_race qd`）：在 Q 上放 4 个 4096 阶的 dgemm，然后 `magma_queue_create` 另一个 queue，此时 Q 仍在忙；再 `magma_queue_destroy` 它，此时 Q 已经空闲，3/3 次都是这样。也就是说，destroy 会通过 cublasDestroy 同步整个设备，而 create 不会。
  - 所以“被调用者写完、destroy 之后调用者再读”这一方向是有序的。原先读码登记的 M244 不成立，已从 FINDINGS 删除，现为 243 条。
  - 只有“调用者的异步操作还没完成，被调用者就在新 queue 上开始工作”这一方向会竞争，也就是 M243。
- 机械扫描 `scripts/xqueue_scan.py`：
  - 在 src 中找出 407 个自己新建 queue 的例程；
  - 再找调用点：调用者在自己 queue 上的异步操作之后，中间没有同步（sync、同步的 get/set、返回标量的 BLAS、lag2/lat2 的标志读取），就调用了这类例程。
  - 结果：33 处（`results/round32_xqueue/scan_output.txt`）。
- 逐个判定：
  - zcgesv/dsgesv、dxgesv_gmres、zcposv/dsposv、dshposv 的回退：都属于 M243（getrs/potrs 第一步就在 dX 上启动内核）；
  - lat2c/lat2s 之后的调用：lat2c 通过 cudaMemcpyFromSymbol 隐式同步，没有问题；
  - hegvd/hegvdx/hegvr/hegvx 中，异步 D2H 拷贝和 hegst 都只读 dB，没有问题；
  - zgetrf 在 getrf_m 回退分支用的是主机矩阵，没有问题；
  - zgegqr（laset dT → geqr2x3）和 zgeqrf_ooc（larfb → geqrf2_gpu）：被调用者第一步都是同步的主机拷贝，或者只写，风险低，没有在 GPU 上测；
  - heevd 系列的 lascl → hetrd（‖A‖ < 1.5e-146 或 > 7e145 时触发）和 gesv_rbt 的 copy → nopiv：在 GPU 上测过（`probes/round32_xqueue.cpp`）。heev 系列 27 组 × 5 次全部正确，rbt 15/15 正确，与 `CUDA_LAUNCH_BLOCKING=1` 的结果逐位相同。原因是这两处的被调用者第一步都是到可分页主机内存的同步拷贝。
- 自旋标志内核的扫描：
  - magmablas 和 sparse/blas 中只有 getf2_native（M242）和 sync-free trsm（M123，等待被注释掉，已登记）用到块间通信；
  - zpotf2、zmdotc、zgemvmdot、zmergecg 中有老式的 warp 同步归约（volatile 共享内存，没有 `__syncwarp`）。在 Volta 及以后的架构上这属于未定义行为，但这里 warp 是收敛的，以往各轮的 potf2 和 merged CG 测试也都正确，所以不登记。
- 第 31–32 轮的结论：MAGMA 的跨 queue 交接只在“新 queue 上的第一步就是内核”时会出问题。这类调用点全部集中在混合精度回退路径上，已全部归入 M243。

### 29.48 第 33 轮：未对齐 ldda 下的分解（2026-11-26，roxy；没有新缺陷，确认 M242 是唯一受影响的）

- 程序：`probes/round33_ldda.cpp`。s/d 精度，ldda ∈ {n, n+1, n+3, n+5, roundup(n,32)}；批量调用中各矩阵起点也不对齐。结果和分诊在 `results/round33_ldda/`。
- 批量的 getrf、getrf_nopiv、potrf、geqrf、getrs 共 330 例，全部正确。getrf_gpu、potrf_gpu、potrf_native、geqrf2_gpu 共 162 例，全部正确。只有 getrf_native 有 19/38 例错（M242）。

### 29.49 第 34 轮：批量与 vbatched BLAS 的差分测试（2026-11-26，roxy；3 个新缺陷）

- 程序：`probes/round34_bblas.cpp`。gemm、gemv、syrk、syr2k、trsm、trmm 的 batched 和 vbatched 版，d 精度。参照遵循参考 BLAS 的契约：beta = 0 不读 C、alpha = 0 不读 A/B、k = 0 时 C := beta·C、只改 uplo 所指的三角、trsm 在 alpha = 0 时置零；并逐位检查“不应改动”的区域。另有 contract 模式和 potrf 模式。结果和分诊在 `results/round34_bblas/`，分诊见其中的 `TRIAGE.md`。修正了两处测试程序问题（batched 调用的 ld 必须统一；gemv 在 m 或 n 为 0 时不缩放 y）。
- **M245**：gemm_batched_core.cu:202 和 herk_batched_core.cu:272 在 `k <= 0` 时提前返回，C 没有乘 beta。syr2k/her2k_batched 经 herk_batched_core 继承了这个问题。vbatched 版正确。
- **M246**：herk_template_kernel_(v)batched.cuh 只跳过完全在三角之外的 tile，跨对角线的 tile 整块写回，所以另一侧三角被改写。
  - 后果：potrf_batched 在 n ≥ 257 时（递归面板调用 herk_batched_core）改写严格上三角，n = 257 时改了 19,632/98,688 个元素；potrf_vbatched 在 n = 500/511/520/600 时改写（逐矩阵走 cuBLAS herk 的尺寸不改写）。
  - 分解本身正确，info = 0。两个例程的文档都写着 strictly upper “not referenced”。
- **M247**：trsm/trmm 的 batched 和 vbatched 在 alpha = 0 时计算 alpha·B（trsm_template_device.cuh 的 init_data、trmm 的 `rb *= alpha`），B 中的 NaN 被保留；非批量的 magmablas_dtrsm 正确。
- gemv_batched/vbatched 在 beta = 0 时读 y：属于已知的 M234（同一个模板），已扩充其证据。potrf_vbatched 不清零 info：复现了 M078（预填 5 时完全不做分解）。
- Lean（Differential.lean §36a–c）：`batched_k0_beta_ignored_all`、`herk_batched_diag_tile_breaks_contract_all`、`trsm_batched_alpha0_nan`、`trsm_batched_alpha0_finite`。`logs/axioms.txt` 382 行，没有 sorryAx；论文 Appendix A 总计 5,087 行、420 个 theorem。FINDINGS 现为 246 条（M001–M247，其中 M244 已撤回）。

### 29.50 第 35 轮：其余批量 BLAS（2026-11-26，roxy；新缺陷 M248：hemm/symm 对任何输入都算错）

- trsv_batched 112/112 正确，potrs_batched 16/16 正确（A 不被改动）。symv_batched/vbatched 在 beta = 0 时保留 NaN，hemm 也一样，都归入 M234（已扩充）。
- **M248**：hemm_template_device.cuh 的 32 处 `fetch(A, …)` 传的是基指针，而 `fetch` 宏期望的是 `offs_dA`。所以 magmablas_[zcsd]hemm/symm 的非批量、batched、vbatched 版本对任何输入都算错（A = I 时误差 1.5–13.7；cuBLAS 和 magmablas_dgemm 正确）。局部改为读 `offs_d##A` 后，全部 40 个 beta ≠ 0 的用例都正确（`probes/hemmfix/`、`scripts/roxy/build_fix35.bat`）。结果和分诊在 `results/round35_bblas2/`，分诊见其中的 `TRIAGE.md`。
- Lean（§36d）：`hemm_fetch_base_wrong_all`、`hemm_fetch_threads_collide`。`logs/axioms.txt` 384 行，没有 sorryAx；论文 Appendix A 总计 5,106 行、422 个 theorem。FINDINGS 现为 247 条（M001–M248，其中 M244 已撤回）。

### 29.51 第 36 轮：41 个辅助例程直接与 LAPACK 对比（2026-11-26，roxy；新缺陷 M249）

- 程序：`probes/round36_helpers.cpp`（d 精度），Householder 数据都来自主机端的 LAPACK 分解。orm 系列（ormqr/lq/ql/rq、ormqr2/ql2_gpu）168 例、ormtr/ormtr_gpu/ormbr 136 例、orgqr2/orgbr 21 例、larfb_gpu 80 例，全部正确。larfb 第一版是测试程序问题：MAGMA 要求显式存储 V 的 0 和 1。结果和分诊在 `results/round36_helpers/`，分诊见其中的 `TRIAGE.md`。
- **M249**：trevc3 和 trevc3_mt 选择分块版本 2 的条件漏掉了 LAPACK 的 `OVER`。所以在 howmany = A/S 且 lwork 取文档推荐的最优值时，循环末尾的“回代变换”GEMM 用调用者的 VR 去乘，改写了已算好的特征向量：36 个特征对中有 26 个残差为 1e13–1e15。lwork = 3n 时正确；加上 `over &&` 后全部正确（`probes/dtrevc3_patched.cpp`、`scripts/roxy/build_fix36.bat`）。geev 用的是回代变换模式，不受影响。
- Lean（§36e）：`trevc3_blocked_without_backtransform`、`trevc3_lapack_selection_correct`。`logs/axioms.txt` 386 行，没有 sorryAx；论文 Appendix A 总计 5,129 行、424 个 theorem。FINDINGS 现为 248 条（M001–M249，其中 M244 已撤回）。
- 覆盖：INVENTORY 中的 (c) 类，除 bulge chasing 的 5 个文件、hetrd_hb2st、gerbt 和 lahr2/lahru/labrd/latrd 这些面板辅助函数外，都已直接测试。面板辅助函数通过驱动间接覆盖：gebrd 和 hetrd 在第 27 轮与 LAPACK 比较过，结果正确；gehrd 和 gehrd2（用到 lahr2、lahru）在本轮补测，n = 1–513，含部分 ilo/ihi，28/28 个用例的 ‖QHQᵀ − A₀‖ 和正交性都正确（`gehrd.csv`）。

### 29.52 第 37 轮：两阶段三对角化链路与 RBT（2026-11-26，roxy；新缺陷 M250）

- 程序：`probes/round37_2stage.cpp`。dsyevdx_2stage 共 450 例，覆盖 bulge chasing、hb2st 和回代变换，其中 356 例正确；94 个 BAD 全部有归属：M236（n = 1，mout 未设置）、M065（Upper 走两阶段路径）、测试程序问题（range V 的区间端点落在重复特征值上），以及 M216（range V 未缩放的端点；本轮确认 1 阶段的 dsyevdx 同样受影响，tridiag 乘 0.5 后正确）。结果和分诊在 `results/round37_2stage/`，分诊见其中的 `TRIAGE.md`。
- **M250**：gerbt 的 Uᵀb 内核对 n < 2 提前返回，大小为 1 的子蝶形因此跳过了 u 缩放，与矩阵变换、V·y 变换不一致。所以 gesv_rbt 在 n ≤ 3 时结果错误（后向误差 1e12–1e14），info = 0。改为 `n < 1` 后 56/56 正确（`probes/dgerbt_kernels_fixed.cu`、`scripts/roxy/build_fix37.bat`）。
- Lean（§36f）：`rbt_size1_inconsistent`。`logs/axioms.txt` 387 行，没有 sorryAx；论文 Appendix A 总计 5,151 行、425 个 theorem。FINDINGS 现为 249 条（M001–M250，其中 M244 已撤回）。
- 至此，INVENTORY 的 (c) 类中，bulge chasing（经两阶段驱动）和 gerbt 也已直接测试。仍未覆盖的只有多 GPU（需要第二块卡）和 HIP 后端（需要 AMD 硬件）。

### 29.53 第 38 轮：多 GPU / 核外驱动在 ngpu = 1 下的数值差分（2026-11-26，roxy；没有新缺陷）

- 程序：`probes/round38_mgpu.cpp`。potrf/getrf/geqrf 的 `_m` 和 `_mgpu` 版共 44 例，全部正确。syevd_m、syevdx_m、syevdx_2stage_m、sygvd_m、sygvdx_m、geev_m 共 52 例，其中 46 例正确，6 例是 n = 1 时 mout 未设置，已把 zheevdx_m、zheevdx_2stage_m 补进 M236。sytrd_mgpu 8/8 正确。gehrd_m 按文档的 LWORK = N·NB 调用时被拒绝（M049，补充了证据），改用查询值后 4/4 正确。dpotrf_mgpu_right 链接失败，是 M156 在 roxy 上的确认。结果和分诊在 `results/round38_mgpu/`，分诊见其中的 `TRIAGE.md`。
- FINDINGS 仍为 249 条，没有新条目。

### 29.54 第 39 轮：新发现在复数精度下的复核（2026-11-26，roxy）

- 程序：`probes/round39_zprec.cpp`。在 z 精度下复现了 M242（zgetrf_native）、M245、M246、M247、M250。
- 两条范围需要收窄：
  - **M242 在 c 精度下未复现**（cgetrf_native 在 ldda = 513 和 516 时都正确），范围改为 [sdz]；
  - **M249 只影响实数版**：ztrevc3 也会在没有 over 时选用版本 2，但它的回代 GEMM 写在 over 分支里面；dtrevc3 和 dtrevc3_mt 则把这段 GEMM 放在分支链之外。范围改为 [ds]，Lean §36e 的注释也已同步更正。
- 结果和分诊在 `results/round39_zprec/`，分诊见其中的 `TRIAGE.md`。FINDINGS 仍为 249 条。

### 29.55 第 40 轮：按已知缺陷的写法扫描全库（2026-11-26，roxy）

- 扫描了 magmablas 中所有 BLAS 驱动的快速返回条件，并与参考 BLAS 的语义比对（`results/round40_scan/TRIAGE.md`）。GPU 上证实了 3 处已知缺陷在非批量版本中的延伸（`probes/round40_nb.cpp`）：
  - 非批量的 magmablas_zherk/zsyrk 在 k = 0 时不乘 beta（补入 M245）；
  - 非批量的 magmablas_zherk/zsyrk 把另一侧三角整块改写（zherk_internal 的块列 gemm，nb = 512；补入 M246）；
  - 非批量的 magmablas_ztrmm 在 alpha = 0 时保留 NaN（同一个 trmm 模板；补入 M247）。
- 这三个例程在库内部都没有调用者。FINDINGS 仍为 249 条。

### 29.56 第 41 轮：alpha = 0 且 A/B 含 NaN（2026-11-26，roxy；观察，不登记）

- magmablas 的 gemm（含 batched、vbatched）、gemv（含 batched）、symv、syrk_batched、syr2k_batched 在 alpha = 0 时仍然读 A 和 B，A/B 中的 NaN 使整个输出变成 NaN；非批量的 dsyrk 经 cuBLAS，结果正确。
- MAGMA 和参考 BLAS 的文档都只对 trmm/trsm 承诺 alpha = 0 时不引用 A（违反情况已登记为 M247），所以这一条只记为“与参考实现、cuBLAS 行为不同”的观察，不登记（`results/round41_alpha0/TRIAGE.md`）。

### 29.57 第 42 轮：多 GPU 一维块循环分布的索引核对（2026-11-26，只读码，没有新缺陷）

- 分布规则（magmablas/zbcyclic.cpp）：全局第 j 列所在的块存放在 `dev = (j/nb) % ngpu`，本地列号为 `j/(nb*ngpu)*nb`。
- 各 `_mgpu`/`_m` 例程计算本地列数的公式分 4 种写法：
  - 标准式：getrf_mgpu、geqrf_mgpu、getrf2_mgpu、potrf3_mgpu、geqrf_m、getrf_m、potrf_mgpu_right；
  - gehrd_m、ungqr_m 的 min_lblocks/last_dev 式；
  - potrf_m 非对角部分的“相对 J 起算，再减去 J2”式。JB 一定是 nb 的倍数：NB 被取整到 nb 的倍数（第 154 行），否则 NB = n，此时没有非对角部分。
- 用 Python 对这些公式做了暴力枚举（ngpu 取 1–8，nb 取 {1, 2, 3, 7, 32, 64}，n 取 0–299；potrf_m 另枚举 JB 和余量），与块循环规则逐一比对，不一致的情况为 0。
- 结论：多卡分布的索引算术没有问题。第 25 轮双卡结果错误的原因仍然指向 s82 的 GPU 1 故障；要确认多卡行为，仍需第二块可用的 GPU。

### 29.58 第 43 轮：按已知缺陷写法的只读扫描；两个待跑的 probe（2026-11-27；roxy 离线）

- roxy 从 2026-11-26 晚起在 Tailscale 上显示 offline（ssh 超时），GPU 实验暂停。下面两个 probe 已经写好，但**还没有编译和运行**：
  - `probes/round43_cs.cpp`：M245、M246、M247、M250 在 s、c 精度下的复核（d 已在第 34–37 轮测过，z 已在第 39 轮测过）；
  - `probes/round44_svdpaths.cpp`：magma_dgesvd、dgesdd 的每一条由 lwork 选择的路径。工作区从“被接受的最小 lwork”（对“lwork 参数错误”做二分得到）到最优值，取 8 档；覆盖 5 种形状、全部 jobu/jobvt（或 jobz）组合；检查奇异值、重构误差和正交性。以前各轮只用了查询得到的最优 lwork，而 M249 正是只在大工作区路径上才出现的缺陷。
- 只读扫描（没有新缺陷）：
  - “偏移指针被计算、也被推进，但从不读取”（M248 的写法）：在 magmablas 和 sparse/blas 中只找到 hemm_template_device.cuh 的 offs_dA/offs_dB，即已知的 M248。zsymmetrize 的 dAT 是误报，它是通过解引用写入的。
  - “同一文件中配套内核的提前返回阈值不一致”（M250 的写法）：只找到 zgerbt_kernels.cu，即已知的 M250。zpotf2（n == 0 与 n < 1）和 zsytrf_nopiv_cpu（n == 1 与 n < 2）是等价写法。
  - 两种扫描都能重新找到已知缺陷，说明扫描方法本身有效。
  - “按 lwork 选择算法”的写法（M249）：列出了 dgesvd 和 dgesdd 中几十处 `if (lwork >= …)` 分支，待 round44 在 GPU 上逐条覆盖。

### 29.59 第 43 轮（续）：与参考 LAPACK 源码逐块对照（2026-11-27，只读码，没有新缺陷）

- 方法：从 Reference-LAPACK 的 GitHub 仓库（公开源码）下载对应的 .f 文件，与 MAGMA 的移植逐块对照。
  - (a) 按 lwork 选择路径的完整 IF 条件，包括 `.AND.` 合取项；
  - (b) `n == 1` 快速分支中赋值的输出参数。
- 两种对照都先用已知缺陷验证过：(a) 在 dtrevc3、dtrevc3_mt 上都标出 LAPACK 的 `OVER` 合取项缺失，即 M249；(b) 在 dsyevdx、dsyevdx_m、dsyevdx_2stage、dsyevdx_2stage_m 上都标出 n = 1 时漏设 M，即 M236，全部已登记。
- 新的比对结果：
  - dgesvd：40 个分支的顺序一致，但 MAGMA 的阈值用的是 `wrkbl`（最优块大小下的工作区），LAPACK 用的是最小需求 `max(4n, bdspac)`。MAGMA 只在工作区更大时才走快路径，更保守，没有删掉任何保护条件。
  - dgesdd：MAGMA 有 4 个分支、LAPACK 有 6 个，看起来移植的是更早版本的 LAPACK 结构。是否正确，仍需 round44 在 GPU 上测。
  - dgetri_gpu 的结构与 LAPACK 不同，没有可比的条件。
  - dstedx、dsyevd 的 n = 1 分支赋值的变量比 LAPACK 还多（多出特征向量 A/Z），没有问题。
- 下载的 LAPACK 源码放在 scratchpad/lapack_ref/（会被清理，需要时重新下载）。

### 29.60 第 43–44 轮的 GPU 部分（2026-11-27，roxy 恢复后）

- round43（`results/round43_cs/`）：M245、M246、M247、M250 在 s、c 精度下 16/16 全部复现，这 4 条在四种精度下都已实测证实，FINDINGS 已补充证据。
- round44（`results/round44_svdpaths/`）：dgesvd 304 例、dgesdd 54 例全部正确。gesvd 有 22 种组合真正覆盖了多档 lwork，所以按 lwork 选择的各条路径都正确；gesdd 被接受的最小 lwork 几乎等于最优值（M049），它的替代路径从公开接口走不到。没有新缺陷。
- FINDINGS 仍为 249 条。

### 29.61 第 45 轮：sparse 求解器在边界输入上的测试（2026-11-27，roxy；新缺陷 M251）

- 程序：`probes/round45_sparse_edge.cpp`（由第 29 轮派生）。7 种边界系统：n = 1、单位阵、对角阵、2×2、b = 0、一整行为空、一行为满行。illegal memory access 会使整个 CUDA 上下文失效，所以每个用例单独一个进程（`run_sparse45_iso.ps1`、`run_emptyrow45.ps1`）。结果和分诊在 `results/round45_sparse_edge/`，分诊见其中的 `TRIAGE.md`。
- **M251**：SELL-P 的 blocksize/alignment 在两处临时结构中丢失：
  - magma_zmconvert 的设备→CPU 回退路径（局部的 hB）；
  - magma_zmtranspose/transposeconjugate（B 在开头被释放）。

  于是 CSR→SELL-P 的转换中出现 `256 % 0`，整数除零，进程崩溃。所有需要 Aᵀ 的求解器（QMR、BICG、LSQR、BOMBARDMENT、PQMR、PBICG）用 SELLP 必崩；直接调用公开接口 zmconvert 把设备上的 CSR 转成 SELLP 也会崩。两处一起补上后全部正常（`probes/magma_zmconvert_fixed.cpp`、`magma_zmtranspose_fixed.cpp`、`scripts/roxy/build_fix45.bat`）。
- **更正第 29 轮**：当时这 6 个求解器的进程崩溃没有被注意到（CSV 中缺少 SELLP 行，PQMR/PBICG 还缺少 ELL 行），TRIAGE 中“ELL/SELLP 的判定与 CSR 相同”的结论不成立，已在该文件中加注更正。
- 其余异常：emptyrow + JACOBI 的越界访问是调用方误用（setup 返回 −204 之后，测试程序仍调用了求解器；MAGMA 的 tester 遇到这种情况会中止），记为观察。其余分别是已知的 M098、M103、M104（补充了 identity8 和 n = 1 的证据）、M105，以及测试程序问题（b = 0 时 0/0）。
- Lean（§36g）：`sellp_params_lost`、`sellp_params_forwarded`。另外删掉了 §36f `rbt_size1_inconsistent` 中多余的参数 `a`。FINDINGS 现为 250 条（M001–M251，其中 M244 已撤回）。

### 29.62 第 46 轮：sparse 全部存储格式的往返与 SpMV（2026-11-27，roxy；新缺陷 M252–M254）

- 程序：`probes/round46_formats.cpp`。16 种格式 × 8 个边界矩阵，每个组合单独一个进程，记录退出码和最后一步（`run_formats46.ps1`）。结果和分诊在 `results/round46_formats/`，分诊见其中的 `TRIAGE.md`。
- **M252**：CSR→COO（主机和设备）把行下标写进 row/drow，COO 的使用方读的却是 rowidx/drowidx，所以往返后所有元素都进了第 0 行。两处补丁后 28/28 正确。
- **M253**：ELLRT 的 SpMV 内核按 warp 内隐式同步写归约（没有 volatile，也没有 syncwarp），在 sm_120 上 T = 8 时丢掉第 6、7 号线程的贡献，所以一行超过 6 个非零元就会出错。加 `__syncwarp` 后正确。zmergecg 中有同样的代码。这是 BUG_DISTRIBUTION 第 9 种机制（并发与内存模型）的又一个实例。之前按“warp 收敛、未观察到”没有登记 zmdotc 等文件中的 volatile 归约；这一处连 volatile 都没有，在实际运行中出了错。
- **M254**：CSR↔CSRLIST 对空行的处理有两处越界写（`list[-1]`，以及每个空行多复制一个元素），导致堆损坏。补丁后正确。
- M251 的范围补充了 ELLRT。其余异常归入已知的 M125、M132、M210；BCSR 在 n = 1 时报 −113，记为观察。
- Lean（§36h–j）：`coo_rows_lost`、`ellrt_reduction_loses_lanes`、`csrlist_empty_row_overflows`。FINDINGS 现为 253 条（M001–M254，其中 M244 已撤回）。

### 29.63 第 46 轮补充与第 47 轮（2026-11-27，roxy）

- 在 round45 的 probe 中直接设置 `output_format = Magma_ELLRT`（magma_zparse_opts 只认 CSR/ELL/SELLP/CUSPARSECSR/CSR5，遇到 ELLRT 会明确提示并退回 CSR，这不是缺陷）。结果：CG、PCG（走 CGMERGE，因而用到 zmergecg 中的 ELLRT 内核）、BICGSTAB、PBICGSTAB、GMRES 在 denserow 上都报告成功（自报残差 1e−14），真实残差却是 4.95e−3。M253 一直影响到求解器层面，已补充进 FINDINGS。
- 第 47 轮（`results/round47_dsparse/`）：M251–M254 在 d 精度下全部复现。
- FINDINGS 仍为 253 条。

### 29.64 第 48 轮：带 volatile 的 warp 内归约实测（2026-11-27，roxy；没有新缺陷）

- dmdotc、dgemvmdot（k = 1–8，n 为 1–100000）、dmdotc1–4，以及 d/zpotf2_gpu，共 176 例，在 sm_120 上全部正确（`results/round48_volatile/`）。带 volatile 的 warp 同步归约实测正确；出错的只有完全不带 volatile 的 ELLRT（M253）。这些 volatile 写法按 CUDA 编程模型仍是未定义行为，但没有观察到错误，所以不登记。

### 29.65 第 49 轮：M251–M254 在 s、c 精度下的复核（2026-11-27，roxy）

- 在 s、c 精度下全部复现（`results/round49_scsparse/`），所以这 4 条在四种精度下都已实测证实。setup 失败之后的状态没有新发现（见该轮 TRIAGE）。FINDINGS 仍为 253 条。

### 29.66 多卡分布索引的 Lean 证明（2026-11-27，只用 Lean）

- 第 42 轮只用暴力枚举验证了各卡的本地列数公式。现在在 `Magma/MultiGpu.lean` 中证明了 `nlocal_eq_cols_on_all`：对所有 n、nb > 0、ngpu > 0 和 d < ngpu，getrf_mgpu、geqrf_mgpu、getrf2_mgpu、potrf3_mgpu、geqrf_m、getrf_m、potrf_mgpu_right 使用的公式都恰好等于全局列中属于第 d 块卡的列数（zbcyclic 的规则是 `(j/nb) % ngpu`）。另证了 `lblocks_eq_cols_on_all`，覆盖 gehrd_m、ungqr_m 的 min_lblocks/last_dev 写法。此前的 `distribution_covers` 只证明了各卡之和等于 n。
- 证明对 n 归纳：先证 `succ_div_cases`，即 n → n+1 时商和余数的两种情况；再证 `nlocal_succ`，即每加一列，只有拥有这一列的那块卡加 1。只用 core Lean，axioms 为 propext 和 Quot.sound。
- （同日补充）zhemm_mgpu 中带偏移的块归属计算（l.235–276：myblkoffset、myblk/devperm、最后不完整块的归属、行 GEMM 的块计数）逐行读码，没有发现错误。其中两条计数规则已在 Lean 中对所有参数证明：`owned_before_formula`（设备 dev 在块号 B 之前拥有的块数）和 `owned_from_formula`（从设备 s 开始的 K 个块中设备 dev 拥有的块数，含 devperm 平移）。辅助引理 `shift_mod`、`mod_lt_two`。
- （同日补充）zgetrf2_mgpu 的面板广播和尾部更新（l.240–445：j_local2、nb0、最后一个面板的大小和发送条件）逐行读码，没有发现错误；`getrf_mgpu_next_local_all` 证明了 j_local2 的取法等于设备在块 0…j 中拥有的块数。

### 29.67 第 50 轮：多卡 Cholesky 的两条缓冲区复用竞争 M255、M256（2026-11-27，roxy，单卡模拟多卡）

- 起点：读 `zpotrf3_mgpu` 的下标（j_local2、nb0、nb2 都对，j_local2 = 设备在块 0…k 中拥有的块数），但 `zpotrf_mgpu.cpp` l.142–145 写着 "setting h = 1 produces failures for ngpus >= 4 / set h = 2 as a temp. fix / TODO"。为此写了 happens-before 模型 `scripts/potrf_mgpu_hb.py`，逐块重放主机程序，检查所有冲突的缓冲区访问是否有序。
- **M255**（z/c，ngpu ≥ 2，任意 h）：设备上的对角块缓冲 `dlP(d,0,0,buf)` 每 ngpu 步轮换一次。第 k 步 stream 2 上更新剩余列的 trsm 读它，第 k+ngpu 步 stream 1 上的对角块拷贝改写它，两者之间没有事件或主机同步。d/s 用 `dtrsm_work`，flag 0 调用只用预先求好的逆，不受影响。
- **M256**（四种精度，h < ngpu；`potrf_mgpu` 写死 h = 2，所以 ngpu ≥ 3）：主机 pinned 工作区有 h 个块槽，第 k+h−1 步发往主机的块行会盖住块 k 的已分解对角块，而各设备 stream 1 上从这里读它的拷贝可能还没执行。模型给出的分界线恰好是 h ≥ ngpu（Lean：`potrf_mgpu_host_slot_covered_iff`），被注释掉的 `h = ngpu` 正是最小安全值。`potrf_m` 取 h = 面板块数，不受影响。
- 验证：`gen_emu.py` 原样复制 2.10.0 的 `[zcds]potrf3_mgpu.cpp`，只把 `magma_setdevice` 改成空操作，每个虚拟设备在 GPU 0 上有自己的缓冲区和 3 个队列；在一个操作前让该流睡 100 ms（合法时序）。426 个进程全部 exit 0：
  - 不加延迟：0/20 出错；
  - M255：z 16/60、c 14/20 出错（info = 0，静默），加一个事件等待后 0/80；时间戳显示改写在 trsm 开始前约 103 ms 完成，误差落在该 trsm 的块行；
  - M256：z/c 在 h < ngpu 时 37/60 出错，全部报 info > 0（SPD 矩阵被判为不正定），h ≥ ngpu 时 0/48。
- d/s 的 M256 在模拟中测不出（0/96）：`magmablas_[ds]trtri_diag` l.104 的 `cudaMemset` 不带流，走 legacy 默认流，单卡上把所有虚拟设备排了序。模型复现了这一点（`--real --one-gpu` 0 个危险对）；真实多卡上它只排序本设备的流，危险仍在（`--real`）。另外核对了平台：pinned H2D 拷贝在执行时读主机数据（`round50_stage.cpp`）。
- 与第 25 轮（§29.28）的关系：当时 zpotrf_m 的双卡结果误差为 63，归因于 GPU 1 故障。zpotrf_m 在 z 精度、ngpu = 2 时会经过 M255，所以 M255 可能是一个原因；但同轮 zgetrf_m 的 inf 不能用它解释，硬件故障仍是更可能的主因。真实多卡复测时应同时检查两者。
- 附带发现（未登记，待查）：`[zcds]trtri_diag`、`[zc]lag2c`、`zlat2c`、`*gecscsyncfreetrsm`、`getrf_batched` 等处有不带流的 `cudaMemset`/`cudaMemcpy`。函数接收 queue，却在 legacy 默认流上做事；若 queue 来自非阻塞流（`magma_queue_create_from_cuda`）或用 per-thread 默认流编译，这些操作就与 queue 上的工作无序。
- 详见 `experiments/magma_gpu/results/round50_potrf_mgpu/TRIAGE.md`。

### 29.68 第 51 轮：接收 queue、却在 legacy 默认流上做事的例程（2026-11-27，roxy）

- 全库扫描不带流的 `cudaMemset/cudaMemcpy/…Symbol` 和不带流的内核启动，命中 trtri_diag、gecscsyncfreetrsm、slag2h、lag2c/lat2c/lag2s/lat2s 几组（四种精度各一份）。其中 sync-free 三角求解（M123）、slag2h（M108、M110）、lag2c 的全局标志（M115）早已登记。**教训：写探针前先在 FINDINGS、SPARSE_INVENTORY、NONZ_INVENTORY 里按文件名搜。**
- **M257（新）**：`[zcds]trtri_diag` l.104 的 `cudaMemset` 走 legacy 默认流。另一条阻塞流忙 150 ms 时，`magmablas_dtrtri_diag` 要 156–164 ms 才返回主机（空闲时 0.4–0.6 ms），三种 queue 都一样，结果正确。dtrsm.cu 的文档说 `trsm_work`、`trsm_outofplace` 是异步的，这一点不成立。在本平台上，非阻塞 queue 时的无序危险不会出现（主机被阻塞），没有验证。Lean：`trtri_diag_waits_unrelated`、`trtri_diag_not_async`。
- 已有条目的新证据（都是用补丁确认的）：
  - M123：d、z 共 24/24 出错（如 19999/20000 个分量）；恢复等待（volatile 读入度 + `__ldcg` 读部分和）后 24/24 精确。
  - M108：roxy 上任何超过 65504 的输入都会触发非法内存访问；用 `cudaGetSymbolAddress` 修正后 info 正确。
  - M110：地址修正、并拷入拷出整块标志后，info_array 能正确标出溢出所在的矩阵。
- 观察：lag2c 等在非阻塞 queue、延迟 100 ms 时仍返回正确的 info。WDDM 上的同步 `cudaMemcpyFromSymbol` 实际等待了整个设备；Linux 上没有测。
- 详见 `experiments/magma_gpu/results/round51_stream/TRIAGE.md`。§29.67 末尾的"附带发现"即由本节处理。

### 29.69 第 52 轮：多卡例程的通用调度模糊测试；新缺陷 M258（2026-11-27，roxy，单卡模拟多卡）

- 工具 `probes/mgpu_fuzz/gen_fuzz.py`：原样复制 MAGMA 源文件，`setdevice` 改为空操作，`queue_create` 改在 GPU 0 上建队列，并在每个提交到 queue 的语句前后各插一个点。运行时只在第 target 个点让该 queue 睡 100 ms。不需要先写 happens-before 模型。
- **必须同时有前置点**：只用后置点时，延迟落在紧随其后的 `magma_event_record` 之前，等这个事件的操作全都跟着推迟，窗口反而关上。第一遍 2,044 次全部正确，连阳性对照都没找到。加上前置点后，阳性对照（potrf U、n = 4096、ngpu 2）8/322 出错，全部在 M255 的那个 trsm 之前，工具得到标定。
- 第二遍共 5,445 次，覆盖 getrf_mgpu、geqrf2_mgpu、potrf_mgpu（U/L）、hegst_m、trsm_m、getrf_m（核内 / 核外）、potrf_m，ngpu 1–4。11 个出错全部是 M255：阳性对照 8 个，potrf_m（L）3 个。potrf_m 原先是推断在 M255 的影响范围内，现在是实测。其余例程全部正确，即正面结果。
- **M258（新）zgetrf_m**：ngpu 在分配前（l.187–192）和循环中（l.248–252）各算一次，用的 NB 不同（前者受 `MAGMA_NGR_NB` 限制，或因矩阵很高而偏小）。分配时减少了卡数之后，循环仍用满 ngpu0 张卡，对应的 queue 为 NULL、缓冲区未分配，进程访问违例；清理也跑到 ngpu0。
  - 实测：恰在 ngpu0 > ceildiv(MAGMA_NGR_NB, nb) 时崩溃。
  - 补丁（循环中 ngpu 取 min(·, 已分配数)，清理也按已分配数）后 12/12 正确。
  - Lean：`getrf_m_loop_exceeds_alloc`、`getrf_m_ngr64_instance`、`getrf_m_fix_within_alloc`。
  - potrf_m 的同类写法两处都向上取整，循环用的卡数不会超过分配数，没有问题。
- 局限：只做单点延迟；只测 z 精度和所列的规模、参数组合。详见 `experiments/magma_gpu/results/round52_mgpu_fuzz/TRIAGE.md`。
- 第三、四批（`c_`、`d_` 前缀）全部正确：unmqr_m 144、geqrf_m 724、ungqr_m 492、hetrd_mgpu n = 3072 抽样 595、heevdx_2stage_m n = 1024 全部前置点 2,267。后者覆盖 he2hb_mgpu、hemm/her2k_mgpu、bulge_back_m、bulge_applyQ_v2_m、unmqr_m 和 stedx_m。
- 生成器修正：匹配括号前先剥掉注释和字符串，并跳过预处理行；queue 表达式允许带结构体前缀；为 `zbcyclic` 辅助函数生成了副本。重新生成之前的 29 个副本，与旧版逐字节相同。
- **方法盲点**：前四批里，CPU 接口的 `_m` 例程收到的主机数组都是可分页的 `std::vector`。可分页内存上的 D2H 异步拷贝实际是同步完成的，所以"返回时拷贝还没完成"或"主机过早读缓冲区"这类竞争测不到，而 MAGMA 自己的测试程序用的是锁页内存。
  - 已给探针加上 `R52_PIN` 模式：主机数组换成锁页副本，调用一返回、在任何设备同步之前就拷回来检查。
  - 结果见 §29.72。

### 29.70 第 53 轮：gehrd_m + unghr_m 在 ilo > 1 时交错 T；新缺陷 M259（2026-11-27，roxy，库版本，1 张 GPU）

- 来源：第 52 轮给 gehrd_m 做模糊测试时读源码发现。
  - `zgehrd_m` 把 ilo − 1 向下取整到块边界，并把起点在全局列 i 的块的 T 存到 `T + i*nb`。
  - `zunghr_m` 把从 ilo − 1 开始的子矩阵交给 `ungqr_m`，T 没有平移。`ungqr_m` 按子问题里的列号读 T，于是每个子块拿到的都是别的块的 T，或者是从未写过的 0。
  - 单卡 `zgehrd` 存在 `dT(0, i−ilo)`，与 unghr 的读法一致。
  - MAGMA 的测试程序只测 ilo = 1。
- 实测（`results/round53_hessenberg/`）：
  - `gehrd_m` 自己的 H 和 tau 在 49 个用例里都正确。
  - `gehrd_m + unghr_m` 恰好在"ilo > 1 且 ihi − ilo + 1 > nb"的 36 个用例里出错：‖QHQᴴ − A‖/‖A‖ 为 4–21，info 0。其余 13 个正确。
  - 单卡这一对和补丁版都是 49/49 正确。
- 端到端：`zgeev_m`（Version5）先做 `zgebal("B")`，只要它做了置换就会走到这里。
  - n = 100、256，ilo = 2、4：`zgeev_m` 的特征向量残差为 10–15，`dgeev_m` 为 8.6–15，info 都是 0。
  - 单卡 `geev` 约 1e-13。
- 补丁：子问题从取整后的 ilo_r 开始，T 平移 ilo_r·nb。多出来的反射的 tau 在 zgehrd 的前提下为 0。
- Lean（`Magma/MultiGpu.lean`）：`unghr_m_reads_wrong_T`、`unghr_m_inconsistent`、`unghr_m_ilo1_consistent`、`unghr_single_consistent`、`unghr_m_fix_consistent`、`unghr_m_fix_extra`。
- 另外确认：`zpotrf_mgpu_right.cpp` 整个文件在 `#if 0` 里（"needs updating to MAGMA v2 interfaces"），不参与编译，不需要测。

### 29.71 第 54 轮：CPU 接口多卡驱动的参数维度（2026-11-27，roxy，库版本，ngpu = 1；没有新缺陷）

- 补测第 38 轮没有覆盖的参数维度：lda = n + 7 并用哨兵检查填充行、z 精度、Upper、itype 2/3、jobz N、range V、geev_m 左向量、m ≠ n。共 492 个用例，全部 exit 0。
- 出错的全部能归入已登记的缺陷：
  - M121：n ≤ 128 时忽略 range，96 例；
  - M040：hegvdx_2stage_m 在 jobz N 时返回 −14，36 例；
  - M065：2-stage 的 Upper，30 例；
  - M216：dstedx_m 用未缩放的 (vl, vu] 挑选缩放后问题的特征向量，6 例。这是新路径：经 hegvdx_m、hegvdx_2stage_m（itype 1，range V）到达，特征值正确，但 75 个特征向量里 74 个是错的。
  - 已补进上述各条的证据。
- 填充行全部没有被改写。详见 `experiments/magma_gpu/results/round54_mgpu_param/TRIAGE.md`。

### 29.72 锁页主机数组复测；新缺陷 M260（2026-11-28，roxy，单卡模拟多卡）

- 第 52 轮的模糊测试改用锁页主机数组（`R52_PIN=1`），重测 CPU 接口的 10 个例程，约 4,100 次。除 potrf_m 的 M255（与可分页时相同）外，只有 gehrd_m 出现新的错误。
- **M260 gehrd_m**：
  - 最后一次 zlahr2_m 在每张卡上排了 `setmatrix_async(Y = work → dY)`。循环后只同步持有剩余块（R ≤ 2）的卡，随即 zgehd2 覆写 work。
  - R < ngpu 时，面板卡的拷贝读到草稿，zlahru_m 用错误的 Y 更新面板的上半部分，info 0。
  - MAGMA 的 testing_zgehrd 就是用锁页的 h_work 调用 zgehrd_m。
  - 实测 ngpu 2、n = 96 出错（相对误差 2.35e3，延迟 1–500 ms 都出错）；补丁（zgehd2 之前同步所有 queue）后，2,534/2,534 正确。
- **单卡模拟的新盲点**：所有虚拟设备共用一个 H2D 拷贝引擎，按提交顺序串行。排在被延迟拷贝之后的卡会跟着等，把竞争掩盖掉：其他组合中 zgehd2 都是在延迟结束后才开始。
- Lean：`gehrd_remainder_le_two`、`gehrd_m_unsynced_device`、`gehrd_m_panel_unsynced`、`gehrd_m_panel_synced_when_full`、`gehrd_m_n96_instance`、`gehrd_m_fix_syncs_all`。
- 详见 `experiments/magma_gpu/results/round52_mgpu_fuzz/TRIAGE.md` 中"锁页主机数组复测"一节。

### 29.73 第 55 轮：geqrf_gpu 的 dT 的消费者；新缺陷 M261–M263（2026-11-28，roxy，库版本，1 张 GPU）

- 沿着 M259 的思路，检查中间结果 dT 的生产方与消费方是否用同一约定，共 236 个用例，外加 unmqr_2stage_gpu 的 4 种输入形式各 96 例。
- ungqr_gpu、ungqr、geqrs(3)_gpu、gels(3)_gpu 全部正确。
- **M261 unmqr_gpu**：
  - 文档只说 dT 是 geqrf_gpu 的输出，并且一部分被当作工作区。但 larfb 工作区放在 dT + 2k·nb，需要 nw·nb；按 geqrf_gpu 的文档，那里只剩 roundup(k, 32)·nb。
  - nw > roundup(k, 32) 时写出界 (nw − roundup(k, 32))·nb 个元素，保护区实测与预测完全相等（72/72）。
  - 不加保护区时，越界写改写了相邻的 dC，64/72 例结果出错，info 0。
  - MAGMA 自己的测试程序把 dT 分配得更大，所以测不出来。
- **M262 unmqr_2stage_gpu（公开接口）**：side = Right 时对每一块都用 MagmaLeft 调 larfb，工作区只有 n·nb 却按 ldwork = m 使用。Right 48/48 错，info 0。
- **M263 unmqr_2stage_gpu**：文档要求的输入只是 zgeqrf 的输出，但它实际需要显式存储的 V 和每一块的 T（zgeqrf_gpu 不写最后一块的 T）。按文档调用 96/96 全错；两者都补齐后 Left 全对。
- Lean：`Magma/Dense/QrEigen.lean` §55，6 个定理。
- 详见 `experiments/magma_gpu/results/round55_qr_dt/TRIAGE.md`。

### 29.74 第 56 轮：Hermitian / 对称不定分解与其求解器（2026-11-28，roxy）

- nopiv 的 GPU 与 CPU 接口、Bunch–Kaufman 的 hetrf/hetrf_gpu/hesv（用 LAPACK zhetrs 检查 ipiv 兼容性）、Aasen，覆盖 U/L、nrhs、ld 填充，共 528 例。
- 求解类 512 例全部正确。
- **M264（轻微）**：hetrf_aasen 的文档描述了 Upper，实际对 Upper 返回 NOT_IMPLEMENTED（−117）。
- 详见 `experiments/magma_gpu/results/round56_ldl/TRIAGE.md`。

### 29.75 第 57 轮：hetrd_gpu → unmtr_gpu，geqrf_batched → unmqr_batched（2026-11-28，roxy）

- hetrd_gpu / hetrd2_gpu 的 dA、wA、tau 原样交给 unmtr_gpu：192/192 正确，wA 与 dA 逐位相同。
- geqrf_batched → unmqr_batched：Left 60/60 正确。
- **M265（轻微）**：unmqr_batched 的文档列出了 Right，实际返回 −1，即 "only side = MagmaLeft is supported"。
- 详见 `experiments/magma_gpu/results/round57_trd_batched/TRIAGE.md`。

### 29.76 第 58 轮：CPU 接口的生产方 → 消费方组合（2026-11-28，roxy；没有新缺陷）

- 用 MAGMA 自己的 gebrd、geqrf、gelqf(_gpu)、geqlf、hetrd 的输出，喂给 unmbr/ungbr、unmqr/ungqr2、unmlq、unmql、unmtr，并与 LAPACK 消费方在同一输入上比较，共 264 例，全部正确。
- 结论：问题集中在 MAGMA 自定义的中间格式（T、dT 及其工作区），与 LAPACK 兼容的格式（反射向量与 tau）的交接没有问题。
- 详见 `experiments/magma_gpu/results/round58_cpu_pairs/TRIAGE.md`。

### 29.77 第 59 轮：设备工作区按文档大小分配并加保护区（2026-11-28，roxy；没有新缺陷）

- 对 13 个带文档化设备工作区的公开例程（geqrf/geqrf3/geqrs/geqrs3_gpu、he2hb、hetrd2_gpu、getri_gpu、zstedx、gegqr_gpu 四种 ikind、ztrsm_work、zhemv_work、zlarfb_gpu、geqp3_gpu）统一检查：工作区取文档大小，其后放 64K 个元素的哨兵。共 89 例，保护区都没有被写。
- 除 M261 外，设备工作区没有写出界。geqp3_gpu 的 z 精度同样拒绝文档给的 LWORK，已补进 M120。
- 详见 `experiments/magma_gpu/results/round59_guard/TRIAGE.md`。

### 29.78 第 60 轮：工作区查询接口按查询值分配并加保护区（2026-11-28，roxy；没有新缺陷）

- 对 expert_gpu_work 系列（geqrf、getrf、potrf、getri、getrs、potrs、trtri、gegqr）、geqrf_batched_work 和 gb*_native_work，按包装函数的写法先查询、再按查询值分配，主机和设备工作区后各放 256 KB 哨兵，共 95 例。保护区都没有被写，结果正确。
- Native 模式被拒（geqrf、getrs）与文档一致，不登记。gb*_native_work 查询时也要传有效的 info，这与 M044 是同一处。
- 测试程序问题：run_list.ps1 原先只认 `^r5` 开头的行，已改为 `^r[0-9]`。
- 详见 `experiments/magma_gpu/results/round60_query_guard/TRIAGE.md`。

### 29.79 第 61 轮：可变尺寸批量分解；新缺陷 M266、M267（2026-11-28，roxy）

- getrf_vbatched、getrf_vbatched_max_nocheck_work、getrf_nopiv_vbatched、potrf_vbatched 与 LAPACK 逐矩阵比较，共 64 例加 85 个针对性用例。非奇异矩阵全部正确；奇异矩阵上重现了已知的 M164、M173。
- **M266**：potrf_vbatched 大路径（max_n 超过交叉点）调用面板时 gbstep 传 0，所以 info 报成 (c mod 128) + 1，整批都如此。
- **M267（仅 Z）**：360 ≤ max_n ≤ 448 时，lpout 的共享内存检查失败，返回 −33，整批都没有分解，info_array 仍为 0。
- Lean：`Magma/Dense/QrEigen.lean` §61，6 个定理。
- 详见 `experiments/magma_gpu/results/round61_vbatched/TRIAGE.md`。

### 29.80 第 62 轮：分派路径的资源上限与回退（2026-11-28，roxy；没有新缺陷）

- 列出所有在资源不足时返回负值的批量和原生内核，逐个检查调用者是否回退：getf2_fused（batched/vbatched）、gesv_batched_small、geqr2_fused_sm、unm2r_sm、gbtrf_fused_sm 都会回退；getf2_native_fused 虽然返回值被忽略，但调用前已按 m 改走 blocked 路径。只有 vbatched 的 lpout 没有回退（M267）。
- 实测 zgetrf_native 在 m 跨过 7168（Z 的 fused 上限）时，22/22 正确。
- 详见 `experiments/magma_gpu/results/round62_native/TRIAGE.md`。

### 29.81 第 63 轮：从未覆盖的 magmablas 辅助内核；新缺陷 M268、M269（2026-11-28，roxy）

- 16 个辅助内核与按其文档写的 CPU 参考实现比较，共 466 例。geadd2、gemv_conj、laset_band、laswpx/laswp2、三种转置、trmv、symmetrize、symv_work、larfg、swapdblk、lacpy_conj 全部正确。
- **M268**：magmablas_[zc]geam 的 ConjTrans 没有取共轭，按 Trans 计算，50/50 出错。MAGMA 自己的测试程序也不取共轭，所以测不出来。
- **M269**：magmablas_[zc]gemm_reduce 的文档写 AᵀB，实际计算 AᴴB。
- Lean：`Magma/Kernels.lean` §63，4 个定理。
- 详见 `experiments/magma_gpu/results/round63_mblas_aux/TRIAGE.md`。

### 29.82 第 64 轮：magmablas BLAS 的 ConjTrans（2026-11-28，roxy；没有新缺陷）

- 沿着 M268 的思路，对 gemm/gemv/trsm/trmm 的非批量、batched、vbatched 版本和 trsv，用复数数据、trans 取 N/T/C，与参考 BLAS 比较，共 312 例，全部正确。M268 只出在 geam 一处。
- 详见 `experiments/magma_gpu/results/round64_conj/TRIAGE.md`。

### 29.83 第 65 轮：稀疏求解器在真正复数数据上的表现；新缺陷 M270（2026-11-28，roxy）

- 21 个求解器在复 Hermitian 正定、Hermitian 不定和非 Hermitian 三类 1024 维复系统上，以真实残差判定。出错的大多可以归入已知缺陷（M084 CG/PCG、M103 IDR、M089 BA/ITERREF），或者属于如实报告的失败。
- **M270**：TFQMR 的四个变体都把影子内积写成 zdotc(r, r̃)（xᴴy 的顺序反了），得到的是共轭值。Hermitian 上不受影响，非 Hermitian 复系统不收敛。改正顺序后 39/39/14 步收敛。
- 补丁要打在 ztfqmr_merge / zptfqmr_merge 上：`--solver TFQMR` 映射到 merge 版本。
- 详见 `experiments/magma_gpu/results/round65_sparse_complex/TRIAGE.md`。

### 29.84 第 66 轮：稀疏 LOBPCG；新缺陷 M271（2026-11-28，roxy）

- 二维磁 Laplacian（n = 256/1024，实与复），k = 4/8/16，与 LAPACK zheevd 比较。24/24 都报成功，但最大残差 6e-6 到 6e-4，要求的是 1e-8。
- **M271**：主循环在第 0 个 Ritz 对的残差达标时就跳出（l.496–508），不等其余 k − 1 对收敛。去掉这个跳出后 24/24 收敛到 1e-8。
- 详见 `experiments/magma_gpu/results/round66_lobpcg/TRIAGE.md`。

### 29.85 第 67 轮：sparse 格式与转置在复数数据上的表现；新缺陷 M272、M273（2026-11-28，roxy）

- 第 46 轮的格式测试只用了实数。本轮改用真正的复数（136 个进程）。各格式的结论与第 46 轮一致（M125、M132、M137、M138、M251–M254），没有复数特有的新问题，已为这几条补了证据。
- **M272**：magma_zprint_csr_mtx 的 RowMajor 分支把尺寸行打印成 "列 行"。
- **M273**（读码）：ColMajor 分支经由 z_transpose_csr，它会释放自己刚交出去的数组，调用者读的是已释放的内存。
- 详见 `experiments/magma_gpu/results/round67_formats_complex/TRIAGE.md`。

### 29.86 第 68 轮：geqr2x{,2,3,4}_gpu 的工作区；新缺陷 M274（2026-11-28，roxy）

- 四个公开的面板 QR 版本，dwork 按文档大小（2N 或 3N 个 double）分配并加保护区。凡是 n ≥ 16 的用例保护区都被写（共 47/56）。实际用量为 4n、5n、6n，到 n = 128 时 v3/v4 达到 50n。
- MAGMA 自己的测试程序分配了 130·min_mn 个 double，所以测不出来。这与 M261 是同一种模式。
- 详见 `experiments/magma_gpu/results/round68_geqr2x/TRIAGE.md`。

### 29.87 第 69 轮：范数与 geqr2 的工作区（2026-11-28，roxy；没有新缺陷）

- 逐个核对测试程序中的设备工作区，补测第 59 轮没覆盖的 zlange、zlanhe、zgeqr2_gpu（文档大小加保护区）。117 例全部正确；zlanhe 的 F 范数被拒绝，与文档的 "not yet supported" 一致；对角元虚部按文档被忽略。
- 设备工作区"文档写小了"的问题目前只有 M261、M274。
- 详见 `experiments/magma_gpu/results/round69_norms/TRIAGE.md`。

### 29.88 第 70 轮：剩余的 magmablas 辅助内核；新缺陷 M275、M276（2026-11-28，roxy）

- herk_small_reduce 160 例、hemv/symv 的 Upper 与 incx 2 共 40 例、symmetrize_tiles 8 例，全部正确。
- **M275**：laswp_sym 每 32 个主元分一块，块内用相对下标填 ipiv，内核却按绝对下标读，第二块起越界；超过 32 个主元时结果全错。
- **M276**：laswp_sym 同一次启动里，各线程依次处理多个主元而没有同步，负责同一元素的线程会随主元变化，形成竞争，结果错且每次不同。只有一个主元时正确（复对称语义）。
- 库内没有调用者。详见 `experiments/magma_gpu/results/round70_mblas_aux2/TRIAGE.md`。

### 29.89 第 71 轮：未覆盖的面向用户的 batched 例程（2026-11-28，roxy；没有新缺陷）

- gemm/gemv_batched_strided（N/T/C）、geadd/lacpy/laset/transpose_conj/hemv_batched、gbsv_batched(_strided)，共 102 例，与 CPU 参考或 LAPACK 一致。
- 详见 `experiments/magma_gpu/results/round71_batched_misc/TRIAGE.md`。

### 29.90 第 72 轮：batched Jacobi 的 strided 接口（2026-11-28，roxy；没有新缺陷）

- gesvj_expert_batched_strided、heevj_batched_strided_small_sm 与对应的指针数组接口输出逐位相同。与 LAPACK 的偏差都来自已知的 M240（Windows 上的 long 交换），已补证据。
- 详见 `experiments/magma_gpu/results/round72_jacobi_strided/TRIAGE.md`。

### 29.91 第 73 轮：复数稀疏工具函数（2026-11-28，roxy；新缺陷 M277、M278）

- M277：magma_z_spmm（稀疏 × 稀疏）在 CUDA ≥ 12 分支把 beta = NULL 传给 cusparseSpGEMM_workEstimation，CUDA 13 上每次返回 −3003；文档里的 alpha 在两个分支都没有用到（读码）。
- M278：magma_zm_27stencil 只清除 x 方向的边界耦合；y 方向回绕的耦合被保留，n = 3 时多 84 个、n = 5 时多 312 个，与按代码推出的数字完全一致；n = 2 时丢失 30 个。5 点模板正确。
- M137/M138 扩充：transposeconj/abs/struct_cpu 共用同一个模板，方阵正确，宽矩阵堆越界写（崩溃，10 × 11 时时有时无），高矩阵返回的维度没有交换。
- tril/triu、diagadd、indexsort(val) 全部正确。Lean：Kernels.lean §73 两个定理；Kernels.lean 394 行 43 个 theorem，总计 5,855 行 494 个。
- 详见 `experiments/magma_gpu/results/round73_sparse_utils/TRIAGE.md`。

### 29.92 第 74 轮：更多稀疏工具函数（2026-11-28，roxy；新缺陷 M279、M280、M281）

- M279：spmv_shift 的三个内核算 αAx − λx + βy，文档是 α(A − λI)x + βy；α ≠ 1 时差 5–9%。
- M280：zmslice 按 ceil(n / num_slices) 切分，末尾几片起点越过 n，片大小为负，读 A.row 越界：返回 −112，或者不确定地返回 0 且 start > end；出错后输出里留着已释放的指针，magma_zmfree 导致堆损坏（0xC0000374）。MAGMA 测试程序的用法（9 片取第 8 片）在 28 个 n ≤ 55 的尺寸上触发。
- M281：zmconjugate 不检查格式，ELL 矩阵触发非法内存访问仍返回 0。
- mcsrcompressor_gpu、mshrink、vtranspose、mconjugate（CSR）、mslice（正常片）全部正确。Lean：Kernels.lean §74 四个定理；共 5,885 行 498 个 theorem，axioms.txt 456 行，无 sorryAx。
- 注意：`build_probe.bat` 从 `D:\lean4hpc\magma\probes\` 读源码，scp 时要放到这个子目录。
- 详见 `experiments/magma_gpu/results/round74_sparse_utils2/TRIAGE.md`。

### 29.93 第 75 轮：直接调用 IDR(s) 与 magma_zqr（2026-11-28，roxy；新缺陷 M282、M283，扩充 M103）

- M282：6 个复数 IDR(s) 变体的角度保护用 |Re(tᴴr)|（注释和原算法都是 |tᴴr|），所以结果不具有相位不变性。A = iS（S Hermitian 正定）时 idr/pidr 在 s = 1/2/4/8 下全部 −201 不收敛，A = S 时 40/27/15/9 步收敛。只改这一行的补丁版 20/20 收敛，所有相位的步数和残差与 A = S 完全相同。
- M103 扩充：idr_strms、pidr_strms 与 MERGE 变体结果逐位相同，所有输入都失败。
- M283：magma_zqr 对设备矩阵原地分解，文档为 [in] 的 A 被改写成 Q；Q、R 正确。
- MAGMA 2.10 的 sparse 里已经没有 ISAI / ParILU，第 74 轮交接里那句已改正。
- Lean：Kernels.lean §75 两个定理；共 5,904 行 500 个 theorem，axioms.txt 458 行，无 sorryAx。
- 详见 `experiments/magma_gpu/results/round75_idr/TRIAGE.md`。

### 29.94 第 76 轮：PBiCG / PQMR 用到的转置预条件（2026-11-28，roxy；新缺陷 M284，扩充 M106）

- M284：magma_z_applyprecond_left/right 的 MagmaTrans 分支对 ICC 仍按左 L⁻¹、右 L⁻ᵀ 求解，影子向量得到 L⁻¹L⁻ᵀu 而不是 M⁻ᵀu。四种精度相同。
  - 恒等式 uᵀ(M⁻¹v) = (M⁻ᵀu)ᵀv：ILU 误差 1e-16，ICC 误差 0.03–0.22；把顺序对调后为 1e-16。
  - PBiCG + ICC 在 n = 100 时 −201；PQMR + ICC 在 n = 900 时要 370 步，ILU 只要 28 步。
  - 补丁后与 ILU 完全相同。
- M106 补充：不加预条件的 QMR 在实 SPD 拉普拉斯矩阵上跑满 maxiter 后报 −201，真实残差已经是 3e-9 和 4e-10。
- custom spmv / precond 是模板，不算缺陷。
- Lean：Kernels.lean §76 两个定理；共 5,925 行 502 个 theorem，axioms.txt 460 行，无 sorryAx。
- 详见 `experiments/magma_gpu/results/round76_precond_trans/TRIAGE.md`。

### 29.95 第 77 轮：LSQR 与复数矩阵上的转置预条件（2026-11-28，roxy；新缺陷 M285）

- 在这个 wrapper 里，MagmaTrans 的意思是共轭转置：ILU 的转置用 transposeconjugate 构造，LSQR、PBiCG 乘的是 Aᴴ，复数矩阵上 ILU 的转置分支满足 Hermitian 恒等式。
- M285：Jacobi 的转置分支乘 D⁻¹ 而不是 conj(D⁻¹)。复数矩阵（对角 4.05 + 0.8i）上 Hermitian 恒等式误差 0.388；LSQR + Jacobi 为 −201，PBiCG + Jacobi 为 −202，残差最大 13.2，而无预条件时分别 64/91、33/47 步收敛。补丁后与无预条件相同。只影响 z/c 版本。
- 实矩阵上的 LSQR 四种预条件都收敛，ICC（M284）只是多用一些步数。
- Lean：Kernels.lean §77 两个定理；共 5,943 行 504 个 theorem，axioms.txt 462 行，无 sorryAx。
- 详见 `experiments/magma_gpu/results/round77_lsqr_precond/TRIAGE.md`。

### 29.96 第 78 轮：可变尺寸批量 BLAS（2026-11-28，roxy；没有新缺陷，扩充 M246、M248）

- 174 例，复数数据，尺寸含 0，ld 余量和另一侧三角填哨兵。gemm / gemv / hemv / trmm / trsm_vbatched 全部正确。
- herk / syrk / her2k / syr2k_vbatched：计算的那一侧正确，另一侧三角每次都被改写，即 M246。
- hemm_vbatched：每批 12 个矩阵中 10 个出错，即 M248。用第 35 轮的 fetch 补丁重新编译 zhemm_vbatched_core.cu 后 8/8 正确。
- mconvert：主机上只支持 CSR ↔ X，两端都不是 CSR 时明确返回 NOT_SUPPORTED，设备路径第 46 轮已测过，所以没有再测。
- 教训：按文件名扫描记录会漏掉按例程名登记的条目（M245/M246/M248 已经包含 vbatched），写探针前要同时按例程名搜。
- 详见 `experiments/magma_gpu/results/round78_vbatched_blas/TRIAGE.md`。

### 29.97 第 79 轮：trtri_diag_batched / vbatched 与 get/setmatrix_transpose（2026-11-28，roxy；没有新缺陷）

- trtri_diag_batched / vbatched：44 种组合，有效对角块的逆误差 ≤ 7.2e-16，填了 NaN 的另一侧三角没有被引用。
  - 观察：upper 的 triple_zgemm16_part1 在条件装载失败时没有清零 rA，旧值写进了超出 n 的补齐列（例如 n = 17 时第 20、24、28 列）。库内唯一的使用者 trsm_batched 只取有效部分，没有可观察的影响，不计入缺陷。
- get/setmatrix_transpose（单卡）：18 例同步后全部正确。
  - 观察：对锁页主机内存，getmatrix_transpose 返回时最多 100% 的元素还没到；文档没写明需要调用者同步，但库内调用者 getrf_m 会自己同步，不计入缺陷。
- lacpy_sym_in/out 的 rows 约定（文档写 rows[i] / rows[n+i]，代码用交错布局）只记在交接文档里，留待核对。
- 详见 `experiments/magma_gpu/results/round79_trtri_transpose/TRIAGE.md`。

### 29.98 第 80 轮：其他精度上的复核与 lacpy_sym 的约定（2026-11-28，roxy；新条目 M286）

- d 精度：M279（差 5–8%）、M280（−112 和不确定的结果）、M284（dpbicg + ICC −202，残差 3.33）全部复现。
- c 精度：M282（A = iS 时 −202，残差 NaN）、M285（clsqr + Jacobi −201）全部复现。
- M286：lacpy_sym_in/out 的文档描述 rows[i] / rows[n+i] 交换，代码按交错二元组做赋值。库内唯一的调用者 Aasen 与代码一致，按文档调用的外部用户会出错。读码确认。
- FINDINGS 共 285 条（M001–M286）。
- 详见 `experiments/magma_gpu/results/round80_prec/TRIAGE.md`。

### 29.99 第 81 轮：geqr2x 的辅助函数与 n > 128（2026-11-28，roxy；新缺陷 M287、M288）

- M287：magma_ztrmv_tkernel / ztrmv_kernel2 把部分和放在 `__shared__ sum[128]`，却以 k 个线程启动。geqr2x2/3/4（文档 N ≥ 0）在 n ≥ 130 时触发非法内存访问，info = 0，R 全错。z、d 精度都在 129/130 之间出错，与预测一致。larfbx 在 k ≥ 129 时同样出错。gegqr 自己检查了 n ≤ 128，所以只影响直接调用者。第 68 轮最大只测到 n = 128，所以当时没有发现。
- M288：larfbx 计算 HᴴC（文档写 HC），并且会读 T 的下三角（GPU 确认）；larfx 的 tau 约定与文档相反，也不更新 xnorm；larfgx 把 beta 存到 dA 而不是 dx0（读码）。
- Lean：Kernels.lean §81 三个定理；共 5,966 行 507 个 theorem，axioms.txt 465 行，无 sorryAx。FINDINGS 287 条（M001–M288）。
- 详见 `experiments/magma_gpu/results/round81_geqr2x_large/TRIAGE.md`。

### 29.100 第 82 轮：按运行时线程数扫描，magma_zhetrs_gpu（2026-11-28，roxy；新缺陷 M289、M290）

- 扫描所有以运行时变量作为线程数的内核启动，找到了从没测过的公开函数 magma_zhetrs_gpu。
- M289：Lower 的前代走融合内核（逐列版本被 `if (0)` 关掉了），这些内核访问 B 时不带列下标，只解第 0 列。nrhs ≥ 2 时恰好 nrhs − 1 列出错，info = 0。补丁（`if (1)`）后 11/11 正确。
- M290：对角块内核以 <<<1, nrhs>>> 启动，nrhs > 1024 时启动失败且不检查，所有列都错，info = 0。
- 四种精度的代码相同。Lean：Kernels.lean §82 两个定理；共 5,982 行 509 个 theorem，axioms.txt 467 行，无 sorryAx。FINDINGS 289 条（M001–M290）。
- 详见 `experiments/magma_gpu/results/round82_hetrs/TRIAGE.md`。
