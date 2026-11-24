# lean4HPC：数值规格、证明流程与 V100 校验入口

日期：2026-11-22（America/Chicago）。状态：已有CPU模型实验和源码反例；GPU与Lean检查尚未执行。

## 研究目标

针对开源低精度 GPU 算子，构建可复用的数值规格与 Lean 证明流程，并明确从源码、编译结果到硬件执行的可信边界。

首个算子范围：GEMM。每个实现用 adapter 提供输入域、输出类型、数值模式、运算顺序模型、源码版本与构建标识。规格不绑定 NumPy/cuBLAS 输出；标准库作为差分参考，精确数学语义作为误差参考。

## 当前连接与硬件约束

- 已实际执行 `ssh s82`，返回 `Could not resolve hostname login2.mystic.cs.iit.edu: -65563`，退出码255。未到认证阶段。
- 因此尚未执行登录节点上的 `bash login_s8.sh`，未确认远端CUDA、GPU分配、Lean或编译环境。这不说明服务器本身故障。
- 用户报告8张V100。正式清单须以远端探测为准。先使用调度器分配的一张卡做正确性；其他卡可在后续分发独立测试，首轮无需多卡归约。
- V100/Volta Tensor Core支持FP16输入、FP16或FP32累加。整数Tensor Core是后续Turing新增功能，因此不能在V100上把ozIMMU运行解释为INT8 Tensor Core执行。
- ozIMMU仍保留为整数切片源码建模对象；INT8 cuBLAS API在具体V100软件栈上是否拒绝或采用其他路径，需要运行确认。
- V100实机候选改为明确支持sm_70的FP16多分量/纠错实现。cuMpSGEMM当前README面向A100，不能因其有FP16模式便假设支持V100。先审版本与构建目标；必要时选OzBLAS/CUTLASS的Volta兼容实现。没有完成该选择前不声称已运行替代库。

来源：[Volta Tensor Core说明](https://docs.nvidia.com/cuda/archive/12.8.0/volta-tuning-guide/index.html#tensor-core-operations)；[Turing新增整数Tensor Core](https://docs.nvidia.com/cuda/archive/12.8.0/turing-tuning-guide/index.html#tensor-core-operations)；[cuMpSGEMM官方README](https://github.com/enp1s0/cuMpSGEMM)。

## 可复用的数值契约

参考结果定义为输入浮点数解码到实数之后的 `R = alpha * op(A) * op(B) + beta * C0`。

| 契约字段 | 含义与首轮范围 |
|---|---|
| 输入 | dtype、shape、transpose、leading dimension、alignment；首轮real NN，alpha=1，beta=0 |
| 有效域 | 有限输入、范围/指数跨度、切片数、K、所有中间量的溢出/下溢前提 |
| 算术模式 | FP16转换舍入、FP32/FP64累加、FMA融合、FTZ、归约树/允许顺序集合 |
| 输出 | dtype、全部元素写入、NaN/Inf行为；异常数另设契约，不并入普通相对误差 |
| 数值保证 | 逐元素 `abs(Y_ij - R_ij) <= E_ij(A,B,C0,config)` |
| 辅助观测 | 最大绝对误差、ULP差、非有限数计数、相对 `sum_k abs(A_ik*B_kj)` 的误差 |
| 证据 | 实验结果与定理分开记录；unsupported、unknown、counterexample、proved使用不同状态 |

必须分别计算两种误差：①相对原始高精度输入的误差；②相对量化后输入的误差。否则会漏掉低精度转换造成的信息损失。输入值以原始二进制比特保存，随机种子仅作补充。

FP64/cuBLAS不是精确oracle。小矩阵使用任意精度Fraction计算给定二进制输入的精确乘加；大矩阵可用MPFR，但必须记录精度、舍入与是否有严谨包围区间。接近零输出不以逐元素相对误差单独判错。

误差预算至少分为：输入分解/量化残差、丢弃交叉项、乘加舍入、最终缩放与alpha/beta运算。暂时无法证明的硬件误差预算必须显式列为假设，不能将实测最大误差当作全称上界。

## 证明与实现的对应关系

| 层 | 目标 | 当前证据 | 仍需信任/完成 |
|---|---|---|---|
| 数学规格 | 精确GEMM及可接受误差界 | 本文定义 | 证明规格确实表达用户需求 |
| Lean算术模型 | 分片、无溢出、尾项与舍入误差 | 有待检查的ComplexBeta整数反例草稿 | Lean检查、形式化浮点语义、实际定理假设 |
| 原始源码 | 源码实现形式化模型 | 固定ozIMMU快照、CPU数值模型、抽取函数体重放 | 手工转写/抽取的正确性尚未证明 |
| 编译产物 | 固定编译器/flags产物保持源码语义 | 尚无远端PTX/cubin | 编译器正确性、优化/FMA/未定义行为检查 |
| PTX/SASS与设备 | 目标指令满足算术模型和并行语义 | 尚无GPU结果 | 驱动JIT、硬件指令数值/同步语义 |
| 标准库依赖 | cuBLAS调用符合声明配置 | CPU NumPy差分已做 | 闭源cuBLAS后端选择和内部归约仍为边界 |

保存PTX/cubin、检查SASS和做硬件测试能增加证据，不等于证明编译器或硬件正确。若只证明算术核心，就准确称为“在这些前提下的算术核心证明”。

## 第一批可验收目标

1. **已知反例检查**：在Lean 4检查 `experiments/ComplexBeta.lean`，记录完整工具链版本与axioms；它只证明Int层的具体反例。再为精确可表示的小整数建立浮点对应。
2. **整数安全定理**：若每个乘积绝对值至多M_A*M_B，证明每个长度t<=K的部分和不超过t*M_A*M_B；若K*M_A*M_B<=2^31-1，则任意顺序均不溢出INT32。再证明源码的位宽选择满足前提。
3. **分片定理**：对固定128位提取语义、合法移位和有限共享scale证明重构残差；零/subnormal/极大指数需明确支持、拒绝或分支处理。
4. **误差组合**：将输入残差、未计算交叉项和FP64重组误差连接到GEMM契约。V100 FP16路线使用相应转换/累加模型。
5. **实机差分**：固定提交及全部submodule、工具链、GPU、输入比特；分别调用目标库和未被拦截的标准cuBLAS，核实实际走到目标路径。报告与精确oracle的误差，不能只测两库allclose。

所有定理须经Lean checker通过；禁止把sorry、额外未审计axiom或直接声明误差前提当作所需误差定理。反例、实测和证明分别保存状态。

## 远端恢复连接后的首个动作

本节命令尚未在远端运行。按用户给定路径进入计算节点：

```sh
ssh s82
bash login_s8.sh
```

进入后先只读检查，并尊重当前调度器分配和CUDA_VISIBLE_DEVICES，不覆盖这些值：

```sh
hostname
pwd
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,process_name,used_memory --format=csv
nvcc --version
command -v cmake gcc g++ python3 lean lake cuobjdump nvdisasm compute-sanitizer
printenv CUDA_VISIBLE_DEVICES SLURM_JOB_ID PBS_JOBID
```

根据实际toolkit选择支持sm_70的版本。初次只跑小样本，不启用8卡压力测试。先原样复现，再在独立目录修复候选缺陷并比较，保持原提交不变。

## 每次运行必须保存的清单

`source_commit, submodule_commits, source_hashes, compiler_version, compile_flags, target_sm, cuda_runtime_version, cuda_driver_version, cublas_version, gpu_uuid, binary_hash, input_hash, input_dtype, output_dtype, shape, strides, transpose, alpha, beta, numeric_mode, rounding_assumptions, oracle, metrics, verdict, proof_artifact, proof_checker_version, trusted_assumptions`。

缺失关键字段的记录不能归为可复现硬件验证。当前SSH失败只记为环境阻塞，不生成GPU通过/失败结果。
