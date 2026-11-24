# 开源算子库调研：低精度模拟高精度
调研日期：2026-11-22。范围：真实开源代码的数值核心与可验证性；未运行 GPU 库，未完成形式化证明。

## 选择结论

本轮首选 **enp1s0/ozIMMU**，选定 commit `08eea9231729d54dbfd92955f2cbfc21ec236856`。它直接对应“INT8 Tensor Core 模拟 FP64 GEMM”，数值流程集中在三个文件，可清晰分离分片截断、整数积精确性、FP64 重组误差和外围 BLAS 语义。相比从整个 Triton 编译链开始，更容易形成小规模、可反驳的规格。

当前工作机没有 CUDA，第一阶段只能对已读源码的数学与位操作核心做 CPU 重放，并与 NumPy BLAS 及精确有理数参考比较；不能把结果表述为 ozIMMU GPU 实测或库整体正确性证明。使用 CUDA/cuBLAS 的原库比较仍需 NVIDIA 环境。

## 候选比较

“Lean适合度”是本次判断，不是项目自带能力；以下项目均未因本次调研而获得 Lean 正确性证明。

| 候选及源码 | 许可证 | 机制/目标 | 依赖和本机可运行性 | 数值风险与 Lean 切入口 | 本轮优先级 |
|---|---|---|---|---|---|
| [ozIMMU](https://github.com/enp1s0/ozIMMU)，[split.cu](https://github.com/enp1s0/ozIMMU/blob/08eea9231729d54dbfd92955f2cbfc21ec236856/src/split.cu)、[gemm.cu](https://github.com/enp1s0/ozIMMU/blob/08eea9231729d54dbfd92955f2cbfc21ec236856/src/gemm.cu) | MIT | 按共享指数切成有符号 INT8 分片，INT32 GEMM，截取分片乘积三角区域，FP64 重组；提供3–18片及auto模式 | CMake、CUDA、cuBLAS、cutf子模块；Linux动态拦截；Mac不能执行CUDA库 | 分片位宽、指数跨度、截断尾项、整数累加溢出、FP64重组及alpha/beta语义；核心小，适合逐层证明 | **1：本轮** |
| [cuMpSGEMM](https://github.com/enp1s0/cuMpSGEMM)，[device_tcec_wrapper.hpp](https://github.com/enp1s0/cuMpSGEMM/blob/42dd92ea1c0e3d603bbb3f8f354f53e0967a9f4b/src/device_tcec_wrapper.hpp) | MIT | FP16/TF32 Tensor Core + error correction 模拟SGEMM，含 scaling和自动选择 | CUDA、cuBLAS、wmma_extension、cutf；README面向A100 | 主kernel调用wmma_extension的tcec::mma_sync，不能只验证wrapper便宣称纠错算法已验证；需追踪子模块与转换舍入 | 3 |
| [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)，[crt_tree.hpp](https://github.com/RIKEN-RCCS/GEMMul8/blob/d8d390125964c7f1c60d5f1e0e5d9656e5614852/src/oz2/undo_scaling/crt_tree.hpp) | MIT | Ozaki II：模数域低精度GEMM与CRT重建；INT8/FP8后端 | make/config.mk允许CUDA/HIP选择；GPU工具链必需 | 整数模运算与CRT适合Lean；但重建、定点缩放和舍入路径较多，需约束模数乘积容量；本次读到多字长整数、Montgomery与转换代码 | **2：第二阶段** |
| [OzBLAS](https://github.com/mukunoki/ozblas)，[ozblas_split.cpp](https://github.com/mukunoki/ozblas/blob/a6f6198bc7af6cf288607ef867693f991cf48862/src/cpu/ozblas_split.cpp) | MIT | Ozaki浮点切分；DOT/GEMM等CPU与GPU实现，低精度分片与高精度求和 | CPU需要G++、OpenMP、MKL/OpenBLAS；GPU CUDA/cuBLAS；完整测试另需MPFR/MPLAPACK等 | sigma移位切分的精确性依赖舍入与无溢出；编译器重结合会影响(input+sigma)-sigma。CPU原库值得后续移植，但不是Mac开箱即用 | 2，CPU实际库候补 |
| [mpoz2](https://github.com/tkouya/mpoz2)，[oz2_crt.c](https://github.com/tkouya/mpoz2/blob/2a276ba7484943b3043b0d0a865590ae619ff667/src/oz2_crt.c) | BSD-3-Clause | Ozaki II扩至任意精度、多分量精度和LU；GMP整数CRT后转换MPFR | GMP、MPFR、OpenBLAS、C11/OpenMP；Makefile按aarch64或x86选SVE2 i8mm或AMX | scalar CRT用FP64估计商后整数修正，商误差上界值得证明；Makefile不识别Mac的arm64，不能把“CPU无需GPU”解读成Mac即装即用 | 探索性候补 |
| [CUTLASS](https://github.com/NVIDIA/cutlass)，[mma_tensor_op_fast_f32.h](https://github.com/NVIDIA/cutlass/blob/af951046cd9cac9301d8e3e90863fd8d7edc9fa9/include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h) | BSD-3-Clause（文件许可头） | 已读FastF32路径：FP32拆big/small TF32，3项或4项积；是高精度模拟的真实例子 | CUDA、支持相应Tensor Core的GPU，C++模板工程较大 | 3项路径漏small×small；转换默认big toward-zero、small half-ULP truncate。证明必须对应这组模式，不可统一假设round-to-nearest | 3，SGEMM后续 |
| [Triton](https://github.com/triton-lang/triton)，[03-matrix-multiplication.py](https://github.com/triton-lang/triton/blob/7d4f88e3c7065f134b598b2832a46549babf4368/python/tutorials/03-matrix-multiplication.py) | MIT | 已读教程是低精度输入、FP32累加，再转输出；本身不是多分量高精度模拟库 | Triton、PyTorch、支持的CUDA/HIP GPU | 源代码易读但tl.dot隐藏硬件语义；适合建立前端和边界mask证明。需自行加入多分量算法才能对应本任务 | 不选为首个目标 |

## 源码阅读记录与可复查定位

固定引用均通过 GitHub API/raw 内容读取，未只依赖搜索摘要。当前工作空间 shell 无法解析 github.com，因此未成功git clone。vendor/ozIMMU是选定文件快照，**不是完整可构建仓库**。保存了MIT许可证；不应将vendor目录当作完整依赖安装。

| 库 | 本次读取版本 | 实际读取的代表代码/构建入口 |
|---|---|---|
| ozIMMU | 08eea9231729d54dbfd92955f2cbfc21ec236856 | src/split.cu, src/gemm.cu, src/config.cu, src/utils.hpp, LICENSE |
| cuMpSGEMM | 42dd92ea1c0e3d603bbb3f8f354f53e0967a9f4b | src/cumpsgemm_kernel.cuh, src/device_tcec_wrapper.hpp；纠错实现委托子模块，未深审子模块 |
| OzBLAS | a6f6198bc7af6cf288607ef867693f991cf48862 | src/cpu/ozblas_split.cpp；README的依赖和实验性声明 |
| GEMMul8 | d8d390125964c7f1c60d5f1e0e5d9656e5614852 | src/oz2/undo_scaling/crt_tree.hpp, Makefile, make/config.mk |
| mpoz2 | 2a276ba7484943b3043b0d0a865590ae619ff667 | src/oz2_crt.c, Makefile |
| CUTLASS | af951046cd9cac9301d8e3e90863fd8d7edc9fa9（读取main附近查询的HEAD） | include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h；未进行完整模板实例化 |
| Triton | 7d4f88e3c7065f134b598b2832a46549babf4368（读取main附近查询的HEAD） | python/tutorials/03-matrix-multiplication.py, LICENSE；未编译IR |

后两项代码先从main读取，再查询HEAD；存在读取期间分支变动的理论可能，正式复现实验应重新按固定SHA下载并记录哈希。选定的ozIMMU代码从固定SHA读取，无此歧义。

## ozIMMU：可直接开始的规格

### 数据流

1. `get_bits_per_int8(k)` 计算 b=min(7,floor((31-ceil(log2(k)))/2))，k=0另行返回0。研究首轮限定小正k，避免函数本身的极大k移位边界。
2. `get_exp_max_element` 对A的行/B的列做指数最大值归约，`split_int8_kernel` 将最大指数对应的2幂再乘2作为共享scale。
3. `cut_int8_core<double,__uint128_t>` 将有效尾数左移75位、按指数差右移，再重复取最高b位形成带符号INT8片。
4. `get_split_config` 对s片只计算1-based i+j<=s+1的片积，按照对角线从大项到小项累加；不是s²项全算。
5. `accumulate_in_f64` 以及 `axby_kernel` 执行重组、行列scale恢复和alpha/beta操作。

### 推荐定理和测试顺序

- 整数安全：若每片绝对值不超过2^b-1，证明 k(2^b-1)^2 <= 2^31-1，可排除INT32积累溢出。
- 分片重构：在普通有限数、合法移位、有限scale条件下，证明源码分片对应某个定点截断，给出被截断残差。
- 乘积尾项：对未计算的i+j区域给出绝对误差界；不能从“每个输入保留53位”直接推出乘积结果正确舍入。
- FP64重组：单独计入每次加法和最终scale恢复的误差，并规定溢出/下溢前提。
- 整体GEMM：覆盖alpha、beta、非整齐尺寸、转置、leading dimension；不满足这些条件的规格不能称作完整BLAS正确性。

实验应比较：标准BLAS结果、源码核心重放结果、精确有理数AB结果。标准BLAS自身也有舍入误差，所以两库不同不自动说明低精度实现错误。CPU重放应报告它省略了GPU索引、并发、cuBLAS INT8调用和编译器层。

### 具体审计线索：复数beta更新

选定版本 `src/gemm.cu` 的 `init_c_complex_kernel<false>` 按以下顺序更新：
```cpp
c.x = c.x * beta.x - c.y * beta.y;
c.y = c.y * beta.x + c.x * beta.y;
```
第二条语句读的是已经更新的实部。取原C=1+2i，beta=3+4i，语句计算为-5-14i，而标准复数乘法为-5+10i。这是源码级确定的算术反例，所有数都能精确表示，无需舍入误差才能触发。复数INT8 GEMM路径会先调用该初始化函数再累加乘积贡献。正式上游缺陷报告还应附完整GPU调用复现；本次未运行GPU，不能声称已在GPU实测。

正确修复思路是保留旧实部和旧虚部后同时计算结果。这个外围缺陷虽然不是低精度逼近误差，却说明“BLAS库正确性”同时包含普通程序语义，可以先作为Lean中的小反例。

另有应进一步验证而非直接宣称bug的风险：`cut_int8_core` 的128位右移若指数差>=128，需要确定CUDA编译器/设备语义；非有限数、全部subnormal的共享指数尺度、最大正规数scale溢出，以及signed左移语义均不应被Python无限精度整数重放掩盖。

## 下一轮所需运行条件

- NVIDIA GPU与CUDA/cuBLAS；固定GPU架构、driver、toolkit、cuBLAS和编译flags。
- 完整固定commit仓库及submodules。
- 实数实验先固定alpha=1,beta=0,N/N，小矩阵与多种k；额外复数反例用非零复数beta。
- 可显式选择标准cuBLAS DGEMM作为基线，关闭本库的拦截后运行同一输入，再通过ozIMMU入口执行。
- 输出同时记录最大绝对误差、ULP差、相对于sum(abs(a*b))的误差、NaN/Inf计数，避免仅用相消后结果作相对误差分母。
- 第一阶段无形式化成果；只有通过Lean checker且声明完整前提的定理，才能标记为形式化证明。

