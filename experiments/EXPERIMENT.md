# ozIMMU 第一轮比较实验

日期：2026-11-22（America/Chicago）。选择对象：enp1s0/ozIMMU，固定提交 `08eea9231729d54dbfd92955f2cbfc21ec236856`。

## 已做与未做

已运行两个实验：①按真实源码构造的实数 INT8 切片 GEMM CPU 算术模型，对比 NumPy FP64 matmul 和精确 Fraction 参考；②原样抽取复数 beta*C kernel 的函数体，以单线程 C++ host replay 编译执行，对比 Python 内建 complex。

没有运行 ozIMMU CUDA 库、cuBLAS、GPU 性能测试或 Lean checker。当前 macOS ARM 环境无 CUDA/Lean；shell 无法解析 GitHub，源码由 GitHub 连接获取。不能将这些结果称为原库 GPU 已验证，也不能据此声称证明了误差界。`ComplexBeta.lean` 是没有 sorry 的待编译草稿，不是已检查证明。

## 为什么选择 ozIMMU

它的目标就是用 INT8 单元模拟 FP64 GEMM，契合课题；分片、INT32 点积、FP64 重组有相对清晰的源码边界。整数累加可建立简单的无溢出前提。固定源码和配置顺序均已留存。相比之下，直接启动完整 CUTLASS/Triton 数值语义验证会包含更多并行和浮点指令语义工作。

## 实数模型与结果

`compare_ozimmu.py` 对照 `split.cu:cut_int8_core/get_bits_per_int8`、`config.cu:get_split_config`、`gemm.cu:accumulate_in_f64_kernel/axby_kernel` 实现：

- p=min(7,floor((31-ceil(log2(K)))/2))；A 按行、B 按列取共享指数。
- 用 uint128 截断移位模型抽取 signed INT8 切片，并用独立 Fraction 公式核对每一个 digit。
- 按源码顺序只累加 i+j<s 的交叉项；INT64 计算点积并检查 INT32 范围，随后按源码缩放与 FP64 累加。
- 只覆盖 real NN、alpha=1、beta=0、非零 normal 输入、uint128 移位<128，且本次测试缩放远离溢出/下溢。零、subnormal、过大指数跨度和 max-double 输入会明确拒绝。
- Padding 的零贡献在数学上省略；没有验证 GPU 地址布局、padding写入、实际 cublas 调用或编译器 signed shift 语义。
- 基线为 `numpy.float64 @`，其 BLAS 配置信息留在 results.json；它不是 cuBLAS，也不假定其结果逐位正确。
- 精确参考使用 `Fraction.from_float` 等价转换和任意精度有理数乘加，计算给定二进制输入的精确实数积。误差先在 Fraction 中计算，最后仅为显示转 float。

运行种子 20261122，共 7 组输入 × 6 种实现 = 42 条结果。下表的归一化量是 max-entry 误差 / max-entry 精确值，不是逐元素相对误差，也不是诱导无穷范数。完整结果在 results.json。

| 输入 | 实现 | 最大绝对误差 | 最大误差 / 最大精确结果 |
|---|---|---:|---:|
| uniform_k64 | numpy_fp64 | 1.345e-15 | 1.921e-16 |
| uniform_k64 | ozimmu_cpu_s3 | 3.009e-05 | 4.299e-06 |
| uniform_k64 | ozimmu_cpu_s6 | 3.164e-11 | 4.521e-12 |
| uniform_k64 | ozimmu_cpu_s9 | 3.121e-15 | 4.459e-16 |
| exponent_spread_80 | numpy_fp64 | 1.307e+08 | 9.621e-17 |
| exponent_spread_80 | ozimmu_cpu_s3 | 7.725e+18 | 5.687e-06 |
| exponent_spread_80 | ozimmu_cpu_s6 | 1.207e+13 | 8.888e-12 |
| exponent_spread_80 | ozimmu_cpu_s9 | 1.307e+08 | 9.621e-17 |
| cancellation | numpy_fp64 | 0.000e+00 | 0.000e+00 |
| cancellation | ozimmu_cpu_s3 | 5.684e-14 | 1.000e+00 |
| cancellation | ozimmu_cpu_s6 | 5.684e-14 | 1.000e+00 |
| cancellation | ozimmu_cpu_s9 | 0.000e+00 | 0.000e+00 |

随机测试支持“切片增加能减少截断误差”，不是全输入保证。9、12、18片在这些样本可能相同，是最终FP64累加精度的限制；不能推出所有输入9片足够。

相消例 A=[1,2^-45,-1,2^-45]，B=[1,1,1,1]^T，精确结果为2^-44。3/6片模型输出0，9片恢复精确结果。这是有限切片精度的例子，不能当作违反了库承诺的未知bug。文献已有对 badly scaled 输入的相关分析。

4类输入拒绝和7个K边界的整数上界检查均通过。INT32验证仅覆盖抽样K，不构成全称证明。每次点积的全部部分和也可由绝对值和界约束，这是后续 Lean 的合适目标。

## 实际源码中的独立算术问题

`gemm.cu:init_c_complex_kernel<false>` 依次执行：

```cpp
c.x = c.x * beta.x - c.y * beta.y;
c.y = c.y * beta.x + c.x * beta.y;
```

第二句读取已被覆盖的 c.x。固定源码的函数体被直接抽取到 `complex_beta_replay.cpp`，只以 host stub 替换 CUDA 线程坐标和复数结构，clang++ -std=c++17 -O0 编译执行成功。

| 输入 | 原函数体 host 结果 | Python 标准 complex 结果 |
|---|---|---|
| C=1+2i，beta=3+4i | -5-14i | -5+10i |

这些小整数的乘加都可由 binary64 精确表示，差异来自覆盖顺序；这不是低精度 Tensor Core 误差。调用链检查显示 complex int8 GEMM 会调用这个初始化函数。这里只确认固定版本函数体的算术反例；完整GPU入口复现、所用CUDA版本以及维护者确认仍待做。没有提交外部issue，也没有修改上游文件。

修复方向：同时从旧 x/y 计算两个新分量，最后再写回。`ComplexBeta.lean` 表达同一组整数反例，但本机未安装 Lean，尚未检查。

## 后续形式化与GPU复现

1. 首先编译 ComplexBeta.lean；再给源语句与浮点模型建立对应关系，不把 Int 例子当作完整浮点证明。
2. 证明任意K个切片乘积的每个部分和绝对值不超过 K*M_A*M_B，若小于2^31则INT32无溢出。
3. 证明128-bit提取在合法移位、输入范围、切片数下满足残差界；将所有前提写成检查器。
4. 证明被省略的交叉项与FP64重组的误差，分析相消如何影响相对误差。
5. 在有 CUDA 的 Linux GPU 上固定相同提交、cutf子模块、CUDA/cuBLAS版本，调用 real/complex 入口，比较 cublasDgemm/cublasZgemm 与同一 Fraction oracle。复数beta反例要取合法shape和对齐，先确认进入 int8 路径。

## 复现

在本目录运行：

```sh
python3 compare_ozimmu.py
python3 replay_complex_beta.py
# 装有 Lean 4 时，另行检查：
lean ComplexBeta.lean
```

Python依赖 NumPy；第二项需要 clang++。结果记录Python/NumPy/平台/编译器版本。完整源码快照与其LICENSE位于 ../operators/vendor/ozIMMU。

## 来源

- [ozIMMU 固定提交](https://github.com/enp1s0/ozIMMU/tree/08eea9231729d54dbfd92955f2cbfc21ec236856)
- [split.cu](https://github.com/enp1s0/ozIMMU/blob/08eea9231729d54dbfd92955f2cbfc21ec236856/src/split.cu)
- [gemm.cu](https://github.com/enp1s0/ozIMMU/blob/08eea9231729d54dbfd92955f2cbfc21ec236856/src/gemm.cu)
- [config.cu](https://github.com/enp1s0/ozIMMU/blob/08eea9231729d54dbfd92955f2cbfc21ec236856/src/config.cu)
