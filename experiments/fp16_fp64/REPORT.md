# FP16 模拟 FP64：OzBLAS 最小实验

日期：2026-11-22。结论：找到了真实开源的FP16→FP64 GEMM接口，并已完成CPU数值模型及求和顺序实验；GPU原库执行仍受SSH连接阻塞，未运行Lean。

## 选定的原库路径

库：[mukunoki/ozblas](https://github.com/mukunoki/ozblas)，固定提交 `a6f6198bc7af6cf288607ef867693f991cf48862`，MIT许可证，作者标注实验性alpha实现。

- `cuozblasRgemm<double, half, float>`：FP64输入/输出、FP16分片、FP32片积。
- `make.inc`：`PREC_D_H=yes`，列出 `CC70` 并注明测试过CC70/75/86。
- `src/cuda/Makefile`：CC70映射到 `compute_70/sm_70`。
- `cuozblas_blas_wrapper.cu:210`：调用 `cublasGemmEx`，输入CUDA_R_16F，输出CUDA_R_32F，CUBLAS_COMPUTE_32F，请求Tensor Op算法。该调用本身不是硬件确实用了Tensor Core的证据。

完整数值流程仍使用FP64分解与重组，并不是只靠FP16运算就能实现任意FP64语义。FP64输入被拆成多个缩放后的FP16片，FP32计算片积，最后恢复尺度并FP64求和。

## 模型范围

对应 `splitEpsMode=0, splitShift=1, fastMode=0, sumMode=2, useBatchedGemmFlag=0, alpha=1, beta=0`，实数NN，小矩阵，缩放远离FP64溢出/下溢。求和模式2是可选择的原库路径，不是默认模式0；没有模拟全部库功能。

分片参数为 `rho=ceil(53-(24-log2(max(K,2)))/2)`。这里24来自原库`getEpse<half>()`的FP32片积累加位数，并非FP16自身的有效位数。每轮根据行/列残差最大值计算tau与sigma，以两次FP64舍入求 `(x+sigma)-sigma`，缩放后存入FP16，并保存FP64残差。

CPU模型逐个检查：分片缩放值可由FP16精确表示；旧输入精确等于当前分片加残差；每个FP32片积按正序/逆序/平衡树三种顺序的结果均等于Fraction精确点积。最后按源码的交叉项对角线顺序进行FP64重组，并比较逆序、平衡树和Python math.fsum。

这些检查仅是本次有限样本证据，不是Lean证明，也不建立真实Tensor Core累加等同于CPU FP32的全称语义。V100内部归约、编译器、cuda log2和指令细节仍待核对。

**零残差扩展**：某一行/列提前分解完毕时，模型显式补零片；原源码会继续处理devMax=0并计算log2(0)。因此模型在这类状态不是完全忠实重放。本次spread_40、12片上限触发此扩展（A/B各2个row-step），普通随机和相消样本未触发。该源代码边界应另做GPU复现，不能直接据模型宣称原库结果。

## 已执行的结果

4组输入，每组NumPy FP64、直接FP16/FP32、3/6/12片上限×4种最终求和方式，共56条记录（含一条直接FP16转换溢出的状态）。分片会提前终止，因此“6片”是上限。

指标为 `max_ij abs(Y_ij-exact_ij) / max_ij abs(exact_ij)`；不是逐元素相对误差，也不代表全部输入精度保证。参考值为输入binary64精确解码后、用Python Fraction做的精确GEMM。NumPy结果作为标准库差分参考。

| 输入 | 方法 | 归一化最大误差 |
|---|---|---:|
| uniform_k16 | numpy_fp64 | 2.262e-16 |
| uniform_k16 | naive_fp16_fp32 | 4.899e-04 |
| uniform_k16 | ozblas_model_s3_forward | 1.730e-10 |
| uniform_k16 | ozblas_model_s6_forward | 1.185e-16 |
| uniform_k16 | ozblas_model_s12_forward | 1.185e-16 |
| uniform_k64 | numpy_fp64 | 9.858e-17 |
| uniform_k64 | naive_fp16_fp32 | 2.174e-04 |
| uniform_k64 | ozblas_model_s3_forward | 1.084e-09 |
| uniform_k64 | ozblas_model_s6_forward | 7.385e-17 |
| uniform_k64 | ozblas_model_s12_forward | 7.385e-17 |
| spread_40 | numpy_fp64 | 7.394e-17 |
| spread_40 | naive_fp16_fp32 | conversion_overflow |
| spread_40 | ozblas_model_s3_forward | 3.401e-09 |
| spread_40 | ozblas_model_s6_forward | 1.385e-16 |
| spread_40 | ozblas_model_s12_forward | 1.385e-16 |
| cancellation | numpy_fp64 | 0.000e+00 |
| cancellation | naive_fp16_fp32 | 1.000e+00 |
| cancellation | ozblas_model_s3_forward | 0.000e+00 |
| cancellation | ozblas_model_s6_forward | 0.000e+00 |
| cancellation | ozblas_model_s12_forward | 0.000e+00 |

普通随机K=16在最多6片时实际用5×5个片积；K=64实际用6×6个片积。两组误差均达到binary64舍入量级。这不证明“所有输入6片足够”，也不意味着性能一定超过V100原生DGEMM。

spread_40的原始输入动态范围超过直接FP16可表示范围，直接转换发生溢出；带缩放的分片模型仍可计算。相消例 `[1,2^-45,-1,2^-45]` × 全1列的精确结果为2^-44，直接FP16丢失小项；自适应残差切分模型在2个A片和1个B片时保留它。

## 累加顺序实验

发现7组（输入、分片上限）组合中，最终FP64正序与逆序结果有不同。随机K=16、最多6片的(0,0)元素：

- 正序：`-0x1.5cb20c2086c29p+0`
- 逆序：`-0x1.5cb20c2086c28p+0`
- 差异：1 ULP；该矩阵最大顺序差异为2.220446049250313e-16。

三种FP32片积顺序在本次样本都精确，但FP64重组仍会舍入，故“片积精确”和“整体结果与顺序无关”是两个不同命题。逆序、平衡树、math.fsum是实验变体，不是声称原库提供这些模式。`math.fsum`也不是原库的NearSum：本版本对TYPE1!=TYPE3的NearSum明确报不支持，D_H属于该情况。

## GPU入口与当前阻塞

`gpu_compare.cu` 直接调用原库模板实例及独立的cublasDgemm，避免LD_PRELOAD和类型配置导致误跑。初始化输出为0，检查CUDA错误和分片统计，保存两路原始binary64结果。它只请求一张逻辑可见GPU，不覆盖CUDA_VISIBLE_DEVICES。

`run_gpu.sh` 要求完整、干净、固定版本checkout；以sm_70强制重编译，避免复用仓库自带的.o文件。包含半精度类型头文件的显式编译配置。当前只完成bash语法检查；本地无nvcc，CUDA编译和运行尚未验证，可能需要按远端toolkit解决构建兼容问题。

`analyze_gpu.py` 将GPU输出与同一输入的精确Fraction GEMM比较。已用明确标为synthetic的精确参考文件检查读取和误差计算，并确认截断文件会被拒绝。这些synthetic文件不作为GPU结果保存。

再次执行SSH仍返回登录节点域名解析失败，未进入服务器，也未执行login_s8.sh。

## 复现命令

CPU已执行：

```sh
python3 compare.py
```

远端恢复后，在已分配V100的计算节点上，用支持sm_70的CUDA工具链执行（以下未运行）：

```sh
git clone https://github.com/mukunoki/ozblas.git ozblas-fp16-validation
cd ozblas-fp16-validation
git checkout a6f6198bc7af6cf288607ef867693f991cf48862
cd ..
bash /path/to/fp16_fp64/run_gpu.sh "$PWD/ozblas-fp16-validation"   /path/to/fp16_fp64/uniform_k16.bin "$PWD/results/uniform_k16"
```

原始输入已保存在四个.bin文件里，格式：little-endian uint32 M,N,K，接column-major binary64 A、B。脚本输出前缀包含编译日志、环境、实际运行日志、二进制结果及误差JSON。必须保存CUDA/cuBLAS版本、实际执行路径和二进制hash。后续可用Nsight/指令检查确认Tensor Core实际执行；查看wrapper的SASS不等于检查动态cuBLAS内部kernel。

## Lean切入点

相关原始研究：[DGEMM Using Tensor Cores, and Its Accurate and Reproducible Versions](https://pmc.ncbi.nlm.nih.gov/articles/PMC7295351/)，ISC 2020。论文已测试V100，其DP模式未加速cuBLAS DGEMM；本轮首先关注数值正确性。关于用户提供的TensorLean报道及研究定位，见[补充调研](../../literature/TENSORLEAN_AND_FP16.md)。

1. 分片表达式的舍入语义及输入=片+残差的条件。
2. FP16缩放片的可表示性；FP32片积在位宽/长度条件下对允许的归约顺序精确。
3. 最终FP64重组的误差界；再与分片截断误差组合。
4. 指定执行图的误差契约，与“所有合法重排都在同一误差界内”分别定义。

截至本报告，没有新Lean定理经过checker。下一步先完成原库GPU结果与CPU模型的一致性检查，解释任何差异，再固定形式化模型。

## 固定源码链接

- [GEMM实例与片积顺序](https://github.com/mukunoki/ozblas/blob/a6f6198bc7af6cf288607ef867693f991cf48862/src/cuda/cuozblas_gemm.cu)
- [分片](https://github.com/mukunoki/ozblas/blob/a6f6198bc7af6cf288607ef867693f991cf48862/src/cuda/cuozblas_split.cu)
- [cuBLAS wrapper](https://github.com/mukunoki/ozblas/blob/a6f6198bc7af6cf288607ef867693f991cf48862/src/cuda/cuozblas_blas_wrapper.cu)
- [求和](https://github.com/mukunoki/ozblas/blob/a6f6198bc7af6cf288607ef867693f991cf48862/src/cuda/cuozblas_sum.cu)
- [V100构建配置](https://github.com/mukunoki/ozblas/blob/a6f6198bc7af6cf288607ef867693f991cf48862/make.inc)
