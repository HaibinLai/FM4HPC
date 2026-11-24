# 第五轮调研：可在 roxy（RTX 5070 Ti，sm_120）上测试的 A/B 两类低精度库

日期：2026-11-23。

**选题标准**（PLAN.md 第0节）：
- 属于A类（AI时代写出、要基本可靠）或B类（低精度模拟高精度、要高可靠）；
- 失败真实发生过；
- 验证能给出测试给不了的东西。

用户在本轮指定 **B类五项为主线**；A类作为备查。

**证据等级：**

| 标记 | 含义 |
|---|---|
| **V** | 读过源码、文档或GitHub API输出 |
| **V\*** | 本会话主线程亲自复核过 |
| **S** | 仅来自搜索摘要 |
| **Inf** | 推断，未测 |

本轮**没有在GPU上运行任何东西**。

## B类（主线）

| 对象 | 能否在sm_120上运行 | 精度/回退由什么决定 | 文档的承诺 | 已知的真实失败 | 我们能做什么 |
|---|---|---|---|---|---|
| **cuBLAS FP64仿真**（Fixed-Point，Ozaki-I/II，ADP），闭源 | **能**：文档支持表写“Fixed-Point FP64：8.x, 9.0, 10.x, 11.0, 12.x，CUDA 13.0u2+”（V\*） | ADP（动态尾数控制，默认开启）计算所需尾数位，超过默认值或用户上限就回退原生FP64（V\*）。默认策略PERFORMANT只在“有性能收益时”才仿真，EAGER是“尽可能仿真”（V\*）。环境变量 `CUBLAS_EMULATE_DOUBLE_PRECISION=1` 可以在不改代码的情况下启用（V\*） | “accuracy **equal or better than FP64**”；“support the **full range of finite input values (including denormals)**”；Fixed-Point对Inf/NaN“None”隐式支持，需打开special-values掩码（V\*） | 发布说明里有错误结果类缺陷：实验性ESC结果错误（#6862950）、Ozaki-II ZGEMM“rare cases”错误（CUB-10890）、out-of-place错误（#6854729）（V，子代理读的发布说明，主线程未复核）。ADP论文（arXiv 2511.13778）的ESC在“同号”假设下才保证等价（V，子代理据论文摘要） | 黑盒差分：用第16/17节攻破ozIMMU/GEMMul8的输入（一行一列极端、结构化零、2^±512附近、非规格化数、Inf/NaN、−0），对照精确有理数参考，分别测PERFORMANT/EAGER。理论：在论文的ESC模型下证明“ESC ⇒ 逐元素误差界”，把“同号”假设与文档的无条件承诺对比 |
| **GEMMul8 FP8后端** | 有sm_120专用代码：`mod_hi2mid.hpp:135`（`__CUDA_ARCH__>=1200`）、`crt_group.hpp:327`（V）。README只排除Hopper（V\*，第17节读过） | 与INT8后端相同的 `calc_sft`/`log2P`，另有FP8 limb选择（`common/fp8_limb_selection.hpp`、`fp8_plan.hpp`）（V）；FP8 GEMM用 `CUDA_R_8F_E4M3` + `CUBLAS_COMPUTE_32F`（V） | 推荐模数“不是精度保证”（V\*） | #7/#8：accurate模式复用已缩放操作数时移位公式错误，丢5–7位，2026-11-12修复（V\*）；PR #4：旧fast模式缩放“could fail CRT reconstruction”（S，查询超时未复核） | 直接复用第17节的探针、穷举与Lean（`Gemmul8Range.lean`）；新增：FP8 limb表的穷举，以及“FP8 MMA + FP32累加在sm_120上是否精确”的微基准（k上界在哪里强制，子代理没找到） |
| **cuSOLVER IRS**（闭源）/ **MAGMA IR-GMRES**（BSD-3） | cuSOLVER：CC≥5.0（V）；MAGMA的CMake中Blackwell→`sm_100 sm_120`（V） | cuSOLVER：停止条件 `RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX`；负的niters编码（−2：降精度转换溢出，−50：达到最大迭代）；默认回退到主精度，关闭回退时“不保证解是好的”（V）。MAGMA：`dxgesv_gmres_gpu.cpp` 第492行 `cte = Anrm*eps*sqrt(n)*BWDMAX`，第648行 `if (Rnrm > Xnrm*cte) goto refinement;` 之后直接 `*iter = 0; goto cleanup;`（V\*） | 只承诺**按范数的后向误差**，前向误差仍可达 κ·eps（V/Inf） | **MAGMA的NaN问题**：若Rnrm或Xnrm为NaN，比较为假，于是被当作已收敛（结构V\*，行为未测）；Rnrm取自 `idamax` 选出的元素，idamax遇到NaN的行为依实现而定（Inf） | 定理：“浮点接受测试通过 ⇒ 真后向误差 ≤ √n·eps + 残差/范数计算误差”；NaN安全性证明或反例。黑盒：κ逼近 1/u_low、元素>65504、b含NaN、关闭回退 |
| **cuMpSGEMM / TCEC**（Ootomo，MIT） | 未声明：CMake硬编码 `CMAKE_CUDA_ARCHITECTURES 80`；sm_120落到sm86配置；TCEC把≥900映射到sm_90（V）；靠LD_PRELOAD拦截cuBLAS，只能在Linux/WSL2用（V） | AUTO模式（`src/exp_stats.cu`）按“|a| < 2^-15·max的元素数”决定FP16TCEC还是TF32TCEC；m、n或k≤1024时交给cuBLAS SIMT（V） | 面向A100（V）；最后提交2025-02（V） | 无直接的失败记录 | FP32的hi/lo分解对全部2^32个值做穷举；AUTO规则 ⇒ 下溢比例界。**优先级低**（陈旧，非sm_120目标） |
| **cuBLAS BF16x9**（FP32仿真） | **不能**：支持表“BF16x9 FP32：10.0, 10.3，CUDA 12.9+”（V\*） | — | FP32“exactly represented as three BF16 values”（V\*） | — | 只能测“在sm_120上请求它会返回什么”（报错，还是静默回退）；分解的穷举可以在CPU上做。**排除为主目标** |

### 实测进展（2026-11-23，HANDOFF 第19节）

cuBLAS FP64仿真已在roxy上黑盒测试，只用Python，没有编译。

- **精度**：所有对抗输入都达到FP64级。
- **性能缺陷**：
  - 零块/三角×三角使粗化ESC报出4248位，回退原生后比原生慢约11%；
  - 需要100位以上时同样回退，比原生慢；
  - 回退时位数指针不报−1。
- **第21节（`AdpEsc.lean`）**：
  - 位数公式拟合为 `x+y−est+54`，块长64，零记为 −4195，38/38 与实测一致；
  - Lean证明：零码安全的充要条件，以及“估计 ⇒ 逐元素误差界”的完整链条；
  - 安全的零感知修正：在模型下把零块情形从4248位降到54位；按此位数用固定模式运行，n=4096 时 239 ms → 27 ms，精度不变。
- **第22节（cuSOLVER IRS）**：
  - 经典精化符合文档；
  - **GMRES 精化不执行文档判据**：默认的 `DBgesv`/`DXgesv`/`DSgesv` 在 A、b 缩放 2^40 或 2^−60 时报“收敛”，不回退，返回错误解（DBgesv 相对误差最高 23）；
  - NaN 安全；
  - MAGMA 待 s82 恢复后再测。

### B类建议顺序

1. **cuBLAS FP64仿真（ADP）**：
   - 理由：sm_120上的生产路径；文档对精度和取值范围的承诺是无条件的；源码闭源，独立检验的价值最大；我们已有一整套对抗输入。
   - 很可能**只需Python + CUDA 13 wheel**（cuBLAS 13.0u2+，用环境变量启用，ctypes调用），不需要编译器。
2. **GEMMul8 FP8后端**：资产复用最多（探针、穷举、Lean）；需要编译（建议WSL2 + CUDA 13）。
3. **cuSOLVER IRS / MAGMA**：
   - cuSOLVER可以用ctypes黑盒测；MAGMA的NaN接受问题需要编译后复现；
   - 接受测试的可靠性定理与ozIMMU/GEMMul8的“选参 ⇒ 精度”是同一类问题。
4. cuMpSGEMM/TCEC：陈旧，且只能在Linux上拦截。
5. BF16x9：sm_120不支持。

## A类（备查）

子代理报告，其中主线程复核过的标V\*。

- **SageAttention**（Apache-2.0，2026-01后无推送）：
  - sm120走INT8-QK / FP8-PV的“fp32+fp16”累加路径，依据是注释“sm120 has accurate fp32 accumulator for fp8 mma”（`core.py:153`，V\*）；
  - `MeanScaleKernel` 计算 `recp_scale = scale_max / s_amax_val` 时没有防零（`fused.cu` 第385–399行），而同文件的 `QuantInt8Kernel` 有 `amax_val = 1e-7`（第147行）（V\*）。某通道恒定时会得到 0·inf = NaN，很可能就是 #164（RTX 50xx NaN）的根因（Inf，未复现）；
  - 其他：#392 sm120在CUDA graph重放下输出约90%错误、#386 int32偏移溢出、#406 SA3 FP4掩码错误（V，子代理）。
- **NVFP4/MXFP4块缩放量化器**（vLLM / FlashInfer / TRT-LLM 共用、彼此漂移）：
  - 失败：vLLM #67914、FlashInfer #8083、#9333、QuTLASS #15（V，子代理）；
  - 子代理对全部31743个正的有限fp16 amax做了穷举：代码注释“max_val/scale ≤ 6.0”在7897个情形下不成立，但都没有舍入到超过6（V，子代理，主线程未复核）。
- **llama.cpp MXFP4/NVFP4**：
  - Blackwell上MXFP4张量会静默改走W4A4（#32683）；
  - E8M0的软件回退把0xFF解为+inf，而OCP规范定义为NaN（V，子代理）。
- 不适用：DeepGEMM只支持SM90/SM100（V）。

## 对 roxy 环境的含义

| 阶段 | 内容 | 前提 |
|---|---|---|
| 阶段1，**不装系统软件** | 在 `D:\lean4hpc` 建Python虚拟环境，装 `nvidia-cublas`（13.x）等wheel，用ctypes测cuBLAS FP64仿真（以及cuSOLVER IRS） | 先确认wheel中的cuBLAS版本 ≥ 13.0u2，以及用ctypes调用的可行性 |
| 阶段2 | GEMMul8 FP8、MAGMA需要编译，建议WSL2 + Ubuntu + CUDA 13 | 需管理员权限，可能要重启，**要用户同意** |

## 局限

- 子代理用的是未认证的GitHub API（限流），部分搜索被截断。
- 发布说明里的缺陷编号与ADP论文的细节，主线程没有逐条复核。
- 对所有“sm_120能跑”的判断都来自文档或构建文件，**没有实机验证**。
