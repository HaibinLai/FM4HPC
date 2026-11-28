# 第 35 轮：其余批量 BLAS（symm、symv、trsv、potrs）——分诊

程序：`probes/round34_bblas.cpp`，新增 symm、symv、trsv（含 potrs_batched）三组，以及追查用的 symmdbg、symmstruct、symmeye、symmprec 四种模式；补丁验证程序见 `probes/hemmfix/` 和 `scripts/roxy/build_fix35.bat`。A 不应被引用的那一侧三角填 NaN；beta = 0 时 C/y 填 NaN。

| 例程 | 结果 | 归属 |
|---|---|---|
| dtrsv_batched（n 为 1–257，uplo、trans、diag 各两种，incb ∈ {1, 2}） | 112/112 正确 | — |
| dpotrs_batched（n 为 1–513，nrhs ∈ {1, 3}；A 的上三角放哨兵） | 16/16 正确，A 不被改动 | — |
| dsymv_batched / vbatched | beta = 0 时 y 中的 NaN 被保留（30 例），其余正确 | 归入 M234（hemv_template_device.cuh:53 的 `ry = Y*beta`） |
| **magmablas_[zcsd]hemm/symm**（非批量、batched、vbatched） | **任何输入的结果都错**（见下） | **新 M248** |

## M248：hemm 模板把基指针传给了 fetch()

- 现象：A = I 时 C 应等于 B，但结果错误（`symmeye.csv`）。m 从 1 到 128、L/R、U/L、批量和非批量，全部出错；`symmprec.csv` 中 4 种精度的最大误差为 1.5–13.7。同样的输入下，cuBLAS（magma_dsymm）和 magmablas_dgemm 的结果都精确正确。
- 定位：`gemm_template_device_defs.cuh:70` 定义 `#define fetch(A, m, n, bound) A[min(n*LD##A+m, bound)]`，它的其他使用者 read_gm2sm_* 传入的是每个线程的偏移指针 `offs_dA`。而 `hemm_template_device.cuh` 的 32 处 fetch 调用（如 `fetch(A, m, n, boundA)`） 传的是内核的基指针 A，所以一个线程块中的所有线程每一步只读到循环偏移处的那个元素；`offs_dA` 被计算、也被逐步推进，但从来没有被读取。
- 验证：在 hemm_template_device.cuh 的副本里局部改为 `#define fetch(A, m, n, bound) offs_d##A[...]`，重新编译 dsymm.cu、dsymm_batched_core.cu、dsymm_vbatched_core.cu，链接到库之前。结果：symmprec 3/3 精确；完整的 symm 组中，beta ≠ 0 的 40 例全部正确（`symm_hemmfix.csv`）。
- 剩下的 beta = 0 用例（40 例）仍然得到 NaN：第 252 行的 `memC = alpha*rC + beta*memC` 无条件读 C。这一点归入 M234。
- 影响范围：库内部只有 zhetrd_he2hb_mgpu 用到 hemm，而它走的是自己的 zhemm_mgpu 实现，所以受影响的是直接调用公开接口的用户。MAGMA 自带的 testing_zhemm 拿 magmablas_zhemm 和 blasf77_zhemm 比较，按理应该能发现。我手头的源码副本只有 2.10.0 这一个提交，无法判断是不是这个版本新引入的；roxy 上的二进制只在 sm_120 上测过，但这个错误在源码层面，与架构无关。
