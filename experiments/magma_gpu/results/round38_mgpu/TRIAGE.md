# 第 38 轮：多 GPU / 核外驱动在 ngpu = 1 下的数值差分——分诊

程序：`probes/round38_mgpu.cpp`（roxy 上只有一块 GPU，所以 ngpu = 1）。以前的轮次只检查了这些驱动的状态逻辑。ngpu = 1 时，`_m`/`_mgpu` 仍然走自己的代码路径（各自的面板调度和主机端核外循环），所以值得在数值上与 LAPACK 逐一比较。

| 组 | 例程 | 用例 | 结果 |
|---|---|---|---|
| fact | dpotrf_m、dpotrf_mgpu（L/U，n 为 1–1025），dgetrf_m、dgetrf_mgpu、dgeqrf_m、dgeqrf2_mgpu（含 600×257 和 257×600） | 44 | 全部正确 |
| eig | dsyevd_m、dsyevdx_m、dsyevdx_2stage_m（A/I）、dsygvd_m、dsygvdx_m、dgeev_m | 52 | 46 例正确；6 个 BAD 都是 n = 1 时 mout 未设置：已知 M236 在 `_m` 文件中的同一缺陷，已把 zheevdx_m、zheevdx_2stage_m 补进 M236 |
| red | dgehrd_m、dsytrd_mgpu（L/U） | 12 | dsytrd_mgpu 8/8 正确。dgehrd_m 按文档给 LWORK = N·NB 时一律返回 −8，因为代码要求 n·(nb + nb·ngpu)，这是已知的 M049；改用查询得到的 lwork 后 4/4 正确（`red_query.csv`） |

- dpotrf_mgpu_right 无法链接：它在头文件中有声明，但函数体整个包在 `#if 0` 里（已知 M156），这次在 roxy 上以链接错误的形式得到确认。
- 结论：ngpu = 1 时，多 GPU 驱动的数值路径没有发现新缺陷。多卡之间的数据分布和通信仍然需要第二块 GPU 才能测。
