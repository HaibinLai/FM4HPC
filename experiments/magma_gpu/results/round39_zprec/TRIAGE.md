# 第 39 轮：第 31–37 轮新发现在复数精度下的复核

程序：`probes/round39_zprec.cpp`。每条发现用一个最小触发用例，并配一个对照用例。

| 发现 | 例程 | 触发用例 | 对照用例 | 结论 |
|---|---|---|---|---|
| M242 | zgetrf_native | n = 513、ldda = 513：误差 6.3e9 | ldda = 544：0.14 | 复现 |
| M242 | cgetrf_native | n = 513、ldda = 513：0.156；n = 512、ldda = 516：0.144 | 0.156 | **未复现**：本轮两个配置下 c 精度都正确；第 31 轮 zcgesv（内部调用 cgetrf_native）在 n = 513 时也收敛，与此一致 |
| M245 | zgemm_batched、zherk_batched | k = 0、beta = 0.5：C 没有乘 beta（偏差 0.35） | — | 复现 |
| M246 | zherk_batched | n = 31、k = 17、L：3 个矩阵共 1,395 个严格上三角元素，被改动的计数为 1.4e3（输出按 3 位有效数字打印） | — | 复现 |
| M247 | ztrsm_batched | alpha = 0、B 为 NaN：465 个元素中 465 个不是 0 | — | 复现 |
| M249 | ztrevc3 | howmany = A、lwork = (1+2·128)n：残差 0.0087 | lwork = 2n：0.0087 | **未复现**：复数版也会在没有 `over` 时选用版本 2，但它的回代 GEMM 写在 `else if (version == 2)` 分支里，只有 over 时才执行；实数版（dtrevc3、dtrevc3_mt 第 603 行）把这段 GEMM 放在分支链之外。所以 M249 只影响 [ds]trevc3(_mt) |
| M250 | zgesv_rbt | n = 1、2、3：误差 0.26–0.55 | n = 5：7.9e−17 | 复现 |

FINDINGS 已据此修正：M242 的范围改为 [sdz]（注明 c 未复现），M249 的范围改为 [ds]；其余各条补充了 z 精度的证据。
