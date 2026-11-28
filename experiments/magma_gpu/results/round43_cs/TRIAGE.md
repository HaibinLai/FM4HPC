# 第 43 轮：M245–M247、M250 在 s、c 精度下的复核

程序：`probes/round43_cs.cpp`（roxy，2026-11-27）。16 项全部复现：

| 发现 | s | c |
|---|---|---|
| M245（批量 gemm/herk 在 k = 0 时不乘 beta） | sgemm_batched 偏差 0.25；ssyrk_batched 偏差 0.25 | cgemm_batched 0.34；cherk_batched 0.25 |
| M246（批量 herk/syrk 改写另一侧三角） | ssyrk_batched：1,395 中约 1.4e3 处改动 | cherk_batched：同左 |
| M247（trsm_batched、非批量 trmm 在 alpha = 0 时不置零） | strsm_batched 465/465 为 NaN；magmablas_strmm 155/155 为 NaN | ctrsm_batched 465/465；magmablas_ctrmm 155/155 |
| M250（gesv_rbt 在 n ≤ 3 时出错） | n = 1、2、3 时误差 0.005–0.019（n = 5 时为 3e−8） | 误差 0.26–0.56（n = 5 时为 8e−8） |

d 精度在第 34–37 轮已测，z 精度在第 39 轮已测。因此这 4 条在 s、d、c、z 四种精度下都已实测证实。
