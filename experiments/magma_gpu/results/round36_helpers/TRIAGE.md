# 第 36 轮：辅助例程（INVENTORY 中的 (c) 类）直接与 LAPACK 对比——分诊

程序：`probes/round36_helpers.cpp`（d 精度）。Householder 数据都来自主机端的 LAPACK 分解（geqrf、gelqf、geqlf、gerqf、sytrd、gebrd），所以 MAGMA 和 LAPACK 作用或生成的是同一个 Q，可以逐元素比较。

| 组 | 例程 | 用例 | 结果 |
|---|---|---|---|
| orm | dormqr、dormlq、dormql、dormrq（主机接口），dormqr2_gpu、dormql2_gpu | 168 | 全部正确（包括 k = 0、k = nq、600×70 且 k = 520） |
| ormtrbr | dormtr、dormtr_gpu（L/U × L/R × N/T），dormbr（Q/P，宽矩阵和高矩阵） | 136 | 全部正确 |
| org | dorgqr2、dorgbr（Q/P） | 21 | 全部正确 |
| larfb | dlarfb_gpu（F/B × C/R × L/R × N/T） | 80 | 全部正确。第一版是测试程序问题：MAGMA 文档写明 “All elements including 0's and 1's are stored, unlike LAPACK”，第一版按 LAPACK 的约定传了隐式单位三角 |
| gehrd | dgehrd、dgehrd2（面板辅助函数 lahr2、lahru），n = 1–513，含部分 ilo/ihi；以 LAPACK dorghr 生成的 Q 检查 ‖QHQᵀ − A₀‖ 和正交性 | 28 | 全部正确（`gehrd.csv`） |
| trevc | dtrevc3、dtrevc3_mt（R/L/B × A/B/S，带 2×2 块的拟三角 T） | 54 + 36 个残差 | **新 M249**（见下） |

## M249：trevc3 在不做回代变换时也选用了分块版本

- 判据：与 LAPACK 的向量逐元素比较；另外计算与归一化方式无关的特征残差 ‖Tv − λv‖/(‖T‖‖v‖n·eps)，复数对按实部、虚部形式分别检查。LAPACK 自己的残差最大为 0.54，说明判据本身正确。
- 现象：lwork 取文档推荐的最优值 (1 + 2nb)·n 时，howmany = A（全部）或 S（选定）的特征向量是错的：36 个特征对中有 26 个残差为 1e13–1e15。howmany = B（回代变换，geev 用的就是这种）始终正确。trevc3_mt 的表现完全相同（`trevc_mt_nb*.csv`）。
- 与工作区的关系（`trevc_nb{0,8,16,128}.csv`）：lwork = 3n 或 nb = 8（小于 nbmin = 16）时走版本 1，全部正确；nb = 16 和 nb = 128 时走版本 2，36 个中错 26 个。
- 根因：LAPACK 只在回代变换时选用分块版本 2（`IF( OVER .AND. LWORK .GE. N + 2*N*NBMIN )`），MAGMA 的条件（dtrevc3.cpp:283、dtrevc3_mt.cpp:379、ztrevc3.cpp:243、ztrevc3_mt.cpp:359）漏掉了 `over`。在 A/S 模式下，向量 x 先被正确地复制到 VR 中；但循环末尾的版本 2 代码仍然执行“回代变换”的 GEMM，把调用者传入的 VR（在这两种模式下只是输出空间）当作 Q 去乘，于是改写了已经算好的向量。
- 验证：`probes/dtrevc3_patched.cpp` 在该条件中加上 `over &&`（`scripts/roxy/build_fix36.bat`），nb = 16 和 nb = 128 时都是 36/36 个残差正确、54/54 个与 LAPACK 一致（`trevc_fixed_nb*.csv`）。
- 附带一点：magma_dtrevc3 不支持 lwork = −1 的工作区查询，会报 info = −14；LAPACK 的 dtrevc3 支持。不过 MAGMA 的文档没有承诺支持查询，所以不登记。
