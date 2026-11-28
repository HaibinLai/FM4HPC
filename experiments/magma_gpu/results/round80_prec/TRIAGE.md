# 第 80 轮：近期 sparse 发现在其他精度上的复核，以及 lacpy_sym 的约定（2026-11-28，roxy；新条目 M286）

`probes/round80_prec.cpp`，用例表 `scripts/roxy/r80_cases.txt`，共 16 例。第 74–77 轮的发现都是用 z 精度找到的，本轮在 d 或 c 精度上逐一复现。

| 条目 | 精度 | 结果 |
|---|---|---|
| M279 spmv_shift | d | CSR/ELL/ELLPACKT，α = 0.7：与文档公式差 8.3%（n = 10）、4.9%（n = 200），与 αAx − λx + βy 只差 1e-16 |
| M280 mslice | d | n = 10 分 9 片：第 5 片正确，第 6、8 片返回 −112；n = 5 分 4 片的第 3 片：3 次中 2 次 −112，1 次返回 0 且 start = 6 > end = 5 |
| M284 ICC 转置 | d | 恒等式误差 6.6e-2（g = 10）、6.5e-3（g = 30）；dpbicg + ICC 在 g = 10 时 −202，残差 3.33（比 z 更差）；g = 30 时 30 步收敛 |
| M282 IDR 角度 | c | A = iS：−202，残差 NaN（s = 4 时 36 步，s = 1 时 127 步）；A = S：8 步、18 步收敛 |
| M285 Jacobi 转置 | c | Hermitian 恒等式误差 0.388；clsqr + Jacobi −201，残差 0.27 / 0.49；不加预条件时 38 / 44 步收敛 |

各条目的精度范围（[zcds] 或 [zc]）与读码结论一致，已把证据补进 FINDINGS。

## M286：lacpy_sym_in / out 的文档与代码约定不一致

- 文档说 rows "存放新的主元，使第 rows[i] 行与第 rows[n+i] 行交换"，perm "进入时是恒等排列"。
- 内核实际把 rows 当作交错的二元组 (rows[2jj], rows[2jj+1])，按赋值 perm[rows[2jj+1]] = rows[2jj] 处理，sym_out 再把 perm[rows[2jj+1]] 重置回来。
- 唯一的调用者 zhetrf_aasen（l.405–420）构造的正是这种交错列表：rows[2·count] = perm[ii]，rows[2·count+1] = ii，表示一个完整的排列。所以库内部是自洽的，Aasen 在第 56 轮的结果也正确。
- 但这两个函数在公开头文件 magmablas_z.h 中，按文档的布局和"交换"语义调用，会得到一个不是排列的映射和错误的拷贝。
- 证据为读码，没有做 GPU 复现。

## 文件

- r80.csv、r80.err：16 例。
