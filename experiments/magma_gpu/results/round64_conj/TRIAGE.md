# 第 64 轮：magmablas BLAS 是否正确处理 MagmaConjTrans（2026-11-28，roxy；没有新缺陷）

## 起点

M268 中，geam 把 ConjTrans 当作 Trans 计算，MAGMA 自己的测试程序也犯了同样的错误。扫描源码时发现，不少文件接收 magma_trans_t，却不在本文件里取共轭（共轭可能是在模板头文件里做的）。所以直接实测：复数数据，trans 取 N/T/C，与参考 BLAS（blasf77_*）比较。

## 方法与结果

程序为 `probes/round64_conj.cpp`，用例表为 `scripts/roxy/r64_cases.txt`。batched 为 3 个同形问题，vbatched 走 vbatched 接口。

| 例程 | 参数 | 用例 | 结果 |
|---|---|---|---|
| magmablas_zgemm / _batched / _vbatched | (m,n,k) ∈ {33×17×9, 100×64×128, 5×5×5} × tA,tB ∈ {N,T,C}² | 81 | 全部正确 |
| magmablas_zgemv / _batched / _vbatched | 3 种形状 × {N,T,C} | 27 | 全部正确 |
| magmablas_ztrsm / _batched / _vbatched | (m,n) ∈ {16×8, 33×17, 100×64} × L/R × U/L × {N,T,C} | 108 | 全部正确（含 trsm_small 路径） |
| magmablas_ztrmm / _batched | 同上 | 72 | 全部正确 |
| magmablas_ztrsv | n ∈ {16, 33, 100, 300} × U/L × {N,T,C} | 24 | 全部正确 |

共 312 例，全部 exit 0，误差 < 1e-11。

## 结论

主要 BLAS 例程都正确处理了 ConjTrans，M268 只出在 geam 一处。
