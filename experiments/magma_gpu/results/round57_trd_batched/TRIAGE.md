# 第 57 轮：hetrd_gpu → unmtr_gpu，geqrf_batched → unmqr_batched（2026-11-28，roxy，库版本）

## 方法

`probes/round57_trd_batched.cpp`。生产方的原样输出直接交给消费方，判据按消费方文档写的合同。

- **trd**：
  - 例程：`magma_zhetrd_gpu` / `magma_zhetrd2_gpu`（后者带 dwork）。参数：nh ∈ {33, 300, 513}，另一维 ∈ {nh, 50}，U/L，side L/R，trans N/C，ldda = nh + {0, 5}，共 192 例。
  - 检查三项：
    - d、|e| 与 LAPACK zhetrd 一致；
    - 返回的主机副本 wA 与 dA 在被引用的三角上逐元素相等（文档称其为副本）；
    - `magma_zunmtr_gpu(dA, tau, wA)` 等于 LAPACK zunmtr 作用在 wA 上的结果。
- **bqr**：
  - 例程：`magma_zgeqrf_batched`（3 个矩阵），然后 `magma_zunmqr_batched`，工作区按文档用 lwork_device < 0 查询。
  - 参数：(m, n) ∈ {64×48, 48×64, 200×150, 150×200, 300×300}，k ∈ {nq, nq/2, 7}，L/R，N/C，ldd + {0, 3}，共 120 例。
  - 判据：每个矩阵与 LAPACK zunmqr 作用在下载回来的反射向量上的结果比较。

共 312 例，全部 exit 0，无 CUDA 错误。

## 结果

- trd 192/192 正确：wA 与 dA 逐位相同，unmtr_gpu 的误差 ≤ 3e-15。
- bqr 的 Left 60/60 正确。
- **M265（新，轻微）unmqr_batched Right**：
  - 文档列出了 SIDE = MagmaRight（C·Q、C·Qᴴ，并给出 Right 时 k、ldda 的约束），但代码在参数检查之后，对 Left 以外一律打印 "only side = MagmaLeft is supported" 并返回 −1（l.235–237）。
  - 小规模时，工作区查询已经进入 `zunm2r_reg_batched` / `zunm2r_sm_batched`，它们也拒绝 Right，各打印 6 次 xerbla。
  - 实测 Right 60/60 返回 −1。会明确报错，归入"拒绝合法调用"。
  - Lean：`unmqr_batched_rejects_documented_right`（`Magma/Dense/QrEigen.lean` §57）。
