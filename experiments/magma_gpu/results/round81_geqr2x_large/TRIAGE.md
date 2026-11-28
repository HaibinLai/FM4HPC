# 第 81 轮：geqr2x 的内部辅助函数与 n > 128（2026-11-28，roxy；新缺陷 M287、M288）

## 起因

第 68 轮测 geqr2x 时最大只用到 n = 128。读 geqr2x 的辅助函数（magma_zlarfbx_gpu、magma_zlarfx_gpu、magma_zlarfgx_gpu，都在公开头文件 magmablas_z.h 里）时发现：
- magma_ztrmv_tkernel 和 magma_ztrmv_kernel2（zlarfx.cu）把部分和放在 `__shared__ sum[128]` 里，却以 k 个线程启动，k 是当前已有的反射向量个数。
- geqr2x2/3/4 的文档只写了 N ≥ 0，只有 v1 检查了 N ≤ min(M, 128)。

## M287：n ≥ 130 时共享数组越界

- 预测：第 i 步以 i 个线程启动，i 最大为 n − 1，所以 n ≥ 130 时越界。
- r81.csv、r81b.csv（z，m = 2000，复用第 68 轮的探针，dwork 每列 60 个 double，避开 M274）：
  - n = 128、129：R 误差 6.6e-15，正确；
  - n = 130、131、132、136、144、160、180、199、200、256、513、1100：v2、v3、v4 都报 "an illegal memory access was encountered"，info = 0，R 的相对误差为 1。
- r81c.csv（`probes/round81_larfbx.cpp`）：
  - d 精度的 geqr2x2/3/4：n = 129 正确，n = 130 和 200 出现同样的错误；
  - magma_zlarfbx_gpu：k = 128 能运行，k = 129 和 200 触发非法内存访问。
- 边界正好在预测的 129/130 之间。四种精度的代码相同。
- 库内调用者 magma_zgegqr_gpu 自己检查了 n ≤ 128，所以只有直接调用者受影响。
- Lean：`geqr2x_tkernel_overflow_iff`（存在 i < n 使 i > 128，当且仅当 n ≥ 130）、`geqr2x_tkernel_instances`。

## M288：辅助函数的文档与代码不一致

- magma_zlarfbx_gpu：文档写 C = H C，H = I − V T Vᴴ，T 为上三角。
  - 实际计算 (I − V Tᴴ Vᴴ)C = HᴴC。m = 500、k = 64、128 时与 Hc 差 0.79 / 1.04，与 Hᴴc 只差 8e-16。
  - 会读 T 的严格下三角：在那里放 1e3 后误差为 9.9e3。
- magma_zlarfx_gpu（读码）：
  - 文档写 H = I − tau v vᴴ，"要应用 Hᴴ 就传 conjg(tau)"，代码用 −conj(*tau)，即传 tau 得到的是 Hᴴ；
  - 文档说 xnorm 会被更新，相关代码已被注释掉；
  - 还会把 v[0] 改写为 1，并按 geqr2x 的列编排处理 C。
- magma_zlarfgx_gpu（读码）：文档写 "stores beta over dx0"，实际存到 dA。数值上与 LAPACK zlarfg 等价，只是 x 为零时符号约定不同，反射向量仍然有效。
- geqr2x 内部的使用与代码一致，所以只影响按文档直接调用的人。
- Lean：`larfbx_applies_adjoint`（标量模型：1 − conj t = 1 − t 当且仅当 t 为实数）。

## 文件

- r81.csv / .err：18 例。
- r81b.csv / .err：12 例，确定边界。
- r81c.csv：14 例，d 精度与 larfbx。
