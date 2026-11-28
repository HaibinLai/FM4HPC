# 第 82 轮：按"运行时线程数"扫描内核启动，magma_zhetrs_gpu（2026-11-28，roxy；新缺陷 M289、M290）

## 扫描

延续第 81 轮的教训：在 magmablas 和 sparse 里列出所有以运行时变量作为线程数的启动（`<<<…, 变量>>>`）。候选如下：
- zgeqr2x v4 / zlarfx / zlarfbx 的 trmv 内核：即 M287；
- zhetrs.cu 中以 `<<<1, nrhs>>>` 启动的 6 个内核；
- zgetf2_native_init_kernel（`max(n, npages)` 个线程）、zswapdblk（nb 个线程）、zpotf2（threadSize），这几个的线程数受调用方的块大小限制。

zhetrs_gpu 在公开头文件 magmablas_z.h 里，库内唯一的调用（zchesv_gpu）已被注释掉，记录里也从没出现过，于是本轮测它。

## 方法

`probes/round82_hetrs.cpp`：
- A 为 Hermitian 矩阵，两种：对角占优（只有 1×1 主元），以及对角很小的不定矩阵（会出现 2×2 主元）；
- 用 LAPACK zhetrf 分解，把分解后的 A 拷到设备，主机 ipiv 直接传给 magma_zhetrs_gpu；
- 结果与 LAPACK zhetrs 对比，并检查残差 |AX − B|；
- 用例：U/L × 两种矩阵 × nrhs ∈ {1, 2, 3, 16, 1024, 1025}，n = 100，另加 n = 300、nrhs = 5。

## M289：Lower 时只解第 0 列

- Lower 的前代 L D X = B 走的是融合内核（逐列版本被 `if (0)` 关掉了）：magma_zgeru_1、magma_zswap_scal、magma_zgeru_2、magma_zswap_scal_inverseblock_lower。
- 它们接收 nrhs 参数却不使用，访问 B 时不带列下标，所以只解第 0 列。zswap_scal 还以 nrhs 个线程启动，每个线程都对同一组第 0 列元素做交换和缩放。
- r82.csv：
  - Lower、nrhs = 1：正确；
  - nrhs = 2、3、16、1024 以及 n = 300 时的 5：恰好 nrhs − 1 列出错，相对误差 0.5–1.8，残差 9–48；
  - Upper：全部正确。
- 第 0 列碰巧正确：同一个 warp 的线程读到相同的旧值、写入相同的结果。
- **补丁确认**：`probes/zhetrs_fixed.cu` 只把两处 `if (0)` 改成 `if (1)`，走逐列版本，由 `scripts/roxy/build_fix82.bat` 链接。r82f.csv：11/11 正确。
- 四种精度（chetrs、dsytrs、ssytrs）的代码相同。
- Lean：`hetrs_lower_wrong_cols`。

## M290：nrhs > 1024 时启动失败，结果全错

- 对角块求逆的内核以 `<<<1, nrhs>>>` 启动，nrhs > 1024 时这不是合法的启动参数，而代码从不检查启动错误，于是对角缩放被跳过。
- r82.csv：Upper 和 Lower 在 nrhs = 1025 时 1025 列全错，相对误差 17–71，info = 0；Upper 在 nrhs = 1024 时正确。
- Lean：`hetrs_launch_invalid_iff`。

## 文件

- r82.csv、r82.err：26 例。
- r82f.csv：11 例补丁版（Lower，nrhs ≤ 1024）。
