# 第 48 轮：带 volatile 的 warp 内归约在 sm_120 上的实测

程序：`probes/round48_volatile.cpp`。被测内核的归约都是 volatile 共享内存加 warp 内隐式同步，没有 `__syncwarp`：dmdotc.cu（8 处）、dgemvmdot.cu（2 处），这两个文件只在 d/s 精度下这样写，z/c 精度每一步都用 `__syncthreads`；另外还有 zpotf2.cu 的 zdotc 内核。

| 例程 | 用例 | 结果 |
|---|---|---|
| magma_dmdotc、magma_dgemvmdot（k = 1–8 个向量；n ∈ {1, 31, 255, 256, 257, 1000, 4097, 100000}） | 128 | 全部正确（与主机 long double 内积比较，误差 ≤ 64·eps·Σ\|vᵢwᵢ\|） |
| magma_dmdotc1–4 | 32 | 全部正确 |
| magma_dpotf2_gpu、magma_zpotf2_gpu（n 为 1–128） | 16 | 全部正确，info = 0 |

结论：在 sm_120 上，带 volatile 的 warp 内归约实测正确，在这些内核里 warp 在归约时是收敛的。出错的只有完全不带 volatile 的 ELLRT 内核（M253），因为编译器可以把共享内存的值留在寄存器里，再加上线程各自调度，就会读到旧值。这些 volatile 写法按 CUDA 编程模型仍属于未定义行为（Volta 之后官方要求用 `__syncwarp`），但没有观察到错误，所以不登记。
