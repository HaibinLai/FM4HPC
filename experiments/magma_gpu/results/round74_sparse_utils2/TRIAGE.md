# 第 74 轮：更多稀疏工具函数（2026-11-28，roxy；新缺陷 M279、M280、M281）

`probes/round74_sparse_utils2.cpp`：以前只读过代码、没有在 GPU 上跑过的稀疏工具函数，复数数据，参考值按文档在 CPU 上计算。用例表 `scripts/roxy/r74_cases.txt`，48 例，一个用例一个进程。

## 正确的函数

| 函数 | 用例 | 结果 |
|---|---|---|
| magma_zmcsrcompressor_gpu（设备 CSR） | n = 10、100、1000；存储值中约三分之一为 0，三分之一纯虚，三分之一纯实 | 只删除真正的复数零，纯虚、纯实的项保留，行指针正确 |
| magma_zmshrink | 10×15、40×41、20×20，CPU 与设备 | 正确（保留前 m 列） |
| magma_zvtranspose | 7×3、64×5，CPU 与设备 | 正确，结果标记为 RowMajor |
| magma_zmconjugate（设备 CSR） | n = 50、1000 | 正确 |
| magma_zmslice（起点在 n 以内的片） | n = 100 分 9 片取 0、1、8（测试程序的用法），n = 100 分 1 片，n = 10 分 9 片取 0–5，n = 7 分 3 片 | B、ALOC、ANLOC、通信表 comm_i / comm_v、start / end 全部正确 |

## M279：spmv_shift 的 λ 项没有乘 α

- 文档（wrapper 和三个内核）都写 y = α (A − λI) x + β y。三个内核（CSR、ELLPACKT、ELL）算的都是 `dot * alpha - lambda * dx[offset+row] + beta * dy[row]`，也就是 αAx − λx + βy，差值正好是 (α − 1)λx。
- r74.csv：α = 0.7 − 0.3i 时，三种格式、n = 10 和 200，与文档公式差 5–9%，与 αAx − λx + βy 只差 1e-16；α = 1 时正确。
- 四种精度的代码相同。库内没有调用者，是公开 API。
- Lean：`Kernels.lean` §74 `spmv_shift_lambda_unscaled`（差值恒等于 α(λx) − λx）、`spmv_shift_instance`。

## M280：zmslice 末尾的片起点越过 n

- 行按每片 ceil(n / num_slices) 行切分。num_slices 不整除时，末尾几片的 lstart = slice·size 会大于 n，而 lend = min(…, n)，于是片的大小为负。代码随后计算 nnz = A.row[lend] − A.row[lstart] + …，读 A.row 越界。
- 大小 ≤ −2：分配 size + 1 个元素失败，返回 MAGMA_ERR_HOST_ALLOC（−112），可请求本身是合法的（n = 10 分 9 片，第 6、7、8 片）。
- 大小 = −1（n = 5 分 4 片，第 3 片）：结果不确定，取决于越界读到的值。6 次运行中 2 次返回 0 且 start = 6 > end = 5，4 次返回 −112。
- 出错时 B、ALOC、ANLOC 里留着已经释放的 col / val / row 指针。调用者按惯例用 magma_zmfree 清理，进程以 0xC0000374（堆损坏）结束（R74_FREE，n = 10 第 6、8 片）。
- MAGMA 自己的测试程序用 9 片取第 8 片：n ≤ 100 中有 28 个尺寸（都 ≤ 55）会触发。
- Lean：`mslice_size_negative`（起点越过 n 时大小必为负）、`mslice_instances`（28 个尺寸，以及 −2、−6、−1 三个实例）。

## M281：zmconjugate 不检查格式

- 文档只写 "conjugates a matrix"，没有格式或位置限制，但内核把 A->drow 当 CSR 行指针来遍历。对设备上的 ELL 矩阵（drow 为空），内核触发 "an illegal memory access was encountered"，函数仍返回 0，此后 CUDA 上下文不可用，结果既不是共轭也不是原值。
- 库内唯一的调用者（zmtransposeconjugate）传的是 CSR，所以只影响直接调用者。

## 对第 73 轮的补充

round67_trans_rect 用当前源码重新构建后再跑了一遍（之前一次构建把源码放错了目录，用的仍是旧源码）：方阵三种变体都正确；宽矩阵 30×47 三种都崩溃；10×11 这次三种都没崩溃，只越界一个元素，崩不崩溃取决于堆布局；高矩阵仍是维度未交换。结论不变。

## 文件

- r74.csv、r74.err：48 例。
