# 第 55 轮：消费 `magma_zgeqrf_gpu` 的 dT 的例程（2026-11-28，roxy，库版本，1 张 GPU）

## 起点

M259 说明，中间结果的生产方和消费方可能用的是不同约定。`zgeqrf_gpu` 产出的 dT 有 MAGMA 自己的布局：
- 依次是 T 块、R 对角块的逆、工作区；
- 只为 i < minmn − nb 的块写 T（l.279–325），最后一块在 CPU 上用 `lapackf77_zgeqrf` 分解，不写 T（l.356–362）。

消费 dT 的例程以前只在默认参数下测过：第 36 轮比较了 Q 的生成与作用，但用的是主机端 LAPACK 的反射向量，不经过 dT。

## 方法

`probes/round55_qr_dt.cpp <op> <m> <n> <k|nrhs> <pad> [side] [trans]`：
- 先做 `zgeqrf_gpu`（geqrs3 用 `zgeqrf3_gpu`），下载 MAGMA 自己的反射向量，再让 LAPACK 在**同一组反射向量**上计算参考结果。这样只检验 dT 的交接和分块。
- 覆盖非方阵、k < n、ldda = 行数 + 5、L/R、N/C。
- 用例由 `scripts/roxy/r55_cases.txt` 给出，共 236 个，每个用例一个进程（`run_list.ps1`），全部 exit 0，无 CUDA 错误。

## 结果

| 例程 | 用例 | 结果 |
|---|---|---|
| ungqr_gpu、ungqr（CPU 接口） | 60 | 全部正确 |
| geqrs_gpu、geqrs3_gpu、gels_gpu、gels3_gpu | 80 | 全部正确 |
| unmqr_gpu | 96 | **64 错误**（M261） |
| unmqr_2stage_gpu（另测 96 × 4 种输入形式） | 384 | **M262、M263** |

### M261：unmqr_gpu 把 dT 的末尾当作 larfb 工作区，超出 geqrf_gpu 文档给的大小

- unmqr_gpu 的文档只说 dT 是 "the output (the 9th argument) of magma_zgeqrf_gpu. Part used as workspace"。
- 但它用 `dwork = dT + 2k·nb` 作 larfb 工作区，ldwork = nw（Left 时为 n，Right 时为 m），所以总共需要 (2k + nw)·nb。
- geqrf_gpu 文档给的 dT 是 (2k + roundup(k, 32))·nb。当 nw > roundup(k, 32) 时，写出界 (nw − roundup(k, 32))·nb 个元素。
- MAGMA 自己的 `testing_zunmqr_gpu` 分配的是 (2·min(m,k) + roundup(max(m,n), 32))·nb，所以它测不出来。geqrs_gpu 文档给出了自己更大的 dT，gels_gpu 也自己放大了，这两个不受影响。
- 实测：
  - `r55_unguarded.csv`：dA、dT、dC 依次分配，dT 取文档大小。nw > roundup(k, 32) 的 72 例中有 64 例 C 出错，相对误差 0.35–2.7，info 0。其余 24 例全部正确。写出界的部分落进了紧接着分配的 dC。
  - `r55_unmqr_guarded.csv`：在文档大小之后放一段哨兵保护区。C 全部正确，而保护区被写的元素数在预测的 72 例中**恰好等于** (nw − roundup(k, 32))·nb，其余 24 例为 0。例如 m = 600、n = 70、Right、k = 35 时写出界 17,152 个元素。

### M262：unmqr_2stage_gpu 的 side = Right 没有实现

- 该函数通过 `magma_v2.h → magma_bulge.h → magma_zbulge.h` 公开，文档写明 Right 时计算 C·Q 或 C·Qᴴ。
- 代码对每一块都调用 `magma_zlarfb_gpu(MagmaLeft, …)`（l.199–201）；私有工作区只有 n·nb（l.157），却按 ldwork = nw = m 使用。
- 即使用它真正需要的输入形式（见 M263），实测（`r55_unmqr2s_explicitV_fullT.csv`）结果也是：
  - Left 48/48 正确，误差约 1e-15；
  - Right 48/48 错误，误差 1.2–171，并且随 ldda 填充而变化，说明读到了工作区之外的数据。
- info 都是 0。库内唯一的调用者 zheevdx_2stage 只用 Left。

### M263：unmqr_2stage_gpu 的文档前提不足

- 文档要求的输入只是"zgeqrf 的输出"：dA 中的反射向量，以及 zgeqrf_gpu 的 dT。
- 但它对每一块都用 `dT + i·nb` 调 larfb，并按存储原样读 V。zgeqrf_gpu 的最后一块（nb ≥ minmn 时就是整个 A）在 CPU 上分解：这一块没有 T，而且对角线上方存着 R。前面各块的 V 是显式存储的，它们的 R 对角块存在 dT 里（第 27 轮已注意到 R 的对角块不在 dA 中）。所以出问题的只是最后一块。只有 zhetrd_he2hb 交给 zheevdx_2stage 的那种形式（V 显式存储，每块都有 T）才能用。
- 四种输入形式的对比（每种 96 例）：

| 输入 | Left | Right |
|---|---|---|
| zgeqrf_gpu 的原样输出（文档写法） | 全错 | 全错 |
| 只把 V 改为显式 | 全错 | 全错 |
| 只补齐每块的 T（LAPACK zlarft） | 全错 | 全错 |
| 两者都做 | **全对** | 全错（M262） |

## Lean

`Magma/Dense/QrEigen.lean` §55：
- `unmqr_gpu_dT_overflow`：nw > roundUp32 k 时写出界，且恰好写出 (nw − roundUp32 k)·nb 个元素；
- `unmqr_gpu_tester_size_suffices`：MAGMA 测试程序用的大小总是够用；
- `unmqr_gpu_overflow_instance`：实测的 17,152；
- `unmqr_2stage_right_applied_left`、`unmqr_2stage_right_dwork_short`：M262；
- `unmqr_2stage_reads_unwritten_T`：M263，最后一块的 T 被读到却从未被写。

## 局限

- 只测了 z 精度；s/c/d 由同一模板生成。
- unmqr_gpu 越界写在真实程序中会落到哪里，取决于分配顺序；这里只演示了紧邻分配的 dC 被改写。
