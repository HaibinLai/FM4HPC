# 第 50 轮：多卡 Cholesky `[zcds]potrf3_mgpu` 的缓冲区复用竞争（2026-11-27，roxy，单卡模拟多卡）

## 结论

两条新缺陷，都是同一类：一个轮换使用的缓冲区被下一次写入时，上一次的读还可能没有完成，源码中没有任何事件或主机同步把两者排序。

| 编号 | 缓冲区 | 条件 | 精度 | 表现 | 证据 |
|---|---|---|---|---|---|
| M255 | 设备上的对角块缓冲 `dlP(d,0,0,buf)`，`buf = (j/nb) % ngpu` | ngpu ≥ 2，与 h 无关 | z、c（d、s 走 `dtrsm_work`，用 flag 0 时不再读 dlP） | 因子静默出错，info = 0 | 模拟：z 16/60、c 14/20 个单点延迟出错；加一个事件等待后 0/80 |
| M256 | 主机 pinned 工作区 `Aup/Alo`，h 个块槽 | h < ngpu；`magma_[zcds]potrf_mgpu` 写死 h = 2，所以 ngpu ≥ 3 | 四种（d、s 在单卡模拟中被掩盖，见下） | 对 SPD 矩阵报 info > 0（"不正定"），因子错误 | 模拟：z/c 在 h < ngpu 时 37/60 出错，h ≥ ngpu 时 0/48 |

`zpotrf_mgpu.cpp` l.142–145 的注释是："setting h = 1 produces failures for ngpus >= 4 / set h = 2 as a temp. fix / TODO: investigate the correct value for h"，后面是 `h = 2; //ngpu;`。模型和实测都表明：**h ≥ ngpu 是不出错的充要条件**，被注释掉的 `h = ngpu` 恰好是最小的安全值，h = 2 只对 ngpu ≤ 2 成立。M255 与 h 无关，开发者的临时修复对它没有作用。

影响范围：M255 经 `magma_[zc]potrf_mgpu` 和 `magma_[zc]potrf_m`（后者被 `[zc]hegvd_m`、`hegvdx_m`、`hegvdx_2stage_m` 调用）触达；M256 只在 `potrf_mgpu`，因为 `potrf_m` 取 h = 面板块数，不会复用槽。

## 方法

1. **happens-before 模型** `scripts/potrf_mgpu_hb.py`：按源码（upper，l.225–500）逐块重放主机程序，记录每个操作所在的队列、等待的事件、读写的缓冲区。规则：同队列按序；`wait_event` 之后的工作在该事件记录的全部工作之后；未记录的事件等待不起作用；`magma_queue_sync` 之后主机发出的一切都在该队列已有工作之后；CPU 上的 potrf 同理。然后检查所有冲突的访问对（同一缓冲区、块行重叠、至少一个写）是否有序。lower 与 upper 结构对称，没有单独建模，但下面的实测覆盖了 lower。
   - d/s 路径：`dtrsm_work` 的 flag 0 调用只用 `dinvA` 中预先求好的逆，不读面板（第一版模型把它算成了读，已修正）；flag 1 调用里的 `magmablas_dtrtri_diag` 有一个不带流的 `cudaMemset`（legacy 默认流），建模为该设备所有阻塞流上的屏障。
   - 结果见 `hb_tables.txt`：z/c 路径 P 类对所有 ngpu ≥ 2 存在，A 类恰在 h < ngpu 时存在；d/s 路径只有 A 类，条件相同。
2. **单卡模拟多卡**（`probes/potrf_emu/`）：`gen_emu.py` 把 2.10.0 的 `[zcds]potrf3_mgpu.cpp` 原样复制，只改函数名并插入 `emu_hooks.h`：`magma_setdevice` 变成空操作，每个虚拟设备有自己的 dlA、dlP 和 3 个队列，全部在 GPU 0 上。队列、事件、主机同步的语义与跨设备时相同。为了打开竞争窗口，在某一个操作之前让该流睡 100 ms（`cudaLaunchHostFunc`），这只改变时间，CUDA 本来就允许任意时序。
   - kind 5：只延迟第 c 个读 dlP 的 stream-2 trsm；kind 6：只延迟第 c 个把对角块从主机工作区拷进拥有者 dlA 的 stream-1 拷贝。
   - 修复变体（fix 1）：在每个读 dlP 的 stream-2 trsm 之后记录事件，下一次往该设备 dlP 写对角块的 stream-1 拷贝先等它。
   - 参照解为主机 LAPACK potrf；n = 1024，nb = 64，A = G·Gᴴ（每 7 列取一列）+ n·I。判定阈值：双精度相对误差 1e-10，单精度 1e-5（实测 OK 的最大值 2.2e-16 / 1.2e-7，出错的最小值 1.1e-4）。
3. **机理确认**：给延迟的操作和之后改写同一缓冲区的操作打时间戳（cudaEvent），并定位最大误差所在的块。

## 结果（`r50.csv` z/d 共 324 个进程，`r50cs.csv` c/s 共 102 个进程；全部 exit 0，无缺行）

- A 组（不延迟，MAGMA 自己的 h = 2，ngpu 1–4，四种精度 × U/L）：0/20 出错。自然时序下在单卡上不出现。
- B 组（M255，h = ngpu 以排除 M256）：
  - z：U ngpu 2/3/4 = 6/10、1/10、0/10；L = 6/10、2/10、1/10；fix 1 全部 0/60。c（ngpu 2）：U 6/10，L 8/10，fix 1 0/20。
  - 出错时，改写 dlP 的拷贝**在被延迟的 trsm 开始前约 103 ms 就完成了**，最大误差位于该 trsm 负责的块行（k = 2、3、6 时分别是块行 2、3、6）；不出错的点，改写发生在 trsm 之后，即窗口没有打开（延迟期间主机被别的依赖拖住）。ngpu 越大，主机等待链上覆盖的设备越多，窗口越少打开。
  - d、s：0/30，与模型一致（flag 0 不读 dlP）。
- C 组（M256）：
  - z/c：h < ngpu 时 37/60 出错（h = 1、ngpu 2：每组 5/6；h = 2、ngpu 3：3–4/6；h = 2、ngpu 4：3–4/6；h = 3、ngpu 4：2–3/6）；h ≥ ngpu 时 0/48。37 个出错全部伴随 info > 0（例如 193、257、321），即对 SPD 矩阵报"不正定"，相对误差 ≥ 2.6e26。时间戳：写入同一主机行的 D2H 在被延迟的拷贝开始前约 100–115 ms 完成。
  - d/s：0/96。原因是 `magmablas_[ds]trtri_diag` 第 104 行的 `cudaMemset(d_dinvA, …)` 走 legacy 默认流，它与该设备上所有阻塞流同步。模拟时所有虚拟设备在同一块 GPU 上，这个 memset 就把它们全部排了序。模型复现了这一点（`--real --one-gpu`：0 个危险对）；而在真实多卡上 memset 只排序本设备的流，危险对不变（`--real`）。所以 d/s 的 M256 结论来自模型，模拟无法检验。
- 平台核对 `round50_stage.cpp`：`magma_zsetmatrix_async` 从 pinned 内存拷贝时，读主机数据的时刻是执行时而不是入队时（4 KB–4 MB、lda = m 与 lda > m 都是 "new"），排除了驱动提前暂存数据的可能。

## Lean（`Magma/MultiGpu.lean`）

- `add_mod_eq_self_iff`：`(k+i) % g = k % g ↔ i % g = 0`。
- `potrf_mgpu_dlp_reuse_at`、`potrf_mgpu_dlp_not_reused_before`：dlP 对角块缓冲在第 k 步写入后，恰好在第 k + ngpu 步再次写入，此前不会。
- `potrf_mgpu_host_slot_alias`：第 k+h−1 步发往主机的块列 k+h 的第 0…k+h−1 块行落在块 k 的槽里，覆盖它的对角行。
- `potrf_mgpu_host_slot_covered_iff`：能覆盖这次读的排序只有两种：第 k+i 步（1 ≤ i < h）对设备 (k+i) % g 的 stream 1 做主机同步，或者写者就是读者所在的设备（i = h）。存在这样的 i 当且仅当 g ≤ h。`potrf_mgpu_h2_uncovered`（ngpu ≥ 3 时 h = 2 不被覆盖），`potrf_mgpu_h_ngpu_covered`（h = ngpu 被覆盖）。
- 注意：这几条是排序论证的算术核心；“源码里只有这些排序”这一前提来自 happens-before 模型，没有在 Lean 中形式化。

## 局限

- 没有在真实多卡上运行（s82 不可用）。模拟证明的是：按 CUDA 语义，这些操作之间没有顺序，某些合法时序会算错；不能说明真实多卡在自然时序下出错的频率。M256 的 h = 1 版本是开发者在真实多卡上见过的。
- 自然时序下（A 组）单卡模拟不出错。真实多卡上，当某张卡的 stream 2 积压（大 n 时 trsm/gemm 变长）而主机已经推进 ngpu 步时，M255 的窗口会打开。

## 文件

`r50.csv`、`r50.err`（z/d）、`r50cs.csv`、`r50cs.err`（c/s）、`hb_tables.txt`。探针 `probes/round50_potrf_emu.cpp`、`probes/round50_stage.cpp`、`probes/potrf_emu/{emu_hooks.h,gen_emu.py,emu_[zcds]potrf3_mgpu.cpp}`；脚本 `scripts/roxy/build_r50.bat`、`run_r50.ps1`、`run_r50cs.ps1`；模型 `scripts/potrf_mgpu_hb.py`。
