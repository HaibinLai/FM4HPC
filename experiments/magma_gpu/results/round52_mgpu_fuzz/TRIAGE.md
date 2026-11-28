# 第 52 轮：多卡例程的通用调度模糊测试（2026-11-27，roxy，单卡模拟多卡）

## 方法

第 50 轮的验证要先写 happens-before 模型，再在模型指出的位置延迟。本轮把它做成不依赖模型的通用工具：

- `probes/mgpu_fuzz/gen_fuzz.py` 原样复制 MAGMA 源文件，只做三处改动：
  - 换掉入口函数名，并用 `extern "C" decltype(原名) 新名;` 补上原型；
  - `magma_setdevice/getdevice` 变成空操作，`magma_queue_create(d, …)` 改为在 GPU 0 上建队列；
  - 在每个向 queue 提交工作的语句**之前和之后**各插一个 `emu_point(queue)`。
- 运行时，`emu_point` 按执行顺序计数。到第 `target` 个点时，往该 queue 放一个 100 ms 的 `cudaLaunchHostFunc` 睡眠，其余点不做任何事。只改变时序，所以只要某个 target 让结果出错，就说明源码没有给某两个操作排序。
- 每个 target 一个进程；先空跑（target = −1）得到点数 P，再跑 0…P−1。脚本为 `scripts/roxy/run_r52.ps1`，检查退出码和缺行。
- **前置点是必要的**（第一遍只有后置点）：
  - 代码里常见"提交操作，紧接着 `magma_event_record`"的写法。后置点会落在 record 之前，于是所有等这个事件的操作都跟着被推迟，窗口反而关上了。
  - 第一遍 2,044 个进程，包括阳性对照 potrf n = 4096，全部正确，没能找到第 50 轮的 M255。
  - 加上前置点之后，阳性对照中 8/322 出错，而且全部落在 M255 那个 stream-2 trsm 之前的点上。工具由此得到标定。
- 被测例程与检查方式（z 精度）：

| 例程 | 规模 | ngpu | 检查 |
|---|---|---|---|
| magma_zgetrf_mgpu（含 zgetrf2_mgpu） | n = 256，nb = 32 | 1–4 | P·A·x 与 L·(U·x) |
| magma_zgeqrf2_mgpu | n = 256，nb = 32 | 1–4 | R、tau 与 LAPACK zgeqrf |
| magma_zpotrf_mgpu（含 potrf3_mgpu），U / L | n = 1024，nb = 256；阳性对照 U n = 4096 | 2–4；对照 2 | 与 LAPACK zpotrf |
| magma_zhegst_m（itype 1，L） | n = 1024 | 1–4 | 与 LAPACK zhegst |
| magma_ztrsm_m（L L N N） | n = 1024，nb = 128 | 1–4 | 与 blas ztrsm |
| magma_zgetrf_m（含 get/setmatrix_transpose_mgpu），核内 / 核外（`MAGMA_NGR_NB=64`） | n = 256 | 1–4 | P·A·x 与 L·(U·x) |
| magma_zpotrf_m（L） | n = 1024 | 1–4 | 与 LAPACK zpotrf |
| magma_zunmqr_m（L C） | n = 512 | 1–4 | 与 LAPACK zunmqr |
| magma_zgeqrf_m | n = 256 | 1–4 | R、tau 与 LAPACK zgeqrf |
| magma_zungqr_m（T 来自 magma_zgeqrf_gpu） | n = 256 | 1–4 | Q 与 LAPACK zungqr |
| magma_zhetrd_mgpu（含 zlatrd_mgpu、zhemv_mgpu，nqueue 2） | n = 3072（n < 3000 时会改走 LAPACK） | 2–4，随机抽样前置点 | d、e、tau 与 LAPACK zhetrd |
| magma_zheevdx_2stage_m（V、All、L；含 he2hb_mgpu、hemm/her2k_mgpu、bulge_back_m、bulge_applyQ_v2_m、unmqr_m、stedx_m） | n = 1024 | 1–4，偶数 target | 特征值与 LAPACK，残差 ‖AV − VΛ‖ |
| magma_zgehrd_m（含 zlahr2_m、zlahru_m） | n = 96 全扫；n = 256 抽样 | 1–4 | H（次对角线及以上）、tau 与 LAPACK zgehrd |

## 结果

- 第一遍（只有后置点）：2,044 个进程，0 个出错。
- 第二遍（前置点 + 后置点）：5,445 个进程，11 个出错，全部 exit 0，无缺行，没有 CUDA 错误：
  - 阳性对照 potrfU n = 4096、ngpu 2：8/322，都在 `fuzz_zpotrf3_mgpu.cpp:511`，即 upper 情形 stream-2 剩余列 trsm 之前，也就是 M255。
  - zpotrf_m（L）：ngpu 2 为 2/162，ngpu 3 为 1/206，都在 `:806`，即 lower 情形的同一个 trsm 之前，相对误差约 7e-6。第 50 轮推断 potrf_m 在 M255 的影响范围内，这里实测确认，已补进 M255。
  - 其余例程全部正确：getrf_mgpu 540、geqrf2_mgpu 298、potrf_mgpu U/L 1024 共 210、hegst_m 317、trsm_m 518、getrf_m 核内 1,296、核外 1,216、potrf_m ngpu 1/4 共 360。这些是**正面结果**：在这种单点延迟下，没有发现无序的冲突访问。
- **新缺陷 M258（zgetrf_m）**：不是延迟引起的，是空跑时直接崩溃。
  - `getrfmooc` 在 ngpu 3、4 时先打印 "magma_setmatrix_async_internal got NULL queue"，然后访问违例（0xC0000005）。
  - 原因：ngpu 在 l.187–192（分配前）和 l.248–252（循环中）各算一次。第一次用的 NB 受 `MAGMA_NGR_NB` 限制（不设这个变量时，矩阵非常高也会让它变小），之后 NB 被重新计算且不再受限；循环里用的是最终的 NB。所以分配时把 ngpu 减少之后，循环仍可能用满 ngpu0 张卡，而 dA、queues、events 只给前面那几张建了。另外，清理循环也跑到 ngpu0，会销毁从未创建的 queue。
  - `getrfm_ngpu.txt`：n = 256、nb = 32，崩溃恰好发生在 ngpu0 > ceildiv(MAGMA_NGR_NB, 32) 时（64 → ngpu 3、4、5；128 → ngpu 5；不设 → 都不崩）。
  - 补丁 `fixed_zgetrf_m.cpp.txt`：循环中的 ngpu 取 min(·, 已分配的卡数)，清理也按已分配的卡数。修正后 12/12 正确（`getrfm_ngpu_fixed.txt`）。核外 ngpu 3、4 的模糊测试用的就是这个修正版（上面的 1,216 次里，ngpu 3、4 各 284 次）。
  - Lean（`Magma/MultiGpu.lean`）：`getrf_m_loop_exceeds_alloc`（只要分配时减少了卡数、而最终面板至少有 ngpu0 块，循环就用 ngpu0 张卡，比分配的多）、`getrf_m_ngr64_instance`、`getrf_m_fix_within_alloc`。
  - 证据记为 gpu-emulated。崩溃本身是主机端的计数逻辑，与模拟无关；但运行用的是单卡上的虚拟设备。
- **第三批（`c_` 前缀）**：全部 exit 0，无缺行，全部正确。
  - unmqr_m：144 次（每个 ngpu 的点数为 24/40/40/40）。
  - geqrf_m：724 次（点数 118/160/202/244）。
  - ungqr_m：492 次（点数 78/108/138/168）。
  - hetrd_mgpu n = 3072：点数 47,590/68,070/88,550，全扫不现实，所以随机抽样前置点。ngpu 2 抽 296 个，ngpu 3、4 各抽 149、150 个。
- **第四批 heevdx_2stage_m（`d_` 前缀）**：
  - ngpu 1–4 的点数为 412/890/1,350/1,882。所有偶数 target（即全部前置点）共 2,267 次，全部正确，exit 0，无缺行，无 CUDA 错误。
  - 相对残差约 1.3e-14。
  - 被延迟的点的分布：bulge_applyQ_v2_m 900、hemm_mgpu 514、he2hb_mgpu 437、unmqr_m 312、her2k_mgpu 64、dlaex3_m 32、bcyclic 8。
  - n = 1024 时 stedx_m 的多卡分治只走到 dlaex3_m 的少量点，所以对它的覆盖很薄。
- **gehrd_m（`e_` 前缀）**：全部正确，exit 0，无缺行。
  - n = 96：ngpu 1–4 的点数为 570/1,122/1,542/1,834，所有偶数 target 共 2,534 次。
  - n = 256：点数为 1,990/3,932/5,742/7,420，每个 ngpu 随机抽 300 个前置点，去重后共 1,127 次。
  - 被延迟的点绝大多数在 lahr2_m（3,447），其余在 lahru_m（167）、bcyclic（25）、gehrd_m 本身（22）。
  - 读源码时还发现了 gehrd_m 与 unghr_m 之间的 T 交接错误（ilo > 1），与调度无关，单独登记为 M259，见第 53 轮。

### 生成器修正（第三批之前）

- 先剥掉 `/* */` 注释和字符串字面量，并跳过预处理行，再匹配括号。原因：
  - zunmqr_m 有一段被注释掉的死代码；
  - he2hb_mgpu 有括号不配对的 OpenCL `#define`。
  - 两者都曾让生成器算错括号深度。
- queue 表达式允许带结构体前缀（`data.queues[…]`、`data->queues[…]`）。
- library 里的 `zbcyclic` 辅助函数会在真实设备号上建 queue，报 "invalid device ordinal"，所以也生成了可模糊的副本。
- 修正后重新生成之前的 29 个副本，与旧版逐字节相同，所以第一、二批结果不受影响。

## 锁页主机数组复测（`p_`、`q_` 前缀）与新缺陷 M260

- 前四批里，CPU 接口的 `_m` 例程收到的主机数组都是可分页的 `std::vector`。可分页源的 `cudaMemcpyAsync` 在入队时就把数据暂存好，D2H 实际上也是同步的，所以"异步拷贝还没执行，主机就改写了缓冲区"这类竞争测不到。MAGMA 自己的测试程序用的是锁页内存。
- 探针新增 `R52_PIN=1`：主机数组换成 `magma_malloc_pinned` 副本，调用一返回、在任何设备同步之前就拷回来检查（`H()` / `unpin()`）。程序为 `round52_mgpu_fuzz_pin.exe`，即 _fix 源码，由 `build_r52pin.bat` 构建；脚本为 `run_r52_pin.ps1`，只跑前置点。
- 结果：
  - 全部正确：trsm_m 518、hegst_m 317、getrf_m 648、getrf_m 核外 608、unmqr_m 72、geqrf_m 362、ungqr_m 246、heevdx_2stage_m（抽样）505、hetrd_mgpu（抽样）99。
  - potrf_m：3/364 出错，都在 `:806`，与可分页时相同，是 M255。
  - **gehrd_m：1/345 出错**。ngpu 2、n = 96、target 1090，在 `fuzz_zlahr2_m.cpp:435` 之前，相对误差 2.35e3。可分页内存或 0 ms 延迟时都正确。
- **M260 机制**：
  - 最后一次 `zlahr2_m` 在**每张卡**上排了 `magma_zsetmatrix_async(Y = work → dY)`。
  - gehrd_m 在循环之后只同步持有 R 个剩余块的那几张卡（R ≤ 2），随即调用 `lapackf77_zgehd2(…, work, …)`，把 work 当草稿区覆写。
  - `zlahru_m` 只用 dY 更新本卡上列号 ≥ k 的列，也就是面板块和剩余块。所以当 R < ngpu 时，面板所在的卡没有被同步：它的拷贝读到 zgehd2 的草稿，再用错误的 Y 更新面板的上半部分。
  - 实测出错的区域正是第 0–31 行 × 第 33–63 列。第 32 列不受影响，因为 V 的第一行为 0。
- 单卡模拟的另一个盲点：
  - 其他"面板卡未被同步"的组合在模拟中都正确，例如 ngpu 3 n = 96 的 target 1494、ngpu 2 n = 256 的 target 3896。
  - 计时表明这些情况下主机被挡住了：zgehd2 在 712–823 ms 才开始，而出错的那一例是 218 ms。
  - 原因是所有虚拟设备共用一个物理 H2D 拷贝引擎，按提交顺序执行。排在被延迟拷贝之后的其他卡的拷贝跟着等，持有剩余块的卡的同步拷贝也就跟着等。只有当面板卡是最后一个提交拷贝的卡时（ngpu 2、n = 96），竞争才显现。
  - 真实多卡没有这种串行化。
- 范围（`q_gehrdpin_tail.csv`）：n ∈ {96, 256} × ngpu 1–4，每个组合最后 200 个点里的所有前置点共 800 次，只有上面那一个出错；延迟 1、10、100、500 ms 都会出错。
- 补丁 `fixed_zgehrd_m.cpp.txt`：zgehd2 之前同步所有 queue。`round52_mgpu_fuzz_pinfix.exe`（`build_r52pinfix.bat`）在 n = 96、ngpu 1–4 的所有前置点上 2,534/2,534 正确（`q_gehrdpinfix96.csv`）。
- Lean（`Magma/MultiGpu.lean`）：`gehrd_remainder_le_two`、`gehrd_m_unsynced_device`、`gehrd_m_panel_unsynced`（R < ngpu 时面板卡未被同步）、`gehrd_m_panel_synced_when_full`、`gehrd_m_n96_instance`、`gehrd_m_fix_syncs_all`。
- 自然时序下的出现频率没有测；面板卡的拷贝只有几微秒，窗口很小。

## 局限

- 每次只延迟一个点、只延迟 100 ms。需要两个操作同时被延迟才暴露的竞争测不到。
- 单卡模拟测不到跨设备的硬件行为（P2P、不同 GPU 的时钟）。所有虚拟设备共用一个拷贝引擎，拷贝按提交顺序串行，会掩盖一部分竞争（见 M260）。legacy 默认流上的操作会把所有虚拟设备排序（见第 50 轮 d/s 的 M256）。
- 只测了 z 精度和上面列出的规模与参数组合。hegst 只测了 itype 1、Lower，trsm 只测了一种 side/uplo/trans 组合。

## 文件

`*.csv`、`*.err`（第一遍无前缀，第二遍以 `b_` 开头，阳性对照为 `potrfU4096*.csv`）、`getrfm_ngpu*.txt`。工具在 `probes/mgpu_fuzz/`（`gen_fuzz.py`、`fuzz_hooks.h`、生成的 `fuzz_*.cpp`、`fixed_zgetrf_m.cpp.txt`），探针 `probes/round52_mgpu_fuzz.cpp`，脚本 `scripts/roxy/{build_r52.bat, run_r52.ps1, run_r52_all.ps1, run_r52_all2.ps1, getrfm_env_r52.ps1, one_r52.ps1, stop_r52.ps1}`。
