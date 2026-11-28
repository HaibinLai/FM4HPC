# 第 45 轮：sparse 求解器在边界输入上的测试——分诊

程序：`probes/round45_sparse_edge.cpp`，由第 29 轮的 probe 派生，只有问题列表不同。7 个问题：n1、identity8、diag64、tridiag2、zero_rhs（b = 0）、emptyrow（第 37 行为空，矩阵奇异，b 相容）、denserow。21 个求解器 × CSR/ELL/SELLP × NONE/JACOBI/ILU。

## 方法上的教训

第一次运行（`*.csv`）中出现了大量 FAIL 和 LIE，原因是某个用例触发了 `an illegal memory access`：这种错误会让整个 CUDA 上下文失效，此后同一进程里的所有调用都会失败。因此改为按（求解器, 问题, 格式）每个用例单独一个进程（`scripts/roxy/run_sparse45_iso.ps1`，结果在 `iso/`；对 emptyrow 又按预条件子细分，`scripts/roxy/run_emptyrow45.ps1`，结果在 `emptyrow/`）。另外，`Start-Process` 带重定向后再 `WaitForExit` 时，这台机器上每次都要多耗约 2 分钟，所以改成了直接同步调用。

## 结果

| 现象 | 用例 | 判定 |
|---|---|---|
| **进程整数除零崩溃（0xC0000094）**：QMR、PQMR、BICG、PBICG、LSQR、BOMBARDMENT 在 SELLP 格式下，对所有 7 个问题都崩溃 | 42 | **新缺陷 M251**（见下） |
| emptyrow + JACOBI：10 个求解器 × 3 种格式都出现 illegal memory access，而且求解器返回 info = 0、x 为 NaN | 30 | **调用方误用，记为观察**：Jacobi 的 setup 如实返回了 −204（MAGMA_ERR_BADPRECOND，因为找不到对角元），不设置 precond->d；测试程序仍然调用了求解器。MAGMA 自己的 testing_zsolver 用 TESTING_CHECK 包住 setup，失败就中止。求解器本身不检查预条件子是否建好，属于防御性检查的缺口，但不违反成文契约 |
| PBICG、PQMR + emptyrow 在 CSR/ELL 下栈溢出（0xC00000FD） | 4 | 已知 M098（ILU 因子没有建成，转置时无限递归） |
| BICGSTAB/PBICGSTAB 在 zero_rhs、identity8、n1 上返回 −202 | — | 已知 M104（没有中断检查）。identity8 和 n1 是在第一次迭代中途残差恰好为 0 时中断，已补进 M104 的证据 |
| JACOBI 在 zero_rhs 上返回 −202 | — | 已知 M105 |
| IDR/PIDR 在 diag64、tridiag2、denserow 上失败 | — | 已知 M103 |
| BA、ITERREF 在所有问题上 LIE | — | 与第 29 轮相同（已知） |
| zero_rhs 上大量 LIE（rel = NaN） | 104 | 测试程序问题：‖b‖ = 0 时相对残差成了 0/0；求解器返回的 x = 0 是对的 |

## 新缺陷 M251：SELL-P 参数在临时结构中丢失，导致整数除零

- CSR → SELL-P 的转换要读目标矩阵的 `blocksize`/`alignment`（magma_zmconvert.cpp 第 634 行：`256 % (B->blocksize)`）。有两条路径传给它的目标是刚初始化的结构，blocksize 为 0：
  1. **magma_zmconvert 的设备端回退路径**（第 1957–1960 行）：GPU 不支持某种格式时，先传回 CPU，用一个局部临时结构 `hB` 转换，而 `hB` 没有继承调用者在 `B` 上设置的 blocksize/alignment；
  2. **magma_zmtranspose / magma_zmtransposeconjugate**（第 332、441 行）：函数开头已经释放了 B，最后再把结果转回 A 的格式。
- 后果：
  - 所有需要 Aᵀ 的求解器（QMR、BICG、LSQR、BOMBARDMENT、PQMR、PBICG）用 SELLP 必崩；
  - 直接调用公开接口 `magma_zmconvert`，把设备上的 CSR 转成 SELLP，即使目标的参数设对了也会崩（`round45_sellp_iso` 第 5 步）。
- 定位：`probes/round45_sellp_iso.cpp` 把 zmtransposeconjugate 的步骤逐个执行。设备端 SELLP→CSR 正常（第 1、2 步）；转置后转回 SELLP 时崩溃（第 3 步）；CSR 输入的转置正常（第 4 步）。
- 验证：两处补丁（`probes/magma_zmconvert_fixed.cpp`、`probes/magma_zmtranspose_fixed.cpp`，`scripts/roxy/build_fix45.bat`）一起链接后，第 3、5 步都正确返回 0；QMR、BICG、LSQR、PQMR、PBICG 在 SELLP 下全部 OK，BOMBARDMENT 返回 −202（如实的失败）。两个补丁缺一不可：只修 transpose 时，崩溃移到了 zmconvert 的回退路径里。
- 第 29 轮漏掉了这些崩溃，那份 TRIAGE 已经更正。
