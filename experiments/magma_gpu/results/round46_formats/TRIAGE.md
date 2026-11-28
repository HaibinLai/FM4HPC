# 第 46 轮：sparse 全部存储格式在边界矩阵上的往返与 SpMV——分诊

程序：`probes/round46_formats.cpp`。16 种目标格式（ELLPACKT、ELL、DENSE、BCSR、CSC、COO、ELLRT、SELLP、ELLD、CSRLIST、CSRD、CSRL、CSRU、CSRCOO、CUCSR、CSR5），7 个矩阵（n1、poisson、random、emptyrow、denserow、rect、zero），另加第 0 行为空的 emptyrow0。每个（格式, 矩阵）单独一个进程，依次执行 4 步：
1. 主机 CSR→F；
2. F→CSR 往返，按稠密形式比较（CSRL/CSRU 与下/上三角比较）；
3. F 上设备 SpMV，与主机参考结果比较；
4. 设备上 CSR→F→CSR。

脚本是 `scripts/roxy/run_formats46.ps1`，它记录每个进程的退出码和 stderr 中最后一条 STEP；结果在 `all.csv`、`status.txt`。

| 现象 | 判定 |
|---|---|
| ELLRT、SELLP 在第 4 步整数除零（0xC0000094），7/7 个矩阵 | M251（ELLRT 已补进范围）：用 M251 的两处补丁后 ELLRT/SELLP 全部正常（`round46_formats_m251fix`） |
| **COO：第 2、4 步往返在所有非平凡矩阵上都错**（误差为元素量级；n1、zero 正常） | **新 M252**：CSR→COO 在主机和设备两处都把行下标写进 `row`/`drow`，而 zmtransfer(COO) 和 COO→CSR 读的是 `rowidx`/`drowidx`。3×3 例子往返后行指针变成 [0 5 5 5]（`probes/round46_coo.cpp`）。补丁（`probes/magma_zmconvert_coofix.cpp`，`build_fix46.bat`）后 28/28 正确；只修主机那处时，设备路径仍然错 |
| **ELLRT 在 denserow 上 SpMV 错**（误差 24.8，5 次运行结果相同） | **新 M253**：只有 40 个非零元的那一行出错，结果正好等于 j mod 8 ∈ {0…5} 的元素之和，也就是 alignment = 8 时第 6、7 号线程的贡献丢失了。zgeellrtmv 的每行归约没有 volatile，也没有 `__syncwarp`。改写为扁平归约并在步与步之间加 `__syncwarp` 后，误差为 1.4e−14（`probes/zgeellrtmv_sync.cu`，`build_fix46c.bat`）。zmergecg 中的 ELLRT 内核是同一份代码（只读码确认，未单独测：`magma_zparse_opts` 不认识 `--format ELLRT`，会悄悄退回 CSR） |
| **CSRLIST：emptyrow 在第 2 步、zero 在第 4 步出现堆损坏（0xC0000374）；emptyrow、emptyrow0、zero 主机往返出错** | **新 M254**：CSR→CSRLIST 对空行写 `list[row[i+1]-1]`（第 0 行为空时就是 `list[-1]`）；CSRLIST→CSR 用 do-while，每个空行都多复制一个元素，结果写越界并且空行里多出一个元素。补丁（`probes/magma_zmconvert_listfix.cpp`，`build_fix46d.bat`）后主机往返全部正确 |
| CSRLIST 第 4 步（设备往返）在所有矩阵上都错；第 3 步 SpMV 返回 −115 | 已知 M210：zmtransfer 没有 CSRLIST 分支 |
| DENSE 在非对称矩阵上 SpMV 错（对称的 poisson 正确） | 已知 M125（实际计算的是 Aᵀx） |
| ELLD、CSRD 在 rect 上往返出错 | 已知 M132（缺少对角元的行） |
| BCSR 在 n = 1 时第 1、4 步返回 −113（MAGMA_ERR_DEVICE_ALLOC） | 如实报告的错误，记为观察，未深究 |
| 其余（ELLPACKT、ELL、CSC、CSRL、CSRU、CSRCOO、CUCSR、CSR5、BCSR 在 n > 1 时）| 主机往返全部正确；有些格式的 SpMV 返回 MAGMA_ERR_NOT_SUPPORTED（如实报告，共 56 例） |

## 补充：M253 在求解器层面的后果

在 round45 的 probe 中，解析参数之后直接设置 `output_format = Magma_ELLRT`（magma_zparse_opts 只认 CSR/ELL/SELLP/CUSPARSECSR/CSR5，遇到 ELLRT 会打印 “invalid format, use default (CSR)”，这是如实的提示），然后在 denserow 上运行：CG、PCG、BICGSTAB、PBICGSTAB、GMRES 都返回 info = 0，求解器自报的残差为 1e−14，但真实相对残差为 4.95e−3（LIE）；对角矩阵 diag64 上全部正确。CG 和 PCG 走的是合并实现（CGMERGE），所以 zmergecg.cu 中那份 ELLRT 内核也在 GPU 上证实出错。
