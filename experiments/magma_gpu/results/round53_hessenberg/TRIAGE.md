# 第 53 轮：Hessenberg 约化在 ilo > 1 时的 T 交接（2026-11-27，roxy，库版本，1 张 GPU）

## 起点

第 52 轮给 `zgehrd_m` 做调度模糊测试时读源码，注意到两处：

- `zgehrd_m` l.260 把 `ilo − 1` 向下取整到块边界（`ilo = (ilo/nb)*nb`）。之后把起点在全局列 i 的块的 T 存到 `T + i*nb`（l.277–278）。
- `zunghr_m` 先把反射向量右移一列，再调用 `magma_zungqr_m(nh, nh, nh, A(ilo,ilo), lda, tau+ilo−1, T, nb)`，T 没有平移。`ungqr_m` 对子问题中起点为 c 的块，读 T 的第 c 列（`dT(d,0,i)`），而它需要的是首个反射为 c + (ilo − 1) 的那个块。

单卡 `zgehrd` 的块从 `ilo − 1` 开始，T 存在 `dT(0, i−ilo)`，与 `zunghr` 的读法一致。

MAGMA 自己的 `testing_zgehrd` 只用 ilo = 1、ihi = n。`zgeev_m`（编译的是 Version5，即 `gehrd_m` + `unghr_m`）先调用 `zgebal("B")`。只要平衡做了置换（ilo > 1），并且要求特征向量，就会走到这个情形。

查重：FINDINGS 里 geev/unghr 相关的只有 M157（分配失败状态被丢）和 M162（内层 iinfo 被丢），与本问题无关。

## 方法

`probes/round53_hessenberg.cpp <n> <ilo> <ihi>`：
- A 随机，并在第 1:ilo−1 行列和第 ihi+1:n 行列上取上三角（LAPACK zgehrd 的前提，zgebal 的输出就是这样）。
- 每个用例比较三组：
  - `gehrd_m + unghr_m`；
  - 单卡 `gehrd + unghr`；
  - `gehrd_m + fixed_unghr_m`（补丁，见下）。
- 指标：
  - H 和 tau 与 LAPACK zgehrd 的差；
  - Q 与"LAPACK zunghr 作用在同一组反射向量上"的结果之差，这样只检验 T 的交接；
  - 合同：‖Q H Qᴴ − A‖/‖A‖ 与 ‖QᴴQ − I‖。

`probes/round53_geev.cpp <z|d> <n> <kind>`：端到端比较 `magma_[zd]geev_m` 与 `magma_[zd]geev`，JOBVR = V，指标为特征向量残差 max_j ‖A v_j − w_j v_j‖/(‖A‖‖v_j‖)。
- kind 0：随机；
- kind 1：第 0 列在对角线以下为 0，gebal 给出 ilo = 2；
- kind 2：前 3 列在对角线以下为 0，ilo = 4；
- kind 3：最后一行在对角线左侧为 0，ihi = n − 1。

两个探针都是单进程，用 `scripts/roxy/one_any.ps1` 运行，全部 exit 0，无 CUDA 错误。

## 结果

- **`r53.csv`**：n ∈ {40, 100, 256, 300}，ilo ∈ {1, 2, 5, 32, 33, 40, 129}，ihi ∈ {n, n − 20}，共 49 个用例，nb = 32。
  - `gehrd_m` 自己的 H 和 tau 在 49 个用例里都与 LAPACK 一致（≤ 1e-12）。
  - `gehrd_m + unghr_m` 在"ilo > 1 且 nh = ihi − ilo + 1 > nb（分块路径）"的 36 个用例里**全部出错**，其余 13 个全部正确，预测与观测完全吻合。
    - 出错时 ‖Q H Qᴴ − A‖/‖A‖ = 4.3–21，‖QᴴQ − I‖ 最大 118。
    - 当 ilo − 1 = 128 是 nb 的倍数时，Q 是酉矩阵，但仍然错（残差 6.7–8.5）。原因：`ungqr_m` 读到的是 T 的第 0、32、64 列，而 `gehrd_m` 从未写过这几列，被 laset 置为 0，于是这几个块都当作单位变换。
  - 单卡 `gehrd + unghr` 和补丁版都是 49/49 正确，误差约 1e-15。
- **`geev.csv`**：n = 100、256。
  - ilo = 2 和 4 时，`zgeev_m` 的残差为 10.6、9.8、14.9、14.5，`dgeev_m` 为 8.6、8.8、13.2、15.0，info 全部为 0。
  - 单卡 `zgeev`、`dgeev` 在同样的输入上残差约 1e-13。
  - ilo = 1 的情形（kind 0 和 3）两者都正确。
- **登记 M259**：
  - 分类：dense-eigen、silent-wrong-result、normal-input、gpu-confirmed。
  - 不依赖多卡：1 张 GPU 的库版本就会出错。
  - s/c/d 由同一模板生成；d 已经端到端实测，s、c 只读了源码。

## 补丁

`round53_hessenberg.cpp` 里的 `fixed_zunghr_m`：移位部分不变。设 `ilo_r = gehrd_m 取了分块路径 ? (ilo−1)/nb*nb : ilo−1`，然后调用 `ungqr_m(k, k, k, A(ilo_r+1, ilo_r+1), lda, tau+ilo_r, T+ilo_r*nb, nb)`，其中 k = ihi − 1 − ilo_r。

这样多出来的是第 ilo_r … ilo−2 个反射。在 zgehrd 的前提下，这些列的次对角线及以下都是 0，所以 tau = 0，补上它们不影响结果。

## Lean

`Magma/MultiGpu.lean` 中 M259 一节，把"读法 (off, tOff)"与"存储映射 stored : 列 ↦ 首个反射"是否一致，定义为 `readerConsistent`：

- `unghr_m_reads_wrong_T`、`unghr_m_inconsistent`：ilo > 1 时，每个子块都读到错误的 T。
- `unghr_m_ilo1_consistent`：ilo = 1 时一致，这也是 MAGMA 测试程序唯一覆盖的情形。
- `unghr_single_consistent`：单卡这一对对任何 ilo 都一致。
- `unghr_m_fix_consistent`、`unghr_m_fix_extra`：补丁的读法一致，每个块都是 gehrd_m 写过的，并且只多出 (ilo−1) mod nb < nb 个反射。

## 局限

- ihi < n 只测了 ihi = n − 20 和 kind 3（ihi = n − 1）。
- 只测了 nb = 32（sm_120 上 `magma_get_zgehrd_nb` 的取值）。
- `geev_m` 的左特征向量（JOBVL = V）走同一个 `unghr_m`，没有单独测。
