# 第 54 轮：CPU 接口多卡驱动的参数维度（2026-11-27，roxy，库版本，ngpu = 1）

## 起点

第 38 轮只测了 d 精度、itype 1、range A/I、lda = n、随机输入（ilo = 1），所以漏掉了第 53 轮的 M259。本轮补上那一轮没有用到的参数维度：

- lda = n + 7（ldb、ldvl、ldvr 相同），填充行里放哨兵，检查有没有被改写；
- z 精度；
- Upper；
- itype 2、3；
- jobz N；
- range V；
- geev_m 的左特征向量；
- getrf_m、geqrf_m 的 m ≠ n。

## 方法

`probes/round54_mgpu_param.cpp`，参数为 `<routine> <n> <pad> <uplo> <itype> <jobz> <range> [m]`。

- 特征值驱动：特征值与 LAPACK zheevd / zhegvd 比较。range I 取第 n/4+1 … n/2 个特征值；range V 选同一组，vl、vu 取在相邻特征值的中点。特征向量按各 itype 的残差检查。
- geev_m：检查左、右特征向量的残差。
- getrf_m、potrf_m、geqrf_m、gehrd_m：与 LAPACK 在同样的 lda 下逐元素比较。

用例由 `scripts/roxy/r54_cases.txt` 给出，共 492 个，覆盖 n ∈ {33, 300} × pad ∈ {0, 7} 以及各种参数组合。每个用例一个进程，由 `scripts/roxy/run_list.ps1` 运行，全部 exit 0，无缺行，无 CUDA 错误。

## 结果：没有新缺陷

| 例程 | 正确 | 出错，归入已登记的缺陷 |
|---|---|---|
| heevd_m | 16 | — |
| hegvd_m（itype 1–3） | 48 | — |
| heevdx_m | 48 | — |
| heevdx_2stage_m | 36 | M065 12（Upper：he2hb 返回的 NOT_IMPLEMENTED 被丢弃，特征值误差 0.41–1） |
| hegvdx_m | 92 | M121 48（n = 33 ≤ 128 时忽略 range）、M216 4 |
| hegvdx_2stage_m | 40 | M121 48、M040 36（jobz N 时用查询到的 lwork 也返回 −14）、M065 18、M216 2 |
| geev_m（左/右向量） | 8 | — |
| getrf_m、geqrf_m（m = n、n+40、n−20） | 24 | — |
| potrf_m（L/U） | 8 | — |
| gehrd_m | 4 | — |

- 所有用例的填充行都没有被改写。
- **M216 的新路径**：hegvdx_m、hegvdx_2stage_m 在 itype 1、jobz V、range V、n = 300 时，特征值完全正确（75/75），但第 0–73 列特征向量是错的，残差 1.3e-4（相对 ‖A‖‖B‖n‖v‖）。
  - 用 `R54_DBG` 逐列打印残差，只有最后一列正确。
  - 原因与 M216 相同：`dstedx_m` l.322–326 先把三对角矩阵除以 orgnrm，再把未缩放的 (vl, vu] 传给 `dlaex0_m` 去挑选特征向量。广义问题化成标准问题后，特征值约为 0.017，与 orgnrm 差得多，所以挑错了。
  - range I、itype 2/3 都正确。
  - 已把这条路径补进 M216，没有另立新条目。
- M121、M040、M065 也补上了 `_m` 版本的实测证据。

## 局限

- 只测 ngpu = 1（真实库）。多卡分布与 lda 的交互，要用第 52 轮的虚拟设备工具另外测。
- n 只有 33 和 300 两种。
