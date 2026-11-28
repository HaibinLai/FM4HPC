# 第 67 轮：sparse 存储格式与转置在真正复数数据上的表现（2026-11-28，roxy，库版本）

## 起点

第 46 轮的 16 种格式测试所有值的虚部都是 0，x 也是实向量，所以丢掉虚部或错误取共轭的内核都发现不了。本轮的 `probes/round67_formats_complex.cpp` 使用相同的矩阵模式，但每个非零元都是真正的复数，x 也是复向量。另外在 CSR 上增加 magma_zmtranspose（Aᵀ，不取共轭）和 magma_zmtransposeconjugate（Aᴴ）的主机和设备检查。用例表为 `scripts/roxy/r67_cases.txt`（17 × 8 = 136 个进程），运行脚本为 `run_multi.ps1`。

## 结果：与第 46 轮（实数）对照

- 各格式的往返和 SpMV 结论与第 46 轮一致，**没有复数特有的新问题**：
  - DENSE 的 SpMV 错误是 M125（实际算 Aᵀx）。复化后的 poisson 不再对称，所以这一例也错了。
  - COO 是 M252，CSRLIST 是 M254，ELLRT denserow 是 M253，ELLRT/SELLP 在第 4 步除零是 M251。
  - CSRD、ELLD 在 rect 上的问题是 M132。它在复数数据下从"往返出错"变成了堆损坏（0xC0000374），根源（缺少对角元时溢出一个槽位）不变，已补证据。
- 转置：方阵上主机和设备的 Aᵀ、Aᴴ 全部正确，共轭处理正确。
  - rect 上主机 magma_zmtranspose 访问违例。用 `probes/round67_trans_rect.cpp` 单独复现：宽矩阵 3/3 崩溃，高矩阵 3/3 维度没有交换。这两种都是已知的 M138、M137，已补证据，并写明根源在 magma_z_mtrans_template：row_ptr 和 B->row 按行数分配，却按列号写入。
  - 主机 magma_zmtransposeconjugate（经由设备）和两个设备版本对所有形状都正确。
- **M272（新，轻微）**：magma_zprint_csr_mtx 的 RowMajor 分支把 MatrixMarket 的尺寸行打印成 "n_col n_row nnz"（magma_zmio.cpp l.640/660）。3×4 矩阵输出的是 "4 3 5"，元素本身正确。magma_zwrite_csr_mtx 打印的是正确的 "num_rows num_cols"（l.473）。`probes/round67_print_colmajor.cpp`
- **M273（新，读码）**：ColMajor 分支调用 z_transpose_csr。它把三个新数组通过输出指针交出去之后，又在 cleanup 中把它们释放掉，调用者随后读的就是已释放的内存。在 roxy 上被释放的块还没有被重用，所以两次打印碰巧都正确。
- Lean：`Magma/Kernels.lean` §67 中的 `print_header_swapped`、`z_transpose_csr_returns_freed`。
- 教训（再一次）：rect 转置的问题早已登记为 M137/M138。写探针之前应先在 FINDINGS 里按函数名查一遍。
