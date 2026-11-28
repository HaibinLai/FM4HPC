# FM4HPC：用形式化方法检验“低精度模拟高精度”的数值算子

- 一页简介：`BRIEF_zh.md`；失效类型分类与传统方法：`operators/TAXONOMY_zh.md`；完整工作记录：`HANDOFF.md`。
- Lean 4 证明（只用 core Lean，不用 Mathlib）：`lean/` 是一个 Lake 包，`cd lean && lake build` 检查全部模块；模块说明见 `lean/OptimiKahan/README.md`，MAGMA 部分在 `lean/OptimiKahan/Magma/`。
- MAGMA 审计：覆盖表与清点在 `operators/magma/`，GPU 探针、脚本与结果在 `experiments/magma_gpu/`（`probes/`、`scripts/`、`shims/`、`inputs/`、`results/`）。
- 实验与结果：`experiments/`，各子目录对应 HANDOFF 的章节：
  - `cublas_emu`：cuBLAS FP64 仿真；
  - `cusolver_irs`：cuSOLVER IRS；
  - `magma_fgmres`：MAGMA FGMRES-IR，含白盒模型的 GPU 确认、轨迹验证和修复（`fix/`）；
  - `ozimmu`、`gemmul8`、`fp16_fp64`：Ozaki 系列。
- 未包含：
  - 尚未提交的厂商 bug 报告草稿；
  - `experiments/fp16_fp64` 的原始 `.bin` 输出（约 330 MB，可由脚本重新生成）；
  - Lean 工具链。
- 集群连接信息在公开版本中替换成了 `<NODE_IP>`、`<KEY_PATH>`。

---

# lean4HPC 初始调研与实验

当前主线：FP16→FP64，选择OzBLAS；INT8硬件实验暂缓。

- experiments/fp16_fp64/REPORT.md：已完成的CPU数值模型、56条实验记录、求和顺序反例与可信边界。
- experiments/fp16_fp64/compare.py：固定输入、精确有理数参考与分片模型。
- experiments/fp16_fp64/run_gpu.sh：V100原库对比cuBLAS DGEMM的待运行入口，尚未编译。
- literature/TENSORLEAN_AND_FP16.md：TensorLean相关工作及对本项目定位的影响。
- operators/vendor/ozblas：固定提交的选定源码与LICENSE，非完整可构建checkout。

以下是保留的INT8调研与CPU实验记录：

- literature/REVIEW.md：12组文献与形式化工具调研。
- operators/REVIEW.md：7个算子库源码调研，初始INT8候选为ozIMMU。
- experiments/EXPERIMENT.md：已运行的比较实验、范围限制与后续计划。
- experiments/compare_ozimmu.py：CPU算术模型 vs NumPy FP64和精确有理数。
- experiments/replay_complex_beta.py：直接抽取原函数体、编译host replay，对比Python complex。
- experiments/ComplexBeta.lean：待检查的Lean 4整数反例，无sorry；本机未运行Lean。
- operators/vendor/ozIMMU：固定版本选定源码与LICENSE，非完整可构建checkout。

已复现源码级复数beta更新顺序反例。未运行GPU、cuBLAS或形式化检查。所有数值测试结论仅覆盖报告列明的输入与CPU模型。
