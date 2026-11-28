# 第 58 轮：CPU 接口的生产方 → 消费方组合（2026-11-28，roxy，库版本；没有新缺陷）

## 方法

`probes/round58_cpu_pairs.cpp`：
- 第 36 轮给消费方喂的是 LAPACK 生成的反射向量。本轮换成 **MAGMA 自己的生产方输出**，同一份输出再交给 LAPACK 的消费方作对照，所以只检验消费方和交接这一步。
- 生产方本身也与 LAPACK 生产方比较：QR 类比较反射向量与 tau，gebrd 和 hetrd 比较 |d|、|e|。

| 生产方 | 消费方 | 参数 | 用例 | 结果 |
|---|---|---|---|---|
| magma_zgebrd | unmbr（Q/P × L/R × N/C）、ungbr（Q/P） | (m, n) ∈ {300×200, 200×300, 257×257, 64×40}，ld + {0, 5}，C 的另一维 50 | 64 + 16 | 全部正确 |
| magma_zgeqrf | unmqr（L/R × N/C）、ungqr2 | 同上 | 32 + 8 | 全部正确 |
| magma_zgelqf、magma_zgelqf_gpu | unmlq（L/R × N/C） | 同上 | 32 + 32 | 全部正确 |
| magma_zgeqlf | unmql（L/R × N/C；反射向量在最后 k 列） | 同上 | 32 | 全部正确 |
| magma_zhetrd（U/L） | unmtr（L/R × N/C） | n ∈ {33, 300, 513} | 48 | 全部正确 |

共 264 例，每例一个进程（`run_list.ps1`，用例表为 `scripts/roxy/r58_cases.txt`），全部 exit 0，无 CUDA 错误，无 xerbla 输出。误差都在 1e-14 量级以下。

## 结论

CPU 接口的这些组合，用的都是与 LAPACK 兼容的格式（反射向量与 tau），没有发现约定不一致。第 53、55 轮的问题（M259、M261–M263）都出在 MAGMA 自己定义的中间格式上，即 T、dT 和 dT 的工作区部分。这与本轮的正面结果是一致的。
