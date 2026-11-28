# Round 28 sweep (structured inputs): triage

`probes/round28_sweep.cpp` on roxy, 5,125 cases. Every input has a prescribed spectrum: A = U·diag(σ)·Vᴴ with Haar
unitary factors, or Q·diag(λ)·Qᴴ. The profiles are:
- rank n/2 and rank 1;
- σ geometric from 1 to 1e−12;
- one σ = 1e−8;
- a gap at n/2;
- the Kahan matrix;
- rows graded over 1e±100;
- clustered, ill-conditioned, two-cluster and low-rank Hermitian spectra.

Verdicts are as in round 27. Outputs after info > 0 are not judged.

| Group | Routine | Cluster | Verdict | Reason / evidence |
|---|---|---|---|---|
| qrcp | zgeqp3 (CPU interface) | 92 of 238 WRONG. rank 1, cond 1e12 and graded fail at every n ≥ 63 (22/22 each); Kahan 10; low-rank and gap partly. 90 have A·P ≠ Q·R (ratio 1e10–1e14), 2 only fail to reveal the rank. | **KNOWN (round 27), now shown to be systematic** | `round28_fixed.exe` links the same sweep with `probes/zlaqps_patched2.cpp`: 714 of 714 OK (`qrcp_patched.csv`). So the two magma_zlaqps root causes account for every failure. Ill-conditioned matrices trigger the norm recomputation as well, not only rank-deficient ones. |
| qrcp | zgeqp3_gpu | with a host `tau`, every call (even n = 1) gives an illegal memory access and info = 0, and the CUDA context is lost | **NEW (API)** | `tau` must be a device array, as MAGMA's own `testing_zgeqp3_gpu.cpp` uses `dtau`. But magma_z.h declares `magmaDoubleComplex *tau` (a host type, not `_ptr`), and the documentation just says "COMPLEX_16 array". With a device tau, all cases are OK (`qp3small.txt`). The GPU panel `zlaqps2_gpu` does not have the zlaqps defects. |
| qrcp | zgeqrf | — | OK | |
| lu | zgesv, zgesv_gpu, zgetrf_native + getrs | — | OK (357) | Graded and ill-conditioned systems are solved backward-stably. |
| chol | zposv, zposv_gpu, zpotrf_native + potrs | — | OK (204) | |
| batched | zgesv_batched, zgetrf_batched + getrs_batched | rank 1 / rank n/2 at n = 2, 3, 8: info 0 where LAPACK gives an exact-zero pivot | NOT A BUG | Rounding order, as in round 27. |
| batched | zposv_batched | — | OK | |
| heig | zheevr(V) | cond 1e12: orthogonality 31–553·n·eps | NOT MAGMA | LAPACK zheevr gives identical numbers on the same matrices (`heevr_vs_lapack.txt`). This is MRRR (zstemr) behaviour. |
| heig | zheevdx_2stage(V) | rank n/2, n = 200: orthogonality 47·n·eps | WARN (mild) | Not investigated further. |
| heig | zheevd, zheevd_gpu, zheevdx, zheevx | — | OK | |
| svd | zgesvd, zgesdd | — | OK (460) | |

The first qrcp run lost its CUDA context in the 18×1 case. That was the illegal access left by the preceding zgeqp3_gpu call with a host tau, and it explains the "cusparseCreate: CUDA context cannot be initialized" message and the −113 that followed.
