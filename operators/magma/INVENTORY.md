# MAGMA 2.10.0 src/z*.cpp inventory (202 files)

Categories: (a) acceptance/status logic, audited earlier; (b) acceptance/status logic, audited now (task-1 multi-GPU/OOC list); (c) computational helper without a status decision; (d) NOT yet audited (status logic outside earlier families; read in this pass).

| file | family | category |
|---|---|---|
| zbulge_applyQ_v2.cpp | 2-stage bulge chasing / back-transform | (c) computational helper without a status decision: applies V2/T2 of bulge chasing to Z; info only alloc/arg |
| zbulge_applyQ_v2_m.cpp | 2-stage bulge chasing / back-transform | (b) acceptance/status logic, audited now (helper; no own status decision) |
| zbulge_aux.cpp | 2-stage bulge chasing / back-transform | (c) computational helper without a status decision: index/size helpers for bulge chasing, no status |
| zbulge_back.cpp | 2-stage bulge chasing / back-transform | (c) computational helper without a status decision: back-transform with V2 (single GPU); info only alloc |
| zbulge_back_m.cpp | 2-stage bulge chasing / back-transform | (b) acceptance/status logic, audited now (helper; no own status decision) |
| zbulge_kernel.cpp | 2-stage bulge chasing / back-transform | (c) computational helper without a status decision: bulge-chasing kernels, void/no status |
| zbulge_kernel_v2.cpp | 2-stage bulge chasing / back-transform | (c) computational helper without a status decision: bulge-chasing kernels, void/no status |
| zcgeqrsv_gpu.cpp | mixed-precision QR least-squares IR (not in the earlier IR list) | (d) NOT yet audited (read now; see findings) |
| zcgesv_gpu.cpp | mixed-precision IR | (a) acceptance/status logic, audited earlier |
| zcgetrs_gpu.cpp | mixed-precision IR | (a) acceptance/status logic, audited earlier |
| zchesv_gpu.cpp | mixed-precision IR | (a) acceptance/status logic, audited earlier |
| zcposv_gpu.cpp | mixed-precision IR | (a) acceptance/status logic, audited earlier |
| zgbsv_batched.cpp | banded LU (gbsv/gbtrf/gbtrs) | (a) acceptance/status logic, audited earlier |
| zgbsv_gpu.cpp | banded LU (gbsv/gbtrf/gbtrs) | (a) acceptance/status logic, audited earlier |
| zgbtrf_batched.cpp | banded LU (gbsv/gbtrf/gbtrs) | (a) acceptance/status logic, audited earlier |
| zgbtrf_gpu.cpp | banded LU (gbsv/gbtrf/gbtrs) | (a) acceptance/status logic, audited earlier |
| zgbtrs_batched.cpp | banded LU (gbsv/gbtrf/gbtrs) | (a) acceptance/status logic, audited earlier |
| zgebrd.cpp | bidiagonal reduction (gebrd) | (a) acceptance/status logic, audited earlier |
| zgeev.cpp | nonsymmetric eig (geev) | (a) acceptance/status logic, audited earlier |
| zgeev_m.cpp | nonsymmetric eig (geev) | (b) acceptance/status logic, audited now |
| zgegqr_gpu.cpp | gegqr | (a) acceptance/status logic, audited earlier |
| zgehrd.cpp | Hessenberg reduction (gehrd) | (a) acceptance/status logic, audited earlier |
| zgehrd2.cpp | Hessenberg reduction (gehrd) | (a) acceptance/status logic, audited earlier |
| zgehrd_m.cpp | Hessenberg reduction (gehrd) | (b) acceptance/status logic, audited now |
| zgelqf.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgelqf_gpu.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgels.cpp | least squares (gels) | (a) acceptance/status logic, audited earlier |
| zgels3_gpu.cpp | least squares (gels) | (a) acceptance/status logic, audited earlier |
| zgels_gpu.cpp | least squares (gels) | (a) acceptance/status logic, audited earlier |
| zgeqlf.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqp3.cpp | pivoted QR (geqp3) | (a) acceptance/status logic, audited earlier |
| zgeqp3_gpu.cpp | pivoted QR (geqp3) | (a) acceptance/status logic, audited earlier |
| zgeqr2x_gpu-v2.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqr2x_gpu-v3.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqr2x_gpu.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrf.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrf2_gpu.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrf3_gpu.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrf_batched.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrf_expert_batched.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrf_gpu.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrf_m.cpp | QR/LQ/QL/RQ | (b) acceptance/status logic, audited now |
| zgeqrf_mgpu.cpp | QR/LQ/QL/RQ | (b) acceptance/status logic, audited now |
| zgeqrf_ooc.cpp | QR/LQ/QL/RQ | (b) acceptance/status logic, audited now |
| zgeqrf_panel_batched.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrs3_gpu.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgeqrs_gpu.cpp | QR/LQ/QL/RQ | (a) acceptance/status logic, audited earlier |
| zgerbt_batched.cpp | RBT solver (gesv_rbt) | (c) computational helper without a status decision: applies butterfly transform; no status decision (gesv_rbt makes decisions) |
| zgerbt_gpu.cpp | RBT solver (gesv_rbt) | (c) computational helper without a status decision: applies butterfly transform; no status decision (gesv_rbt makes decisions) |
| zgerfs_nopiv_gpu.cpp | mixed-precision IR (gerfs_nopiv) | (a) acceptance/status logic, audited earlier |
| zgesdd.cpp | SVD (gesvd/gesdd/gesvj) | (a) acceptance/status logic, audited earlier |
| zgesv.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgesv_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgesv_gpu.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgesv_nopiv_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgesv_nopiv_gpu.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgesv_rbt.cpp | RBT solver (gesv_rbt) | (a) acceptance/status logic, audited earlier |
| zgesv_rbt_batched.cpp | RBT solver (gesv_rbt) | (a) acceptance/status logic, audited earlier |
| zgesvd.cpp | SVD (gesvd/gesdd/gesvj) | (a) acceptance/status logic, audited earlier |
| zgesvj_batched.cpp | SVD (gesvd/gesdd/gesvj) | (a) acceptance/status logic, audited earlier |
| zgesvj_qr_batched.cpp | SVD (gesvd/gesdd/gesvj) | (a) acceptance/status logic, audited earlier |
| zgetf2_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetf2_native.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetf2_nopiv.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetf2_nopiv_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetf2_nopiv_vbatched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetf2_vbatched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf2_mgpu.cpp | LU (getrf/getrs/gesv) | (b) acceptance/status logic, audited now |
| zgetrf_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_gpu.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_m.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_mgpu.cpp | LU (getrf/getrs/gesv) | (b) acceptance/status logic, audited now |
| zgetrf_nopiv.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_nopiv_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_nopiv_gpu.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_nopiv_panel_vbatched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_nopiv_vbatched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_panel_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_panel_native.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_panel_nopiv_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_panel_vbatched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrf_vbatched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetri_gpu.cpp | LU inverse (getri) | (a) acceptance/status logic, audited earlier |
| zgetri_outofplace_batched.cpp | LU inverse (getri) | (a) acceptance/status logic, audited earlier |
| zgetrs_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrs_gpu.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrs_nopiv_batched.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgetrs_nopiv_gpu.cpp | LU (getrf/getrs/gesv) | (a) acceptance/status logic, audited earlier |
| zgglse.cpp | equality-constrained LS (gglse) | (d) NOT yet audited (read now; see findings) |
| zggrqf.cpp | generalized RQ (ggrqf, used by gglse) | (d) NOT yet audited (read now; see findings) |
| zheevd.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zheevd_gpu.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zheevd_m.cpp | Hermitian eig drivers (heev*) | (b) acceptance/status logic, audited now |
| zheevdx.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zheevdx_2stage.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zheevdx_2stage_m.cpp | Hermitian eig drivers (heev*) | (b) acceptance/status logic, audited now |
| zheevdx_gpu.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zheevdx_m.cpp | Hermitian eig drivers (heev*) | (b) acceptance/status logic, audited now |
| zheevr.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zheevr_gpu.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zheevx.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zheevx_gpu.cpp | Hermitian eig drivers (heev*) | (a) acceptance/status logic, audited earlier |
| zhegst.cpp | generalized reduction (hegst) | (a) acceptance/status logic, audited earlier |
| zhegst_gpu.cpp | generalized reduction (hegst) | (a) acceptance/status logic, audited earlier |
| zhegst_m.cpp | generalized reduction (hegst) | (b) acceptance/status logic, audited now |
| zhegvd.cpp | generalized Hermitian eig (hegv*) | (a) acceptance/status logic, audited earlier |
| zhegvd_m.cpp | generalized Hermitian eig (hegv*) | (b) acceptance/status logic, audited now |
| zhegvdx.cpp | generalized Hermitian eig (hegv*) | (a) acceptance/status logic, audited earlier |
| zhegvdx_2stage.cpp | generalized Hermitian eig (hegv*) | (a) acceptance/status logic, audited earlier |
| zhegvdx_2stage_m.cpp | generalized Hermitian eig (hegv*) | (b) acceptance/status logic, audited now |
| zhegvdx_m.cpp | generalized Hermitian eig (hegv*) | (b) acceptance/status logic, audited now |
| zhegvr.cpp | generalized Hermitian eig (hegv*) | (a) acceptance/status logic, audited earlier |
| zhegvx.cpp | generalized Hermitian eig (hegv*) | (a) acceptance/status logic, audited earlier |
| zhesv.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zhesv_nopiv_gpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zhetrd.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (a) acceptance/status logic, audited earlier |
| zhetrd2_gpu.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (a) acceptance/status logic, audited earlier |
| zhetrd_gpu.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (a) acceptance/status logic, audited earlier |
| zhetrd_hb2st.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (c) computational helper without a status decision: band-to-tridiagonal bulge chasing; no status |
| zhetrd_he2hb.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (a) acceptance/status logic, audited earlier |
| zhetrd_he2hb_mgpu.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (b) acceptance/status logic, audited now |
| zhetrd_mgpu.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (b) acceptance/status logic, audited now |
| zhetrf.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zhetrf_aasen.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zhetrf_gpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zhetrf_nopiv.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zhetrf_nopiv_cpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zhetrf_nopiv_gpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zhetrs_nopiv_gpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zlabrd_gpu.cpp | bidiagonal reduction (gebrd) | (c) computational helper without a status decision: panel helper of a reduction; returns 0 / arg errors only |
| zlahef_gpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zlahr2.cpp | Hessenberg reduction (gehrd) | (c) computational helper without a status decision: panel helper of a reduction; returns 0 / arg errors only |
| zlahr2_m.cpp | Hessenberg reduction (gehrd) | (b) acceptance/status logic, audited now (helper; no own status decision) |
| zlahru.cpp | Hessenberg reduction (gehrd) | (c) computational helper without a status decision: panel helper of a reduction; returns 0 / arg errors only |
| zlahru_m.cpp | Hessenberg reduction (gehrd) | (b) acceptance/status logic, audited now (helper; no own status decision) |
| zlaqps.cpp | pivoted QR (geqp3) | (c) computational helper without a status decision: geqp3 panel helper; no status |
| zlarfb_gemm_batched.cpp | Householder block kernels (larfb/larft) | (c) computational helper without a status decision: BLAS-3 Householder kernel; arg checks only |
| zlarfb_gpu.cpp | Householder block kernels (larfb/larft) | (c) computational helper without a status decision: BLAS-3 Householder kernel; arg checks only |
| zlarfb_gpu_gemm.cpp | Householder block kernels (larfb/larft) | (c) computational helper without a status decision: BLAS-3 Householder kernel; arg checks only |
| zlarft_batched.cpp | Householder block kernels (larfb/larft) | (c) computational helper without a status decision: BLAS-3 Householder kernel; arg checks only |
| zlatrd.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (c) computational helper without a status decision: panel helper of a reduction; returns 0 / arg errors only |
| zlatrd2.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (c) computational helper without a status decision: panel helper of a reduction; returns 0 / arg errors only |
| zlatrd_mgpu.cpp | tridiagonal/band reduction (hetrd/he2hb/hb2st) | (b) acceptance/status logic, audited now (helper; no own status decision) |
| zlatrsd.cpp | nonsymmetric eig (geev) | (c) computational helper without a status decision: scaled triangular solve for trevc; info = arg errors only (scale handles singularity, as LAPACK latrs) |
| zlauum.cpp | lauum | (a) acceptance/status logic, audited earlier |
| zlauum_gpu.cpp | lauum | (a) acceptance/status logic, audited earlier |
| zposv.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zposv_batched.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zposv_gpu.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotf2_batched.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotf2_vbatched.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotrf.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotrf3_mgpu.cpp | Cholesky (potrf/potrs/posv) | (b) acceptance/status logic, audited now |
| zpotrf_batched.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotrf_gpu.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotrf_m.cpp | Cholesky (potrf/potrs/posv) | (b) acceptance/status logic, audited now |
| zpotrf_mgpu.cpp | Cholesky (potrf/potrs/posv) | (b) acceptance/status logic, audited now |
| zpotrf_mgpu_right.cpp | Cholesky (potrf/potrs/posv) | (b) acceptance/status logic, audited now (whole file under #if 0) |
| zpotrf_panel_batched.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotrf_panel_native.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotrf_panel_vbatched.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotrf_vbatched.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotri.cpp | Cholesky inverse (potri) | (a) acceptance/status logic, audited earlier |
| zpotri_gpu.cpp | Cholesky inverse (potri) | (a) acceptance/status logic, audited earlier |
| zpotrs_batched.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zpotrs_gpu.cpp | Cholesky (potrf/potrs/posv) | (a) acceptance/status logic, audited earlier |
| zstedx.cpp | tridiagonal D&C (stedx) | (a) acceptance/status logic, audited earlier |
| zstedx_m.cpp | tridiagonal D&C (stedx) | (b) acceptance/status logic, audited now |
| zsysv_nopiv_gpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zsytrf_nopiv_cpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zsytrf_nopiv_gpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| zsytrs_nopiv_gpu.cpp | LDLT (hetrf/sytrf) | (a) acceptance/status logic, audited earlier |
| ztrevc3.cpp | nonsymmetric eig (geev) | (c) computational helper without a status decision: eigenvector back-substitution; info = arg errors only, as LAPACK trevc3 |
| ztrevc3_mt.cpp | nonsymmetric eig (geev) | (c) computational helper without a status decision: eigenvector back-substitution; info = arg errors only, as LAPACK trevc3 |
| ztrsm_m.cpp | multi-GPU trsm | (b) acceptance/status logic, audited now |
| ztrtri.cpp | trtri | (a) acceptance/status logic, audited earlier |
| ztrtri_gpu.cpp | trtri | (a) acceptance/status logic, audited earlier |
| zungbr.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only)；**独立复核后更正**：丢弃 zungqr2/zunglq 的 iinfo，第十九轮故障注入 ⇒ info 0、结果错误（见末节） |
| zunghr.cpp | Hessenberg reduction (gehrd) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunghr_m.cpp | Hessenberg reduction (gehrd) | (b) acceptance/status logic, audited now |
| zunglq.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zungqr.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zungqr2.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zungqr_2stage_gpu.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zungqr_gpu.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zungqr_m.cpp | apply/generate Q (unmqr/ungqr family) | (b) acceptance/status logic, audited now |
| zungtr.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmbr.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmlq.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmql.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmql2_gpu.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmqr.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmqr2_gpu.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmqr_2stage_gpu.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmqr_batched.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmqr_gpu.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmqr_m.cpp | apply/generate Q (unmqr/ungqr family) | (b) acceptance/status logic, audited now |
| zunmrq.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmtr.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmtr_gpu.cpp | apply/generate Q (unmqr/ungqr family) | (c) computational helper without a status decision: Q generation/application; only propagates (note: several discard sub-call iinfo, alloc-only) |
| zunmtr_m.cpp | apply/generate Q (unmqr/ungqr family) | (b) acceptance/status logic, audited now |

## magmablas：写状态的 z 文件：先前 39 个，复核补入 3 个，共 42 个（逐文件判定；第十二至十四、二十二、二十三轮）

其余 123 个 `magmablas/z*` 文件不写 info 或标志（BLAS 类计算核：gemm、gemv、symv、trsm、herk、copy、transpose 等），不属于接受逻辑。

| 文件 | 判定 |
|---|---|
| zgbtrf_batched_fused_sm.cu | F8 km 多一行（✔ GPU） |
| zgbtrf_batched_sliding_window_sm.cu | F8（✔ GPU）；方阵 info 逻辑正确 |
| zgbtf2_kernels.cu | F2 通用回退路径只看最后一列（✔ GPU） |
| zgbsv_kernels.cu | F11 奇异时仍回代（✔ GPU） |
| zgeqr2_batched_fused_reg.cu | 无状态逻辑 |
| zgeqr2_batched.cu | 无状态逻辑 |
| zgeqr2x_gpu-v4.cu | 只有参数检查，干净 |
| zgeqr2_batched_fused_sm.cu | 只写 info = 0，干净 |
| zgeqr2_batched_fused_reg_tall.cu | 无状态逻辑 |
| zgesvj_kernels.cu | 干净（继承 F9 的收敛判据） |
| zgeqr2_batched_fused_reg_medium.cu | 无状态逻辑 |
| zgeqrf_batched_smallsq.cu | 只写 info = 0，干净 |
| zgetf2_native_kernel.cu | F7 偏移来自调用方（✔ GPU）；F12 疑似竞争（✖ 300 次未复现） |
| zgetf2_kernels_var.cu | F1 vbatched 面板外零主元被清零（✔ GPU）；F14 NaN argmax（✔ GPU） |
| zheevj_batched_small_sm.cu | F9 NaN 视为收敛（✔）、max_sweeps = 0 报成功（✔）、noconv 竞争（✖ 未复现） |
| zgesvj_batched_small_sm.cu | 已知 NaN 收敛（✔ 第三轮）；其余干净 |
| zgetrf_batched_smallsq_noshfl.cu | 干净（顺序扫描，NaN 处理同参考实现，✔ 4×4 对照） |
| zgesv_batched_small.cu | F11（✔ GPU） |
| zgetf2_kernels.cu | F2（✔）、F6 长列走 cublasIzamax 不写 info（✔）、F14（✔） |
| zgetf2.cu | F3 奇异检查被注释（✔ GPU） |
| zgetf2_nopiv_kernels_var.cu | F10 `*info++`（✔ GPU） |
| zgetf2_nopiv_kernels.cu | 状态正确（旁注：首个零主元之后的面板不再更新） |
| zgetf2_nopiv_devicefunc.cuh | 独立复核补入（写 linfo，先前的 grep 未匹配）；读码干净 |
| zhetrs.cu | 只有参数检查，干净 |
| zlag2c.cu | NaN 不报（✔）；全局标志跨队列竞争（✔ 第十四轮） |
| zlascl_2x2.cu | 独立复核后更正：ldda 报 −4（应 −6）、lddw 不查（✔ 第十九轮，`lascl2x2_args`） |
| zlascl2.cu | 越界读（✖ 未复现：编译后读取在界检查内）；参数检查干净 |
| zlat2c.cu | NaN 不报（同 zlag2c 的判据） |
| zlaswp_vbatched.cu | 无状态逻辑 |
| zpotf2_kernels.cu | F5 任意宽度版无 NaN 检查（✔ GPU） |
| zpotf2_kernels_var.cu | F5（✔ GPU） |
| zlascl.cu | F15 参数编号（✔ GPU）；cfrom/cto 检查正确 |
| zlascl_diag.cu | 只有参数检查，干净 |
| zpotf2.cu | 0 起下标（✔ 第一轮）、F4 不报 0/NaN（✔）；warp 归约无 __syncwarp（✖ 1000 次未复现） |
| zlaswp_batched.cu | 无状态逻辑 |
| zgeqr2.cpp | 只有参数与分配检查，干净 |
| zgetf2_devicefunc.cuh | F13 m < n 假 info（✔ GPU）、F14（✔） |
| zpotf2_devicesfunc.cuh | F5（✔ GPU） |
| zlaswp_device.cuh | 无状态逻辑 |
| zgeqr2_batched_fused.cuh | 只写 info = 0，干净 |
| zdiinertia.cu（第一次复核补入） | NaN 计为零特征值（✔ 第二十二轮，`diinertia_nan_as_zero`）；函数未在 magmablas_z.h 中声明 |
| zheinertia.cu（第二次复核补入） | NaN 计为零特征值（✔ 第二十三轮，设备上的 ipiv） |

## sparse/src：38 个 z 源文件与 5 个包装文件（逐文件判定）

32 个线性求解器在 GPU 0 上各跑 6 个输入（`experiments/magma_gpu/results/sweep_results.txt`）。

| 文件 | 判定 |
|---|---|
| zbicg.cpp / zpbicg.cpp | breakdown 被收尾覆盖成成功（`bicg_breakdown_success`）；GPU 扫描：NaN 与极小 b 静默失败 |
| zbicgstab.cpp / zpbicgstab.cpp | 早退出查 ‖b‖（`early_exit_returns_x0`）；精确 x0 ⇒ 0/0；GPU：极小 b 静默失败 |
| zbicgstab_merge.cpp / zpbicgstab_merge.cpp | 同上；精确 x0 误报失败 |
| zcg.cpp / zpcg.cpp | 早退出、`template_ignores_final_res`；GPU：极小 b 静默失败 |
| zcg_merge.cpp / zpcg_merge.cpp | 复数不取共轭（`cg_merge_false_convergence`）；pcg_merge 升级成功（`pcg_merge_upgrade`） |
| zcg_res.cpp | 精确 x0 报非正定（`cgres_exact_x0_nonspd`） |
| zcgs.cpp / zpcgs.cpp / zcgs_merge.cpp / zpcgs_merge.cpp | 早退出；GPU：极小 b 静默失败 |
| zqmr.cpp / zqmr_merge.cpp / zpqmr.cpp / zpqmr_merge.cpp | GPU：无静默失败；QMR 有误报失败 |
| ztfqmr.cpp / ztfqmr_merge.cpp | GPU：无静默失败 |
| zptfqmr.cpp / zptfqmr_merge.cpp | breakdown 被覆盖（`overwrite_group_size`）；GPU：NaN 与极小 b 静默失败 |
| zfgmres.cpp | 粘住的成功（`fgmres_sticky_success`；1152 次扫描中 252 次） |
| zidr.cpp / zpidr.cpp | 参数改写（`idr_params`）；GPU：极小 b 静默失败 |
| zidr_merge.cpp / zpidr_merge.cpp | GPU：所有非平凡输入都失败（扫描中发现） |
| zidr_strms.cpp / zpidr_strms.cpp | 包装层中调用被注释掉（magma_z_solver_wrapper.cpp l.125、130），公开求解器类型不可达 |
| zjacobi.cpp | 存储的 0 对角元（`jacobi_stored_zero_passes`）；精确 x0 误报失败 |
| zlobpcg.cpp | 参数检查失效、NaN 收敛（`lobpcg_check_dead`、`lobpcg_nan_converged`） |
| zlsqr.cpp | GPU：极小 b 静默失败 |
| zbombard.cpp / zbombard_merge.cpp | 残差记录错误（`bombard_wrong_res`，✖ 未复现）；GPU：极小 b 静默失败 |
| zbpcg.cpp | num_vecs 算错，第 2 列不解（`bpcg_col0`，✔） |
| zresidual.cpp / zresidualvec.cpp | 独立复核后更正：多右端只写 res[0]（✔ 第十九轮，`residual_multi_first_only`）；A 0 行时的除零第十九轮未测到（b 也是 0 行），第二十三轮 A 0 行、b 4 行 ⇒ SIGFPE ✔ |
| magma_z_solver_wrapper.cpp | 默认分支返回 0（`wrapper_unhandled_success`）；bpcg 分派 |
| magma_z_precond_wrapper.cpp | setup 失败被覆盖（`precond_setup_info_lost`）；ILU/ICC apply 不写 x、ConjTrans 无分支（`applyprecond_ilu_noop`、`applyright_conjtrans_noop`） |
| magma_zcustomprecond.cpp / magma_zcustomspmv.cpp | 用户钩子的示例实现，返回 0，无状态逻辑 |
| magma_zqr_wrapper.cpp | 读码：zgeqrf_gpu 的 info 被随后 zungqr_gpu 的 info 覆盖（两者只会报参数或分配错误），轻微 |

## 41 个辅助例程（(c) 类）逐文件判定（第十一、十四、十七轮）

辅助例程自身只会报参数错误或分配失败。下表给出每个文件是否分配、是否有公开入口，以及对应的 GPU 结果（分配失败的故障注入）。共 41 个（先前自动提取时的正则漏掉了文件名含大写字母的 zbulge_applyQ_v2.cpp，已补上）。

| 文件 | 判定 |
|---|---|
| zbulge_applyQ_v2.cpp | 内部（无公开入口），有分配；经 zheevdx_2stage(_m) 驱动的故障注入覆盖（§3d） |
| zbulge_aux.cpp | 内部（无公开入口），有分配；经 zheevdx_2stage(_m) 驱动的故障注入覆盖（§3d） |
| zbulge_back.cpp | 内部，有分配；同上 |
| zbulge_kernel.cpp | 内部，有分配；同上 |
| zbulge_kernel_v2.cpp | 内部，无分配；无状态判断 |
| zgerbt_batched.cpp | 有分配，无公开的非批量入口；经 zgesv_rbt_batched 运行（第五轮） |
| zgerbt_gpu.cpp | ✔ 故障注入（第十七轮）：两次分配失败都正确报告 −113 |
| zhetrd_hb2st.cpp | 公开、有分配；输入难以单独构造，经 2-stage 驱动的故障注入覆盖（§3d） |
| zlabrd_gpu.cpp | 公开、无分配；只有参数检查 |
| zlahr2.cpp | 公开、无分配；只有参数检查 |
| zlahru.cpp | 公开、无分配；只有参数检查 |
| zlaqps.cpp | 公开、无分配；只有参数检查 |
| zlarfb_gemm_batched.cpp | 无分配；只有参数检查 |
| zlarfb_gpu.cpp | 无分配；只有参数检查 |
| zlarfb_gpu_gemm.cpp | 无分配；只有参数检查 |
| zlarft_batched.cpp | 无分配；只有参数检查 |
| zlatrd.cpp | 公开、无分配；只有参数检查 |
| zlatrd2.cpp | 公开、无分配；只有参数检查 |
| zlatrsd.cpp | 无分配；只有参数检查 |
| ztrevc3.cpp | ✔ 第十四轮：不做设备分配，结果与未注入时一致 |
| ztrevc3_mt.cpp | ✔ 第十四轮：同上 |
| zungbr.cpp | ✔ 独立复核后更正：内部 zungqr2/zunglq 的 iinfo 不读；第十九轮故障注入 ⇒ info 0、Q/P 相对误差 52 / 8.6（`helper_iinfo_dropped`） |
| zunghr.cpp | ✔ 第十一轮：zungqr 的分配失败 ⇒ info 0、Q 错误（`helper_iinfo_dropped`） |
| zunglq.cpp | ✔ 第十八轮：头文件声明 (…, dT, nb) 与定义 (…, work, lwork) 不一致，按头文件调用段错误；按定义调用时分配失败正确报告 −113 |
| zungqr.cpp | ✔ 第十七轮：分配失败正确报告 −113 |
| zungqr2.cpp | ✔ 第十四轮：分配失败正确报告 −113 |
| zungqr_2stage_gpu.cpp | 内部，有分配；经 2-stage 驱动覆盖 |
| zungqr_gpu.cpp | ✔ 第十七轮：分配失败正确报告 −113 |
| zungtr.cpp | 不在公开头文件中；代码与 zunmtr 同样丢弃 iinfo（`helper_iinfo_dropped`） |
| zunmbr.cpp | ✔ 第十一轮：zunmqr 的分配失败 ⇒ info 0、C 不变 |
| zunmlq.cpp | ✔ 第十四轮：正确报告 −113 |
| zunmql.cpp | ✔ 第十四轮：正确报告 −113/−112 |
| zunmql2_gpu.cpp | ✔ 第十七轮：正确报告 −113 |
| zunmqr.cpp | ✔ 第十四轮：正确报告 −113 |
| zunmqr2_gpu.cpp | ✔ 第十七轮：正确报告 −113 |
| zunmqr_2stage_gpu.cpp | 内部，有分配；经 2-stage 驱动覆盖 |
| zunmqr_batched.cpp | 无分配；只有参数检查 |
| zunmqr_gpu.cpp | 公开、无分配；只有参数检查 |
| zunmrq.cpp | ✔ 第十七轮：正确报告 −113 |
| zunmtr.cpp | ✔ 第十一轮：zunmqr 的分配失败 ⇒ info 0、C 不变（`helper_iinfo_dropped`） |
| zunmtr_gpu.cpp | 公开、无分配；只有参数检查 |
