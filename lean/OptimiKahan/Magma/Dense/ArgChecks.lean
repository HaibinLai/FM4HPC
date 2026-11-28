/-!
# MAGMA argument checks: the reported `−k` against the position of the parameter tested

A LAPACK-style routine reports an invalid argument as `info = −k`, where k is the argument's position. MAGMA often adds
parameters to a LAPACK signature (block sizes, offsets, batch arrays, device workspace) without renumbering the checks
it copied, so the routine names the wrong argument.  Found by `experiments/magma_gpu/scripts/argnum_scan.py`, which
takes each function's usual offset (e.g. MAGMA's numbering that skips a leading `ngpu`) as its convention and reports
checks that deviate from it; every hit below was then compared with the signature by hand (HANDOFF §29.33).
`zlarfb_gpu` (a false positive of the scan, all numbers correct) and `zgeqrs3_gpu` (already `geqrs3_argnums_wrong`)
are not repeated.  Evidence: code reading, z files of MAGMA v2.10.0 (s/c/d are generated from them).  Core Lean only.
-/

namespace MagmaAudit

/-! ## 33a. Generalized eigensolvers: `jobz` and `range` codes swapped -/

/-- `zhegvx.cpp` l.240–246, `zhegvr.cpp` l.304–310, `zhegvdx.cpp` l.254–260, `zhegvdx_2stage.cpp` l.280–286 and the
    `_m` versions: the signature (like LAPACK ZHEGVX) is `itype, jobz, range, uplo, …`, LAPACK reports a bad JOBZ as −2
    and a bad RANGE as −3, but MAGMA tests `range` first and reports −2 for it and −3 for `jobz`.
    `(argument tested, code reported)` -/
def hegvChecks : List (String × Int) := [("itype", -1), ("range", -2), ("jobz", -3), ("uplo", -4), ("n", -5)]
def hegvSignature : List String := ["itype", "jobz", "range", "uplo", "n"]
/-- position (1-based) of an argument in the signature -/
def posIn (sig : List String) (a : String) : Nat := sig.idxOf a + 1

theorem hegv_jobz_range_swapped :
    (hegvChecks.filter (fun c => c.2 ≠ -(posIn hegvSignature c.1 : Int))).map Prod.fst = ["range", "jobz"] := by
  decide

/-! ## 33b. Checks that name the wrong argument -/

/-- each entry: routine, argument tested, code reported, actual position (from the signature; z files):
    the checks copied from a LAPACK routine whose signature has fewer leading parameters. -/
def argnumTable : List (String × String × Nat × Nat) :=
  [ ("zcposv_gpu", "n", 1, 2), ("zcposv_gpu", "nrhs", 2, 3), ("zcposv_gpu", "ldda", 4, 5),
    ("zchesv_gpu", "n", 1, 2), ("zchesv_gpu", "nrhs", 2, 3), ("zchesv_gpu", "ldda", 4, 5),
    ("zgerfs_nopiv_gpu", "n", 1, 2), ("zgerfs_nopiv_gpu", "nrhs", 2, 3), ("zgerfs_nopiv_gpu", "ldda", 4, 5),
    ("zgerfs_nopiv_gpu", "lddb", 8, 7), ("zgerfs_nopiv_gpu", "lddx", 10, 9),
    ("zgesv_batched", "lddb", 6, 7), ("zgetrs_nopiv_batched", "lddb", 8, 7),
    ("zgeqrf_expert_batched", "ldda", 4, 5), ("zgeqrf_expert_batched", "lddr", 6, 7),
    ("zgeqrf_expert_batched", "lddt", 8, 9),
    ("zgetrf2_mgpu", "lddat", 5, 7), ("zgetrf_piv", "lda", 4, 5),
    ("zgetrf_recpanel_batched", "ai", 4, 5), ("zgetrf_recpanel_batched", "aj", 5, 6),
    ("zgetrf_recpanel_batched", "ldda", 6, 7),
    ("zhetrd_he2hb", "lda", 4, 5), ("zhetrd_he2hb", "lwork", 9, 8),
    ("zhetrd_he2hb_mgpu", "lda", 4, 5), ("zhetrd_he2hb_mgpu", "lwork", 9, 8),
    ("zlahr2", "ldy", 13, 14), ("zlarft_internal_batched", "lwork", 10, 15),
    ("zlatrd", "ldda", 11, 13), ("zlatrd", "lddw", 13, 15),
    ("zpotrf3_mgpu", "uplo", 1, 2), ("zpotrf3_mgpu", "n", 2, 4), ("zpotrf3_mgpu", "ldda", 4, 9),
    ("zpotrf_expert_batched", "ldda", 4, 6), ("zpotrf_expert_batched", "batchCount", 6, 8),
    ("zpotrf_rectile_native", "ldda", 4, 5), ("zunmqr_2stage_gpu", "lddc", 10, 9),
    ("zgbtrf_batched_work", "lwork", 13, 10), ("zgbtrf_batched_sliding_window_loopout", "lwork", 13, 10),
    ("zgbsv_native_work", "lwork", 10, 12),
    ("zgeqr2_batched", "ldda", 4, 6), ("zgeqr2_fused_batched", "ldda", 4, 6),
    ("zgeqr2_fused_reg_batched", "ldda", 4, 6), ("zgeqr2_fused_reg_medium_batched", "ldda", 4, 6),
    ("zgeqr2_fused_reg_tall_batched", "ldda", 4, 6), ("zgeqr2_fused_sm_batched", "ldda", 4, 6),
    ("zgesvj_batched_small_sm", "lddv", 9, 11), ("zgesvj_batched_small_sm", "batchCount", 11, 13),
    ("zgesvj_batched_update_vectors", "batchCount", 9, 11),
    ("zheevj_batched_expert_small_sm", "heevj_tol", 8, 10), ("zheevj_batched_expert_small_sm", "sort_flag", 9, 11),
    ("zheevj_batched_expert_small_sm", "nthreads", 10, 13), ("zheevj_batched_expert_small_sm", "batchCount", 11, 14),
    ("zheevj_batched_strided_expert_small_sm", "heevj_tol", 10, 12),
    ("zheevj_batched_strided_expert_small_sm", "sort_flag", 11, 13),
    ("zheevj_batched_strided_expert_small_sm", "nthreads", 12, 15),
    ("zheevj_batched_strided_expert_small_sm", "batchCount", 13, 16),
    ("zpotrf_lpout_batched", "lda", 4, 6), ("zpotrf_lpin_batched", "lda", 4, 6) ]

/-- every listed check reports a position that is not the tested argument's position -/
theorem argnum_all_wrong : argnumTable.all (fun e => e.2.2.1 != e.2.2.2) = true := by decide
/-- 58 checks in 33 routines -/
theorem argnum_counts : argnumTable.length = 58 ∧ (argnumTable.map Prod.fst).eraseDups.length = 33 := by decide
/-! ## 33c. `zcposv_gpu`, `zchesv_gpu`: `uplo` is never checked -/

/-- `zcposv_gpu.cpp` l.163–174: there is no `uplo` check.  An invalid `uplo` passes, `magmablas_zlanhe` complains (its
    own −2), `magmablas_zlat2c` returns −1, which the driver reads as a conversion failure: `iter = −2` ("overflow in
    the conversion"), then the FP64 fallback `zpotrf_gpu` returns −1.  `(iter, info)` -/
def cposvBadUplo : Int × Int :=
  let lat2c := -1                       -- zlat2c: uplo is its argument 1
  let iter := if lat2c ≠ 0 then -2 else 0
  (iter, -1)                            -- fallback zpotrf_gpu: uplo is its argument 1
theorem cposv_bad_uplo_reported_as_conversion : cposvBadUplo = (-2, -1) := by decide

end MagmaAudit

#print axioms MagmaAudit.hegv_jobz_range_swapped
#print axioms MagmaAudit.argnum_all_wrong
#print axioms MagmaAudit.argnum_counts
#print axioms MagmaAudit.cposv_bad_uplo_reported_as_conversion
