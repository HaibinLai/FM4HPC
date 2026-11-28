/* Preprocessor check of the CHECK_INFO blocks in MAGMA 2.10.0 batched solvers (HANDOFF §29.37).
   Line 1-2: the macro from include/magma_copy.h l.29-30 (7 arguments).  Then the call copied verbatim from
   src/zposv_batched.cpp, src/zgesv_batched.cpp, src/zgesv_nopiv_batched.cpp (6 arguments, no queue) and
   src/zgesv_rbt_batched.cpp (7 arguments).  Run: cc -E check_info_pp.c -o /dev/null */
#define magma_getvector(                 n, elemSize, dx_src, incx, hy_dst, incy, queue ) \
        magma_getvector_internal(        n, elemSize, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )
/* zposv_batched */     magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1);
/* zgesv_batched */     magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1);
/* zgesv_nopiv_batched */     magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1);
/* zgesv_rbt_batched */     magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1, queue);
