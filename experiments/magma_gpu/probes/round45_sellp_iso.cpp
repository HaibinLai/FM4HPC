// Round 45 follow-up: where does the integer division by zero (0xC0000094) of QMR/BICG/LSQR/BOMBARDMENT on SELL-P come from?
// Builds the SELL-P device matrix exactly as round29/45 (blocksize 8, alignment 8) and runs the steps of
// magma_zmtransposeconjugate one by one; each step prints before it runs, so the last line shows the crashing step.
//   round45_sellp_iso <step>   step 1: zmconvert(dSELLP -> CSR) on the device; 2: transfer to CPU then convert on CPU;
//                              3: magma_zmtransposeconjugate(dSELLP);  4: same as 3 but with a CSR input (control)
//                              5: public magma_zmconvert(device CSR -> SELLP) with blocksize/alignment set on the target
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
static magma_queue_t Q;
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); int step = argc > 1 ? atoi(argv[1]) : 3; magma_init(); magma_queue_create(0, &Q);
  int n = 64; std::vector<magma_index_t> row(n + 1), col(n); std::vector<magmaDoubleComplex> val(n); for (int i = 0; i < n; i++) { row[i] = i; col[i] = i; val[i] = MAGMA_Z_MAKE(1.0 + i, 0); } row[n] = n;
  magma_z_matrix hA = {Magma_CSR}, hB = {Magma_CSR}, dB = {Magma_CSR}, C = {Magma_CSR}, hC = {Magma_CSR}, T = {Magma_CSR};
  magma_zcsrset(n, n, row.data(), col.data(), val.data(), &hA, Q);
  hB.blocksize = 8; hB.alignment = 8; magma_int_t e = magma_zmconvert(hA, &hB, Magma_CSR, step == 4 ? Magma_CSR : Magma_SELLP, Q);
  printf("host convert -> %s: %lld (blocksize %lld alignment %lld numblocks %lld)\n", step == 4 ? "CSR" : "SELLP", (long long)e, (long long)hB.blocksize, (long long)hB.alignment, (long long)hB.numblocks);
  e = magma_zmtransfer(hB, &dB, Magma_CPU, Magma_DEV, Q); printf("transfer to device: %lld (device blocksize %lld alignment %lld)\n", (long long)e, (long long)dB.blocksize, (long long)dB.alignment);
  if (step == 1) { printf("step 1: zmconvert(dSELLP -> CSR) ...\n"); e = magma_zmconvert(dB, &C, Magma_SELLP, Magma_CSR, Q); printf("  returned %lld\n", (long long)e); }
  if (step == 2) { printf("step 2: transfer dSELLP -> CPU ...\n"); e = magma_zmtransfer(dB, &hC, Magma_DEV, Magma_CPU, Q); printf("  returned %lld (cpu blocksize %lld alignment %lld)\n", (long long)e, (long long)hC.blocksize, (long long)hC.alignment);
    printf("step 2: zmconvert(cpu SELLP -> CSR) ...\n"); e = magma_zmconvert(hC, &C, Magma_SELLP, Magma_CSR, Q); printf("  returned %lld\n", (long long)e); }
  if (step == 3 || step == 4) { printf("step %d: zmtransposeconjugate ...\n", step); e = magma_zmtransposeconjugate(dB, &T, Q); printf("  returned %lld\n", (long long)e); }
  if (step == 5) { magma_z_matrix dCSR = {Magma_CSR}, dS = {Magma_CSR}; magma_zmtransfer(hA, &dCSR, Magma_CPU, Magma_DEV, Q); dS.blocksize = 8; dS.alignment = 8;
    printf("step 5: zmconvert(device CSR -> SELLP, target blocksize 8 alignment 8) ...\n"); e = magma_zmconvert(dCSR, &dS, Magma_CSR, Magma_SELLP, Q); printf("  returned %lld (result blocksize %lld)\n", (long long)e, (long long)dS.blocksize); }
  printf("END\n"); magma_queue_destroy(Q); magma_finalize(); return 0;
}
