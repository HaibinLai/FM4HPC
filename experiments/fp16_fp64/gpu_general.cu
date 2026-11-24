// General harness for the ORIGINAL OzBLAS cuozblasRgemm<double,half,float> vs cuBLAS DGEMM.
// Unlike gpu_compare.cu: transposes, alpha/beta, initial C, non-finite inputs, larger shapes, repeat runs.
// Usage: gpu_general --batch LIST   (one job per line: IN.bin OUT.bin [options])
//    or: gpu_general IN.bin OUT.bin [--ta N|T] [--tb N|T] [--alpha x] [--beta y] [--c0 FILE] [--ws MB] [--repeat R] [--maxsplit S] [--sumMode 0|2] [--fast 0|1] [--batched 0|1] [--epsmode 0|2|9] [--repro 0|1] [--pad P]
// IN.bin: uint32 M,N,K then A and B column-major binary64 AS STORED:
//   A is M x K (lda=M) if ta=N else K x M (lda=K);  B is K x N (ldb=K) if tb=N else N x K (ldb=N).
// --c0 FILE: raw binary64 M*N column-major initial C for both libraries (default zeros).
// OUT.bin: same header, then cuBLAS result (M*N) and OzBLAS result (M*N), column-major binary64.
// stdout: one JSON line (device, split counts, library status, bitwise repeat-differences).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuozblas.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e)); std::exit(2);} } while(0)
#define BK(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS){ std::fprintf(stderr,"cuBLAS %s: %d\n",#x,int(e)); std::exit(3);} } while(0)

static cuozblasHandle_t g_oh; static bool g_oh_init = false; static int g_ws = 0;
static int run_job(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: gpu_general IN.bin OUT.bin [options]\n"); return 1; }
  char ta = 'N', tb = 'N'; double alpha = 1., beta = 0.; std::string c0; int wsMB = 1024, repeat = 1, maxsplit = 12, sumMode = 2, fastMode = 0, batched = 0, epsMode = 0, repro = 1, pad = 0;
  for (int i = 3; i < argc; i++) {
    std::string o = argv[i]; auto nx = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", o.c_str()); std::exit(1);} return argv[++i]; };
    if (o == "--ta") ta = nx()[0]; else if (o == "--tb") tb = nx()[0];
    else if (o == "--alpha") alpha = std::atof(nx()); else if (o == "--beta") beta = std::atof(nx());
    else if (o == "--c0") c0 = nx(); else if (o == "--ws") wsMB = std::atoi(nx());
    else if (o == "--repeat") repeat = std::atoi(nx()); else if (o == "--maxsplit") maxsplit = std::atoi(nx());
    else if (o == "--sumMode") sumMode = std::atoi(nx()); else if (o == "--fast") fastMode = std::atoi(nx()); else if (o == "--batched") batched = std::atoi(nx());
    else if (o == "--epsmode") epsMode = std::atoi(nx()); else if (o == "--repro") repro = std::atoi(nx());
    else if (o == "--pad") pad = std::atoi(nx());
    else { std::fprintf(stderr, "unknown option %s\n", o.c_str()); return 1; }
  }
  std::ifstream in(argv[1], std::ios::binary); uint32_t d[3] = {};
  in.read((char*)d, sizeof(d)); const int m = d[0], n = d[1], k = d[2];
  if (!in || m < 1 || n < 1 || k < 1 || m > 8192 || n > 8192 || k > (1u << 24)) { std::fprintf(stderr, "bad dimensions\n"); return 1; }
  std::vector<double> a(size_t(m) * k), b(size_t(k) * n), c(size_t(m) * n, 0.0);
  in.read((char*)a.data(), a.size() * 8); in.read((char*)b.data(), b.size() * 8);
  if (!in) { std::fprintf(stderr, "truncated input\n"); return 1; }
  if (!c0.empty()) { std::ifstream cf(c0, std::ios::binary); cf.read((char*)c.data(), c.size() * 8); if (!cf) { std::fprintf(stderr, "bad c0\n"); return 1; } }
  // --pad P: leading dimensions = rows + P; padding filled with a sentinel NaN to detect reads/writes outside the logical matrix.
  const int rowsA = (ta == 'N') ? m : k, colsA = (ta == 'N') ? k : m, rowsB = (tb == 'N') ? k : n, colsB = (tb == 'N') ? n : k;
  const int lda = rowsA + pad, ldb = rowsB + pad, ldc = m + pad;
  double sent; { uint64_t u = 0x7ff4deadbeef1234ULL; std::memcpy(&sent, &u, 8); }
  auto padded = [&](const std::vector<double>& s, int rows, int cols, int ld) { std::vector<double> o(size_t(ld) * cols, sent); for (int j = 0; j < cols; j++) std::memcpy(&o[size_t(j) * ld], &s[size_t(j) * rows], size_t(rows) * 8); return o; };
  std::vector<double> ap = padded(a, rowsA, colsA, lda), bp = padded(b, rowsB, colsB, ldb), cp = padded(c, m, n, ldc);
  CK(cudaSetDevice(0)); cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  double *da, *db, *dr, *dy;
  CK(cudaMalloc((void**)&da, ap.size() * 8)); CK(cudaMalloc((void**)&db, bp.size() * 8));
  CK(cudaMalloc((void**)&dr, cp.size() * 8)); CK(cudaMalloc((void**)&dy, cp.size() * 8));
  CK(cudaMemcpy(da, ap.data(), ap.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, bp.data(), bp.size() * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dr, cp.data(), cp.size() * 8, cudaMemcpyHostToDevice));
  cublasHandle_t ch; BK(cublasCreate(&ch)); BK(cublasSetMathMode(ch, CUBLAS_DEFAULT_MATH));
  BK(cublasDgemm(ch, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, m, n, k, &alpha, da, lda, db, ldb, &beta, dr, ldc));
  CK(cudaDeviceSynchronize());
  if (!g_oh_init || g_ws != wsMB) { if (g_oh_init) { g_oh.useBatchedGemmFlag = 1; cuozblasDestroy(&g_oh); } g_oh = cuozblasHandle_t{}; cuozblasCreate(&g_oh, (size_t)wsMB * 1024ULL * 1024ULL); g_oh_init = true; g_ws = wsMB; }
  cuozblasHandle_t& oh = g_oh;
  oh.nSplitMax = maxsplit; oh.reproMode = repro; oh.fastMode = fastMode; oh.sumMode = sumMode; oh.splitEpsMode = epsMode; oh.useBatchedGemmFlag = batched;
  std::vector<double> first(cp.size()), cur(cp.size(), 0.0), refp(cp.size()); int status = 0, diffs = 0;
  for (int r = 0; r < repeat; r++) {
    CK(cudaMemcpy(dy, cp.data(), cp.size() * 8, cudaMemcpyHostToDevice));
    status = cuozblasRgemm<double, half, float>(&oh, ta, tb, m, n, k, alpha, da, lda, db, ldb, beta, dy, ldc);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(cur.data(), dy, cur.size() * 8, cudaMemcpyDeviceToHost));
    if (r == 0) first = cur; else if (std::memcmp(first.data(), cur.data(), cur.size() * 8) != 0) diffs++;
  }
  CK(cudaMemcpy(refp.data(), dr, refp.size() * 8, cudaMemcpyDeviceToHost));
  // padding must be untouched in C (both libraries) and in A/B (read-only inputs copied back for the check)
  int clob_oz = 0, clob_cu = 0, clob_in = 0;
  for (int j = 0; j < n; j++) for (int i = m; i < ldc; i++) { if (std::memcmp(&first[size_t(j) * ldc + i], &sent, 8)) clob_oz++; if (std::memcmp(&refp[size_t(j) * ldc + i], &sent, 8)) clob_cu++; }
  { std::vector<double> ca(ap.size()), cb(bp.size()); CK(cudaMemcpy(ca.data(), da, ca.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(cb.data(), db, cb.size() * 8, cudaMemcpyDeviceToHost));
    if (std::memcmp(ca.data(), ap.data(), ca.size() * 8) || std::memcmp(cb.data(), bp.data(), cb.size() * 8)) clob_in = 1; }
  std::vector<double> ref(size_t(m) * n), ozl(size_t(m) * n);
  for (int j = 0; j < n; j++) { std::memcpy(&ref[size_t(j) * m], &refp[size_t(j) * ldc], size_t(m) * 8); std::memcpy(&ozl[size_t(j) * m], &first[size_t(j) * ldc], size_t(m) * 8); }
  std::ofstream f(argv[2], std::ios::binary); f.write((char*)d, sizeof(d));
  f.write((char*)ref.data(), ref.size() * 8); f.write((char*)ozl.data(), ozl.size() * 8); f.close();
  std::printf("{\"out\":\"%s\",\"device\":\"%s\",\"splits_A\":%.0f,\"splits_B\":%.0f,\"partial_gemms\":%.0f,\"status\":%d,\"sumMode\":%d,\"fast\":%d,\"batched\":%d,\"epsmode\":%d,\"ta\":\"%c\",\"tb\":\"%c\",\"alpha\":%.17g,\"beta\":%.17g,\"repeat\":%d,\"repeat_bitwise_diffs\":%d,\"pad\":%d,\"pad_clobbered_oz\":%d,\"pad_clobbered_cublas\":%d,\"inputs_modified\":%d}\n",
              argv[2], prop.name, oh.nSplitA, oh.nSplitB, oh.nSplitA * oh.nSplitB, status, sumMode, fastMode, batched, epsMode, ta, tb, alpha, beta, repeat, diffs, pad, clob_oz, clob_cu, clob_in);
  BK(cublasDestroy(ch)); cudaFree(da); cudaFree(db); cudaFree(dr); cudaFree(dy);
  return 0;
}

// --batch LIST: each line = "IN.bin OUT.bin [options]" ; jobs run sequentially in one process.
int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--batch") {
    std::ifstream lf(argv[2]); std::string line; int fails = 0;
    while (std::getline(lf, line)) {
      std::vector<std::string> tok; size_t p = 0;
      while (p < line.size()) { while (p < line.size() && line[p] == ' ') p++; size_t q = line.find(' ', p); if (q == std::string::npos) q = line.size(); if (q > p) tok.push_back(line.substr(p, q - p)); p = q; }
      if (tok.size() < 2) continue;
      std::vector<char*> av; av.push_back((char*)"gpu_general"); for (auto& t : tok) av.push_back((char*)t.c_str());
      int rc = run_job((int)av.size(), av.data());
      if (rc != 0) { fails++; std::printf("{\"out\":\"%s\",\"job_error\":%d}\n", tok[1].c_str(), rc); }
      std::fflush(stdout);
    }
    return fails ? 4 : 0;
  }
  return run_job(argc, argv);
}
