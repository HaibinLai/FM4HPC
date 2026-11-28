// Round 52: generic schedule fuzzing of MAGMA multi-GPU routines with ngpu virtual devices on one GPU.
// gen_fuzz.py copies a MAGMA source file unchanged except that
//  * magma_setdevice / magma_getdevice are no-ops and magma_queue_create(d, q) creates the queue on GPU 0,
//    so every virtual device's buffers and queues live on GPU 0 (queue / event / host-sync semantics unchanged);
//  * after every statement that enqueues work on a queue, emu_point(queue) is called.
// emu_point counts the points; at point number g_fuzz.target the queue sleeps g_fuzz.ms (cudaLaunchHostFunc), so the
// rest of that queue's work runs late.  Only timing changes, which CUDA leaves free: a wrong result under some
// target means two operations the source does not order.
#pragma once
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include <cstdio>
struct FuzzState { long long count = 0, target = -1; int ms = 100; const char* where = nullptr; int line = 0; int ngpu = 1; };
extern FuzzState g_fuzz;
static void CUDART_CB fuzz_sleep(void* ms) { std::this_thread::sleep_for(std::chrono::milliseconds((size_t)ms)); }
static inline void emu_point_at(magma_queue_t q, const char* file, int line) {
  if (g_fuzz.count == g_fuzz.target) { cudaLaunchHostFunc(magma_queue_get_cuda_stream(q), fuzz_sleep, (void*)(size_t)g_fuzz.ms); g_fuzz.where = file; g_fuzz.line = line; }
  g_fuzz.count++; }
#define emu_point(q) emu_point_at((q), __FILE__, __LINE__)
static inline void emu_setdevice(magma_device_t) {}
#define magma_setdevice(d) emu_setdevice(d)
#define magma_getdevice(p) (*(p) = 0)
#undef magma_queue_create
#define magma_queue_create(device, queue_ptr) magma_queue_create_internal(0, queue_ptr, __func__, __FILE__, __LINE__)
// routines that ask magma_num_gpus() (clamped to the physical GPUs) get the virtual count instead
#define magma_num_gpus() ((magma_int_t)g_fuzz.ngpu)
