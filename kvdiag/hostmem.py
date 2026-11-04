"""Pinned host memory as CUDA-addressable tensors, plus tier-0 kernels. Built with load_inline.

STATUS: UNVERIFIED on GH200 as of 2026-11-03. Written on a Mac without CUDA. First GH200 task:
    python -c "import kvdiag.hostmem as h; h.selftest()"

host_tensor(shape, dtype) -> cuda_view
    cudaHostAlloc(mapped|portable) + cudaHostGetDevicePointer + torch::from_blob(device=cuda).
    The returned CUDA tensor carries `.cpu_view`, a CPU tensor over the same bytes, which owns
    the allocation (keep the cuda view alive no longer than it). cpu_view is what memcpy-based
    arms pass to `.copy_(non_blocking=True)` so the copy engine (DMA) is used; passing the cuda
    view would launch an SM copy kernel instead, which is the host_direct path, not staging.
read_reduce(t, blocks, threads, bytes_per_block, shared) -> checksum
    Each block reads bytes_per_block with 16 B loads. shared=True: every block reads the SAME
    region (L2-caching-of-host-lines test); shared=False: block b reads region b.
pointer_chase(next, start, hops) -> last index. next is an int64 cyclic permutation.
pointer_attrs(ptr) -> dict from cudaPointerGetAttributes.
"""
from __future__ import annotations

import functools

import torch

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

static void check(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

std::vector<torch::Tensor> host_alloc(std::vector<int64_t> shape, torch::ScalarType dtype) {
  int64_t numel = 1; for (auto s : shape) numel *= s;
  int64_t bytes = numel * torch::elementSize(dtype);
  void* hptr = nullptr;
  check(cudaHostAlloc(&hptr, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
  void* dptr = nullptr;
  check(cudaHostGetDevicePointer(&dptr, hptr, 0), "cudaHostGetDevicePointer");
  auto cpu = torch::from_blob(hptr, shape, [hptr](void*) { cudaFreeHost(hptr); },
                              torch::TensorOptions().dtype(dtype).device(torch::kCPU));
  auto gpu = torch::from_blob(dptr, shape, [](void*) {},
                              torch::TensorOptions().dtype(dtype).device(torch::kCUDA, 0));
  return {cpu, gpu};
}

std::vector<int64_t> pointer_attrs(int64_t p) {
  cudaPointerAttributes a;
  cudaError_t e = cudaPointerGetAttributes(&a, (const void*)p);
  if (e != cudaSuccess) { cudaGetLastError(); return {-1, -1, 0}; }
  return {(int64_t)a.type, (int64_t)a.device, (int64_t)(a.hostPointer == a.devicePointer)};
}

__global__ void read_reduce_kernel(const int4* __restrict__ p, long long vec_per_block, int shared,
                                   unsigned long long* out) {
  long long base = shared ? 0 : (long long)blockIdx.x * vec_per_block;
  unsigned long long acc = 0;
  for (long long i = threadIdx.x; i < vec_per_block; i += blockDim.x) {
    int4 v = p[base + i];
    acc += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
  }
  __shared__ unsigned long long red[32];
  unsigned long long w = acc;
  for (int o = 16; o > 0; o >>= 1) w += __shfl_down_sync(0xffffffff, w, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = w;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long s = 0;
    for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) s += red[i];
    atomicAdd(out, s);
  }
}

torch::Tensor read_reduce(torch::Tensor t, int64_t blocks, int64_t threads, int64_t bytes_per_block, bool shared) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous());
  int64_t vec_per_block = bytes_per_block / 16;
  int64_t need = shared ? bytes_per_block : bytes_per_block * blocks;
  TORCH_CHECK(need <= t.numel() * t.element_size(), "buffer too small for blocks*bytes_per_block");
  auto out = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(t.device()));
  read_reduce_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int4*)t.data_ptr(), vec_per_block, shared ? 1 : 0, (unsigned long long*)out.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ void chase_kernel(const long long* __restrict__ next, long long start, long long hops, long long* out) {
  long long i = start;
  for (long long h = 0; h < hops; ++h) i = next[i];
  *out = i;
}

torch::Tensor pointer_chase(torch::Tensor next, int64_t start, int64_t hops) {
  TORCH_CHECK(next.is_cuda() && next.dtype() == torch::kInt64);
  auto out = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(next.device()));
  chase_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const long long*)next.data_ptr(), start, hops, (long long*)out.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> host_alloc(std::vector<int64_t> shape, torch::ScalarType dtype);
std::vector<int64_t> pointer_attrs(int64_t p);
torch::Tensor read_reduce(torch::Tensor t, int64_t blocks, int64_t threads, int64_t bytes_per_block, bool shared);
torch::Tensor pointer_chase(torch::Tensor next, int64_t start, int64_t hops);
"""


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name="kvdiag_hostmem", cpp_sources=_CPP, cuda_sources=_SRC,
        functions=["host_alloc", "pointer_attrs", "read_reduce", "pointer_chase"],
        extra_cuda_cflags=["-O3"], verbose=False,
    )


def host_tensor(shape, dtype=torch.bfloat16) -> torch.Tensor:
    cpu, gpu = _ext().host_alloc(list(shape), dtype)
    gpu.cpu_view = cpu  # cpu owns the allocation; keeps it alive while gpu is referenced
    return gpu


PTR_TYPE = {-1: "error", 0: "unregistered", 1: "host", 2: "device", 3: "managed"}


def pointer_attrs(ptr: int) -> dict:
    t, dev, same = _ext().pointer_attrs(int(ptr))
    return {"type": PTR_TYPE.get(int(t), str(t)), "device": int(dev), "host_ptr_eq_dev_ptr": bool(same)}


def read_reduce(t, blocks, threads, bytes_per_block, shared=False):
    return _ext().read_reduce(t, blocks, threads, bytes_per_block, shared)


def pointer_chase(next_idx, start, hops):
    return _ext().pointer_chase(next_idx, start, hops)


def selftest():
    """Host view is readable by a kernel and holds the right bytes; prints pointer attributes."""
    t = host_tensor((1 << 20,), torch.int32)
    t.cpu_view.copy_(torch.arange(1 << 20, dtype=torch.int32))
    torch.cuda.synchronize()
    assert int(t[12345].item()) == 12345, "kernel read of host view returned wrong data"
    s = int(t.to(torch.int64).sum().item())
    assert s == (1 << 20) * ((1 << 20) - 1) // 2
    print("host view OK:", pointer_attrs(t.data_ptr()))
    chk = read_reduce(t, blocks=4, threads=256, bytes_per_block=1 << 20, shared=False)
    torch.cuda.synchronize()
    print("read_reduce OK, checksum", int(chk.item()))
