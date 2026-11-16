# GH200 follow-up runs

## Zero-copy vs copy pipeline vs HBM

| file              |   block_size | arm      |   R | mode   |   tail |   split |   n |   tpot_ms |   step_ms |   tau |   tok_s |   link_mb_step |   peak_gib |   hbm_kv_mb | match   |
|:------------------|-------------:|:---------|----:|:-------|-------:|--------:|----:|----------:|----------:|------:|--------:|---------------:|-----------:|------------:|:--------|
| zc_check_8k.jsonl |            1 | hbm      |  36 | -      |      0 |       0 |   2 |     42.32 |     42.32 |  1    |   23.63 |           0    |      19.1  |        0    | 2/2     |
| zc_check_8k.jsonl |            1 | ours     |   0 | first  |     64 |       0 |   1 |     61.26 |     61.26 |  1    |   16.32 |        1162.69 |      19.18 |       83.69 | 1/1     |
| zc_check_8k.jsonl |            1 | zerocopy |   0 | first  |     64 |       0 |   1 |    162.85 |    162.85 |  1    |    6.14 |        1176.04 |      19.11 |       17.4  | 1/1     |
| zc_check_8k.jsonl |            1 | zerocopy |   0 | first  |     64 |       1 |   1 |    245.73 |    245.73 |  1    |    4.07 |        1162.51 |      19.16 |       18.73 | 0/1     |
| zc_check_8k.jsonl |            1 | zerocopy |   0 | first  |   2048 |       1 |   1 |    193.74 |    193.74 |  1    |    5.16 |         869.95 |      19.14 |      311.28 | 0/1     |
| zc_check_8k.jsonl |           16 | hbm      |  36 | -      |      0 |       0 |   2 |     36.39 |     79.52 |  2.19 |   27.48 |           0    |      21.16 |        0    | 2/2     |
| zc_check_8k.jsonl |           16 | ours     |   0 | first  |     64 |       0 |   1 |     35.43 |     77.42 |  2.19 |   28.23 |        1162.87 |      21.25 |       83.69 | 1/1     |
| zc_check_8k.jsonl |           16 | zerocopy |   0 | first  |     64 |       0 |   1 |    238.38 |    520.9  |  2.19 |    4.19 |        1177.39 |      21.19 |       17.4  | 1/1     |
| zc_check_8k.jsonl |           16 | zerocopy |   0 | first  |     64 |       1 |   1 |    199.22 |    477.37 |  2.4  |    5.02 |        1162.45 |      21.19 |       18.73 | 0/1     |
| zc_check_8k.jsonl |           16 | zerocopy |   0 | first  |   2048 |       1 |   1 |    176.17 |    377.96 |  2.15 |    5.68 |         869.22 |      21.49 |      309.95 | 1/1     |

## Step profile (torch.profiler, decode steps only)

| dataset              | arm      |   resident_layers |   block_size |   tpot_ms_clean |   step_ms_profiled |   gpu_ms_per_step |   memcpy_ms_per_step |   cmdbuf_full_ms_per_step |   cpu_aten_ms_per_step |   gpu_busy_frac |   launches_per_step |   cudaMalloc_per_step |   cudaFree_per_step |   cudaStreamSynchronize_per_step |
|:---------------------|:---------|------------------:|-------------:|----------------:|-------------------:|------------------:|---------------------:|--------------------------:|-----------------------:|----------------:|--------------------:|----------------------:|--------------------:|---------------------------------:|
| speed:throughput_16k | hbm      |                 0 |           16 |          89.401 |            131.919 |           123.946 |                0.153 |                     0.566 |                 47.753 |           0.94  |             2588.21 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |           16 |          88.804 |            128.95  |           124.788 |                0.156 |                    -0.708 |                 35.439 |           0.968 |             2588.14 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                 0 |           16 |          83.833 |            122.182 |           115.989 |               98.538 |                     0.226 |                 39.021 |           0.949 |             2659.86 |                 0     |                   0 |                                1 |
| speed:throughput_16k | zerocopy |                 0 |           16 |         682.873 |            973.386 |           966.824 |                0.152 |                   102.079 |                 46.982 |           0.993 |             2660.14 |                 0     |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |           16 |         153.387 |            238.629 |           232.559 |                0.155 |                   -16.855 |                 56.82  |           0.975 |             3244.69 |                 0.077 |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |           16 |         152.607 |            237.644 |           232.83  |                0.153 |                    16.036 |                 48.166 |           0.98  |             3244.61 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                 0 |           16 |         148.983 |            231.927 |           226.68  |              119.677 |                    22.509 |                 51.469 |           0.977 |             3316.61 |                 0     |                   0 |                                1 |
| speed:throughput_16k | zerocopy |                 0 |           16 |         218.067 |            340.192 |           335.299 |                0.152 |                    42.649 |                 50.634 |           0.986 |             3316.61 |                 0     |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |           16 |          67.403 |            102.148 |            94.05  |                0.157 |                    -6.984 |                 48.746 |           0.921 |             2796.93 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |           16 |          67.164 |             98.2   |            94.453 |                0.156 |                     3.93  |                 37.622 |           0.962 |             2797.14 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                 0 |           16 |          78.995 |            113.019 |            85.056 |               96.321 |                     2.231 |                 38.808 |           0.753 |             2869.14 |                 0     |                   0 |                                1 |
| speed:throughput_16k | zerocopy |                 0 |           16 |         112.859 |            163.092 |           159.389 |                0.185 |                    12.569 |                 38.672 |           0.977 |             2869.14 |                 0     |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |            1 |          35.741 |             79.913 |            38.16  |                0.051 |                     0.712 |                 38.314 |           0.478 |             2074.33 |                 0     |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |           16 |          82.031 |            117.48  |           114.204 |                0.156 |                   -11.955 |                 35.774 |           0.972 |             2588.14 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |            1 |          36.599 |             55.128 |            38.282 |                0.051 |                    -6.209 |                 26.427 |           0.694 |             2074.44 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |           16 |          80.594 |            120.231 |           114.207 |                0.16  |                     9.262 |                 44.522 |           0.95  |             2587.86 |                 0     |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |            1 |          37.782 |             81.583 |            37.746 |                0.051 |                   -31.41  |                 41.842 |           0.463 |             2074.33 |                 0     |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |           16 |         145.313 |            226.403 |           222.3   |                0.153 |                    11.016 |                 46.129 |           0.982 |             3244.61 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |            1 |          52.069 |             65.917 |            38.946 |                0.05  |                     1.992 |                 32.32  |           0.591 |             2074.44 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |           16 |         145.364 |            226.058 |           221.931 |                0.153 |                    -9.431 |                 46.283 |           0.982 |             3244.61 |                 0     |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |            1 |          35.694 |             65.674 |            37.668 |                0.051 |                   -33.549 |                 32.9   |           0.574 |             2074.33 |                 0     |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |           16 |          59.197 |             98.989 |            83.475 |                0.153 |                    -3.729 |                 47.165 |           0.843 |             2797.14 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |            1 |          37.167 |             56.905 |            38.39  |                0.051 |                    16.195 |                 26.788 |           0.675 |             2074.44 |                 0     |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |           16 |          59.475 |             86.359 |            83.069 |                0.158 |                    -6.05  |                 36.175 |           0.962 |             2796.86 |                 0     |                   0 |                                1 |

<details><summary>top kernels: hbm R=0 bs=16</summary>

- 34.032 ms/step x82.3 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 31.819 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.919 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.351 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 13.169 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.307 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 1.849 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`
- 1.481 ms/step x0.1 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8`

</details>

<details><summary>top kernels: ours R=36 bs=16</summary>

- 34.033 ms/step x82.3 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 31.813 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.921 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.351 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 13.171 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.306 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 1.849 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`
- 1.505 ms/step x0.1 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8`

</details>

<details><summary>top kernels: ours R=0 bs=16</summary>

- 98.213 ms/step x72.0 `Memcpy HtoD (Pinned -> Device)`
- 35.056 ms/step x154.3 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 31.895 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 14.538 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.557 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.627 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 2.270 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.866 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`

</details>

<details><summary>top kernels: zerocopy R=0 bs=16</summary>

- 887.212 ms/step x154.3 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 31.704 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.924 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.161 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.324 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 2.266 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.870 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`
- 1.487 ms/step x0.1 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8`

</details>

<details><summary>top kernels: hbm R=0 bs=16</summary>

- 67.046 ms/step x72.0 `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`
- 33.974 ms/step x76.0 `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, `
- 17.760 ms/step x16.6 `void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false, cublasGemvTensorStridedBatched<fl`
- 13.944 ms/step x123.2 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.367 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 13.296 ms/step x277.0 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`
- 12.620 ms/step x46.8 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 9.759 ms/step x19.4 `void gemmSN_NN_kernel<float, 256, 4, 2, 8, 4, 4, false, cublasGemvTensorStridedBatched<flo`

</details>

<details><summary>top kernels: ours R=36 bs=16</summary>

- 67.042 ms/step x72.0 `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`
- 34.069 ms/step x76.0 `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, `
- 17.772 ms/step x16.6 `void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false, cublasGemvTensorStridedBatched<fl`
- 13.941 ms/step x123.2 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.385 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 13.296 ms/step x277.0 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`
- 12.624 ms/step x46.8 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 9.750 ms/step x19.4 `void gemmSN_NN_kernel<float, 256, 4, 2, 8, 4, 4, false, cublasGemvTensorStridedBatched<flo`

</details>

<details><summary>top kernels: ours R=0 bs=16</summary>

- 119.477 ms/step x72.0 `Memcpy HtoD (Pinned -> Device)`
- 69.563 ms/step x72.0 `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`
- 35.133 ms/step x76.0 `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, `
- 17.812 ms/step x16.6 `void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false, cublasGemvTensorStridedBatched<fl`
- 13.983 ms/step x123.2 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.712 ms/step x277.0 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`
- 12.655 ms/step x46.8 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 9.766 ms/step x19.4 `void gemmSN_NN_kernel<float, 256, 4, 2, 8, 4, 4, false, cublasGemvTensorStridedBatched<flo`

</details>

<details><summary>top kernels: zerocopy R=0 bs=16</summary>

- 125.514 ms/step x277.0 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`
- 67.294 ms/step x72.0 `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`
- 33.938 ms/step x76.0 `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, `
- 17.714 ms/step x16.6 `void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false, cublasGemvTensorStridedBatched<fl`
- 13.949 ms/step x123.2 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 12.617 ms/step x46.8 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 9.761 ms/step x19.4 `void gemmSN_NN_kernel<float, 256, 4, 2, 8, 4, 4, false, cublasGemvTensorStridedBatched<flo`
- 6.810 ms/step x37.8 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`

</details>

<details><summary>top kernels: hbm R=0 bs=16</summary>

- 34.067 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.903 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.381 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 13.172 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.306 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 1.851 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`
- 1.505 ms/step x0.7 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 1.496 ms/step x0.1 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8`

</details>

<details><summary>top kernels: ours R=36 bs=16</summary>

- 34.063 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.903 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.389 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 13.172 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.308 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 1.850 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`
- 1.840 ms/step x0.7 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 1.540 ms/step x0.1 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8`

</details>

<details><summary>top kernels: ours R=0 bs=16</summary>

- 95.958 ms/step x72.0 `Memcpy HtoD (Pinned -> Device)`
- 34.091 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 14.560 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.559 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.619 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 2.284 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.978 ms/step x0.7 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 1.907 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`

</details>

<details><summary>top kernels: zerocopy R=0 bs=16</summary>

- 109.092 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 14.056 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.161 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.324 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 2.264 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.946 ms/step x0.7 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 1.872 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`
- 1.495 ms/step x0.1 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8`

</details>

<details><summary>top kernels: hbm R=0 bs=1</summary>

- 13.115 ms/step x72.0 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 11.894 ms/step x108.0 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`
- 4.084 ms/step x36.0 `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, fals`
- 2.214 ms/step x1.0 `void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, `
- 1.290 ms/step x72.0 `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv`
- 1.167 ms/step x0.0 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 0.550 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.496 ms/step x145.0 `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<flo`

</details>

<details><summary>top kernels: hbm R=0 bs=16</summary>

- 34.052 ms/step x154.3 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 31.811 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.920 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.171 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.308 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 2.001 ms/step x92.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.964 ms/step x0.7 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 1.850 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`

</details>

<details><summary>top kernels: ours R=36 bs=1</summary>

- 13.128 ms/step x72.0 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 11.899 ms/step x108.0 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`
- 4.089 ms/step x36.0 `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, fals`
- 2.213 ms/step x1.0 `void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, `
- 1.292 ms/step x72.0 `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv`
- 1.103 ms/step x0.0 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 0.557 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.498 ms/step x145.0 `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<flo`

</details>

<details><summary>top kernels: ours R=36 bs=16</summary>

- 34.053 ms/step x154.3 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 31.836 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.921 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.172 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.308 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 1.990 ms/step x92.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.902 ms/step x0.7 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 1.850 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`

</details>

<details><summary>top kernels: hbm R=0 bs=1</summary>

- 13.116 ms/step x72.0 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 11.893 ms/step x108.0 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`
- 4.084 ms/step x36.0 `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, fals`
- 2.213 ms/step x1.0 `void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, `
- 1.291 ms/step x72.0 `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv`
- 0.825 ms/step x0.0 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 0.549 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.496 ms/step x145.0 `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<flo`

</details>

<details><summary>top kernels: hbm R=0 bs=16</summary>

- 67.067 ms/step x72.0 `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`
- 33.986 ms/step x76.0 `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, `
- 17.853 ms/step x16.6 `void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false, cublasGemvTensorStridedBatched<fl`
- 13.938 ms/step x123.2 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 12.674 ms/step x72.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::direct`
- 12.622 ms/step x46.8 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 9.758 ms/step x19.4 `void gemmSN_NN_kernel<float, 256, 4, 2, 8, 4, 4, false, cublasGemvTensorStridedBatched<flo`
- 6.792 ms/step x37.8 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`

</details>

<details><summary>top kernels: ours R=36 bs=1</summary>

- 13.125 ms/step x72.0 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 11.896 ms/step x108.0 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`
- 4.089 ms/step x36.0 `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, fals`
- 2.213 ms/step x1.0 `void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, `
- 1.356 ms/step x0.0 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 1.291 ms/step x72.0 `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv`
- 0.696 ms/step x0.0 `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false,`
- 0.560 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`

</details>

<details><summary>top kernels: ours R=36 bs=16</summary>

- 67.055 ms/step x72.0 `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`
- 33.867 ms/step x76.0 `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, `
- 17.869 ms/step x16.6 `void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false, cublasGemvTensorStridedBatched<fl`
- 13.941 ms/step x123.2 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 12.672 ms/step x72.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::direct`
- 12.624 ms/step x46.8 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 9.748 ms/step x19.4 `void gemmSN_NN_kernel<float, 256, 4, 2, 8, 4, 4, false, cublasGemvTensorStridedBatched<flo`
- 6.793 ms/step x37.8 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`

</details>

<details><summary>top kernels: hbm R=0 bs=1</summary>

- 13.120 ms/step x72.0 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 11.894 ms/step x108.0 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`
- 4.086 ms/step x36.0 `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, fals`
- 2.213 ms/step x1.0 `void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, `
- 1.292 ms/step x72.0 `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv`
- 0.771 ms/step x0.0 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 0.550 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.498 ms/step x145.0 `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<flo`

</details>

<details><summary>top kernels: hbm R=0 bs=16</summary>

- 34.046 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.898 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.172 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.308 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 2.093 ms/step x0.7 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 2.006 ms/step x92.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.850 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`
- 1.533 ms/step x0.1 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8`

</details>

<details><summary>top kernels: ours R=36 bs=1</summary>

- 13.127 ms/step x72.0 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 11.898 ms/step x108.0 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`
- 4.089 ms/step x36.0 `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, fals`
- 2.213 ms/step x1.0 `void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, `
- 1.292 ms/step x72.0 `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv`
- 1.100 ms/step x0.0 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`
- 0.562 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.499 ms/step x145.0 `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<flo`

</details>

<details><summary>top kernels: ours R=36 bs=16</summary>

- 34.039 ms/step x36.0 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:`
- 13.900 ms/step x123.1 `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`
- 13.172 ms/step x49.6 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- 6.308 ms/step x35.1 `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 2.004 ms/step x92.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.851 ms/step x83.4 `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn`
- 1.546 ms/step x0.1 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8`
- 1.535 ms/step x0.7 `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8`

</details>
