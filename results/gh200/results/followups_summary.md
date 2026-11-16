# GH200 follow-up runs

## Zero-copy vs copy pipeline vs HBM

| file         |   block_size | arm      |   R | mode   |   tail |   n |   tpot_ms |   step_ms |   tau |   tok_s |   link_mb_step |   peak_gib |   hbm_kv_mb | match   |
|:-------------|-------------:|:---------|----:|:-------|-------:|----:|----------:|----------:|------:|--------:|---------------:|-----------:|------------:|:--------|
| zc_16k.jsonl |            1 | hbm      |  36 | -      |      0 |   4 |     59.57 |     59.57 |  1    |   17.55 |           0    |      21.06 |        0    | 4/4     |
| zc_16k.jsonl |            1 | ours     |   0 | first  |     64 |   4 |     68.12 |     68.12 |  1    |   15.21 |        2379.89 |      21.21 |      155.45 | 2/4     |
| zc_16k.jsonl |            1 | ours     |  18 | first  |     64 |   4 |     47.12 |     47.12 |  1    |   21.22 |        1190.21 |      21.86 |     1352.24 | 2/4     |
| zc_16k.jsonl |            1 | zerocopy |   0 | first  |     64 |   4 |     44.96 |     44.96 |  1    |   22.24 |        2402.89 |      22.36 |       18.73 | 2/4     |
| zc_16k.jsonl |            1 | zerocopy |  18 | first  |     64 |   4 |     44.66 |     44.66 |  1    |   22.39 |        1201.45 |      22.97 |     1215.52 | 2/4     |
| zc_16k.jsonl |           16 | hbm      |  36 | -      |      0 |   4 |     56.1  |    102.84 |  1.92 |   19.53 |           0    |      25.24 |        0    | 4/4     |
| zc_16k.jsonl |           16 | ours     |   0 | first  |     64 |   4 |     33.45 |     62.5  |  1.92 |   30.65 |        2379.8  |      25.43 |      155.12 | 4/4     |
| zc_16k.jsonl |           16 | ours     |  18 | first  |     64 |   4 |     31.69 |     59.22 |  1.92 |   32.35 |        1190.4  |      27.65 |     1352.08 | 4/4     |
| zc_16k.jsonl |           16 | zerocopy |   0 | first  |     64 |   4 |     36.29 |     67.77 |  1.92 |   28.29 |        2412.82 |      26.57 |       18.4  | 4/4     |
| zc_16k.jsonl |           16 | zerocopy |  18 | first  |     64 |   4 |     33.79 |     63.09 |  1.92 |   30.39 |        1206.41 |      29.14 |     1215.35 | 4/4     |
| zc_32k.jsonl |            1 | hbm      |  36 | -      |      0 |   3 |     63.16 |     63.16 |  1    |   15.84 |           0    |      24.93 |        0    | 3/3     |
| zc_32k.jsonl |            1 | ours     |   0 | first  |     64 |   3 |     70.64 |     70.64 |  1    |   14.16 |        4787.71 |      25.2  |      289.19 | 2/3     |
| zc_32k.jsonl |            1 | zerocopy |   0 | first  |     64 |   3 |     39.01 |     39.01 |  1    |   25.63 |        4819.63 |      24.96 |       18.73 | 2/3     |
| zc_32k.jsonl |           16 | hbm      |  36 | -      |      0 |   3 |     83.3  |    143.86 |  1.73 |   12.05 |           0    |      33.35 |        0    | 3/3     |
| zc_32k.jsonl |           16 | ours     |   0 | first  |     64 |   3 |     45.91 |     79.3  |  1.73 |   21.85 |        4787.53 |      33.65 |      288.7  | 3/3     |
| zc_32k.jsonl |           16 | zerocopy |   0 | first  |     64 |   3 |     64.93 |    112.15 |  1.73 |   15.45 |        4834.44 |      33.41 |       18.24 | 3/3     |
| zc_8k.jsonl  |            1 | hbm      |  36 | -      |      0 |   4 |     62.09 |     62.09 |  1    |   16.47 |           0    |      19.1  |        0    | 4/4     |
| zc_8k.jsonl  |            1 | ours     |   0 | first  |     64 |   4 |     70.28 |     70.28 |  1    |   14.49 |        1153.51 |      19.18 |       86.95 | 1/4     |
| zc_8k.jsonl  |            1 | ours     |  18 | first  |     64 |   4 |     45.68 |     45.68 |  1    |   21.89 |         576.92 |      19.52 |      669.51 | 1/4     |
| zc_8k.jsonl  |            1 | zerocopy |   0 | first  |     64 |   4 |     43.76 |     43.76 |  1    |   22.85 |        1173.16 |      19.77 |       18.4  | 1/4     |
| zc_8k.jsonl  |            1 | zerocopy |  18 | first  |     64 |   4 |     42.96 |     42.96 |  1    |   23.28 |         586.58 |      20.11 |      600.96 | 2/4     |
| zc_8k.jsonl  |           16 | hbm      |  36 | -      |      0 |   4 |     55.74 |    107.15 |  1.97 |   18.37 |           0    |      21.12 |        0    | 4/4     |
| zc_8k.jsonl  |           16 | ours     |   0 | first  |     64 |   4 |     31.48 |     60.28 |  1.97 |   32.69 |        1153.92 |      21.25 |       86.61 | 4/4     |
| zc_8k.jsonl  |           16 | ours     |  18 | first  |     64 |   4 |     29.84 |     57.15 |  1.97 |   34.47 |         577.29 |      22.32 |      669.34 | 4/4     |
| zc_8k.jsonl  |           16 | zerocopy |   0 | first  |     64 |   4 |     28.8  |     55.16 |  1.97 |   35.73 |        1181.19 |      21.84 |       18.06 | 4/4     |
| zc_8k.jsonl  |           16 | zerocopy |  18 | first  |     64 |   4 |     28.45 |     54.5  |  1.97 |   36.16 |         590.6  |      23.12 |      600.79 | 4/4     |

## Eviction policy v1 (resident set, tail window)

| file            |   block_size | arm      |   R | mode   |   tail |   n |   tpot_ms |   step_ms |   tau |   tok_s |   link_mb_step |   peak_gib |   hbm_kv_mb | match   |
|:----------------|-------------:|:---------|----:|:-------|-------:|----:|----------:|----------:|------:|--------:|---------------:|-----------:|------------:|:--------|
| evict_16k.jsonl |            1 | hbm      |  36 | -      |      0 |  12 |     55.33 |     55.33 |  1    |   19.02 |           0    |      21.06 |        0    | 12/12   |
| evict_16k.jsonl |            1 | ours     |   0 | first  |   2048 |   4 |     62.61 |     62.61 |  1    |   16.67 |        2087.27 |      21.37 |      448    | 3/4     |
| evict_16k.jsonl |            1 | ours     |   0 | first  |   8192 |   4 |     62.03 |     62.03 |  1    |   16.77 |        1181.11 |      22.03 |     1353.97 | 2/4     |
| evict_16k.jsonl |            1 | ours     |  18 | last   |     64 |   4 |     61.61 |     61.61 |  1    |   16.87 |        1190.21 |      22.12 |     1352.24 | 2/4     |
| evict_16k.jsonl |            1 | zerocopy |   0 | first  |   2048 |   4 |     40.24 |     40.24 |  1    |   24.85 |        2402.89 |      21.39 |      311.28 | 2/4     |
| evict_16k.jsonl |            1 | zerocopy |   0 | first  |   8192 |   4 |     40.44 |     40.44 |  1    |   24.73 |        2402.89 |      22.24 |     1217.25 | 2/4     |
| evict_16k.jsonl |            1 | zerocopy |  18 | last   |     64 |   4 |     42.17 |     42.17 |  1    |   23.71 |        1201.45 |      22.66 |     1215.52 | 4/4     |
| evict_16k.jsonl |           16 | hbm      |  36 | -      |      0 |  12 |     53.92 |     98.83 |  1.92 |   20.33 |           0    |      25.24 |        0    | 12/12   |
| evict_16k.jsonl |           16 | ours     |   0 | first  |   2048 |   4 |     30.02 |     56.07 |  1.92 |   34.19 |        2087.13 |      26.22 |      447.67 | 4/4     |
| evict_16k.jsonl |           16 | ours     |   0 | first  |   8192 |   4 |     30.99 |     57.86 |  1.92 |   33.14 |        1180.78 |      28.54 |     1353.64 | 4/4     |
| evict_16k.jsonl |           16 | ours     |  18 | last   |     64 |   4 |     30.92 |     57.76 |  1.92 |   33.18 |        1190.4  |      27.1  |     1352.08 | 4/4     |
| evict_16k.jsonl |           16 | zerocopy |   0 | first  |   2048 |   4 |     36.69 |     68.5  |  1.92 |   27.99 |        2412.82 |      26.03 |      310.95 | 4/4     |
| evict_16k.jsonl |           16 | zerocopy |   0 | first  |   8192 |   4 |     39.58 |     73.9  |  1.92 |   25.94 |        2412.82 |      28.44 |     1216.92 | 4/4     |
| evict_16k.jsonl |           16 | zerocopy |  18 | last   |     64 |   4 |     32.69 |     61.04 |  1.92 |   31.41 |        1206.41 |      28.48 |     1215.35 | 4/4     |

## Step profile (torch.profiler, decode steps only)

| dataset              | arm      |   resident_layers |   block_size |   tpot_ms_clean |   step_ms_profiled |   gpu_ms_per_step |   memcpy_ms_per_step |   cmdbuf_full_ms_per_step |   cpu_aten_ms_per_step |   gpu_busy_frac |   launches_per_step |   cudaMalloc_per_step |   cudaFree_per_step |   cudaStreamSynchronize_per_step |
|:---------------------|:---------|------------------:|-------------:|----------------:|-------------------:|------------------:|---------------------:|--------------------------:|-----------------------:|----------------:|--------------------:|----------------------:|--------------------:|---------------------------------:|
| speed:throughput_16k | hbm      |                 0 |            1 |          61.711 |             75.071 |            10.102 |                0.007 |                     1.738 |                 36.701 |           0.135 |             2254.56 |                     0 |                   0 |                                1 |
| speed:throughput_16k | hbm      |                 0 |           16 |          88.86  |             96.501 |            44.705 |                0.041 |                     1.184 |                 47.939 |           0.463 |             2777.79 |                     0 |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |            1 |          47.967 |             75.949 |            10.207 |                0.006 |                     2.642 |                 36.599 |           0.134 |             2254.44 |                     0 |                   0 |                                1 |
| speed:throughput_16k | ours     |                36 |           16 |          47.3   |             97.033 |            45.036 |                0.04  |                     1.572 |                 48.008 |           0.464 |             2777.79 |                     0 |                   0 |                                1 |
| speed:throughput_16k | ours     |                 0 |            1 |          80.903 |             86.998 |             9.096 |                6.537 |                     1.483 |                 39.436 |           0.105 |             2326.44 |                     0 |                   0 |                                1 |
| speed:throughput_16k | ours     |                 0 |           16 |          52.863 |            108.747 |            39.402 |                6.572 |                     0.645 |                 51.133 |           0.362 |             2849.79 |                     0 |                   0 |                                1 |
| speed:throughput_16k | zerocopy |                 0 |            1 |          49.62  |             80.086 |            14.754 |                0.006 |                    -0.022 |                 38.182 |           0.184 |             2326.44 |                     0 |                   0 |                                1 |
| speed:throughput_16k | zerocopy |                 0 |           16 |          49.42  |            101.143 |            56.436 |                0.04  |                     0.776 |                 49.451 |           0.558 |             2849.79 |                     0 |                   0 |                                1 |

<details><summary>top kernels: hbm R=0 bs=1</summary>

- 2.235 ms/step x72.0 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`
- 2.002 ms/step x108.0 `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT`
- 1.305 ms/step x72.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonym`
- 0.870 ms/step x36.0 `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_35_16x128x128_1x4x1_cga1x1`
- 0.442 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.396 ms/step x72.0 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT`
- 0.358 ms/step x146.0 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`
- 0.336 ms/step x1.0 `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT`

</details>

<details><summary>top kernels: hbm R=0 bs=16</summary>

- 15.026 ms/step x41.0 `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_cga1x1`
- 10.262 ms/step x72.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 8.595 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.283 ms/step x41.0 `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT`
- 1.084 ms/step x35.1 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`
- 0.993 ms/step x53.6 `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT`
- 0.866 ms/step x330.2 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.723 ms/step x23.4 `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT`

</details>

<details><summary>top kernels: ours R=36 bs=1</summary>

- 2.226 ms/step x72.0 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`
- 1.996 ms/step x108.0 `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT`
- 1.306 ms/step x72.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonym`
- 0.872 ms/step x36.0 `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_35_16x128x128_1x4x1_cga1x1`
- 0.439 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.395 ms/step x72.0 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT`
- 0.356 ms/step x146.0 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`
- 0.336 ms/step x1.0 `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT`

</details>

<details><summary>top kernels: ours R=36 bs=16</summary>

- 15.127 ms/step x41.0 `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_cga1x1`
- 10.276 ms/step x72.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 8.623 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.286 ms/step x41.0 `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT`
- 1.084 ms/step x35.1 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`
- 0.993 ms/step x53.6 `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT`
- 0.884 ms/step x330.2 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.725 ms/step x23.4 `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT`

</details>

<details><summary>top kernels: ours R=0 bs=1</summary>

- 6.529 ms/step x72.0 `Memcpy HtoD (Pinned -> Device)`
- 2.206 ms/step x72.0 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`
- 1.929 ms/step x108.0 `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT`
- 0.826 ms/step x36.0 `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_35_16x128x128_1x4x1_cga1x1`
- 0.436 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.394 ms/step x72.0 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT`
- 0.355 ms/step x146.0 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`
- 0.335 ms/step x1.0 `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT`

</details>

<details><summary>top kernels: ours R=0 bs=16</summary>

- 14.989 ms/step x41.0 `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_cga1x1`
- 12.144 ms/step x144.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 6.530 ms/step x72.0 `Memcpy HtoD (Pinned -> Device)`
- 1.442 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.290 ms/step x41.0 `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT`
- 1.084 ms/step x35.1 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`
- 0.994 ms/step x53.6 `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT`
- 0.867 ms/step x330.2 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`

</details>

<details><summary>top kernels: zerocopy R=0 bs=1</summary>

- 6.656 ms/step x36.0 `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_35_16x128x128_1x4x1_cga1x1`
- 2.225 ms/step x72.0 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`
- 1.932 ms/step x108.0 `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT`
- 0.436 ms/step x216.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.391 ms/step x72.0 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT`
- 0.345 ms/step x146.0 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`
- 0.335 ms/step x1.0 `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT`
- 0.311 ms/step x145.0 `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<flo`

</details>

<details><summary>top kernels: zerocopy R=0 bs=16</summary>

- 29.265 ms/step x144.0 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 14.923 ms/step x41.0 `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_cga1x1`
- 1.426 ms/step x164.0 `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespa`
- 1.291 ms/step x41.0 `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT`
- 1.084 ms/step x35.1 `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`
- 0.997 ms/step x53.6 `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT`
- 0.861 ms/step x330.2 `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`
- 0.726 ms/step x23.4 `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT`

</details>

## Per-layer GPU time (measure_layers.py, mean over layers and prompts)

| dataset              |   block_size |   c_attn_ms |   c_other_ms |   step_ms |   sum_layers_ms |   tau |   attn_share_of_layer |
|:---------------------|-------------:|------------:|-------------:|----------:|----------------:|------:|----------------------:|
| speed:throughput_16k |            1 |       1.123 |        0.466 |    59.395 |          57.201 | 1     |                 0.707 |
| speed:throughput_16k |           16 |       1.816 |        0.179 |   101.265 |          71.821 | 1.917 |                 0.91  |
| speed:throughput_8k  |            1 |       1.21  |        0.454 |    62.027 |          59.914 | 1     |                 0.727 |
| speed:throughput_8k  |           16 |       1.76  |        0.242 |   105.753 |          72.104 | 1.97  |                 0.879 |

## sm_read_NVIDIA_GH200_480GB.csv

|   blocks |      bytes |    gbps |     ms | place   | test              |
|---------:|-----------:|--------:|-------:|:--------|:------------------|
|        0 | 1073741824 |  350.07 | nan    | nan     | dma_h2d           |
|       16 | 1073741824 |   77.96 | nan    | nan     | sm_read           |
|       64 | 1073741824 |  204.69 | nan    | nan     | sm_read           |
|      132 | 1073740800 |  281.26 | nan    | nan     | sm_read           |
|      264 | 1073740800 |  332.23 | nan    | nan     | sm_read           |
|      528 | 1073740800 |  322.35 | nan    | nan     | sm_read           |
|     1056 | 1073740800 |  315.85 | nan    | nan     | sm_read           |
|      132 | 1073740800 | 1214.05 | nan    | nan     | sm_read_hbm       |
|     1056 | 1073740800 | 3570    | nan    | nan     | sm_read_hbm       |
|        0 | 1073741824 |  297.83 | nan    | nan     | sm_write          |
|     8192 |   33554432 | 1082.35 |   0.03 | hbm     | sdpa_q1           |
|     8192 |   33554432 |   91.29 |   0.37 | hbm     | sdpa_q16_repeatkv |
|     8192 |   33554432 |  321.64 |   0.1  | hbm     | sdpa_q16_gqa      |
|     8192 |   33554432 |  315.69 |   0.11 | host    | sdpa_q1           |
|     8192 |   33554432 |   50.9  |   0.66 | host    | sdpa_q16_repeatkv |
|     8192 |   33554432 |  101.94 |   0.33 | host    | sdpa_q16_gqa      |
|    16384 |   67108864 | 1999.57 |   0.03 | hbm     | sdpa_q1           |
|    16384 |   67108864 |   94.47 |   0.71 | hbm     | sdpa_q16_repeatkv |
|    16384 |   67108864 |  337.82 |   0.2  | hbm     | sdpa_q16_gqa      |
|    16384 |   67108864 |  331.08 |   0.2  | host    | sdpa_q1           |
|    16384 |   67108864 |   49.87 |   1.35 | host    | sdpa_q16_repeatkv |
|    16384 |   67108864 |  103.62 |   0.65 | host    | sdpa_q16_gqa      |
|    32768 |  134217728 | 2846.88 |   0.05 | hbm     | sdpa_q1           |
|    32768 |  134217728 |   96.11 |   1.4  | hbm     | sdpa_q16_repeatkv |
|    32768 |  134217728 |  348.09 |   0.39 | hbm     | sdpa_q16_gqa      |
|    32768 |  134217728 |  333.7  |   0.4  | host    | sdpa_q1           |
|    32768 |  134217728 |   49.16 |   2.73 | host    | sdpa_q16_repeatkv |
|    32768 |  134217728 |  104.42 |   1.29 | host    | sdpa_q16_gqa      |
