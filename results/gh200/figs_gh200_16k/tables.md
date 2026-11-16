## Table 2 values (paste into the report)

| | PCIe (measured) | GH200 ratio (model) |
|---|---|---|
| Link bandwidth (GB/s) | 341.5 | 415.7 |
| KV per sequence (GB) | 2.40 | 2.40 |
| Fetch time per layer (ms) | 0.20 | 0.16 |
| Ordinary step T_N (ms) | 53.3 | same |
| Speculative step T_S (ms) | 95.5 | same |
| Tokens per step tau | 1.86 | same |
| Offload budget, ordinary | 0.92 | 0.99 |
| Offload budget, speculative | 0.99 | 0.99 |
| Speculative beats ordinary above | 0.00 | 0.00 |

(Budget = largest offloaded fraction staying within 5% of ordinary decoding with nothing offloaded, 53.3 ms/token; 'none' = already above that with nothing offloaded. Crossover 'never' = speculative stays slower.)

## Numbers for the text

- All on GPU: ordinary 53.3 ms/token, speculative 53.1 ms/token -> speculative is 1.00x the speed of ordinary
- Link-to-memory ratio: 341.5/4000 = 1/12 (GH200: 1/10)
- Per-layer fetch 0.20 ms on PCIe vs per-layer compute 1.48 ms (ordinary) / 2.65 ms (speculative); at GH200 ratio the fetch is 0.16 ms
- Model validation (no fitted parameters): mean |error| 43.8%, max 74.7% over 12 measured configurations
- Budgets on PCIe: measured ordinary 0.92, speculative 0.99; model ordinary 0.99, speculative 0.99
- Budgets at GH200 ratio (model): ordinary 0.99, speculative 0.99
- Speculative beats ordinary at the same offloaded fraction above: measured 0.00, model PCIe 0.00, model GH200 ratio 0.00
- Offloaded runs matching all-GPU output token for token: 49/72
- Prompts: 6, mean context while decoding 16268 tokens

## Mean over prompts per configuration (R = layers kept on the GPU)

|                             |   offload_fraction |   tpot_ms |   step_ms |   tau |   h2d_mb_per_step |
|:----------------------------|-------------------:|----------:|----------:|------:|------------------:|
| ('ordinary', 'hbm', 36)     |               0    |     53.31 |     53.31 |  1    |              0    |
| ('ordinary', 'ours', 36)    |               0    |     39.73 |     39.73 |  1    |              0    |
| ('ordinary', 'ours', 32)    |               0.11 |     39.71 |     39.71 |  1    |            265.47 |
| ('ordinary', 'ours', 27)    |               0.25 |     41.34 |     41.34 |  1    |            596.66 |
| ('ordinary', 'ours', 18)    |               0.5  |     42.89 |     42.89 |  1    |           1192.8  |
| ('ordinary', 'ours', 9)     |               0.74 |     44.53 |     44.53 |  1    |           1788.94 |
| ('ordinary', 'ours', 0)     |               0.99 |     60.58 |     60.58 |  1    |           2385.08 |
| ('speculative', 'hbm', 36)  |               0    |     53.15 |     95.5  |  1.86 |              0    |
| ('speculative', 'ours', 36) |               0    |     30.37 |     55.3  |  1.86 |              0    |
| ('speculative', 'ours', 32) |               0.11 |     30.41 |     55.39 |  1.86 |            265.9  |
| ('speculative', 'ours', 27) |               0.25 |     30.43 |     55.42 |  1.86 |            597.07 |
| ('speculative', 'ours', 18) |               0.5  |     30.66 |     55.83 |  1.86 |           1193.17 |
| ('speculative', 'ours', 9)  |               0.74 |     30.86 |     56.19 |  1.86 |           1789.27 |
| ('speculative', 'ours', 0)  |               0.99 |     31.35 |     57.11 |  1.86 |           2385.37 |
