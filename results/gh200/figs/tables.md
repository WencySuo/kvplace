## Table 2 values (paste into the report)

| | PCIe (measured) | GH200 ratio (model) |
|---|---|---|
| Link bandwidth (GB/s) | 176.7 | 89.8 |
| KV per sequence (GB) | 1.18 | 1.18 |
| Fetch time per layer (ms) | 0.19 | 0.36 |
| Ordinary step T_N (ms) | 34.9 | same |
| Speculative step T_S (ms) | 62.9 | same |
| Tokens per step tau | 2.52 | same |
| Offload budget, ordinary | 0.03 | 0.99 |
| Offload budget, speculative | 0.58 | 0.99 |
| Speculative beats ordinary above | 0.00 | 0.00 |

(Budget = largest offloaded fraction staying within 5% of ordinary decoding with nothing offloaded, 34.9 ms/token; 'none' = already above that with nothing offloaded. Crossover 'never' = speculative stays slower.)

## Numbers for the text

- All on GPU: ordinary 34.9 ms/token, speculative 28.5 ms/token -> speculative is 1.22x the speed of ordinary
- Link-to-memory ratio: 176.7/864 = 1/5 (GH200: 1/10)
- Per-layer fetch 0.19 ms on PCIe vs per-layer compute 0.97 ms (ordinary) / 1.75 ms (speculative); at GH200 ratio the fetch is 0.36 ms
- Model validation (no fitted parameters): mean |error| 23.8%, max 60.6% over 12 measured configurations
- Budgets on PCIe: measured ordinary 0.03, speculative 0.58; model ordinary 0.99, speculative 0.99
- Budgets at GH200 ratio (model): ordinary 0.99, speculative 0.99
- Speculative beats ordinary at the same offloaded fraction above: measured 0.00, model PCIe 0.00, model GH200 ratio 0.00
- Offloaded runs matching all-GPU output token for token: 72/72
- Prompts: 6, mean context while decoding 7993 tokens

## Mean over prompts per configuration (R = layers kept on the GPU)

|                             |   offload_fraction |   tpot_ms |   step_ms |   tau |   h2d_mb_per_step |
|:----------------------------|-------------------:|----------:|----------:|------:|------------------:|
| ('ordinary', 'hbm', 36)     |               0    |     34.87 |     34.87 |  1    |              0    |
| ('ordinary', 'ours', 36)    |               0    |     35.91 |     35.91 |  1    |              0    |
| ('ordinary', 'ours', 32)    |               0.11 |     39.26 |     39.26 |  1    |            129.68 |
| ('ordinary', 'ours', 27)    |               0.25 |     46.84 |     46.84 |  1    |            291.41 |
| ('ordinary', 'ours', 18)    |               0.49 |     60.18 |     60.18 |  1    |            582.52 |
| ('ordinary', 'ours', 9)     |               0.74 |     73.39 |     73.39 |  1    |            873.63 |
| ('ordinary', 'ours', 0)     |               0.98 |     89.02 |     89.02 |  1    |           1164.74 |
| ('speculative', 'hbm', 36)  |               0    |     28.5  |     62.91 |  2.52 |              0    |
| ('speculative', 'ours', 36) |               0    |     28.55 |     63.02 |  2.52 |              0    |
| ('speculative', 'ours', 32) |               0.11 |     29.42 |     64.96 |  2.52 |            130.09 |
| ('speculative', 'ours', 27) |               0.25 |     31.54 |     69.68 |  2.52 |            291.79 |
| ('speculative', 'ours', 18) |               0.49 |     35.32 |     78.1  |  2.52 |            582.84 |
| ('speculative', 'ours', 9)  |               0.74 |     39.11 |     86.53 |  2.52 |            873.89 |
| ('speculative', 'ours', 0)  |               0.99 |     42.89 |     94.95 |  2.52 |           1164.94 |
