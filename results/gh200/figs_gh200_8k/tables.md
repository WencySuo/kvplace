## Table 2 values (paste into the report)

| | PCIe (measured) | GH200 ratio (model) |
|---|---|---|
| Link bandwidth (GB/s) | 341.5 | 415.7 |
| KV per sequence (GB) | 1.18 | 1.18 |
| Fetch time per layer (ms) | 0.10 | 0.08 |
| Ordinary step T_N (ms) | 52.8 | same |
| Speculative step T_S (ms) | 91.8 | same |
| Tokens per step tau | 2.50 | same |
| Offload budget, ordinary | 0.91 | 0.98 |
| Offload budget, speculative | 0.99 | 0.98 |
| Speculative beats ordinary above | 0.00 | 0.00 |

(Budget = largest offloaded fraction staying within 5% of ordinary decoding with nothing offloaded, 52.8 ms/token; 'none' = already above that with nothing offloaded. Crossover 'never' = speculative stays slower.)

## Numbers for the text

- All on GPU: ordinary 52.8 ms/token, speculative 43.5 ms/token -> speculative is 1.21x the speed of ordinary
- Link-to-memory ratio: 341.5/4000 = 1/12 (GH200: 1/10)
- Per-layer fetch 0.10 ms on PCIe vs per-layer compute 1.47 ms (ordinary) / 2.55 ms (speculative); at GH200 ratio the fetch is 0.08 ms
- Model validation (no fitted parameters): mean |error| 44.1%, max 84.2% over 12 measured configurations
- Budgets on PCIe: measured ordinary 0.91, speculative 0.99; model ordinary 0.98, speculative 0.98
- Budgets at GH200 ratio (model): ordinary 0.98, speculative 0.98
- Speculative beats ordinary at the same offloaded fraction above: measured 0.00, model PCIe 0.00, model GH200 ratio 0.00
- Offloaded runs matching all-GPU output token for token: 56/72
- Prompts: 6, mean context while decoding 7988 tokens

## Mean over prompts per configuration (R = layers kept on the GPU)

|                             |   offload_fraction |   tpot_ms |   step_ms |   tau |   h2d_mb_per_step |
|:----------------------------|-------------------:|----------:|----------:|------:|------------------:|
| ('ordinary', 'hbm', 36)     |               0    |     52.79 |     52.79 |   1   |              0    |
| ('ordinary', 'ours', 36)    |               0    |     40.03 |     40.03 |   1   |              0    |
| ('ordinary', 'ours', 32)    |               0.11 |     41.05 |     41.05 |   1   |            129.63 |
| ('ordinary', 'ours', 27)    |               0.25 |     41.87 |     41.87 |   1   |            291.27 |
| ('ordinary', 'ours', 18)    |               0.49 |     43.44 |     43.44 |   1   |            582.22 |
| ('ordinary', 'ours', 9)     |               0.74 |     44.66 |     44.66 |   1   |            873.17 |
| ('ordinary', 'ours', 0)     |               0.98 |     60.34 |     60.34 |   1   |           1164.12 |
| ('speculative', 'hbm', 36)  |               0    |     43.52 |     91.77 |   2.5 |              0    |
| ('speculative', 'ours', 36) |               0    |     23.19 |     50.72 |   2.5 |              0    |
| ('speculative', 'ours', 32) |               0.11 |     23.63 |     51.66 |   2.5 |            130.09 |
| ('speculative', 'ours', 27) |               0.25 |     24.09 |     52.7  |   2.5 |            291.72 |
| ('speculative', 'ours', 18) |               0.49 |     24.86 |     54.35 |   2.5 |            582.64 |
| ('speculative', 'ours', 9)  |               0.74 |     25.45 |     55.68 |   2.5 |            873.57 |
| ('speculative', 'ours', 0)  |               0.99 |     26.08 |     57.09 |   2.5 |           1164.49 |
