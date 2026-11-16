# Review: T4 cost-model (`task/cost-model` @ 59052ce)

**Verdict: REVISE** (small, mostly text)

I re-ran the tests: pytest 25 passed, and `test_kvoffload.py` ends with `identical across arms: True`. `cpu_model.py` regenerates `cpu_fit_gh200.md` byte-identical.

## Quality: acceptable
- The numbers are correct and reproducible, and `analyze.py` is untouched, so the L40S fit is unchanged.
- **The `a, u` fit has 0 degrees of freedom.** G and N come from the 16k profile for both contexts, so the "4 points" are 2 equations in 2 unknowns. An exact solve gives the same a=0.119 and u=17.1 us. `a` is algebra, not "GPU time not overlapped". Real signal: ~17 us/launch dispatch hiding most GPU work. The danger is that the model says T1's 19 ms GPU saving is worth about 2 ms/step. It cannot predict GPU-side changes or context scaling (32k is off by -28 to -35%).
- `fit()` reads `fit.bw`/`fit.b`, function attributes set by the caller (hidden global state). `import glob` is unused.

## Scope: acceptable (not under-delivery)
A separate module is a fair reading of the brief: add a term, re-run it, report the error, leave the L40S fit unchanged, add no new figures. Holding out the `arms_*`/`zc_*`/`evict_*` rows is better than the brief asked.

## Alignment: acceptable
- **Old `hbm` error = 0 is circular: correct.** `analyze.Model` takes the file's `hbm` step as T.
- **The doc's 44.1% is not the engineer's 42.8%.** `load_runs` gives `hbm` R=36, so `measured_curves` pools `hbm` with `ours` R=36 into one configuration; I reproduced 44.1 exactly. 42.8% counts the `hbm` rows separately at 0 error. Only held-out errors are comparable.
- **Most of the gain needs no new parameters.** Ellie's unchanged model, using the sweep's `ours` R=36 step as its reference (as the GH200 doc already advises), gives a **20.8%** held-out `ours` error. So 43.8 → 12.0 is really 43.8 → 20.8 from fixing the reference, then → 12.0 from the CPU term. That second step comes mostly from `pi` at ordinary R=0, and ordinary R=0 itself does not reproduce across files (46-70 ms). The ~10% held-out speculative bias is run-to-run drift that neither model removes. The weak-spot list does not make this clear enough for Ellie.

## Required changes
1. State in `cpu_fit_gh200.md` and in the docstring that `a, u` are exactly determined (0 dof). Drop the physical reading of `a`. Say the model cannot predict GPU-time reductions (T1, CUDA graphs) or context scaling.
2. Add a `ref36` column to the error summary: the old model with `ours` R=36 as its reference, zero new parameters.
3. Label the in-sample error as fit data, and note that the doc's 44.1% pools `hbm` with R=36.
4. Pass `bw, b` to `fit()` as arguments and remove `import glob`.

## CPU-aware budget/crossover in `analyze.py` tables: No (Later, not now)
The budget and crossover would be driven by `pi`, which does not reproduce, and by a T_res that ignores GPU work. T1 and T2 will change the step anyway. Refit after the T1 GPU run, with G and N profiled per context, before anything goes in the report tables.

## Later, not now
- Profile at 8k and 32k, then refit T_res in a `max()` form.
- Model resident-first vs resident-last layers.
- Add a CPU term to the zero-copy branch of `simulate.py`.
