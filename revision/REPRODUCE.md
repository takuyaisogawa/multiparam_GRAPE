# Reproduction guide

Run these commands from the repository directory after installing the dependencies.

## Numerical calculations

These commands use the released inputs and write new results under `build/`:

```sh
python reproduce.py baselines
python reproduce.py coupling
python reproduce.py initialization-checks
python reproduce.py holevo --id static_J0.2_T8
python reproduce.py mismatch --id static_000_saved
python reproduce.py propagation --id xz_J0.1_T2
python reproduce.py initialization --id static_target_j3_b600_direct_p1
python reproduce.py replacements --id new_xz_J0.1_T2
python reproduce.py reuse --model static --arm saved_previous
```

For `holevo`, `mismatch`, `propagation`, `initialization` and `replacements`,
omitting `--id` reruns all released cases in that study. The last three commands
above rerun control optimizations and take longer than rebuilding the figures.
`--workers N` is available for `mismatch`, `propagation` and `initialization`.
The initialization task includes 640 single-target comparisons. The `mismatch`, `propagation` and `initialization` tasks
reuse completed records in their output directory. Set `NJP_OUTPUT_DIR` to a new
directory for an independent run.

The reuse test applies the original finite-difference gradient ascent of
`f0 = 1 / sum(1 / F[i,i])`, with the original learning rates. Cached propagators
accelerate the finite-difference calculation. The additional `--literal` option
for the static model repeats the full original repropagation at each update.
Preparation of the saved starting waveform is excluded from this comparison.

The initialization comparison instead minimizes `-log(f0)` using L-BFGS-B and
an adjoint gradient. Its evaluation budget includes preparation along the
continuation path and all four random starts. The three replacement points also
use `-log(f0)` and L-BFGS-B, with refined propagation. Numerical settings, seeds,
evaluation counts, recorded CPU times and convergence records accompany the
data. CPU times describe the original runs and depend on the machine.

The reported cost is `C = Tr(F_cl^-1)`. Representative CFIM and Holevo results
use `(J,T)=(0.2,8), (10,8), (10,6), (20,1.4)` for static, circular xz,
circular xy and linear fields. The revised linear-field point in Fig. 4 and the
reuse test use `T=1.6`. The saved-control reuse values retain the original
evaluation convention; the initialization comparison and replacement points
include refined endpoint evaluations.

## Data format

JSON records retain full numerical precision; CSV files provide compact tables.
In `initialization.json`, `native` is the optimization-grid evaluation and
`reference` is the refined endpoint evaluation. `job`, `J_path` and `logs`
specify the allocation and actual evaluations for each run.

Load NPZ files with `numpy.load(..., allow_pickle=False)`. Keys in
`initialization_controls.npz` are the run IDs in `initialization.json`, with
arrays of shape `(N,3)` for the x, y and z controls. `control_index.json` lists
the keys, shapes and hashes in `controls.npz`. Original/replacement and mismatch
controls have shape `(N,3)`; reuse controls have shape `(3,N)`. Keys ending in
`__psi` and `__ds` contain states and parameter derivatives for the Holevo checks.
Parameters and time use the dimensionless units defined in the article.

Figures 2 and 4 retain the original vector artwork, with the recorded line-style
and point changes applied by `figures.py`. Figure 3 is redrawn from tabulated
curves and points; `baselines` reevaluates its xz reference curves with 500
control intervals. The original xy reference values are retained.

## Figures and tables

| Article | Data in `data/` |
| --- | --- |
| Appendix B, Fig. 6; SI Table S2 and Fig. S2 | `initialization.json`, `initialization_controls.npz`, `initialization_settings.json` |
| Appendix B, Table I | `reuse.json`, `controls.npz` |
| SI Table S1; Appendix C, Table II | `representatives.json`, `holevo.json` |
| SI Fig. S1 | `mismatch.json`, `mismatch_settings.json` |
| Figs. 3–4 and propagation checks | `replacement_points.json`, `saved_controls.json`, `figure_data.json` |

`python reproduce.py tables`, `figures` and `verify` run the corresponding parts
of the default command separately. Files ending in `_checks.json` contain
numerical validation records. `provenance.json` and `control_index.json` record
the origin of the code and arrays; their archived source paths are not needed
to run the calculations. `MANIFEST.json` lists the released file checksums.
