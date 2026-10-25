# Optional Jackknife cross-pair correction

The Jackknife reconstruction interfaces accept the keyword-only parameter
`correction=None` or `correction='mohammad21'`. The default preserves the
previous ordinary delete-one implementation, including its floating-point
summation order and existing zero-separation handling. Other API values raise
`ValueError`. Raw Corrfunc counts, cross-block multiplicities, MPI scheduling /
buffer transport, RR cache formats and subsample calculations retain their
existing definitions.

## Interfaces and use

The parameter is exposed by:

- `LSS_python.cov.run_jackknife_tpCF`
- `LSS_python.cov.compute_jackknife_xi`
- `LSS_python.cov.compute_jackknife_deductions`
- `LSS_python.tpcf_mpi.run_jackknife_tpCF_mpi`

The private shared workflow `cov._run_jackknife_tpCF` forwards the same choice.
The full-sample xismu result and the raw internal/cross maps are unchanged when
the option is enabled. Raw `compute_DD_JK` / `compute_DR_JK` / `compute_RR_JK`
counters do not apply reconstruction corrections. Existing return structures
and positional arguments are retained.

```python
# Existing measurement arguments and catalogs are supplied by the caller.
from LSS_python.cov import run_jackknife_tpCF
from LSS_python.tpcf_mpi import run_jackknife_tpCF_mpi

samples = run_jackknife_tpCF(
    data, random, sedges, mubin, with_weight,
    boxsize, ngrids, correction='mohammad21',
)
# All ranks enter the MPI variant; only root supplies catalogs/settings.
mpi_samples = run_jackknife_tpCF_mpi(
    data, random, sedges, mubin, with_weight,
    boxsize, ngrids, correction='mohammad21', mpicomm=comm, root=0,
)
```

Both `bin/cal_cov_source_tpcf` and `bin/cal_cov_source_tpcf_mpi` expose
`--correction {none,mohammad21}`, default `none` (converted to Python `None`).
`mohammad21` is valid only for `jk`. For example, from the repository root:

```sh
mpiexec -n 4 python bin/cal_cov_source_tpcf_mpi \
  /absolute/data.npy /absolute/random_full.npy 2000 8 jk -n 8 -v \
  --correction mohammad21 \
  -rr /absolute/results/RR_JK.pkl \
  -tpcf /absolute/results/tpCF_JK_mohammad21_mpi.pkl
```

The example filenames are placeholders and have not been generated. The MPI
command uses four ranks and eight OpenMP threads per rank. Serial use omits the
MPI launcher and selects `bin/cal_cov_source_tpcf`. No-argument help remains
available. Existing weight-method/power options are unchanged.

Choose a new sample-output filename or use `-f` when changing correction;
existing sample outputs still trigger the normal filename-based skip. Raw RR
caches can be reused across correction choices; `-frr` is unnecessary solely
for changing correction. No new automatic suffix or correction metadata field
is added to result objects. Returned samples are saved by the existing CLI as
joblib object arrays; the library functions return them to the caller.

## Formula and normalization

The option follows Mohammad & Percival,
[arXiv:2109.07071, Section 4.5, Eq. 27](https://arxiv.org/html/2109.07071v2#S4.SS5),
as implemented by
[`pycorr.JackknifeTwoPointCounter.realization`](https://github.com/cosmodesi/pycorr/blob/main/pycorr/twopoint_jackknife.py).
For the **total** number of Jackknife regions `N` (not divisions per axis),

```text
alpha = N / (2 + sqrt(2) * (N - 1))
H_k = H_full - H_internal,k - alpha * H_incident_cross,k
```

This is applied to DD, DR and RR. Cross DD/RR already include both point
orderings in the stored maps; alpha is applied once to those values. Both
directed DR entries are scaled once. Internal deductions and the native
self-pair content of raw counts are retained. The existing zero-bin add-back
step is preserved after the deductions. At least three regions are required
for this correction; the current grid-based entry has eight or more regions
when using at least two divisions per axis.

Normalizations use all possible catalog pairs, including spatially distant
regions whose pairs do not occur in a measured separation bin. Let `D`, `R`
be full-catalog sums of weights and `d`, `r` those of region k. Without weights,
these are counts. Let `Q_D`, `Q_R` be full sums of squared weights and `q_d`,
`q_r` the region sums. First compute the existing delete-one normalizations:

```text
L_DD = (D-d)^2 - (Q_D-q_d)
L_DR = (D-d)*(R-r)
L_RR = (R-r)^2 - (Q_R-q_r)
```

Retaining `1-alpha` of incident cross pairs then gives:

```text
norm_DD = L_DD + (1-alpha) * 2*d*(D-d)
norm_DR = L_DR + (1-alpha) * (d*(R-r) + r*(D-d))
norm_RR = L_RR + (1-alpha) * 2*r*(R-r)
```

`compute_jackknife_deductions` returns scaled count deductions and the original
excluded catalog moments. `compute_jackknife_xi` combines them with the above
normalizations. Its full-sample object continues to use the full normalization.
The total region count is obtained from catalog lists; the deductions-only
interface can infer it as `ngrids_1d**3` if lists are absent. Current cubic-grid
and `smax <= min(boxsize/ngrids)` requirements remain those of the existing
spatial task / reconstruction implementation.

## Focused validation and actual files (2026-10-25)

`tests/validate_jackknife_correction.py` performs two focused checks on one
weighted float64 catalog: independent component/region-pair reconstruction,
then serial/MPI equality for the two correction modes. It does not run a broad
regression or smoke suite. Before code changes, the original workflow was run
once and saved with its source snapshots as the reference. The driver expects
`catalogs.npz`, `baseline_result.pkl` and `RR.pkl` staged in its output directory.

It requires the installed project, NumPy/joblib/numba/tqdm/Corrfunc, mpi4py and
a compatible system MPI. Only comparison/error-propagation helpers are imported
from `validate_resampling_mpi.py`; that driver's main is not executed. No
arguments prints help before MPI initialization. `--output-dir` selects the
staged inputs and generated outputs; `--root` defaults to 1. To repeat using the
actual staged directory, from the repository root:

```sh
mpiexec -n 2 python -B tests/validate_jackknife_correction.py \
  --output-dir /mnt/xiaodong/xiaoliang/software/LSS_python/validation/jackknife_correction/20261025_2bway6gp
```

The driver reads staged inputs, compares a default call with the pre-change
reference, compares the corrected samples against an independent retained-pair
sum and a sum over all ordered region-pair normalization masses, and saves both
serial modes. It then calls MPI for both choices, saves root's outputs and
checks the RR cache hash is unchanged. Reusing this directory overwrites the
five generated result/expected files and `summary.json`; it does not overwrite
the staged baseline/catalogs/RR. The driver creates no separate log files or
checkpoints. The execution wrapper captured stdout/stderr and environment as
listed below. Before using a different input sample, prepare a matching
pre-change baseline and RR cache; driver geometry is fixed at boxsize 20,
grid 2, four mu bins, one thread.

Actual validation: 97 data points and 385 random points, grid 2 (eight regions),
boxsize 20, six separation bins over 0.5..5, four mu bins, nonuniform weights,
empty and singleton regions, seed 20261024. Here alpha is
`0.6722974414137588`. The default matches the pre-change baseline exactly.
Corrected results match the independent sum with maximum absolute errors of
`4.37e-11` in normalization, `4.34e-18` in normalized counts and `2.11e-15` in xi
(within `rtol=2e-12`, `atol=1e-10`). Both serial/MPI comparisons have zero
difference. The raw components, full sample and RR cache are unchanged. Unknown
corrections are rejected. Help was checked for both CLIs and the driver.

Artifacts are under:

`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/jackknife_correction/20261025_2bway6gp/`

| Relative filename | Contents and actual status |
| --- | --- |
| `catalogs.npz` | Copied synthetic data/random/sedges from the previous small-sample input; generated copy, read-only during validation |
| `before_src__LSS_python__cov.py`, `before_src__LSS_python__tpcf_mpi.py`, `before_bin__cal_cov_source_tpcf`, `before_bin__cal_cov_source_tpcf_mpi` | Generated pre-edit source snapshots for provenance |
| `change_src__LSS_python__cov.py.diff`, `change_src__LSS_python__tpcf_mpi.py.diff`, `change_bin__cal_cov_source_tpcf.diff`, `change_bin__cal_cov_source_tpcf_mpi.diff` | Generated unified diffs against those snapshots, isolating this task's code changes |
| `baseline_result.pkl` | Generated before edits using the then-current serial workflow, full output |
| `RR.pkl` | Generated pre-edit raw effective internal/cross RR cache; reused unchanged for all validation measurements |
| `serial_none.pkl`, `serial_mohammad21.pkl` | Generated full-output serial dictionaries, including full and eight resampled objects |
| `mpi_none.pkl`, `mpi_mohammad21.pkl` | Generated corresponding MPI root outputs |
| `expected_mohammad21.pkl` | Generated eight-object array from independent retained-component / ordered-region-pair sums |
| `summary.json` | Generated settings, alpha and both outcomes/errors |
| `baseline_execution.json`, `execution.json`, `interface_check.json` | Generated baseline provenance, MPI command/environment/exit code, and help checks |
| `{baseline,validation,serial_help,mpi_help,validation_help}.{stdout,stderr}.log` | Generated logs; all recorded invocations succeeded; baseline stderr contains tqdm progress bars, other stderr logs are empty |
| `file_inventory.csv` | Generated complete absolute source/input/output/log/temp manifest, with origin, format, actual state and overwrite/cleanup policy |
| `tmp/tmp*` | Corrfunc per-call temporary bin text files; random basenames were not logged; retained `tmp/` is empty after execution |

The validation used two ranks on the local host with root 1. `TMPDIR` pointed to
the run's `tmp/`, `OMP_NUM_THREADS=1`, and Python bytecode writing was disabled.
No production catalogs/results were changed, no checkpoint or Slurm output was
generated, and no job was submitted. This checks implementation equivalence and
the correction algebra; it does not establish covariance accuracy against
independent simulations. Float32, zero separation, cross-node performance and
CLI end-to-end corrected measurements were not separately run in this check.
The CLI changes expose/forward the option and retain the existing weight logic.
