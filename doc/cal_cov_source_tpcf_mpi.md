# MPI covariance-source command

`/mnt/xiaodong/xiaoliang/software/LSS_python/bin/cal_cov_source_tpcf_mpi`
is the MPI counterpart of `bin/cal_cov_source_tpcf`. It generates Jackknife or
subsample correlation measurements for subsequent covariance analysis. It saves
the same joblib object array as the serial command, with canonical
`LSS_python.xismu.xismu` objects, and does not calculate a covariance matrix.

## Invocation and dependencies

From `/mnt/xiaodong/xiaoliang/software/LSS_python`, examples are:

```sh
mpiexec -n 4 python bin/cal_cov_source_tpcf_mpi \
  /absolute/data.npy /absolute/random_full.npy 2000 8 jk \
  -n 8 --sbin 150 --mubin 120 --s_max 150 -v \
  -rr /absolute/results/RR_JK_mpi.pkl \
  -tpcf /absolute/results/tpCF_JK_mpi.pkl

mpiexec -n 4 python bin/cal_cov_source_tpcf_mpi \
  /absolute/data.npy /absolute/random_subbox.npy 2000 8 subsample \
  -n 8 -ww -weipow 1.5 -v \
  -rr /absolute/results/RR_sub_mpi.pkl \
  -tpcf /absolute/results/tpCF_sub_mpi.pkl
```

These are examples using user-provided catalogs; the example paths have not been
created. In the second example the random catalog covers one 250-wide subbox,
while data covers the full 2000-wide box. Substitute the scheduler's MPI launcher
when required by the cluster; no Slurm script is supplied or submitted here.
Four ranks with `-n 8` use up to 32 OpenMP threads in total. Cross-node execution
requires compatible MPI/Python environments and access to the project on every
node. Rank zero must also access the input and output paths.

Dependencies are the installed `LSS_python` project, its numerical stack
(including NumPy, joblib, numba, tqdm and Corrfunc), `mpi4py`, and a compatible
system MPI launcher. The script imports canonical token helpers from the package
module `LSS_python.weight_token`. MPI is imported after argument processing: running the command
without arguments, or with `--help`, prints help without initializing MPI.
The file is executable and can also be launched directly in place of
`python bin/cal_cov_source_tpcf_mpi` when its shebang selects the intended Python.

## Parameters and processing

All option names and aliases from the current serial script are retained.

| Input or option | Meaning |
| --- | --- |
| `data_filename random_filename boxsize ngrid {jk,subsample}` | `.npy` catalogs, full cubic box size, subdivisions per axis and resampling method |
| `--sbin`, `--mubin`, `--s_max` | Defaults 150, 120 and 150; separation edges always start at 0 |
| `--correction {none,mohammad21}` | Jackknife reconstruction correction; default none; mohammad21 requires jk |
| `-n / --nthreads` | OpenMP threads **per rank**, default 32 |
| `-ww / --with_weight` | Use column 4 in both catalogs; a missing weight column is rejected unless weights are explicitly set to one |
| `-weightone / --set_weight_to_one` | Set both enabled weight columns to one; permits three-column inputs |
| `-weipow / --weight-power` | Raise data weights only to this power, default 1; ignored when weights are disabled |
| `--weight-method` | Declare the raw mark method n<k>, gn<k>, or r<radius>; requires -ww, default n30 |
| `--weight-token` | Alternative single canonical method/power identity; conflicts and joint tokens are rejected |
| `-rr / --rr_filename` | RR joblib cache, default `RR_temp.pkl` in the working directory |
| `-tpcf / --tpcf_filename` | Sample-array output, default `tpCF_temp.pkl` in the working directory |
| `-f / --force` | Recompute an existing sample output; RR remains reusable unless `-frr` is also set |
| `-frr / --force_rr` | Recompute and overwrite RR when the measurement actually runs |
| `-v / --verbose` | Configuration, catalog sizes/dtypes, cache decisions, assignments and progress |

Non-AP measurement scripts now share the contract documented in
`/mnt/xiaodong/xiaoliang/software/LSS_python/doc/weight_tokens.md`. Raw data marks
are powered once; random weights are unchanged. Result xismu objects record the
declared method, canonical power, token and unit-weight override. Six-column gn
data use the Euclidean norm of columns 4-6 before the power; existing scalar
gradient marks are also accepted. AP sources and cache policies are unchanged.

The workflow is:

1. Validate positive dimensions/bin counts/thread counts. Jackknife additionally
   requires `s_max <= boxsize/ngrid`, matching its existing neighbor stencil.
2. Rank zero checks the sample output. If it exists without `-f`, all ranks skip
   before catalog loading. `-frr` alone does not override this whole-run skip.
3. Rank zero loads the `.npy` arrays, retains float32/float64 precision, selects
   three/four columns, applies the existing weight rules, and raises only data
   weights to the requested power. The shared library handles common precision.
4. All ranks collectively enter `run_jackknife_tpCF_mpi` or
   `run_subsample_tpCF_mpi` in `src/LSS_python/tpcf_mpi.py`. The shared workflows in
   `cov.py` own RR caching, normalization and reconstruction. Jackknife distributes
   labelled internal/cross tasks; subsample distributes independent DD/DR subboxes,
   sharing a local random template. Subsample's common RR is counted once on root
   or read from cache. Catalog/result arrays use the existing bounded MPI buffers.
5. Rank zero creates the output parent directory and saves the returned object
   array with `joblib.dump`. Python input/import/save failures are communicated
   before peers proceed or exit. Native process termination still depends on the
   MPI launcher/scheduler's termination policy.

The output has `ngrid**3` slots. Empty/singleton independent subboxes retain
`None`; Jackknife defaults to ordinary delete-one samples. With
`--correction mohammad21`, it retains part of the incident cross-region pairs
and adjusts their normalizations; see [the formulas and API](jackknife_correction.md).
RR formats and existing zero-separation handling match the serial workflows.
The serial `cal_cov_source_tpcf` exposes the same option. Changing correction
requires a new sample-output path or `-f`; the raw RR cache can be reused.

Both output filenames are explicit: `_mpi` is not appended automatically. Relative
output/cache names are resolved against rank zero's working directory. Use
separate RR paths for Jackknife and subsample because their cache schemas differ.
Existing outputs are replaced only with `-f`; existing RR is replaced only when
counting runs with `-frr`. No backup is created. Saving follows the serial
joblib behavior, without an atomic replacement step. An RR cache saved before a
later failure remains available. The command itself creates no log file,
checkpoint or persistent transport file. Redirect stdout/stderr or use the
scheduler's log configuration as needed. Corrfunc manages its temporary bin files
under the process temporary directory and removes them after successful calls.

## Actual small-sample validation and file inventory

Actual artifacts are under:

`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/cov_source_tpcf_mpi/20261024_6pg_7mv2/`

The reference was the current `bin/cal_cov_source_tpcf`, including its existing
weight-power support. Two MPI ranks on this local machine were compared with
one serial process for each method. Input data had 97 points, including empty
and singleton subboxes; random had 385 full-box points for Jackknife and 384
local-template points for subsample. Both runs used nonuniform float64 weights,
data weight power 1.5, boxsize 20, grid 2, six separation bins over 0..5, four mu
bins and one thread per process. Returned normalized DD/DR/RR, normalizations,
xi and the saved RR cache fields all matched with maximum absolute difference
**0** (`rtol=2e-12`, `atol=1e-10`; integer fields checked exactly). There were
eight Jackknife objects and six valid subsample objects plus two `None` slots.

One additional rerun checked whole-run skipping with missing input filenames:
the existing Jackknife output caused all ranks to skip without reading catalogs,
and neither the sample file nor RR was rewritten. The first temporary skip-check
command had a mispositioned boxsize argument and was correctly rejected by the
argument parser. It was corrected without rerunning the passed Jackknife
measurements; both the rejected command's logs and the successful corrected logs
are retained. This was a launch-wrapper typo, not a script defect.

| Artifact relative to the absolute directory above | Contents and status |
| --- | --- |
| `data.npy`, `random.npy`, `sub_random.npy` | Generated input copies from the previous synthetic `catalogs.npz`; unmodified on disk by weight-power processing |
| `jk_serial.pkl`, `jk_mpi.pkl` | Generated serial/MPI arrays of eight Jackknife objects |
| `subsample_serial.pkl`, `subsample_mpi.pkl` | Generated serial/MPI arrays with eight slots, including `None` at 0 and 1 |
| `jk_RR_serial.pkl`, `jk_RR_mpi.pkl` | Generated RR effective internal/cross arrays, total, edges and weight mode |
| `subsample_RR_serial.pkl`, `subsample_RR_mpi.pkl` | Generated structured RR result, edges, weight mode and subbox dimensions |
| `summary.json` | Generated settings, outputs, comparison errors and successful skip check |
| `execution.json` | Generated exact commands, exit codes, logs, environment and launch-wrapper correction record |
| `{help,jk_serial,jk_mpi,subsample_serial,subsample_mpi,jk_skip,jk_skip_corrected}.{stdout,stderr}.log` | Generated logs for all seven invocations; `jk_skip` records the rejected temporary command; corrected skip returned 0 |
| `file_inventory.csv` | Generated complete absolute source/input/output/log/temporary-file inventory, with origins and overwrite/cleanup policies |
| `tmp/tmp*` | Corrfunc's temporary bin text files; random basenames were not logged; the retained `tmp/` directory is empty after execution |

The two `not_present_*.npy` paths used in the corrected skip check were never
created. The validation used a unique directory: it did not overwrite earlier
results or read production catalogs. `TMPDIR` was this run's `tmp/`, Python
bytecode writing was disabled, and no checkpoint or Slurm stdout/error file was
created. The complete input/source paths and actual generation states are in
`file_inventory.csv`. No broader regression suite, smoke test, cross-node
benchmark or production job was run. Force/cache-recompute combinations and
float32 were not separately tested in this CLI check.
