# Pair-counting system

`LSS_python.pair_counting` contains the counting cores and the surrounding block
workflow in one file. All public entry points are listed in `__all__`. Jackknife
counters in `tpcf.py` use its task builder and counting core; `cov.py` retains
resampling statistics. The complete `count_pairs_in_blocks` workflow remains a
separate convenience entry point. Counting uses non-periodic
`Corrfunc.theory.DDsmu` with the global z line of sight. MPI orchestration is provided separately by
`LSS_python.tpcf_mpi.run_tpCF_mpi`; see [the MPI guide](tpcf_mpi.md). The counting
cores do not import MPI or apply statistical corrections.

## Organization and dependencies

| Layer | Public interfaces | Responsibility |
| --- | --- | --- |
| Types | `PairCounts`, `PairTask` | Raw histogram/binning data and spatial task metadata |
| Counting core | `count_pairs` | Count the supplied point set or pair of point sets |
| Merging core | `merge_pair_counts` | Add raw histograms using explicit integer multiplicities |
| Preparation | `prepare_catalogs`, `split_catalog_into_blocks` | Prepare precision/weights and optionally partition coordinates |
| Task workflow | `build_pair_tasks`, `count_pair_task`, `sum_pair_counts` | Choose inputs, track tasks, and supply multiplicities to the cores |
| Result policies | `pair_normalizations`, `finalize_pair_counts` | Compute normalizations and apply final weight compatibility |
| Complete workflow | `count_pairs_in_blocks` | Compose the preceding functions for serial spatial counting |

Dependencies run from the complete workflow through peripheral tools to the
two cores. The cores do not depend on block geometry, task IDs, or finalization.
`PairCounts` stores copied `npairs`, `wpairs`, `sedges`, `muedges`, and
`with_weight`. Histograms have shape `(len(sedges) - 1, mubin)`; `npairs` is
uint64 and `wpairs` is float64. Task results and merged core results are raw
`PairCounts`, while finalized global results use dictionaries.

## Use the complete block workflow

```python
from LSS_python.pair_counting import count_pairs_in_blocks

result = count_pairs_in_blocks(
    data, random, sedges, mubin=4, boxsize=900.0, ngrids=3,
    with_weight=False, nthreads=1,
)
dd = result["DD"]["npairs"]
weighted_dd = result["DD"]["wpairs"]
dd_normalization = result["DD"]["normalization"]
```

The block workflow accepts `(N, 3)` or `(N, 4)` catalogs, fills missing weight
columns with ones when requested, and uses a common float32/float64 precision.
Spatial splitting retains global coordinates and empty blocks. Positions must
lie in `[0, boxsize]`; upper-edge positions belong to the last block. Scalar and
three-axis box sizes and grid shapes are supported.

The neighbor stencil requires `sedges[-1] <= min(boxsize / ngrids)`; larger
separations raise an error. Internal DD/RR tasks use autocorrelation. Each
unordered cross-block DD/RR task is submitted once, so its workflow multiplicity
is two. DR keeps both block directions, each with multiplicity one. The core
merger never infers these rules. `sum_pair_counts` rejects duplicate tasks.

The result includes global DD/DR/RR histogram dictionaries with normalizations,
binning, geometry, block populations, tasks, and their unscaled `PairCounts`.

## Choose point sets and call the cores directly

This example partitions a catalog by row order rather than spatial geometry.
There is no block-width constraint in either core.

```python
import numpy as np
from LSS_python.pair_counting import (
    count_pairs, merge_pair_counts, prepare_catalogs,
    finalize_pair_counts, pair_normalizations,
)

d, r = prepare_catalogs(data, random, with_weight=True)
a, b = np.array_split(d, 2)
options = dict(sedges=sedges, mubin=4, with_weight=True, nthreads=1)

aa = count_pairs(a, autocorr=True, **options)
bb = count_pairs(b, autocorr=True, **options)
ab = count_pairs(a, b, autocorr=False, **options)
dr = count_pairs(d, r, autocorr=False, **options)
rr = count_pairs(r, autocorr=True, **options)

raw = merge_pair_counts(
    {"DD": [(aa, 1), (bb, 1), (ab, 2)], "DR": [(dr, 1)], "RR": [(rr, 1)]},
    sedges=sedges, mubin=4, with_weight=True,
)
counts = finalize_pair_counts(raw)
norms = pair_normalizations(d, r, with_weight=True)
```

`count_pairs` requires an explicit `autocorr` flag. Autocorrelation requires
`data2=None`; cross-correlation requires a second catalog. Its inputs must
already be float32/float64 catalogs of matching precision with finite positions
and, if enabled, finite weights in the fourth column. It does not fill weights,
convert precision, impose `[0, boxsize]`, or require non-negative coordinates.
Use `prepare_catalogs` when input preparation is needed.

`merge_pair_counts` accepts streams of `(PairCounts, non-negative integer)`
contributions, grouped by DD/DR/RR. Missing groups are zero. It checks exact bin
edges, histogram shapes, and weight modes; it does not infer a multiplicity from
the group name or modify the inputs. Results can arrive in any order. For a
two-stage reduction, merge each partial output as a new contribution with
multiplicity one: multiplicities already applied must not be applied again.

## Jackknife integration

`compute_DD_JK`, `compute_DR_JK`, and `compute_RR_JK` in `LSS_python.tpcf` now use
`build_pair_tasks` and `count_pair_task`. Their signatures and returned internal/
cross dictionaries are unchanged. The common task loop counts internal tasks
first and neighboring tasks second; it retains the legacy `None` entries for
missing or fewer-than-two-point DD/RR autocorrelations. A singleton still
participates in cross-region and DR counts.

The public `tpcf.pair_counts_to_corrfunc` adapter converts a raw `PairCounts` to
the existing flat Corrfunc structured dtype. It applies no statistical policy.
The Jackknife adapter multiplies cross DD/RR `npairs` by the task's explicit
multiplicity once; DR retains both directions without this factor. Raw core
results are untouched. Existing `extract_npairs` compatibility, per-component
zero-weight fallback, RR cache keys, normalization, self-pair handling, and
Jackknife reconstruction remain in their existing layers. Subsample and MPI
implementations are unchanged by this migration.

The reconstruction layer now separately accepts `correction=None` or
`correction='mohammad21'`; see [Jackknife correction](jackknife_correction.md).
This option changes cross deductions and normalizations, while the raw pair
cores and the reusable internal/cross maps retain their counting definitions.

The existing covariance reconstruction assumes `n x n x n` divisions, so this
entry path now explicitly rejects unequal axis division counts. One or three
physical box lengths are supported. Neighbor coverage also explicitly requires
`sedges[-1] <= min(boxsize / ngrids)`. Spatial partitioning still uses the
existing `cov.get_sub_box_shift`; the allocation and coordinate-boundary policy
of that partitioner were not replaced.

No CLI arguments were added. Existing `run_jackknife_tpCF` callers and the
Jackknife branch of `bin/cal_cov_source_tpcf` automatically use the migrated
counters. Runtime dependencies are the installed LSS_python package, NumPy,
Corrfunc, joblib, numba, and tqdm; MPI is not needed for this serial migration.
The workflow is unchanged: prepare and partition inputs, read or compute RR,
count DD/DR, reconstruct full and leave-one-region-out results, then save the
desired result with joblib.

Acceptance used one weighted float64 sample with seed 20261024: 97 data points,
385 random points, boxsize 20, `2 x 2 x 2` regions, six separation bins over
0.5..5, four mu bins, and one thread. One region is empty and one contains one
point. It produced a full result and eight Jackknife objects. Counts,
normalizations, and xi matched the pre-migration result exactly in this sample;
loading the pre-migration RR cache also matched and left that file unchanged.
Only this sample was run, without a regression suite or smoke test. Zero
separation and other configurations were not separately exercised.

Actual inputs, outputs, logs, and the source snapshot are under
`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/jackknife_pair_counting/20261024_y73z2_9o/`.
`catalogs.npz` contains data, random, and sedges; `parameters.json` records the
settings. `jackknife_result.pkl` contains full output and
`jackknife_samples.pkl` contains just the eight-object array. `RR.pkl` is the
new cache; `baseline_RR.pkl` and `baseline_result.pkl` preserve the pre-migration
reference. `tpcf_before.py` is a source snapshot, not a standalone caller.
`summary.json` records comparisons and public-core call counts. The
`baseline.stdout.log`/`baseline.stderr.log` and
`validation.stdout.log`/`validation.stderr.log` files capture the two phases.
`file_inventory.csv` lists every absolute path and actual status.

To calculate from these small saved inputs, use the existing API in the current
Python environment (substitute new absolute output paths if retaining results):

```python
import joblib
import numpy as np
from LSS_python.cov import run_jackknife_tpCF

inputs = np.load(
    "/mnt/xiaodong/xiaoliang/software/LSS_python/validation/jackknife_pair_counting/"
    "20261024_y73z2_9o/catalogs.npz"
)
result = run_jackknife_tpCF(
    inputs["data"], inputs["random"], inputs["sedges"], 4, True,
    boxsize=20., ngrids=2, rr_filename="/absolute/new/output/RR.pkl",
    nthreads=1, full_output=True,
)
joblib.dump(result, "/absolute/new/output/jackknife_result.pkl")
```

The example was not executed with those placeholder paths. Reusing a result
path in `joblib.dump` overwrites it; an existing RR cache is read unless
`force_rr=True`, which overwrites that cache after recomputation. Actual
acceptance files were written to a new directory. Corrfunc temporary bin files
were confined to that directory's `tmp/tmp<random>` paths and cleaned after
successful counting; `tmp/` was empty at completion. No checkpoint or additional
cache is created, and no production catalog or result was accessed.

## Weights, finalization, and zero separation

Weighted core results contain the raw sum `npairs * weightavg`; mean pair weights
are never averaged between tasks. The existing all-zero weighted-result fallback
to unweighted counts is preserved by `finalize_pair_counts`, once per complete
DD/DR/RR histogram. Do not finalize partial results: a weighted singleton's
native zero-weight self-pair must not trigger a local fallback. Finalization
returns copied arrays and leaves the raw results intact.

No self-pair subtraction is performed. If the first separation edge is zero,
histograms retain Corrfunc's native self-pair output. Normalizations exclude
identical objects, so a physical estimator using the zero bin requires separate
treatment. The default end-to-end validation uses positive separations.

## Reproduce validation

```sh
python -m pytest -q tests/test_pair_counting.py
python tests/validate_pair_counting.py --output-dir validation/pair_counting
```

The validation uses 256 data points, 512 random points, a 900-unit box, `3 x 3 x 3`
blocks, four separation bins between 0.5 and 200, and four mu bins. Three cases
cover unweighted data, nonuniform positive product weights, and absent weight
columns with weights enabled. Direct whole-catalog Corrfunc calls provide the
reference. The test suite also exercises independent cores and hierarchical
merges without spatial task metadata.

`report.json` records settings, versions, task counts, timings, normalization and
histogram errors, and uncorrected Landy-Szalay xi errors where RR is positive.
Each NPZ contains catalogs, global and task histograms, multiplicities,
normalizations, and xi arrays. Integer counts must match exactly; float64
weighted counts and xi use `rtol=2e-12, atol=1e-10`. Small-sample timings are not
an acceleration benchmark. With no arguments the script prints help and exits.
