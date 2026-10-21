# Pair-counting system

`LSS_python.pair_counting` contains the counting cores and the surrounding block
workflow in one file. All public entry points are listed in `__all__`. Existing
production `cov.py`, `tpcf.py`, and command-line tasks do not call this experimental
system. It currently uses non-periodic `Corrfunc.theory.DDsmu` with the global z
line of sight. MPI orchestration is provided separately by
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
