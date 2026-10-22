# MPI pair counting for run_tpCF

`LSS_python.tpcf_mpi.run_tpCF_mpi` is a separate MPI entry point for non-periodic
Cartesian DD/DR/RR counting. It uses the cores and block tools in
`LSS_python.pair_counting`. The command-line box workflow is available separately
as `bin/cal_MCF_box_mpi`.

## Install and call collectively

Install a system MPI implementation and the optional Python dependency:

```sh
python -m pip install '.[mpi]'
```

Every rank must call the function with the same communicator and root. The
catalogs and remaining configuration are read only from root. Other ranks may
pass `None` for the catalogs. For example, save this caller as `run_pairs.py`:

```python
import numpy as np
from mpi4py import MPI
from LSS_python.tpcf_mpi import run_tpCF_mpi

comm = MPI.COMM_WORLD
data = np.load("data.npy") if comm.rank == 0 else None
random = np.load("random.npy") if comm.rank == 0 else None

result = run_tpCF_mpi(
    data, random, np.linspace(0.5, 200.0, 5), 4, False,
    boxsize=900.0, ngrids=3, mpicomm=comm,
    nthreads=4,
    output_dict={"DD": "pairs/DD.pkl", "DR": "pairs/DR.pkl", "RR": "pairs/RR.pkl"},
)
if comm.rank == 0:
    print(result["DD"]["npairs"])
```

Launch four ranks with:

```sh
mpiexec -n 4 python run_pairs.py
```

For MPICH/Hydra, an example with two ranks on each of two allocated nodes is:

```sh
mpiexec -hosts nodeA,nodeB -ppn 2 -n 4 python run_pairs.py
```

Replace the example hostnames with the nodes assigned to your job. Use your
cluster's scheduler launcher when required by its MPI configuration.

The MPI launcher or cluster scheduler determines which nodes host those ranks.
All nodes need compatible MPI/Python/Corrfunc environments and access to the
caller and package. `nthreads` is the OpenMP count **per rank**; choose the rank
and thread counts to fit the allocated cores.

## Command-line box workflow

From the repository root, run:

```sh
mpiexec -n 4 python bin/cal_MCF_box_mpi data.npy random.npy 4 \
  --boxsize 900 --ngrids 3 \
  -smin 0.5 -smax 150 -sbin 150 -mubin 120 \
  --with_weight --save_dd_dr \
  --rr_filename results/RR.pkl --tpcf_filename results/tpCF.pkl
```

The positional arguments and existing option names follow `cal_MCF_box`.
`nthreads` is per rank, so this example uses four MPI ranks with four OpenMP
threads each. The additional required options `--boxsize` and `--ngrids` each
accept one value or three axis values, for example `--boxsize 900 780 660
--ngrids 2 3 2`. The same block-width restriction applies.

Only rank zero loads catalogs and cached counts, selects work, calls the existing
`cal_tpCF_from_pairs` estimator, and writes outputs. All ranks receive skip/error
decisions and collectively participate in any required counting. Final tpCF
results keep the existing estimator's normalization and zero-separation handling.

Cache and output behavior follows the original box script:

- Defaults are `RR_temp.pkl` and `tpCF_temp.pkl` in the working directory.
- `--save_dd_dr` saves newly computed counts as `DD_temp.pkl` and `DR_temp.pkl`.
  Existing DD/DR/RR files are reused when their respective force flags are absent.
- An existing tpCF output skips the whole run unless `-force` is set. `-force`
  recomputes DD, DR, and tpCF; add `--force_rr` to recompute RR as well.
- `--only_run_rr` ignores the positional data filename. It skips an existing RR
  output unless `--force_rr` is supplied.

Cache selection uses filenames and force flags; it does not establish whether a
cache was produced from the current catalogs, binning, or weight settings. Set
the corresponding force flags or choose new output paths when those inputs change.

With `--with_weight`, existing fourth-column weights are retained and missing
weight columns receive ones. `--set_weight_to_one` forces both catalogs' enabled
weights to one. Without `--with_weight`, only position columns are used. Calling
the command without arguments prints help before MPI initialization.

Add `--verbose` / `-v` to print the configuration, catalog shapes/dtypes/sizes,
cache decisions, block populations, task counts, and per-rank assignments. Each
rank then reports task progress and elapsed times at roughly 5% task intervals
or after 30 seconds between reports, checked when a task completes. Progress
counts tasks rather than predicting remaining runtime; a single Corrfunc call
finishes before its completion can be reported. Logs include rank prefixes and
are flushed immediately for batch-job output. Reduction and file writes also
have start/completion messages and timings.

## Interface and outputs

The first ten parameters follow `run_tpCF`: catalogs, `sedges`, `mubin`,
`with_weight`, `run_parts`, `refine_factors`, `output_dict`, `nthreads`, `verbose`.
MPI-specific keyword arguments are:

| Argument | Meaning |
| --- | --- |
| `boxsize`, `ngrids` | Required scalar or three-axis box lengths and grid divisions |
| `mpicomm=None` | Defaults to `MPI.COMM_WORLD`; subcommunicators are supported |
| `root=0` | Rank that provides inputs, receives results, and writes files |
| `broadcast_result=False` | If true, return the complete result on every rank |

By default root returns `{"DD": ..., "DR": ..., "RR": ...}` and other ranks
return `None`. Requested entries are flattened Corrfunc structured arrays with
fields `smin`, `smax`, `savg`, `mu_max`, `npairs`, and `weightavg`; unused entries
are `None`. As in the default serial calls, `savg` is zero. These arrays can be
passed to the existing result-processing functions and saved as joblib files.
Only root writes requested files, creating missing parent directories.

`run_parts` accepts a count name, a sequence of names, or `all`. RR-only accepts
`data_catalog=None`; DD-only accepts `random_catalog=None`. An empty selection
returns all entries as `None`. Missing weight columns receive unit weights when
enabled, and catalog precisions are unified using `prepare_catalogs`.

## Distribution and correctness

1. Root validates and partitions catalogs, keeping global coordinates.
2. Internal and neighboring cross-block tasks are grouped by their first block
   and assigned using estimated pair-product costs. Grouping reuses input blocks;
   root computes assigned work as well.
3. Each rank receives its task list and the distinct blocks required by those
   tasks. Task lists, block IDs, shapes, and dtypes are scattered as small Python
   metadata. Catalog arrays are sent separately through direct MPI buffers.
   The complete catalog is not unconditionally broadcast to every rank.
4. Ranks use `count_pair_task` and `sum_pair_counts`. DD/RR off-diagonal
   multiplicity is applied once, and both DR block directions remain distinct.
5. MPI reduces raw uint64 counts and float64 weight sums with SUM. Root restores
   `weightavg = wpairs / npairs` where counts are nonzero.

No local all-zero-weight fallback, normalization, or special statistical
correction is applied. Native zero-bin self-pairs are retained. This preserves
the complete-result compatibility convention in existing consumers rather than
triggering it on partial rank results.

Positions must lie in `[0, boxsize]`. The 26-neighbor stencil requires
`smax <= min(boxsize / ngrids)`. This implementation distributes block tasks; it
does not use a fully distributed catalog reader or fine-grained halo exchange.
Root holds the input catalogs, and a block may be replicated to several ranks
when their tasks require it. The communication benefit therefore depends on
geometry, task placement, and catalog size.

Python exceptions during root preparation, worker counting, and root output are
propagated collectively as `RuntimeError` before the next collective phase.
Every rank must participate; fatal native failures and MPI transport errors
remain subject to the MPI runtime's failure behavior.

## Bounded array-buffer transfers

Catalog distribution uses `Send`/`Recv` with `MPI.BYTE`, iterating over contiguous
byte views of each required block. Each message is at most 256 MiB, including
when a block or a rank's total catalog payload exceeds 2 GiB. No per-rank joined
catalog or large pickle stream is created. Root retains its original block
references. Array dtype, coordinates, weights, and point order are unchanged;
the existing spatial partition and counting rules are unchanged.

After receiving metadata, non-root ranks allocate their final arrays. Allocation
errors are exchanged before any catalog buffers are sent, so a sender does not
wait for a receiver that already failed to allocate. An exception during an
in-progress buffer transfer prints the failing rank and calls `Abort`; a process
killed directly by the OS still requires MPI launcher/scheduler termination.
The buffers retain the existing same-platform dtype representation expected
across the cluster's compatible environments.

`broadcast_result=True` also sends only small layout metadata through pickle;
the Corrfunc structured arrays are broadcast in bounded byte buffers. Their
dtype descriptions are preserved. Raw histogram `Reduce` calls remain as before.
The public function and command-line arguments are unchanged. The chunk limit
is an internal constant, not a new command-line option. These changes reduce
serialization overhead; root catalog loading and spatial partition temporaries
still use the existing implementation.

The focused validation performs exactly two checks:

```sh
mpiexec -n 2 python tests/validate_mpi_buffers.py --output-dir /absolute/new/output
```

Run from the repository root with Python, mpi4py/system MPI, NumPy, Corrfunc,
joblib, and the installed project available. No arguments prints help before MPI
initialization. `--output-dir` selects the validation output directory. Both
ranks temporarily use 128-byte messages, so small inputs exercise multiple
messages without multi-GiB allocations; production still defaults to 256 MiB.

1. Send deterministic float32 three-column and float64 four-column catalog views,
   including an empty block, using rank 1 as root. Check shape, dtype, contiguity,
   every element, every byte, and root's reuse of the original arrays.
2. Generate 256 data and 512 random points with nonuniform float64 weights,
   seed 20261022, boxsize 24, grid 2, six separation bins over 0..6, six mu bins,
   and one thread. Compare MPI DD/DR/RR with whole-catalog serial Corrfunc:
   integer counts must match exactly; weighted sums and xi use `rtol=2e-12`,
   `atol=1e-10`. Exercise result-buffer broadcasting and compare its bytes across
   ranks. Only comparison helpers are reused from `validate_tpcf_mpi.py`; its
   broader validation routine is not invoked.

Synthetic inputs are generated in memory. The driver writes
`<output-dir>/summary.json` (settings, two outcomes, hosts, transfer hashes, and
errors) and `<output-dir>/sample_counts.npz` (data, random, bin edges, and the
MPI/serial structured counts). Reusing an output directory overwrites these two
files; use a new directory to retain previous evidence. Stdout/stderr use the
launcher's streams; the driver creates no independent log files or checkpoints.
Corrfunc's temporary separation-bin file, if used, follows its existing cleanup
behavior. This local check does not establish cross-node performance or an
actual multi-GiB transfer.

On 2026-10-22 this focused two-step driver passed with two ranks on the same
local host. DD/DR/RR integer counts were identical; the maximum weighted-sum
difference was `9.09e-13` and the maximum xi difference was `2.39e-15`.
The actual run used `/home/xiaoliang/.conda/envs/xiaoliang/bin/python` and
`/software/mpich4.3-gnu12/bin/mpiexec`. Evidence is under
`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/mpi_buffer_transfer/20261022_y_t59es5/`:
`summary.json` records the two checks, `sample_counts.npz` retains their counting
inputs/results, `execution.json` records the command, environment and exit code,
and `validation.stdout.log`/`validation.stderr.log` capture launcher output.
`file_inventory.csv` lists all absolute input/output paths and their actual
status. `TMPDIR` was directed to this run's `tmp/` subdirectory; Corrfunc's
temporary `tmp<random>` bin files were created there and the directory was empty
after the successful run. No checkpoint or Python bytecode cache was generated.
No broader regression suite, smoke test, production-catalog read, or Slurm
submission was performed for this change. The validation used a new directory
and did not overwrite earlier validation or production results.

## Validate

```sh
python -m pytest -q tests/test_pair_counting.py tests/test_tpcf_mpi.py \
  tests/test_cal_2pcf_box_mpi.py
mpiexec -n 2 python tests/validate_tpcf_mpi.py --output-dir validation/tpcf_mpi/n2
```

The validation compares small random catalogs against serial `run_tpCF`, including
nonuniform weights and missing weight columns. It also exercises partial count
selection, nonzero roots, broadcast results, empty inputs, zero separation,
rectangular grids, file output, and collective error propagation. Reports record
rank hostnames so a local multi-process check is not confused with a multi-node
benchmark. No arguments prints help without initializing MPI.

The command-line tests also compare saved xi results with the serial estimator,
exercise cache skip/reuse and force flags, and check root input/output failures.

Local checks with 1, 2, and 4 ranks passed for 256 data and 512 random points.
Integer counts matched exactly; the largest absolute weighted-count difference
was `1.14e-12`, and the largest uncorrected xi difference was `2.09e-15`, within
`rtol=2e-12, atol=1e-10`. Reports and input/output arrays are saved under
`validation/tpcf_mpi/n1`, `n2`, and `n4`. All recorded ranks ran on the same host;
these results establish local MPI correctness, without measuring cross-node
performance.
