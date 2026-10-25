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

The box CLI also accepts `--weight-method n30 -weipow 1` or an equivalent single
`--weight-token n30_weipow1`; the method is no longer fixed to n30. Explicit
methods/tokens require `--with_weight`, compatible duplicate arguments are
allowed, and conflicting or joint identities are rejected. Data marks are
powered once, while random weights retain their original values. Explicit
method selection requires a data mark column. Saved result dictionaries include
`weight_method`, `weight_power`, `weight_token`, and `weight_set_one` in addition
to the existing numerical keys. Scalar gn marks and six-column gn vectors are
supported; vectors are reduced to their Euclidean norm before the power. See
`/mnt/xiaodong/xiaoliang/software/LSS_python/doc/weight_tokens.md` for the common
contract across all five non-AP measurement CLIs. This does not change the
library counting API or the cache policy.

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

## Jackknife and subsample MPI entry points

The command-line counterpart of `cal_cov_source_tpcf` is now
`bin/cal_cov_source_tpcf_mpi`; see [its invocation and output documentation](cal_cov_source_tpcf_mpi.md).

`LSS_python.tpcf_mpi` also exports `run_jackknife_tpCF_mpi` and
`run_subsample_tpCF_mpi`. Their positional parameters, RR cache formats and
root return values follow `cov.run_jackknife_tpCF` and `cov.run_subsample_tpCF`:

```python
from mpi4py import MPI
from LSS_python.tpcf_mpi import run_jackknife_tpCF_mpi, run_subsample_tpCF_mpi

# All ranks enter the call; data and random need exist only on root.
result = run_jackknife_tpCF_mpi(
    data, random, sedges, mubin, with_weight,
    boxsize, ngrids, refine_factors=(2, 2, 1),
    rr_filename=rr_filename, force_rr=False, nthreads=1,
    verbose=True, full_output=True, mpicomm=MPI.COMM_WORLD, root=0,
)
```

The example uses application-provided arrays/settings and does not create a
caller file. Replace the function with `run_subsample_tpCF_mpi` for independent
subvolumes, supplying a **single subvolume-sized** random catalog. Every rank
must use the same entry point, communicator and root; remaining settings are
read from root. Non-root catalogs and other mandatory measurement arguments may
be `None`. MPI is still imported only when an MPI entry point is called.

| Parameter | Meaning |
| --- | --- |
| `data`, `random` | Root's input arrays; enabled fourth-column weights are preserved, missing weights receive ones as in `cov` |
| `sedges`, `mubin`, `with_weight` | Existing separation/mu binning and weight mode |
| `boxsize`, `ngrids` | Full box geometry and subvolume divisions |
| `refine_factors` | Existing Corrfunc spatial refinement factors |
| `rr_filename=None` | Root alone loads/saves RR; `None` disables persistent RR caching |
| `force_rr=False` | Reuse an existing compatible RR cache; true recomputes and overwrites it |
| `nthreads=1` | OpenMP threads per MPI rank |
| `verbose=False` | Cache decisions, assignment sizes, rank progress and completion |
| `full_output=False` | Root returns the object array; true returns the existing full-output dictionary |
| `correction=None` (Jackknife only) | Ordinary delete-one samples; `'mohammad21'` scales cross deductions and their pair normalizations |
| `mpicomm=None`, `root=0` | Communicator defaults to `MPI.COMM_WORLD`; any valid root rank is supported |

Other ranks return `None`. These two entries do not broadcast `xismu` objects.
The default result is an object array of length `prod(ngrids)`. With
`full_output=True`, Jackknife returns `xismu_jk`, `xismu_full`, `sedges`,
`muedges`, and the DD/DR internal/cross maps; subsample returns `xismu_sub`,
`sedges`, `muedges`, `DD_sub`, and `DR_sub`. As in serial, Jackknife RR maps live
in its optional RR cache. The caller decides whether/where to save the returned
objects; the entry points themselves write only the requested RR cache.

The core call chains are:

```text
run_jackknife_tpCF_mpi
  -> cov._run_jackknife_tpCF (shared serial/MPI workflow on root)
     -> _count_jackknife_mpi
        -> tpcf._prepare_jackknife_tasks -> pair_counting.build_pair_tasks
        -> distribute buffers -> pair_counting.count_pair_task -> Corrfunc
        -> gather labelled buffers -> restore internal/cross maps
     -> cov.compute_jackknife_xi

run_subsample_tpCF_mpi
  -> cov._run_subsample_tpCF (shared serial/MPI workflow on root)
     -> _count_subsample_mpi
        -> shared RR: tpcf.compute_subsample_pairs on root, or existing cache
        -> DD/DR: distribute independent subboxes
           -> tpcf.compute_subsample_pairs on each rank -> Corrfunc
           -> gather buffers by global subbox ID
     -> existing per-subvolume normalization and xismu construction
```

`cov`'s public serial signatures are preserved. Its two shared private workflows
accept counting callbacks so preparation, cache checks, zero-weight
compatibility, self-pair handling, normalizations and reconstruction execute the
same processing steps for serial and MPI. Jackknife defaults to
`correction=None`; the optional `'mohammad21'` reconstruction is documented in
[Jackknife correction](jackknife_correction.md). The previously validated
`run_tpCF_mpi` counting/Reduce path is retained.

For Jackknife, global coordinates and the existing cubic grid / 26-neighbor
selection are retained; `smax <= min(boxsize / ngrids)` is required. Tasks are
balanced by estimated pair products, grouped by first region, with root also
counting. Cross DD/RR counts receive multiplicity two exactly once and both DR
directions remain separate. Gathered results retain region identities and are
restored to serial component order before summation and Jackknife deductions.

For subsample, root retains the existing local-coordinate partition. A nonempty
subbox's DD and DR go to the same rank, with assignments balanced by estimated
`N*(N-1)/2 + N*Nrandom` cost. Empty and singleton subboxes remain `None` in their
original global slots. Each working rank receives the shared random template;
independent samples are gathered without summation. The common RR is one serial
Corrfunc call on root, unless loaded from cache. This leaves RR acceleration
outside this subsample implementation. There is no new neighbor/smax restriction
for independent-subbox counting.

Catalogs and structured result arrays use direct `MPI.BYTE` buffers, with the
same production 256 MiB chunk limit as the full-count entry. Only commands,
settings, task labels and layouts use Python metadata messages. Receive
allocation and counting errors are exchanged before buffer traffic; exceptions
inside an active buffer transfer call `Abort`. Root announces workflow failures
to waiting workers. A process killed by the OS still requires the MPI launcher
or scheduler to terminate peers; this implementation does not add process
recovery. Root still owns all original catalogs and the existing partition
allocation. It also gathers the component maps needed by the existing RR cache
and result construction.

Only root accesses cache paths. Existing RR files are read when allowed by the
serial cache checks; `force_rr=True` replaces them. No backup/checkpoint is
created. As in the serial workflow, a newly saved RR cache remains if a later
DD/DR or reconstruction stage fails. Output-object saving and any Slurm
submission remain the caller's responsibility. This change adds library entry
points and the focused validation driver; production command-line entry points
are not connected to resampling MPI by this change.

### Focused validation and actual files (2026-10-24)

The new driver runs one weighted float64 sample per method, with the same samples
repeated using their RR caches. It depends on this installed project, NumPy,
joblib, numba, tqdm, Corrfunc, mpi4py and a compatible system MPI. Run from the
repository root:

```sh
mpiexec -n 2 python -B tests/validate_resampling_mpi.py \
  --output-dir /absolute/new/output --root 1
```

No arguments prints help before importing MPI. `--output-dir` is the artifact
directory and `--root` selects the input/output rank (default 1, requiring at
least two ranks). Both ranks temporarily use 128-byte chunks to exercise
multi-message catalog/result transfers; the production default stays 256 MiB.
The script generates catalogs in memory, saves them to `catalogs.npz`, runs the
serial and MPI measurements with separate fresh RR caches, compares all returned
counts and objects, and checks a second MPI call with cache reuse and the default
array return. Floating arrays use `rtol=2e-12, atol=1e-10`; exposed integer count
fields use exact comparison. Its only two methods are Jackknife and subsample;
it does not invoke other validation scripts or regression suites.

The actual local two-rank run used root 1, seed 20261024, boxsize 20, grid 2,
97 data points, 385 full-box random points for Jackknife, and 384 local-template
random points for subsample. Data subbox populations were `[0,1,16,16,16,16,16,16]`,
so the result includes empty/singleton handling. Six separation bins span
0.5..5 and there are four mu bins. All returned pair arrays, normalizations,
normalized counts and xi had maximum absolute difference **0**. Both RR cache
comparisons passed, cached MPI results matched fresh results, and the cache
hashes were unchanged. Jackknife returned eight exclusion samples; subsample
returned six valid objects and `None` at slots 0 and 1. This sample did not test
zero separation, float32 inputs, actual multi-GiB transfers, cross-node
performance or fatal-process recovery.

All actual artifacts are under:

`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/resampling_mpi/20261024_anv2vbyw/`

| Filename relative to that absolute directory | Source and contents | Actual status |
| --- | --- | --- |
| `catalogs.npz` | Driver-generated data (97x4), full random (385x4), shared random (384x4), sedges | Generated |
| `jackknife_serial.pkl`, `jackknife_mpi.pkl` | Full serial/MPI output dictionaries containing the full object and eight Jackknife objects, edges and effective DD/DR maps | Generated |
| `subsample_serial.pkl`, `subsample_mpi.pkl` | Full serial/MPI output dictionaries with eight slots, edges and structured DD/DR maps | Generated |
| `jackknife_RR_serial.pkl`, `jackknife_RR_mpi.pkl` | Existing-format effective RR internal/cross maps, full RR, edges and weight mode | Generated; cache reuse did not overwrite |
| `subsample_RR_serial.pkl`, `subsample_RR_mpi.pkl` | Existing-format structured RR, edges, weight mode and subbox size | Generated; cache reuse did not overwrite |
| `summary.json` | Settings, hosts, output paths and comparison metrics | Generated |
| `execution.json` | Launcher command, environment, elapsed time and exit code 0 | Generated by the execution wrapper |
| `validation.stdout.log` | MPI progress and both PASS messages | Generated by redirecting launcher stdout |
| `validation.stderr.log` | tqdm progress bars (no exception traceback) | Generated by redirecting launcher stderr |
| `file_inventory.csv` | Absolute paths, roles, contents, generation status and overwrite/cleanup policy | Generated after the run |
| `tmp/tmp*` | Corrfunc's per-call temporary separation-bin text files; random suffixes were not logged | Temporary files managed by Corrfunc; no files remain in `tmp/` |

The full absolute input/source/output inventory is `file_inventory.csv` in that
directory. Input catalogs were synthetic; no production catalog or production RR
was read. A unique directory was used, so earlier artifacts and production
results were not overwritten. If the driver is rerun in the same directory, its
catalog/result/summary filenames are overwritten and its existing RR caches are
reused under serial cache rules; use a new directory for a fresh reference run.
The driver writes no separate logs or execution record itself: the two log files
and `execution.json` above came from the launch wrapper. `TMPDIR` was set to the
run's `tmp/`, `OMP_NUM_THREADS=1`, and Python bytecode writing was disabled. No
checkpoint, retained Corrfunc temporary file, or Slurm stdout/error file was
created. No production job was submitted.
