# Method and weight tokens

The canonical implementation is
`/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/weight_token.py`.
It uses only the Python standard library. Import APIs from
`LSS_python.weight_token`; the bin script is only a CLI entry point and does not
re-export the old helper functions. This change migrates callers in this project;
callers in other projects must update their imports separately.

## Grammar and scientific conventions

| Method | Meaning | Raw generator columns |
| --- | --- | --- |
| `n<k>` | Density from k nearest points, including self; adaptive h = max distance / 2 | x, y, z, rho |
| `gn<k>` | Gradient magnitude divided by paired density by default; raw fields use the existing Fortran kernel-derivative convention and same neighbor rule | x, y, z, rho, drhox, drhoy, drhoz |
| `r<radius>` | Density within the full support radius; h = radius / 2; includes self | x, y, z, rho |

Counts are positive integers. Radii are positive finite decimals in the same
length units as positions. The generator additionally requires 2 <= k <= N
for adaptive smoothing and radius < BoxSize/2 for its periodic box.

Canonical spellings have no leading zeros, plus sign, scientific notation, or
unnecessary fractional zeros: `n30`, `gn30`, `r10`, `r0.5`. Builders normalize
numeric inputs such as radius `10.0`; parsers reject noncanonical token text.

Full weight tokens are `<method>_weipow<alpha>`, for example
`n30_weipow-0.5`, `gn30_weipow1`, and `r10_weipow-0.5`. Powers are finite signed
decimals, normalized by the same rules; zero is `0`. Existing canonical
`n30_weipow...` text and the default builder calls are preserved. As before,
noncanonical text such as `n30_weipow1.0` is rejected.

Joint tokens contain ordered powers for one method, e.g.
`r10_weipow-1_0_1`. Order and repeated powers are preserved. These denote an
ordered set, not a product of marks or a combination of different methods.

The short method grammar fixes the default self and smoothing conventions.
The generator rejects the old `--ignore-self`, `--no-adaptive-h`, and
`--h-fixed` options because they would change a calculation without changing its
token. Custom scientific parameters remain available through the existing
`LSS_python.marks` functions; no new token syntax for them is introduced here.
Thread and chunk settings are runtime options, not part of scientific identity.
For gn measurements, `--no-divide-by-rho` explicitly selects raw gradient
magnitude. The token spelling is unchanged; the saved gn result additionally
records `gradient_divide_by_rho` as true/false. Different overrides therefore
require distinct output paths and this metadata when comparing results.

## Python APIs

```python
from LSS_python.weight_token import (
    make_mark_token, parse_mark_token, make_weight_token,
    make_joint_weight_token, parse_weight_token,
)

make_mark_token("r", "10.0")                 # r10
parse_mark_token("gn30")                      # MarkMethod(kind="gn", value=30, token="gn30")
make_weight_token(-1)                         # n30_weipow-1
make_weight_token(-0.5, mark="r10")            # r10_weipow-0.5
make_joint_weight_token([-1, 0, 1], "gn30")    # gn30_weipow-1_0_1
parse_weight_token("r10_weipow-0.5").method   # MarkMethod(kind="r", value=Decimal("10"), token="r10")
```

Parsed weight tokens preserve the `mark`, `powers`, `token`, `power`, and
`is_joint` interfaces and expose the parsed method through `.method`.
`.power` rejects joint tokens. The count/radius is available as `.method.value`;
radius values use Decimal to preserve their canonical text without binary
floating-point rounding. Decimal formatting does not round to the current
Decimal context precision.

The numerical API `LSS_python.marks.create_drho(...)` now returns
`(rho_array, gradient_array)`, with shapes `(N,)` and `(N,3)`. Both are raw
fields from the same neighbor query. Update callers that previously received
only the gradient. The shared measurement APIs `prepare_box_catalog` and
`prepare_mock_catalog` accept `divide_by_rho=True` (default); false uses the
raw norm. They normalize vectors before taking the norm, mathematically
equivalent to `norm(gradient)/rho`, to avoid avoidable raw-norm overflow.

## Command-line token tools

The following examples print text or JSON to stdout; they do not write files:

```sh
python /mnt/xiaodong/xiaoliang/software/LSS_python/bin/mcf_weight_token.py method-token r 10.0
python /mnt/xiaodong/xiaoliang/software/LSS_python/bin/mcf_weight_token.py parse-method gn30
python /mnt/xiaodong/xiaoliang/software/LSS_python/bin/mcf_weight_token.py weight-token -0.5 --method r10
python /mnt/xiaodong/xiaoliang/software/LSS_python/bin/mcf_weight_token.py joint-weight-token -1 0 1 --method gn30
python /mnt/xiaodong/xiaoliang/software/LSS_python/bin/mcf_weight_token.py parse n30_weipow-1
```

`python -m LSS_python.weight_token` exposes the same CLI. No arguments prints
help and exits successfully for both entry points.

## Raw mark generation

The executable is `/mnt/xiaodong/xiaoliang/software/LSS_python/bin/create_rho`.
Its positional arguments are input .npy path, scalar periodic BoxSize, and an
optional output path. Without the third argument, the output is placed beside
the input as `<input_stem>_<method>.npy`: for example, input `snapshot.npy`
with `-method n30` produces `snapshot_n30.npy`; `snapshot.v2.npy` with
`-method r0.5` produces `snapshot.v2_r0.5.npy`. The stem removes the input's
last suffix before adding the selected method. The input must be float32/float64 with shape (N, >=3); only
its first three position columns are used. Existing additional columns are not
treated as neighbor weights and are not copied to the output.

`--rho-method`, `--method`, and `-method` select one canonical method token,
defaulting to `n30`. `--nthreads`/`-n` selects backend threads. `--chunk-size`
applies only to radius queries and defaults to 100000. `--verbose`/`-v` prints
progress. `--force`/`-f` permits output replacement. No arguments prints help.

Processing: parse method through the package token module; validate input and
runtime settings; call create_rho/create_drho/create_rho_fix_r; combine positions
with the raw result; save NPY. Density output has four columns and gradient
output has seven: xyz, rho, gx, gy, gz. Input precision is preserved. A missing `.npy` suffix on an
explicit output is added before checking whether output exists. Existing output
is skipped without `--force`, without asking for name confirmation because no
calculation will run. Otherwise the output basename must contain the exact
selected method token, or the user must enter `y` or `yes` to proceed. Matching
is case-sensitive for canonical tokens; for example, n30 does not match gn30 or
n300, and r2 does not match r2.5. Parent-directory names do not count. Confirmation
accepts surrounding whitespace and uppercase Y/YES. Any other reply, EOF, or
Ctrl-C cancels with exit code 1 before catalog loading/calculation and writes no
output. `--force` permits overwriting but does not bypass a missing-token prompt;
the output can include the input itself if that path is deliberately supplied
and the required confirmation is given. Parent directories must already exist.
There is no automatic deletion, checkpoint, or science cache.

```sh
create_rho /absolute/path/snapshot.npy 2000 -method n30 -n 8
create_rho /absolute/path/snapshot.npy 2000 /absolute/path/custom_n30.npy -method n30 -n 8
create_rho /absolute/path/snapshot.npy 2000 /absolute/path/custom.npy -method n30 -n 8
```

The first command chooses `/absolute/path/snapshot_n30.npy`; the second uses
the matching explicit name without a prompt; the third requires y/yes. These
paths are examples and are not generated by this documentation. The complete
actual naming/confirmation verification and file inventory are in
`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/create_rho_output_naming/README.md`.

The generator accepts method tokens only: it rejects a full weight token rather
than silently discarding its power. It never raises raw marks to a power.
The non-AP measurement contract is described below. Seven-column gradient
outputs retain both raw density and raw vectors. Measurement CLIs use
`sqrt(drhox**2 + drhoy**2 + drhoz**2) / rho` by default before applying the
requested power. The underlying vector kernel and norm convention remain the
ones previously checked against
`/Raid6/1/teamxdli_a/xiaoliang/software/LSSCode/src_gfortran/LSS_grad_fields.f90:350`
and `LSS_chisq.f90:198`; density division is now explicitly applied at the
measurement boundary. Existing six-column files lack rho: regenerate them
(use --force to replace an existing output), or choose --no-divide-by-rho
to measure the original raw-gradient magnitude. No density is inferred from
the filename and there is no extra neighbor query at measurement time.

Generator dependencies are NumPy, SciPy, Numba and tqdm, plus freud-analysis for
radius queries. No dependency installation or scheduler submission is needed
for the token helpers. Actual verification inputs, commands, outputs and logs
are inventoried in
`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/weight_refactor_stage2/README.md`.

## Non-AP measurement commands

The following five scripts share their argument and weight preparation logic:
`cal_MCF_box`, `cal_MCF_box_mpi`, `cal_MCF_mock`, `cal_cov_source_tpcf`, and
`cal_cov_source_tpcf_mpi`, all under
`/mnt/xiaodong/xiaoliang/software/LSS_python/bin/`. AP scripts are unchanged.
The FFT power scripts' general particle-weight/value fields are separate from
these marked-correlation identities and are not changed by this migration.

Use `--with_weight`/`-ww` with either:

- `--weight-method r10 --weight-power -0.5`, or
- `--weight-token r10_weipow-0.5`.

The method defaults to n30 and power to 1, preserving existing weighted calls.
`-weipow` remains an alias for `--weight-power`. When both forms are supplied,
their methods and powers must agree. Joint tokens are rejected because one
measurement invocation writes one result; joint tokens still identify ordered
sets of separate results. Explicit methods/tokens require `-ww`. For legacy
compatibility, `-weipow` and the unit-weight flag alone remain ignored without
`-ww`. Numerical powers must be representable as finite floats without
underflow to zero.

The method describes the supplied raw mark. These commands do not regenerate
marks, infer their method from filenames, or prove their provenance. The
generator still outputs raw marks. Apply their power exactly once at the shared
measurement loading boundary; there is no second script-level power operation.

Cartesian data use `[x,y,z,raw_mark]`, or `[x,y,z,rho,gx,gy,gz]` for gn methods.
The default gn data weight is `(norm(gradient)/rho)**alpha`. Add
`--no-divide-by-rho` to get `norm(gradient)**alpha`; this option requires -ww
and a gn method, and also accepts old six-column xyz,gx,gy,gz input in that
explicit raw mode. Division always precedes the mark power, and rho must be
finite and strictly positive when used. The norm uses successive hypot
operations to avoid unnecessary square overflow. Prepared scalar gn marks
remain supported: their sole mark column must already represent norm/rho
in default mode, or raw norm in explicit raw mode; it is powered once with
no further density division because the scalar layout contains no density.
Other enabled data layouts
are rejected. Legacy box calls
may still supply three-column data and receive unit weights; explicit methods
require a mark column. Covariance inputs retain their stricter requirement for
an enabled weight column. Random weights are read from their fourth column and
are not raised to the data power.

Survey/mock data use `[RA,DEC,R,base_weight,raw_mark,...]`. The effective data
weight is `base_weight * raw_mark**alpha * product(extra_factors)`. The power
does not affect base weights or extra factors. For n/r methods, existing extra
factors are preserved. Four-column base-only legacy mock data are still valid
with the default power 1; explicit method selection or a nonunit power requires
a separate mark. Random mock weights remain the product of their existing
weight columns, without the data exponent. Both pair counting and normalization
receive the same prepared four-column catalog, and the input arrays/files are
not mutated. Scalar gn mock input has exactly five columns. The new raw-field
gn mock layout has eight: `[RA,DEC,R,base_weight,rho,gx,gy,gz]`; its final data
weight is `base_weight * (sqrt(gx**2 + gy**2 + gz**2)/rho)**alpha` by default.
With --no-divide-by-rho, rho is ignored and the raw norm is powered; old
seven-column base+gradient input is accepted only in that explicit raw mode.
Other gn column counts are
rejected to keep the mark layout unambiguous. A zero gradient with a negative
power and non-finite or overflowing weights are rejected.

For example, generate paired raw fields once and measure both modes with:

```sh
create_rho /absolute/path/positions.npy 2000 -method gn30 -n 8
cal_MCF_box /absolute/path/positions_gn30.npy /absolute/path/random.npy 8 -ww --weight-token gn30_weipow-0.5 -tpcf /absolute/path/normalized.pkl -rr /absolute/path/rr.pkl
cal_MCF_box /absolute/path/positions_gn30.npy /absolute/path/random.npy 8 -ww --weight-token gn30_weipow-0.5 --no-divide-by-rho -tpcf /absolute/path/raw.pkl -rr /absolute/path/rr.pkl
```

Replace the illustrative absolute paths with existing inputs and writable output
locations. These example files are not created by the documentation. No
intermediate scalar-mark file is written: the norm and powered four-column
catalog exist only in memory. The gradient integration checks and complete
actual file inventory for the current normalization change are recorded in
`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/gradient_density_normalization/README.md`.

`--set_weight_to_one`/`-weightone` explicitly sets enabled effective data and
random weights to one before counting. This now also applies consistently to
the serial box random catalog; previously that CLI did not forward the flag to
its random loader. Weight identity metadata records this override separately.

Saved correlation dictionaries and each nonempty covariance xismu object have
`weight_method`, `weight_power` (canonical decimal text), `weight_token`, and
`weight_set_one`. The first three are None for unweighted results. Other result
fields and the covariance array layout are unchanged. Existing consumers that
assert an exact dictionary key set must allow these four additional fields.
Enabled gn results additionally have `gradient_divide_by_rho`, recording the
explicitly selected normalization convention. n/r and unweighted results retain
the existing four-field metadata layout.
No pair-count cache policy, cache naming or AP computation is changed.

The shared implementation is
`/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/measurement_weights.py`.
It depends on NumPy and the standard-library token module. Actual small-data
serial, survey, covariance and two-rank MPI checks and their complete file
inventory are recorded in
`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/weight_refactor_stage3/README.md`.
