"""Non-periodic Corrfunc counting, from individual point sets to block workflows.

The public API is organized as result/task types, two counting cores, peripheral
tools, and a complete serial workflow. The dependency direction is
``count_pairs_in_blocks -> peripheral tools -> count_pairs / merge_pair_counts``.
The cores do not select blocks, interpret task IDs, normalize counts, or apply
weight compatibility rules. Callers can use them without the block workflow.

Raw histograms retain Corrfunc's native self-pairs at zero separation. No MPI,
Jackknife covariance correction, or self-pair subtraction is performed here.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from .corrfunc import call_DDsmu


__all__ = [
    "PAIR_TYPES", "PairCounts", "PairTask", "count_pairs", "merge_pair_counts",
    "prepare_catalogs", "split_catalog_into_blocks", "get_neighbors", "build_pair_tasks",
    "count_pair_task", "sum_pair_counts", "pair_normalizations",
    "finalize_pair_counts", "count_pairs_in_blocks",
]

PAIR_TYPES = ("DD", "DR", "RR")


# Result and task types.

@dataclass(frozen=True)
class PairCounts:
    """Raw 2D counts and their binning, with no task or normalization state.

    Arrays are copied on construction. ``wpairs`` contains raw product-weight
    sums when weighted, and a floating copy of ``npairs`` otherwise. No all-zero
    fallback is applied; that belongs to :func:`finalize_pair_counts`.
    """

    npairs: np.ndarray
    wpairs: np.ndarray
    sedges: np.ndarray
    muedges: np.ndarray
    with_weight: bool = False

    def __post_init__(self):
        npairs = np.asarray(self.npairs)
        if (npairs.ndim != 2 or not np.issubdtype(npairs.dtype, np.integer)
                or np.any(npairs < 0)):
            raise ValueError("npairs must be a non-negative 2D integer array")
        wpairs = np.asarray(self.wpairs)
        if np.iscomplexobj(wpairs):
            raise ValueError("wpairs must be real-valued")
        wpairs = np.asarray(wpairs, dtype=np.float64)
        sedges = _edges(self.sedges, "sedges")
        muedges = _edges(self.muedges, "muedges")
        shape = (len(sedges) - 1, len(muedges) - 1)
        if npairs.shape != shape or wpairs.shape != shape:
            raise ValueError("histogram shape does not match the binning")
        if not isinstance(self.with_weight, (bool, np.bool_)):
            raise ValueError("with_weight must be boolean")
        object.__setattr__(self, "npairs", npairs.astype(np.uint64, copy=True))
        object.__setattr__(self, "wpairs", wpairs.copy())
        object.__setattr__(self, "sedges", sedges.copy())
        object.__setattr__(self, "muedges", muedges.copy())
        object.__setattr__(self, "with_weight", bool(self.with_weight))


@dataclass(frozen=True)
class PairTask:
    """A block-workflow task; its multiplicity depends on task coverage."""

    pair_type: str
    first: int
    second: int

    def __post_init__(self):
        if self.pair_type not in PAIR_TYPES:
            raise ValueError("pair_type must be DD, DR, or RR")
        if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
               or n < 0 for n in (self.first, self.second)):
            raise ValueError("block IDs must be non-negative integers")
        if self.pair_type != "DR" and self.first > self.second:
            raise ValueError("DD/RR tasks require first <= second")

    @property
    def autocorr(self):
        return self.pair_type != "DR" and self.first == self.second

    @property
    def multiplicity(self):
        # This workflow submits each unordered DD/RR block pair only once.
        return 2 if self.pair_type != "DR" and not self.autocorr else 1


# Core operations: no catalog preparation, spatial selection, or finalization.

def count_pairs(data1, data2=None, *, autocorr, sedges, mubin,
                with_weight=False, boxsize=None, nthreads=1,
                refine_factors=(2, 2, 1)):
    """Count one supplied point set or point-set pair and return PairCounts.

    ``autocorr=True`` requires data2=None; otherwise data2 is required. Inputs
    must already be real float32/float64 catalogs of matching precision, with
    a fourth weight column when weights are enabled. Use prepare_catalogs for
    coercion or unit weights. Coordinates may be negative: no block geometry,
    position range, or block-width constraint is imposed by this core.
    """
    edges, muedges = _binning(sedges, mubin)
    if not isinstance(autocorr, (bool, np.bool_)):
        raise ValueError("autocorr must be boolean")
    if autocorr and data2 is not None:
        raise ValueError("data2 must be None for autocorrelation")
    if not autocorr and data2 is None:
        raise ValueError("data2 is required for cross-correlation")
    _positive_integer(nthreads, "nthreads")
    if len(refine_factors) != 3:
        raise ValueError("refine_factors must contain three positive integers")
    for factor in refine_factors:
        _positive_integer(factor, "refine_factors")
    data1 = _count_catalog(data1, with_weight)
    if not autocorr:
        data2 = _count_catalog(data2, with_weight)
        if data1.dtype != data2.dtype:
            raise ValueError("catalog precisions must match; use prepare_catalogs first")
    shape = (len(edges) - 1, mubin)
    if len(data1) == 0 or (not autocorr and len(data2) == 0):
        return PairCounts(np.zeros(shape, dtype=np.uint64), np.zeros(shape),
                          edges, muedges, with_weight)
    result = call_DDsmu(
        data1, data2, edges, mubin, with_weight, boxsize,
        refine_factors=refine_factors, nthreads=nthreads, autocorr=autocorr,
    )
    npairs = result["npairs"].reshape(shape)
    wpairs = npairs.astype(np.float64)
    if with_weight:
        wpairs *= result["weightavg"].reshape(shape)
    return PairCounts(npairs, wpairs, edges, muedges, with_weight)


def merge_pair_counts(contributions, *, sedges, mubin, with_weight=False):
    """Merge named streams of (PairCounts, non-negative integer multiplicity).

    Keys are DD/DR/RR; omitted or empty streams produce zero histograms. Only
    explicitly supplied multiplicities are applied. Exact bin edges and weight
    modes must match. Inputs are untouched and outputs remain raw PairCounts,
    suitable for subsequent merges without premature compatibility fallback.
    """
    edges, muedges = _binning(sedges, mubin)
    if set(contributions) - set(PAIR_TYPES):
        raise ValueError("contribution keys must be DD, DR, or RR")
    shape = (len(edges) - 1, mubin)
    merged = {}
    for name in PAIR_TYPES:
        npairs = np.zeros(shape, dtype=np.uint64)
        wpairs = np.zeros(shape, dtype=np.float64)
        for counts, multiplicity in contributions.get(name, ()):
            if not isinstance(counts, PairCounts):
                raise ValueError("each contribution must contain PairCounts")
            if (not isinstance(multiplicity, (int, np.integer))
                    or isinstance(multiplicity, (bool, np.bool_)) or multiplicity < 0):
                raise ValueError("multiplicity must be a non-negative integer")
            if (counts.npairs.shape != shape or counts.wpairs.shape != shape
                    or not np.array_equal(counts.sedges, edges)
                    or not np.array_equal(counts.muedges, muedges)):
                raise ValueError("contribution binning does not match the requested binning")
            if counts.with_weight != with_weight:
                raise ValueError("contribution weight mode does not match with_weight")
            npairs += int(multiplicity) * counts.npairs
            wpairs += int(multiplicity) * counts.wpairs
        merged[name] = PairCounts(npairs, wpairs, edges, muedges, with_weight)
    return merged


# Peripheral tools: input preparation, spatial tasks, and result policies.

def _edges(values, name):
    values = np.asarray(values)
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values))
            or values[0] < 0 or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} must be finite, non-negative, and strictly increasing")
    return values


def _positive_integer(value, name):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or value < 1):
        raise ValueError(f"{name} must be a positive integer")


def _binning(sedges, mubin):
    _positive_integer(mubin, "mubin")
    return _edges(sedges, "sedges"), np.linspace(0.0, 1.0, mubin + 1)


def _count_catalog(data, with_weight):
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] not in (3, 4):
        raise ValueError("catalogs must have shape (N, 3) or (N, 4)")
    if data.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("catalogs must have real float32/float64 dtype; use prepare_catalogs")
    if with_weight and data.shape[1] != 4:
        raise ValueError("weighted catalogs need a fourth column; use prepare_catalogs")
    if np.any(~np.isfinite(data[:, :4 if with_weight else 3])):
        raise ValueError("positions and enabled weights must be finite")
    return data


def prepare_catalogs(data, random=None, *, with_weight=False):
    """Prepare a catalog pair with common precision and optional unit weights.

    Return (data, random), retaining None when no second catalog is supplied.
    No spatial range is imposed. Preparation never edits the input arrays.
    """
    catalogs = [np.asarray(data)]
    if random is not None:
        catalogs.append(np.asarray(random))
    for catalog in catalogs:
        if catalog.ndim != 2 or catalog.shape[1] not in (3, 4):
            raise ValueError("catalogs must have shape (N, 3) or (N, 4)")
        if np.iscomplexobj(catalog):
            raise ValueError("catalog positions and weights must be real-valued")
    dtype = np.result_type(*(catalog.dtype for catalog in catalogs))
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        dtype = np.dtype("float64")
    prepared = []
    for catalog in catalogs:
        catalog = np.asarray(catalog, dtype=dtype)
        if with_weight and catalog.shape[1] == 3:
            catalog = np.column_stack((catalog, np.ones(len(catalog), dtype=dtype)))
        prepared.append(_count_catalog(catalog, with_weight))
    return prepared[0], prepared[1] if random is not None else None


def _grid_shape(ngrids):
    grid = np.asarray(ngrids)
    if grid.ndim == 0:
        grid = np.repeat(grid, 3)
    if (grid.shape != (3,) or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid <= 0)):
        raise ValueError("ngrids must be a positive integer or three positive integers")
    return tuple(int(n) for n in grid)


def _geometry(boxsize, ngrids):
    box = np.asarray(boxsize, dtype=np.float64)
    if box.ndim == 0:
        box = np.repeat(box, 3)
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0):
        raise ValueError("boxsize must contain one or three finite positive lengths")
    return box, _grid_shape(ngrids)


def split_catalog_into_blocks(data, boxsize, ngrids, with_weight=False):
    """Return C-ordered blocks with global coordinates, retaining empty blocks.

    Only this spatial workflow requires positions in [0, boxsize]. Upper-edge
    positions belong to the last block. Missing weights are filled with ones.
    """
    box, shape = _geometry(boxsize, ngrids)
    data, _ = prepare_catalogs(data, with_weight=with_weight)
    if np.any((data[:, :3] < 0) | (data[:, :3] > box)):
        raise ValueError("non-periodic block positions must lie within [0, boxsize]")
    cells = np.floor(data[:, :3] / (box / np.asarray(shape))).astype(np.int64)
    cells = np.minimum(cells, np.asarray(shape) - 1)
    labels = np.ravel_multi_index(cells.T, shape)
    order = np.argsort(labels, kind="stable")
    ordered = np.ascontiguousarray(data[order])
    offsets = np.concatenate(([0], np.cumsum(np.bincount(labels, minlength=np.prod(shape)))))
    return tuple(ordered[start:stop] for start, stop in zip(offsets[:-1], offsets[1:]))


def get_neighbors(idx, n):
    """
    Get all neighboring subvolume indices for a given subvolume.
    Includes all 26 neighbors (face, edge, and corner adjacent).

    Parameters
    ----------
    idx : int
        Linear index of the subvolume.
    n : int
        Number of divisions along each axis.

    Returns
    -------
    neighbors : list of int
        List of neighboring subvolume indices.
    """
    ix = idx // (n * n)
    iy = (idx // n) % n
    iz = idx % n

    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = ix + dx, iy + dy, iz + dz
                # Check boundaries (no periodic)
                if 0 <= nx < n and 0 <= ny < n and 0 <= nz < n:
                    neighbor_idx = nx * n * n + ny * n + nz
                    neighbors.append(neighbor_idx)

    return neighbors


def build_pair_tasks(data_blocks, random_blocks, ngrids):
    """Describe internal/adjacent tasks without running Corrfunc.

    The caller enforces smax <= min(boxsize / ngrids). DD/RR unordered block
    pairs occur once; DR retains both block directions. Empty inputs are skipped.
    """
    shape = _grid_shape(ngrids)
    nblocks = int(np.prod(shape))
    if len(data_blocks) != nblocks or len(random_blocks) != nblocks:
        raise ValueError("block lists must match ngrids")
    tasks = []
    for pair_type, first_blocks, second_blocks in (
        ("DD", data_blocks, data_blocks),
        ("DR", data_blocks, random_blocks),
        ("RR", random_blocks, random_blocks),
    ):
        for first, block in enumerate(first_blocks):
            if len(block) == 0:
                continue
            cell = np.unravel_index(first, shape)
            ranges = [range(max(0, c - 1), min(n, c + 2)) for c, n in zip(cell, shape)]
            for neighbor in product(*ranges):
                second = int(np.ravel_multi_index(neighbor, shape))
                if pair_type != "DR" and second < first:
                    continue
                if len(second_blocks[second]):
                    tasks.append(PairTask(pair_type, first, second))
    return tuple(tasks)


def count_pair_task(task, data_blocks, random_blocks, sedges, mubin,
                    with_weight, boxsize, nthreads=1, refine_factors=(2, 2, 1)):
    """Select task inputs and return (task, raw PairCounts), without scaling."""
    first_blocks = random_blocks if task.pair_type == "RR" else data_blocks
    second_blocks = data_blocks if task.pair_type == "DD" else random_blocks
    counts = count_pairs(
        first_blocks[task.first], None if task.autocorr else second_blocks[task.second],
        autocorr=task.autocorr, sedges=sedges, mubin=mubin, with_weight=with_weight,
        boxsize=boxsize, nthreads=nthreads, refine_factors=refine_factors,
    )
    return task, counts


def sum_pair_counts(task_results, sedges, mubin, with_weight=False):
    """Reject duplicate tasks and pass explicit workflow multiplicities to merge.

    Results may arrive in any order. This returns raw PairCounts; finalization
    must wait until the complete DD/DR/RR streams have been merged.
    """
    contributions = {name: [] for name in PAIR_TYPES}
    seen = set()
    for task, counts in task_results:
        if task in seen:
            raise ValueError(f"duplicate pair task: {task}")
        seen.add(task)
        contributions[task.pair_type].append((counts, task.multiplicity))
    return merge_pair_counts(contributions, sedges=sedges, mubin=mubin,
                             with_weight=with_weight)


def pair_normalizations(data, random, with_weight=False):
    """Return global ordered-pair normalizations, excluding identical objects."""
    data, random = prepare_catalogs(data, random, with_weight=with_weight)

    def weight_sums(catalog):
        if not with_weight:
            return float(len(catalog)), float(len(catalog))
        weights = catalog[:, 3].astype(np.float64)
        return float(np.sum(weights)), float(np.sum(weights * weights))

    wd, qd = weight_sums(data)
    wr, qr = weight_sums(random)
    return dict(zip(PAIR_TYPES, (wd * wd - qd, wd * wr, wr * wr - qr)))


def finalize_pair_counts(counts):
    """Copy completed histograms to dictionaries, applying weight compatibility.

    For each complete DD/DR/RR result, an entirely zero weighted histogram falls
    back to unweighted counts. Apply this only after the last merge, never to
    intermediate results. Raw PairCounts and their bin metadata remain intact.
    """
    finalized = {}
    for name, histogram in counts.items():
        npairs = histogram.npairs.copy()
        wpairs = histogram.wpairs.copy()
        if not histogram.with_weight or not np.any(wpairs != 0):
            wpairs = npairs.astype(np.float64)
        finalized[name] = {"npairs": npairs, "wpairs": wpairs}
    return finalized


# Complete experimental workflow; existing production paths do not call it.

def count_pairs_in_blocks(data, random, sedges, mubin, boxsize, ngrids,
                          with_weight=False, nthreads=1, refine_factors=(2, 2, 1)):
    """Run the serial adjacent-block workflow and return complete DD/DR/RR.

    The largest separation must not exceed the smallest block width. Global
    histogram dictionaries contain npairs, finalized wpairs, and normalization.
    Other fields record binning, geometry, populations, tasks, and their raw
    PairCounts. Normalizations exclude self-pairs; native zero-bin self-pairs
    remain in the histograms and require separate physical-estimator handling.
    """
    box, shape = _geometry(boxsize, ngrids)
    edges, muedges = _binning(sedges, mubin)
    if edges[-1] > np.min(box / np.asarray(shape)):
        raise ValueError("smax must not exceed the smallest block width for adjacent-block counting")
    _positive_integer(nthreads, "nthreads")
    data, random = prepare_catalogs(data, random, with_weight=with_weight)
    data_blocks = split_catalog_into_blocks(data, box, shape, with_weight)
    random_blocks = split_catalog_into_blocks(random, box, shape, with_weight)
    tasks = build_pair_tasks(data_blocks, random_blocks, shape)
    task_counts = dict(count_pair_task(
        task, data_blocks, random_blocks, edges, mubin, with_weight, box,
        nthreads=nthreads, refine_factors=refine_factors,
    ) for task in tasks)
    merged = sum_pair_counts(task_counts.items(), edges, mubin, with_weight)
    counts = finalize_pair_counts(merged)
    for name, normalization in pair_normalizations(data, random, with_weight).items():
        counts[name]["normalization"] = normalization
    counts.update({
        "sedges": edges.copy(), "muedges": muedges,
        "boxsize": box.copy(), "ngrids": np.asarray(shape),
        "with_weight": bool(with_weight), "tasks": tasks, "task_counts": task_counts,
        "block_sizes": {"data": np.array([len(block) for block in data_blocks]),
                        "random": np.array([len(block) for block in random_blocks])},
    })
    return counts
