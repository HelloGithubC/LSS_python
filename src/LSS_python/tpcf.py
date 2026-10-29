"""Corrfunc measurement entry points for whole catalogs and resampling counts.

Result objects and profile analysis live in LSS_python.xismu. This module owns
ordinary/mocks measurements, raw Jackknife/subsample counters and the existing
Corrfunc-pairs-to-xi dictionary adapter. Resampling policies, cache handling and
statistical reconstruction remain in LSS_python.cov. Legacy result utilities
are direct compatibility exports; only the old xismu class lookup warns.
"""

import warnings

import joblib
import numpy as np
from Corrfunc.theory import DDsmu
from Corrfunc.mocks import DDsmu_mocks
from tqdm import tqdm

from .corrfunc import call_DDsmu
from .pair_counting import PairCounts, build_pair_tasks, count_pair_task, get_neighbors
from .xismu import (
    average_xismus_dict_list, cal_convert_factor, get_diff_array,
    get_diff_array_2d, meannorm, packarray1d, packarray2d,
)

__all__ = [
    "run_tpCF", "run_tpCF_mock", "cal_tpCF_from_pairs", "compute_DD_JK",
    "compute_DR_JK", "compute_RR_JK", "compute_subsample_pairs", "extract_npairs",
    "pair_counts_to_corrfunc",
    "xismu", "meannorm", "packarray1d", "packarray2d", "cal_convert_factor",
    "average_xismus_dict_list", "get_diff_array",
    "get_diff_array_2d",
]

_legacy_xismu_warned = False


def __getattr__(name):
    """Resolve historical class references without wrapping or duplicating it."""
    if name != "xismu":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .xismu import _note_legacy_xismu_reference, xismu as result_class

    _note_legacy_xismu_reference()

    global _legacy_xismu_warned
    if not _legacy_xismu_warned:
        warnings.warn(
            "LSS_python.tpcf.xismu is deprecated; use "
            "'from LSS_python.xismu import xismu'. "
            "For saved object files, run 'migrate_xismu FILE' to switch "
            "to the new layout (overwrites the file in place).",
            FutureWarning,
            stacklevel=2,
        )
        _legacy_xismu_warned = True
    return result_class

def run_tpCF(data_catalog, random_catalog, sedges, mubin, with_weight, run_parts=["all"], refine_factors=(2, 2, 1), output_dict=None, nthreads=1, verbose=False):
    """
    data_catalog & random_catalog: ndarray, three cols: [x, y, z](without weight) or four cols: [x, y, z, weight]
    run_parts: List. Support all, ALL, DD, DR, RR. If including all or ALL, run_parts will be set to ["DD", "DR", "RR"]
    """

    if "all" in run_parts or "ALL" in run_parts:
        run_parts = ["DD", "DR", "RR"]
    if isinstance(run_parts, str):
        run_parts = [run_parts]

    if "DD" in run_parts or "DR" in run_parts:
        if data_catalog.dtype != random_catalog.dtype:
            if data_catalog.dtype == np.float32:
                random_catalog = random_catalog.astype(np.float32)
            elif data_catalog.dtype == np.float64:
                random_catalog = random_catalog.astype(np.float64)
            else:
                raise ValueError("The data type of data_catalog and random_catalog should be np.float32 or np.float64")

    if output_dict is not None:
        output_DD = output_dict.get("DD", None)
        output_DR = output_dict.get("DR", None)
        output_RR = output_dict.get("RR", None)
    else:
        output_DD = None
        output_DR = None
        output_RR = None

    x_refine_factor, y_refine_factor, z_refine_factor = refine_factors
    result_dict = {
        "DD": None,
        "DR": None,
        "RR": None,
    }
    for run_part in run_parts:
        if run_part == "DD":
            if data_catalog is None:
                raise ValueError("DD: data_catalog is not set")
            else:
                autocorr = True
                if verbose:
                    print("Now running DD", flush=True)
                if with_weight:
                    DD_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=data_catalog[:,0], Y1=data_catalog[:,1], Z1=data_catalog[:,2], weights1=data_catalog[:,3], weight_type="pair_product", verbose=verbose, periodic=False, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    DD_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=data_catalog[:,0], Y1=data_catalog[:,1], Z1=data_catalog[:,2], verbose=verbose, periodic=False, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["DD"] = DD_result
                if output_DD is not None:
                    joblib.dump(DD_result, output_DD)

        if run_part == "DR":
            if data_catalog is None or random_catalog is None:
                raise ValueError("DR: data_catalog or random_catalog is not set")
            else:
                if verbose:
                    print("Now running DR", flush=True)
                autocorr = False
                if with_weight:
                    DR_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=data_catalog[:,0], Y1=data_catalog[:,1], Z1=data_catalog[:,2], weights1=data_catalog[:,3], X2=random_catalog[:,0], Y2=random_catalog[:,1], Z2=random_catalog[:,2], weights2=random_catalog[:,3], weight_type="pair_product", verbose = verbose, periodic=False, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    DR_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=data_catalog[:,0], Y1=data_catalog[:,1], Z1=data_catalog[:,2], X2=random_catalog[:,0], Y2=random_catalog[:,1], Z2=random_catalog[:,2], verbose = verbose, periodic=False, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["DR"] = DR_result
                if output_DR is not None:
                    joblib.dump(DR_result, output_DR)

        if run_part == "RR":
            if random_catalog is None:
                raise ValueError("RR: random_catalog is not set")
            else:
                if verbose:
                    print("Now running RR", flush=True)
                autocorr = True
                if with_weight:
                    RR_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=random_catalog[:,0], Y1=random_catalog[:,1], Z1=random_catalog[:,2], weights1=random_catalog[:,3], weight_type="pair_product", verbose = verbose, periodic=False, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                else:
                    RR_result = DDsmu(autocorr, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, X1=random_catalog[:,0], Y1=random_catalog[:,1], Z1=random_catalog[:,2], verbose = verbose, periodic=False, xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["RR"] = RR_result
                if output_RR is not None:
                    joblib.dump(RR_result, output_RR)
    return result_dict

def run_tpCF_mock(mock_catalog, random_catalog, sedges, mubin, with_weight, run_parts=["all"], refine_factors=(2, 2, 1), output_dict=None, nthreads=1, verbose=False):
    """
    mock_catalog & random_catalog: ndarray, three cols: [x, y, z](without weight) or four cols: [x, y, z, weight]
    run_parts: List. Support all, ALL, DD, DR, RR. If including all or ALL, run_parts will be set to ["DD", "DR", "RR"]
    """

    if "all" in run_parts or "ALL" in run_parts:
        run_parts = ["DD", "DR", "RR"]
    if isinstance(run_parts, str):
        run_parts = [run_parts]

    if "DD" in run_parts or "DR" in run_parts:
        if mock_catalog.dtype != random_catalog.dtype:
            if mock_catalog.dtype == np.float32:
                random_catalog = mock_catalog.astype(np.float32, copy=False)
            elif mock_catalog.dtype == np.float64:
                random_catalog = mock_catalog.astype(np.float64, copy=False)
            else:
                raise ValueError("The data type of mock_catalog and random_catalog should be np.float32 or np.float64")

    if output_dict is not None:
        output_DD = output_dict.get("DD", None)
        output_DR = output_dict.get("DR", None)
        output_RR = output_dict.get("RR", None)
    else:
        output_DD = None
        output_DR = None
        output_RR = None

    x_refine_factor, y_refine_factor, z_refine_factor = refine_factors
    result_dict = {
        "DD": None,
        "DR": None,
        "RR": None,
    }

    if with_weight:
        if "DR" in run_parts or "DD" in run_parts:
            mock_weight = mock_catalog[:,3]
            n_columns = mock_catalog.shape[1] - 4
            for i in range(n_columns):
                mock_weight *= mock_catalog[:,4+i]
        if "DR" in run_parts or "RR" in run_parts:
            random_weight = random_catalog[:,3]
            n_columns = random_catalog.shape[1] - 4
            for i in range(n_columns):
                random_weight *= random_catalog[:,4+i]
    else:
        if "DR" in run_parts or "DD" in run_parts:
            mock_weight = mock_catalog[:,3]
        if "DR" in run_parts or "RR" in run_parts:
            random_weight = random_catalog[:,3]

    for run_part in run_parts:
        if run_part == "DD":
            if mock_catalog is None:
                raise ValueError("DD: mock_catalog is not set")
            else:
                autocorr = True
                if verbose:
                    print("Now running DD", flush=True)
                DD_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=mock_catalog[:,0], DEC1=mock_catalog[:,1], CZ1=mock_catalog[:,2], is_comoving_dist=True, weights1=mock_weight, weight_type="pair_product", verbose=verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["DD"] = DD_result
                if output_DD is not None:
                    joblib.dump(DD_result, output_DD)

        if run_part == "DR":
            if mock_catalog is None or random_catalog is None:
                raise ValueError("DR: mock_catalog or random_catalog is not set")
            else:
                if verbose:
                    print("Now running DR", flush=True)
                autocorr = False
                DR_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=mock_catalog[:,0], DEC1=mock_catalog[:,1], CZ1=mock_catalog[:,2], is_comoving_dist=True, weights1=mock_weight, RA2=random_catalog[:,0], DEC2=random_catalog[:,1], CZ2=random_catalog[:,2], weights2=random_weight, weight_type="pair_product", verbose = verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["DR"] = DR_result
                if output_DR is not None:
                    joblib.dump(DR_result, output_DR)

        if run_part == "RR":
            if random_catalog is None:
                raise ValueError("RR: random_catalog is not set")
            else:
                if verbose:
                    print("Now running RR", flush=True)
                autocorr = True
                RR_result = DDsmu_mocks(autocorr, cosmology=1, nthreads=nthreads, binfile=sedges, mu_max=1.0, nmu_bins=mubin, RA1=random_catalog[:,0], DEC1=random_catalog[:,1], CZ1=random_catalog[:,2], is_comoving_dist=True, weights1=random_weight, weight_type="pair_product", verbose = verbose,  xbin_refine_factor=x_refine_factor, ybin_refine_factor=y_refine_factor, zbin_refine_factor=z_refine_factor)
                result_dict["RR"] = RR_result
                if output_RR is not None:
                    joblib.dump(RR_result, output_RR)
    return result_dict

def cal_tpCF_from_pairs(DD_result, DR_result, RR_result, data, random, sbin, mubin, with_weight=False):
    result_dict = {}
    result_dict["DDnpairs"] = DD_result["npairs"].reshape(sbin, mubin)
    result_dict["DRnpairs"] = DR_result["npairs"].reshape(sbin, mubin)
    result_dict["RRnpairs"] = RR_result["npairs"].reshape(sbin, mubin)
    if with_weight:
        for key_str, result_temp in zip(["DDwpairs", "DRwpairs", "RRwpairs"], [DD_result, DR_result, RR_result]):
            if np.all(result_temp["weightavg"] == 0):
                result_dict[key_str] = result_temp["npairs"].reshape(sbin, mubin)
            else:
                result_dict[key_str] = (result_temp["npairs"] * result_temp["weightavg"]).reshape(sbin, mubin)
        result_dict["with_weight"] = True
    else:
        result_dict["with_weight"] = False

    if with_weight:
        DD = result_dict["DDwpairs"]
        DR = result_dict["DRwpairs"]
        RR = result_dict["RRwpairs"]
    else:
        DD = result_dict["DDnpairs"]
        DR = result_dict["DRnpairs"]
        RR = result_dict["RRnpairs"]

    result_dict["sedges"] = np.append(DD_result["smin"].reshape(sbin, mubin)[:,0], DD_result["smax"].reshape(sbin, mubin)[-1,0])
    mumax_str = "mumax" if "mumax" in DD_result.dtype.names else "mu_max"
    result_dict["muedges"] = np.append([0], DD_result[mumax_str].reshape(sbin, mubin)[0])

    dataNum = data.shape[0]
    randomNum = random.shape[0]
    if with_weight:
        randomWeight = random[:,3]
        dataWeight = data[:,3]

    sum_wd = np.sum(dataWeight) if with_weight else dataNum
    sum_wr = np.sum(randomWeight) if with_weight else randomNum
    sum_wd2 = np.sum(dataWeight**2) if with_weight else dataNum
    sum_wr2 = np.sum(randomWeight**2) if with_weight else randomNum
    sum_wr_2 = sum_wr
    sum_wd_2 = sum_wd

    norm_d1d2 = sum_wd * sum_wd - sum_wd2
    norm_r1r2 = sum_wr * sum_wr - sum_wr2
    norm_d1r2 = sum_wd * sum_wr_2
    norm_r1d2 = sum_wr * sum_wd_2

    if abs(result_dict["sedges"][0] - 0.0) < 1e-10 and abs(result_dict["muedges"][0] - 0.0) < 1e-10:
        DD[0,0] -= sum_wd2
        RR[0,0] -= sum_wr2

    dd1d2 = DD / norm_d1d2
    dd1r2 = DR / norm_d1r2
    dr1d2 = DR / norm_r1d2
    dr1r2 = RR / norm_r1r2

    dr1r2_remove_0 = np.copy(dr1r2)
    dr1r2_remove_0[dr1r2_remove_0 == 0] = 1e-15

    result_dict.update({
        "sum_wd": sum_wd,
        "sum_wr": sum_wr,
        "sum_wd2": sum_wd2,
        "sum_wr2": sum_wr2,
        "norm_d1d2": norm_d1d2,
        "norm_r1r2": norm_r1r2,
        "norm_d1r2": norm_d1r2,
        "norm_r1d2": norm_r1d2,
        "tpCF": (dd1d2 - dr1d2 - dd1r2 + dr1r2) / dr1r2_remove_0,
    })

    return result_dict

def pair_counts_to_corrfunc(counts):
    """Adapt raw PairCounts to the legacy flat DDsmu structured result.

    No multiplicity, normalization, zero-weight fallback, or self-pair
    correction is applied. Task callers supply their own explicit scaling.
    """
    if not isinstance(counts, PairCounts):
        raise TypeError("counts must be PairCounts")
    result = np.zeros(counts.npairs.size, dtype=[
        ("smin", np.float64), ("smax", np.float64), ("savg", np.float64),
        ("mu_max", np.float64), ("npairs", np.uint64), ("weightavg", np.float64),
    ])
    mubin = counts.npairs.shape[1]
    result["smin"] = np.repeat(counts.sedges[:-1], mubin)
    result["smax"] = np.repeat(counts.sedges[1:], mubin)
    result["mu_max"] = np.tile(np.arange(1, mubin + 1) * (1.0 / mubin),
                               len(counts.sedges) - 1)
    result["npairs"] = counts.npairs.ravel()
    if counts.with_weight:
        np.divide(counts.wpairs.ravel(), result["npairs"],
                  out=result["weightavg"], where=result["npairs"] != 0)
    return result


def _prepare_jackknife_tasks(pair_type, data_list, random_list, sedges,
                             with_weight, boxsize, ngrids):
    """Share region selection and geometry checks between serial and MPI paths."""
    grid = np.asarray(ngrids)
    if grid.ndim == 0:
        grid = np.repeat(grid, 3)
    if (grid.shape != (3,) or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid <= 0)):
        raise ValueError("ngrids must contain one or three positive integers")
    # The existing covariance reconstruction uses one scalar grid division.
    if np.any(grid != grid[0]):
        raise ValueError("Jackknife reconstruction currently requires a cubic grid")
    box = np.asarray(boxsize, dtype=np.float64)
    if box.ndim == 0:
        box = np.repeat(box, 3)
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0):
        raise ValueError("boxsize must contain one or three finite positive lengths")
    edges = np.asarray(sedges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("sedges must be a one-dimensional array of bin edges")
    if edges[-1] > np.min(box / grid):
        raise ValueError("smax must not exceed the smallest block width for Jackknife counting")

    n_cubes = int(np.prod(grid))
    empty = np.empty((0, 4 if with_weight else 3), dtype=np.float64)
    data_blocks = ((empty,) * n_cubes if data_list is None else
                   tuple(empty if block is None else block for block in data_list))
    random_blocks = ((empty,) * n_cubes if random_list is None else
                     tuple(empty if block is None else block for block in random_list))
    first_blocks = random_blocks if pair_type == "RR" else data_blocks
    tasks = sorted(
        (task for task in build_pair_tasks(data_blocks, random_blocks, grid)
         if task.pair_type == pair_type
         and (not task.autocorr or len(first_blocks[task.first]) >= 2)),
        key=lambda task: (task.first != task.second, task.first, task.second),
    )
    return tasks, data_blocks, random_blocks


def _compute_jackknife_pairs(pair_type, data_list, random_list, sedges, mubin,
                             with_weight, boxsize, refine_factors, nthreads,
                             verbose, ngrids):
    """Evaluate public block tasks and preserve the legacy internal/cross maps."""
    tasks, data_blocks, random_blocks = _prepare_jackknife_tasks(
        pair_type, data_list, random_list, sedges, with_weight, boxsize, ngrids,
    )
    internal = dict.fromkeys(range(len(data_blocks)))
    cross = {}
    for task in tqdm(tasks, desc=f"Jackknife {pair_type}", disable=not verbose):
        _, counts = count_pair_task(
            task, data_blocks, random_blocks, sedges, mubin, with_weight, boxsize,
            nthreads=nthreads, refine_factors=refine_factors,
        )
        result = pair_counts_to_corrfunc(counts)
        # Legacy cross maps already contain ordered DD/RR counts. Leave the
        # public core result raw and scale only the adapted output, once.
        result["npairs"] *= task.multiplicity
        if task.first == task.second:
            internal[task.first] = result
        else:
            cross[(task.first, task.second)] = result
    return internal, cross


def compute_DD_JK(data_list, sedges, mubin, with_weight, boxsize,
               refine_factors, nthreads, verbose, ngrids):
    """
    Compute DD for data catalog.
    Computes internal pairs for each subvolume + cross-pairs between neighboring subvolumes.
    This ensures the same geometric structure as RR.

    Parameters
    ----------
    data_list : list of ndarray
        List of n^3 subvolume data arrays.
    sedges : array-like
        Separation bin edges.
    mubin : int
        Number of mu bins.
    with_weight : bool
        Whether to use weights.
    boxsize : float
        Box size.
    refine_factors : tuple
        (x_refine, y_refine, z_refine) factors.
    nthreads : int
        Number of threads.
    verbose : bool
        Whether to print progress.
    ngrids : array-like
        Number of divisions along each axis.

    Returns
    -------
    DD_internal : dict
        Dictionary mapping subvolume index to DD result (internal pairs).
    DD_cross : dict
        Dictionary mapping (idx1, idx2) tuple to DD result (cross pairs).
        Only includes pairs where idx1 < idx2.
    """

    return _compute_jackknife_pairs(
        "DD", data_list, None, sedges, mubin, with_weight, boxsize,
        refine_factors, nthreads, verbose, ngrids,
    )

def compute_DR_JK(data_list, random_list, sedges, mubin, with_weight, boxsize,
                         refine_factors, nthreads, verbose, ngrids):
    """
    Compute DR for data-random cross-correlation.
    Computes internal pairs (same position) + cross-pairs between neighboring subvolumes.
    This ensures the same geometric structure as DD and RR.

    Parameters
    ----------
    data_list : list of ndarray
        List of n^3 subvolume data arrays.
    random_list : list of ndarray
        List of n^3 subvolume random arrays (same spatial division as data).
    sedges : array-like
        Separation bin edges.
    mubin : int
        Number of mu bins.
    with_weight : bool
        Whether to use weights.
    boxsize : float
        Box size.
    refine_factors : tuple
        (x_refine, y_refine, z_refine) factors.
    nthreads : int
        Number of threads.
    verbose : bool
        Whether to print progress.
    ngrids : array-like
        Number of divisions along each axis.

    Returns
    -------
    DR_internal : dict
        Dictionary mapping subvolume index to DR result (data_i vs random_i).
    DR_cross : dict
        Dictionary mapping (idx1, idx2) tuple to DR result (data_i vs random_j).
        Both directed region pairs (idx1, idx2) and (idx2, idx1) are retained.
    """

    return _compute_jackknife_pairs(
        "DR", data_list, random_list, sedges, mubin, with_weight, boxsize,
        refine_factors, nthreads, verbose, ngrids,
    )

def compute_RR_JK(random_list, sedges, mubin, with_weight, boxsize,
                refine_factors, nthreads, verbose, ngrids):
    """
    Compute RR for random catalog.
    Computes internal pairs for each subvolume + cross-pairs between neighboring subvolumes.
    This ensures the same geometric structure as DD.

    Parameters
    ----------
    random_list : list of ndarray
        List of n^3 subvolume random arrays.
    sedges : array-like
        Separation bin edges.
    mubin : int
        Number of mu bins.
    with_weight : bool
        Whether to use weights.
    boxsize : float
        Box size.
    refine_factors : tuple
        (x_refine, y_refine, z_refine) factors.
    nthreads : int
        Number of threads.
    verbose : bool
        Whether to print progress.
    ngrids : array-like
        Number of divisions along each axis.

    Returns
    -------
    RR_internal : dict
        Dictionary mapping subvolume index to RR result (internal pairs).
    RR_cross : dict
        Dictionary mapping (idx1, idx2) tuple to RR result (cross pairs).
        Only includes pairs where idx1 < idx2.
    RR_full : ndarray
        Combined RR full result as flattened array (sum of internal + cross pairs).
    """

    internal, cross = _compute_jackknife_pairs(
        "RR", None, random_list, sedges, mubin, with_weight, boxsize,
        refine_factors, nthreads, verbose, ngrids,
    )
    # Preserve per-component weight compatibility and the legacy cache format.
    total = np.zeros((len(sedges) - 1, mubin), dtype=np.float64)
    for results in (internal, cross):
        for result in results.values():
            if result is not None:
                total += extract_npairs(result, with_weight).reshape(total.shape)
    return internal, cross, total.ravel()

def extract_npairs(result, with_weight):
    """
    Extract npairs and weighted pairs from DDsmu result.

    Parameters
    ----------
    result : structured array, simple ndarray, or dict of these
        DDsmu result. Can be:
        - Structured array with 'npairs' and 'weightavg' fields
        - Simple numpy array (already flattened)
        - Dict of the above types (e.g., RR_internal, RR_cross)
    with_weight : bool
        Whether to use weights.

    Returns
    -------
    npairs : ndarray or dict of ndarray
        Number of pairs (or weighted pairs). Returns None if result is None.
    """
    if result is None:
        return None

    if isinstance(result, dict):
        # result is a dict like RR_internal or RR_cross
        return {k: extract_npairs(v, with_weight) for k, v in result.items()}

    if isinstance(result, np.ndarray) and result.dtype.names is not None:
        # result is a structured array
        npairs = result['npairs']

        if with_weight:
            weightavg = result['weightavg']
            if not np.all(weightavg == 0):
                npairs = npairs * weightavg

        return npairs
    else:
        # result is already a simple array (from extract_npairs or direct computation)
        return result

def compute_subsample_pairs(
    data_list, random, sedges, mubin, with_weight, boxsize,
    refine_factors=(2, 2, 1), nthreads=1, verbose=False, *,
    run_parts=("DD", "DR", "RR"),
):
    """Count raw pairs for already prepared independent subvolumes.

    ``random`` is one subvolume-sized catalog shared by the data subvolumes.
    ``boxsize`` is the subvolume size passed to the existing Corrfunc adapter.
    Requested DD/DR entries retain the legacy None value for subvolumes with
    fewer than two points. Unrequested dictionaries are empty and an
    unrequested RR_result is None. No cache I/O, normalization, self-pair
    subtraction, weight fallback, or xismu construction occurs here.

    Request RR separately when its cache must be saved before counting DD/DR.
    Catalog precision and coordinates must already be prepared by the caller.
    """
    parts = (run_parts,) if isinstance(run_parts, str) else tuple(run_parts)
    if any(not isinstance(part, str) for part in parts):
        raise ValueError("run_parts must contain DD, DR, RR, or all")
    parts = {part.upper() for part in parts}
    if parts - {"DD", "DR", "RR", "ALL"}:
        raise ValueError("run_parts must contain DD, DR, RR, or all")
    if "ALL" in parts:
        parts = {"DD", "DR", "RR"}
    if parts & {"DD", "DR"} and data_list is None:
        raise ValueError("data_list is required for DD or DR")
    if parts & {"DR", "RR"} and random is None:
        raise ValueError("random is required for DR or RR")

    result = {"DD_sub": {}, "DR_sub": {}, "RR_result": None}
    if "RR" in parts:
        result["RR_result"] = call_DDsmu(
            random, None, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=True,
        )
    if not parts & {"DD", "DR"}:
        return result

    iterator = tqdm(
        enumerate(data_list), total=len(data_list),
        desc="Processing subvolumes (DD+DR)", disable=not verbose,
    )
    for idx, data_sub in iterator:
        for part in ("DD", "DR"):
            if part not in parts:
                continue
            key = part + "_sub"
            if data_sub is None or len(data_sub) < 2:
                result[key][idx] = None
                continue
            result[key][idx] = call_DDsmu(
                data_sub, None if part == "DD" else random,
                sedges, mubin, with_weight, boxsize, refine_factors,
                nthreads, autocorr=part == "DD",
            )
    return result
