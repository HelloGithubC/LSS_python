import numpy as np 
from numba import njit, prange, set_num_threads
from scipy.spatial import KDTree
from tqdm import tqdm
import freud

from LSS_python.base import Hz, DA

@njit 
def w_kernel_jit(d_array, h):
    w_array = np.zeros(d_array.shape, dtype=np.float64)
    for i in range(d_array.shape[0]):
        sigma = 1.0 / (np.pi * h**3) 
        q = d_array[i] / h
        if q < 0.0:
            w_array[i] = 0.0
        elif q <= 1.0:
            w_array[i] = sigma * (0.25 *(2.0-q)**3 - (1.0-q)**3)
        elif q < 2.0:
            w_array[i] = sigma * 0.25 *(2.0-q)**3
        else:
            w_array[i] = 0.0
    return w_array

@njit
def dw_kernel_jit(d_array, h):
    dw_array = np.zeros(d_array.shape, dtype=np.float64)
    for i in range(d_array.shape[0]):
        sigma = 1.0 / (np.pi * h**4) 
        q = d_array[i] / h
        if q < 0.0:
            dw_array[i] = 0.0
        elif q <= 1.0:
            dw_array[i] = sigma * (0.25*3.0 * (2.0-q)**2 - 3.0*(1.0-q)**2)
        elif q < 2.0:
            dw_array[i] = sigma * 0.25 * 3.0 * (2.0-q)**2
        else:
            dw_array[i] = 0.0
    return dw_array

@njit
def cal_rho_array(distance_array, h, use_adaptive_h=True):
    """ cal rho to get MCF
    Note:
        distance_array: [n_galaxy, n_neighbor]
        h: The fixed smoothing length. Only used when use_adaptive_h is False
    """
    rho_array = np.zeros(distance_array.shape[0], dtype=np.float64)
    for i in range(distance_array.shape[0]):
        if use_adaptive_h:
            h_need = np.max(distance_array[i]) / 2.0
        else:
            h_need = h
        rho_temp = w_kernel_jit(distance_array[i], h_need)
        rho_array[i] = np.sum(rho_temp)
    return rho_array

@njit(parallel=True)
def cal_rho_ragged(distance_array, offsets, h):
    """cal rho for ragged neighbor counts, summed per galaxy (parallel).

    Note:
        distance_array: [n_total_neighbors] flattened distances.
        offsets: [n_galaxy + 1] boundaries; offsets[i]:offsets[i+1]
            selects the neighbors of galaxy i.
        h: The fixed smoothing length.
        Galaxies are processed in parallel via prange; the number of
        threads is controlled by numba.set_num_threads.
    """
    rho_array = np.zeros(offsets.shape[0] - 1, dtype=np.float64)
    for i in prange(rho_array.shape[0]):
        start = offsets[i]
        end = offsets[i + 1]
        rho_temp = w_kernel_jit(distance_array[start:end], h)
        rho_array[i] = np.sum(rho_temp)
    return rho_array

@njit(parallel=True)
def cal_rho_from_pos(query_pos, data_pos, flat_idx, offsets, h, boxsize3):
    """cal rho directly from neighbor indices and positions (parallel).

    Distances are computed per galaxy inside the parallel loop, so no large
    intermediate distance array is materialized and no single-threaded numpy
    stage (repeat/diff/norm) is needed. Periodic wrap is applied when the
    corresponding boxsize3 component is positive.

    Note:
        query_pos: [n_query, 3] positions whose densities are calculated.
        data_pos: [n_galaxy, 3] positions referenced by flat_idx.
        flat_idx: [n_total_neighbors] flattened neighbor indices.
        offsets: [n_query + 1] boundaries; offsets[i]:offsets[i+1]
            selects the neighbors of query position i.
        h: The fixed smoothing length.
        boxsize3: (3,) box size; values <= 0 mean non-periodic along that axis.
    """
    n_query = offsets.shape[0] - 1
    rho_array = np.zeros(n_query, dtype=np.float64)
    periodic = boxsize3[0] > 0.0
    half = boxsize3 / 2.0
    for i in prange(n_query):
        start = offsets[i]
        end = offsets[i + 1]
        n_nei = end - start
        d_tmp = np.empty(n_nei, dtype=np.float64)
        for j in range(n_nei):
            k = flat_idx[start + j]
            dx = query_pos[i, 0] - data_pos[k, 0]
            dy = query_pos[i, 1] - data_pos[k, 1]
            dz = query_pos[i, 2] - data_pos[k, 2]
            if periodic:
                if dx > half[0]:
                    dx -= boxsize3[0]
                elif dx < -half[0]:
                    dx += boxsize3[0]
                if dy > half[1]:
                    dy -= boxsize3[1]
                elif dy < -half[1]:
                    dy += boxsize3[1]
                if dz > half[2]:
                    dz -= boxsize3[2]
                elif dz < -half[2]:
                    dz += boxsize3[2]
            d_tmp[j] = np.sqrt(dx * dx + dy * dy + dz * dz)
        rho_array[i] = np.sum(w_kernel_jit(d_tmp, h))
    return rho_array

@njit 
def cal_drho_array(distance_array, x0, h, use_max_distance_as_h=False):
    """ cal drho 
    Note:
        distance_array: [n_galaxy, n_neighbor]
    """
    raise NotImplementedError

def w_kernel(d, h):
    sigma = 1.0 / (np.pi * h**3)
    q = d / h
    if q < 0.0:
        return 0.0
    elif q <= 1.0:
        return sigma * (0.25 *(2.0-q)**3 - (1.0-q)**3)
    elif q < 2.0:
        return sigma * 0.25 *(2.0-q)**3
    else:
        return 0.0
    
def cal_rho(distance_array, h, use_max_distance_as_h=False):
    """
    Note:
        distance_array: [n_neighbor]
    """
    if use_max_distance_as_h:
        h = np.max(distance_array) / 2.0
    return np.sum(w_kernel(distance_array, h))

def create_rho(
    pos,
    boxsize,
    k=30,
    nthreads=1,
    only_return_rho=True,
    ignore_self=False,
    use_adaptive_h=True,
    h=None,
):
    """Estimate a local number-density mark for each position.

    Parameters
    ----------
    pos : ndarray
        Positions with shape ``(n_galaxy, 3)``.
    boxsize : float or ndarray or None
        Periodic box size. Pass ``None`` for non-periodic distances.
    k : int, default=30
        Number of neighbours used to estimate each density.
    nthreads : int, default=1
        Number of KDTree query workers.
    only_return_rho : bool, default=True
        Return only the density mark; otherwise append it to ``pos``.
    ignore_self : bool, default=False
        Exclude the query point itself from its neighbour list.
    use_adaptive_h : bool, default=True
        Set ``h`` to half the maximum queried neighbour distance for each point.
    h : float, optional
        Fixed smoothing length when ``use_adaptive_h`` is ``False``.
    """
    if not use_adaptive_h and (h is None or h <= 0.0):
        raise ValueError("h must be positive when use_adaptive_h is False")
    if boxsize is not None:
        boxsize += 1e-2
    kdtree = KDTree(pos, boxsize=boxsize)
    query_k = k + 1 if ignore_self else k
    distance_array, _ = kdtree.query(pos, k=query_k, workers=nthreads)
    if ignore_self:
        distance_array = distance_array[:, 1:]
    rho_array = cal_rho_array(
        distance_array,
        h=h,
        use_adaptive_h=use_adaptive_h,
    ).astype(pos.dtype)
    if only_return_rho:
        return rho_array
    else:
        pos_new = np.concatenate([pos, rho_array[:, None]], axis=1)
        return pos_new

# Legacy SciPy implementation retained as disabled reference.
'''
def create_rho_fix_r(
    pos,
    boxsize,
    r,
    nthreads=1,
    chunk_size=100_000,
    verbose=False,
):
    """Estimate a local number-density mark using a fixed support radius.

    Unlike :func:`create_rho`, which fixes the number of neighbours ``k``,
    this function fixes the search radius ``r`` and includes ALL neighbours
    within it (the query point itself is always included since its distance
    is zero). The kernel support radius is ``r``, so the smoothing length
    ``h`` is set to ``r / 2``. Query positions are processed in chunks to
    limit the peak memory used by ragged neighbour lists and flattened indices.

    Parameters
    ----------
    pos : ndarray
        Positions with shape ``(n_galaxy, 3)``.
    boxsize : float or ndarray or None
        Periodic box size. Pass ``None`` for non-periodic distances.
    r : float
        Fixed search radius and full kernel support radius. The corresponding
        smoothing length is ``h = r / 2``.
    nthreads : int, default=1
        Number of KDTree query workers and Numba calculation threads.
    chunk_size : int, default=100000
        Maximum number of query positions processed at once. Smaller chunks
        reduce peak memory usage but increase per-chunk overhead.
    verbose : bool, default=False
        Display a tqdm progress bar for chunk processing when ``True``.

    Returns
    -------
    rho_array : ndarray
        Number-density mark for each position, shape ``(n_galaxy,)``.
    """
    if r <= 0.0:
        raise ValueError("r must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if boxsize is not None:
        tree_boxsize = np.asarray(boxsize, dtype=np.float64) + 1e-5
        boxsize3 = np.broadcast_to(tree_boxsize, (3,)).copy()
    else:
        tree_boxsize = None
        boxsize3 = np.full(3, -1.0, dtype=np.float64)

    kdtree = KDTree(pos, boxsize=tree_boxsize)
    rho_array = np.empty(pos.shape[0], dtype=pos.dtype)
    set_num_threads(nthreads)

    chunk_starts = range(0, pos.shape[0], chunk_size)
    if verbose:
        chunk_starts = tqdm(
            chunk_starts,
            desc="Calculating density",
            unit="chunk",
        )

    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + chunk_size, pos.shape[0])
        query_pos = pos[chunk_start:chunk_end]
        neighbor_idx = kdtree.query_ball_point(
            query_pos,
            r,
            workers=nthreads,
            return_sorted=False,
        )
        counts = np.fromiter(
            (len(idx) for idx in neighbor_idx),
            dtype=np.int64,
            count=query_pos.shape[0],
        )
        offsets = np.empty(counts.shape[0] + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        flat_idx = (
            np.concatenate(neighbor_idx)
            if offsets[-1] > 0
            else np.empty(0, dtype=np.int64)
        )
        del neighbor_idx, counts

        rho_array[chunk_start:chunk_end] = cal_rho_from_pos(
            query_pos,
            pos,
            flat_idx,
            offsets,
            h=r / 2.0,
            boxsize3=boxsize3,
        )

    return rho_array
'''

def create_rho_fix_r(
    pos,
    boxsize,
    r,
    nthreads=1,
    chunk_size=100_000,
    verbose=False,
):
    """Estimate a fixed-radius density mark using freud NeighborLists.

    The search radius ``r`` is the full cubic-spline kernel support radius,
    and the smoothing length is ``h = r / 2``. freud returns a contiguous
    neighbor-pair list with distances, avoiding Python ragged neighbor lists,
    flattened indices, and a second position-distance calculation.

    Parameters
    ----------
    pos : ndarray
        Positions with shape ``(n_galaxy, 3)``.
    boxsize : float or ndarray or None
        Periodic box size. Pass ``None`` for non-periodic distances. A padded
        internal box is used to emulate non-periodic boundaries.
    r : float
        Fixed search radius and full kernel support radius.
    nthreads : int, default=1
        Number of freud worker threads.
    chunk_size : int, default=100000
        Maximum number of query positions processed at once. Smaller chunks
        reduce peak NeighborList memory usage but increase query overhead.
    verbose : bool, default=False
        Display a tqdm progress bar for chunk processing when ``True``.

    Returns
    -------
    rho_array : ndarray
        Number-density mark for each position, shape ``(n_galaxy,)``.
    """
    if r <= 0.0:
        raise ValueError("r must be positive")
    if nthreads <= 0:
        raise ValueError("nthreads must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (n_galaxy, 3)")
    if pos.shape[0] == 0:
        return np.empty(0, dtype=pos.dtype)

    freud_pos = np.ascontiguousarray(pos, dtype=np.float32)
    if boxsize is None:
        pos_min = np.min(freud_pos, axis=0)
        pos_max = np.max(freud_pos, axis=0)
        boxsize3 = pos_max - pos_min + 2.0 * r + 1e-3
        freud_pos = freud_pos - (pos_min + pos_max) / 2.0
    else:
        boxsize3 = np.broadcast_to(
            np.asarray(boxsize, dtype=np.float32),
            (3,),
        ).copy()
        if np.any(boxsize3 <= 0.0):
            raise ValueError("boxsize values must be positive")

    if r >= np.min(boxsize3) / 2.0:
        raise ValueError("r must be smaller than half the smallest box size")

    freud_box = freud.box.Box(
        Lx=boxsize3[0],
        Ly=boxsize3[1],
        Lz=boxsize3[2],
        is2D=False,
    )
    freud_pos = freud_box.wrap(freud_pos)
    neighbor_query = freud.locality.AABBQuery(freud_box, freud_pos)
    rho_array = np.empty(pos.shape[0], dtype=pos.dtype)
    freud.parallel.set_num_threads(nthreads)

    chunk_starts = range(0, pos.shape[0], chunk_size)
    if verbose:
        chunk_starts = tqdm(
            chunk_starts,
            desc="Calculating density",
            unit="chunk",
        )

    h = r / 2.0
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + chunk_size, pos.shape[0])
        neighbor_list = neighbor_query.query(
            freud_pos[chunk_start:chunk_end],
            {"r_max": r, "exclude_ii": False},
        ).toNeighborList()
        weights = w_kernel_jit(neighbor_list.distances, h)
        rho_array[chunk_start:chunk_end] = np.bincount(
            neighbor_list.query_point_indices,
            weights=weights,
            minlength=chunk_end - chunk_start,
        )

    return rho_array

def create_random(omega_mf, w_f, omega_mm, w_m, redshift, boxsize_source, npar, return_boxsize_new=False):
    Hz_f, Hz_m = Hz(redshift, omega_mf, w_f), Hz(redshift, omega_mm, w_m)
    DA_f, DA_m = DA(redshift, omega_mf, w_f), DA(redshift, omega_mm, w_m)
    convert_factor = np.array([DA_m / DA_f, DA_m / DA_f, Hz_f / Hz_m])
    boxsize = boxsize_source * convert_factor
    random = np.random.uniform(0, boxsize, size=(int(npar * np.prod(boxsize)), 3)).astype(np.float32)
    if return_boxsize_new:
        return random, boxsize 
    else:
        return random
