import numpy as np 
from numba import njit
from Corrfunc.theory import DDsmu
from .tpcf import xismu
from tqdm import tqdm
import time

def get_sub_box_shift(data, boxsize, ngrids, max_points_default=None, perodic=False, shift=False):
    """
    Divide data into subvolumes.
    
    Parameters
    ----------
    data : ndarray
        Data array with shape (N, 3) or (N, 4).
    boxsize : float or array-like
        Box size.
    ngrids : int or array-like
        Number of divisions along each axis.
    max_points_default : int, optional
        Maximum number of points per subvolume.
    perodic : bool
        Whether to apply periodic boundary conditions.
    shift : bool
        If True, convert global coordinates to local coordinates (subtract subvolume origin).
        If False, keep global coordinates.
    
    Returns
    -------
    data_out_array : ndarray of object
        3D array of subvolume data arrays.
    """
    if isinstance(boxsize, float) or isinstance(boxsize, int):
        boxsize = np.array([boxsize, boxsize, boxsize])
    if isinstance(ngrids, float) or isinstance(ngrids, int):
        ngrids = np.array([ngrids, ngrids, ngrids], dtype=np.int32)
    data_new_array, counts_array = get_sub_box_shift_core(data, boxsize, ngrids, max_points_default, perodic, shift)
    data_out_array = np.empty(shape=ngrids, dtype=object)
    for i in range(ngrids[0]):
        for j in range(ngrids[1]):
            for k in range(ngrids[2]):
                data_out_array[i, j, k] = data_new_array[i, j, k, :counts_array[i, j, k]]
    return data_out_array

@njit
def get_sub_box_shift_core(data, boxsize, ngrids, max_points_default=None, perodic=False, shift=False):
    sub_box_size = boxsize / ngrids
    num = data.shape[0]
    elements_size = data.shape[1]

    if max_points_default is None:
        max_points_default = np.int64(num / np.prod(ngrids) * 2)
    else:
        max_points_default = np.int64(max_points_default)
    data_new_array = np.empty(shape=(ngrids[0], ngrids[1], ngrids[2]) + (max_points_default, elements_size), dtype=data.dtype) 
    counts_array = np.zeros(shape=(ngrids[0], ngrids[1], ngrids[2]), dtype=np.int64)

    for i in range(num):
        x_temp, y_temp, z_temp = data[i, :3]
        if perodic:
            x_temp = x_temp % boxsize[0]
            y_temp = y_temp % boxsize[1]
            z_temp = z_temp % boxsize[2]
        else:
            if x_temp < 0 or x_temp > boxsize[0] or y_temp < 0 or y_temp > boxsize[1] or z_temp < 0 or z_temp > boxsize[2]:
                continue
            if x_temp == boxsize[0]:
                x_temp -= 1e-10 
            if y_temp == boxsize[1]:
                y_temp -= 1e-10
            if z_temp == boxsize[2]:
                z_temp -= 1e-10
        x_i = np.int32(x_temp / sub_box_size[0])
        y_i = np.int32(y_temp / sub_box_size[1])
        z_i = np.int32(z_temp / sub_box_size[2])

        # Only shift coordinates if shift=True
        if shift:
            x_temp -= x_i * sub_box_size[0]
            y_temp -= y_i * sub_box_size[1]
            z_temp -= z_i * sub_box_size[2]

        count_temp = counts_array[x_i, y_i, z_i]
        if count_temp >= max_points_default:
            raise RuntimeError("Too many points in a sub-box (larger than max_points_default)")
        data_new_array[x_i, y_i, z_i, count_temp, :3] = x_temp, y_temp, z_temp
        data_new_array[x_i, y_i, z_i, count_temp, 3:] = data[i, 3:]
        counts_array[x_i, y_i, z_i] += 1

    return data_new_array, counts_array 

def run_jackknife_tpCF(data, random, sedges, mubin, with_weight,
                       boxsize, ngrids, refine_factors=(2, 2, 1), RR_result=None, nthreads=1, verbose=False,
                       full_output=False):
    """
    Main function to compute Jackknife two-point correlation function.
    
    Parameters
    ----------
    data : ndarray
        Complete data array with shape (N, 3) or (N, 4).
        Will be divided into ngrids[0] x ngrids[1] x ngrids[2] subvolumes internally.
    random : ndarray
        Random catalog array with shape (M, 3) or (M, 4).
    sedges : array-like
        Separation bin edges.
    mubin : int
        Number of mu bins.
    with_weight : bool
        If True, use the 4th column of data as weights.
    boxsize : float or array-like
        Box size. If float, assumes cubic box.
    ngrids : int or array-like of 3 ints
        Number of divisions along each axis.
        If int, assumes same number for all axes (ngrids^3 total subvolumes).
    refine_factors : tuple of 3 ints
        (x_refine_factor, y_refine_factor, z_refine_factor) for DDsmu.
    RR_result : None or rr
        If not None, will not calculate rr
    nthreads : int
        Number of threads for parallel computation.
    verbose : bool
        Whether to print progress information.
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'xismu_jk': ndarray of xismu objects (n_cubes,) - xismu for each JK sample
        - 'xismu_full': xismu object for full sample
        - 'sedges': separation bin edges
        - 'muedges': mu bin edges
        - 'DD_internal': dict of internal DD results
        - 'DD_cross': dict of cross DD results
        - 'DR_internal': dict of internal DR results (data_i vs random_i)
        - 'DR_cross': dict of cross DR results (data_i vs random_j, j is neighbor of i)
        - 'RR_result': RR result
    """
    # Normalize ngrids to array
    if isinstance(ngrids, int):
        ngrids = np.array([ngrids, ngrids, ngrids], dtype=np.int32)
    elif isinstance(ngrids, (list, tuple)):
        ngrids = np.array(ngrids, dtype=np.int32)
    
    n_cubes = int(np.prod(ngrids))
    
    if verbose:
        print(f"Dividing data into {ngrids[0]}x{ngrids[1]}x{ngrids[2]} = {n_cubes} subvolumes...")
    
    # Divide data into subvolumes
    data_array = get_sub_box_shift(data, boxsize, ngrids)
    
    # Divide random into subvolumes (same grid as data)
    random_array = get_sub_box_shift(random, boxsize, ngrids)
    
    # Convert 3D array to list (ordered as [sub(0,0,0), sub(0,0,1), ..., sub(0,1,0), ...])
    data_list = []
    random_list = []
    for i in range(ngrids[0]):
        for j in range(ngrids[1]):
            for k in range(ngrids[2]):
                data_list.append(data_array[i, j, k])
                random_list.append(random_array[i, j, k])
    
    if verbose:
        print(f"Number of subvolumes: {n_cubes} = {ngrids[0]}x{ngrids[1]}x{ngrids[2]}")
    
    # Step 1: Compute DD (internal + cross between neighboring subvolumes)
    DD_internal, DD_cross = compute_DD(
        data_list, sedges, mubin, with_weight, boxsize,
        refine_factors, nthreads, verbose, ngrids
    )
    
    # Step 2: Compute DR (internal + cross between neighboring subvolumes)
    DR_internal, DR_cross = compute_DR(
        data_list, random_list, sedges, mubin, with_weight, boxsize,
        refine_factors, nthreads, verbose, ngrids
    )
    
    # Step 3: Compute RR (internal + cross between neighboring subvolumes)
    if RR_result is None:
        RR_result = compute_RR(
            random_list, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, verbose, ngrids
        )
    else:
        if verbose:
            print("Using provided RR result...")
    
    if verbose:
        print("Computing Jackknife correlation functions...")
    
    # Step 4: Compute Jackknife correlation functions
    xi_results = compute_jackknife_xi(
        DD_internal, DD_cross, DR_internal, DR_cross, RR_result,
        data_list, random_list, sedges, mubin, with_weight
    )
    
    if verbose:
        print("Done!")
    
    if full_output:
        return xi_results 
    else:
        return {
            'xismu_jk': xi_results['xismu_jk'],
            'xismu_full': xi_results['xismu_full'],
            "RR_result": xi_results["RR_result"]
        }

def run_subsample_tpCF(data, random, sedges, mubin, with_weight,
                           boxsize, ngrids, refine_factors=(2, 2, 1), RR_result=None, nthreads=1, verbose=False, full_output=False):
    """
    Compute two-point correlation function for each subvolume as independent samples.
    Each subvolume is treated as an independent sample with the same random catalog.
    
    Parameters
    ----------
    data : ndarray
        Complete data array with shape (N, 3) or (N, 4).
        Will be divided into ngrids[0] x ngrids[1] x ngrids[2] subvolumes.
    random : ndarray
        Random catalog array with shape (M, 3) or (M, 4).
        Used as-is for each subvolume (not divided). Represents a single subvolume-sized random catalog.
    sedges : array-like
        Separation bin edges.
    mubin : int
        Number of mu bins.
    with_weight : bool
        If True, use the 4th column of data as weights.
    boxsize : float or array-like
        Total box size. If float, assumes cubic box.
    ngrids : int or array-like of 3 ints
        Number of divisions along each axis.
        If int, assumes same number for all axes (ngrids^3 total subvolumes).
    refine_factors : tuple of 3 ints
        (x_refine_factor, y_refine_factor, z_refine_factor) for DDsmu.
    RR_result : structured array, optional
        Pre-computed RR result. If provided, skip RR computation.
    nthreads : int
        Number of threads for parallel computation.
    verbose : bool
        Whether to print progress information.
    full_output : bool
        If True, return full output with DD_sub, DR_sub.
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'xismu_sub': ndarray of xismu objects (n_cubes,) - xismu for each subvolume
        - 'sedges': separation bin edges
        - 'muedges': mu bin edges
        - 'DD_sub': dict of DD results for each subvolume (if full_output)
        - 'DR_sub': dict of DR results for each subvolume (if full_output)
        - 'RR_result': RR result (computed once or provided)
    """
    # Normalize ngrids to array
    if isinstance(ngrids, int):
        ngrids = np.array([ngrids, ngrids, ngrids], dtype=np.int32)
    elif isinstance(ngrids, (list, tuple)):
        ngrids = np.array(ngrids, dtype=np.int32)
    
    # Normalize boxsize to array
    if isinstance(boxsize, (float, int)):
        boxsize = np.array([boxsize, boxsize, boxsize])
    
    n_cubes = int(np.prod(ngrids))
    sub_boxsize = boxsize / ngrids
    
    if verbose:
        print(f"Dividing data into {ngrids[0]}x{ngrids[1]}x{ngrids[2]} = {n_cubes} subvolumes...")
        print(f"Subvolume box size: {sub_boxsize[0]:.2f} x {sub_boxsize[1]:.2f} x {sub_boxsize[2]:.2f}")
    
    # Divide data into subvolumes with shift=True (local coordinates)
    data_array = get_sub_box_shift(data, boxsize, ngrids, shift=True)
    
    # Convert 3D array to list
    data_list = []
    for i in range(ngrids[0]):
        for j in range(ngrids[1]):
            for k in range(ngrids[2]):
                data_list.append(data_array[i, j, k])
    
    # Storage for results
    DD_sub = {}
    DR_sub = {}
    xismu_sub = np.empty(n_cubes, dtype=object)
    
    sbin = len(sedges) - 1
    
    # Compute RR once (or use provided)
    if RR_result is None:
        if verbose:
            print("Computing RR...")
            start_time = time.time()
        RR_result = _call_DDsmu(
            random, None, sedges, mubin, with_weight, sub_boxsize[0],
            refine_factors, nthreads, autocorr=True
        )
        if verbose:
            end_time = time.time()
            print(f"Done! Time elapsed: {end_time - start_time:.2f} seconds")
    else:
        if verbose:
            print("Using provided RR result...")
    
    # Extract RR pair counts (same for all subvolumes)
    RR = extract_npairs(RR_result, with_weight).reshape(sbin, mubin)
    
    # Get mu edges
    mumax_str = "mumax" if "mumax" in RR_result.dtype.names else "mu_max"
    muedges = np.append([0], RR_result[mumax_str].reshape(sbin, mubin)[0])
    
    # Compute normalization for RR (same for all subvolumes)
    if with_weight:
        nr = np.sum(random[:, 3])
        sum_wr2 = np.sum(random[:, 3]**2)
    else:
        nr = len(random)
        sum_wr2 = nr
    norm_RR = nr * nr - sum_wr2
    
    # RR self-pair correction
    RR_corrected = RR.copy()
    if sedges[0] == 0.0 and RR_result is None: # When RR_result is not None, it has been corrected
        RR_corrected[0, 0] -= nr
    
    # Process each subvolume
    iterator = tqdm(enumerate(data_list), total=n_cubes, desc="Processing subvolumes (DD+DR)", disable=not verbose)
    for idx, data_sub in iterator:
        if data_sub is None or len(data_sub) < 2:
            DD_sub[idx] = None
            DR_sub[idx] = None
            xismu_sub[idx] = None
            continue
        
        # Compute DD (autocorrelation within subvolume)
        DD_result = _call_DDsmu(
            data_sub, None, sedges, mubin, with_weight, sub_boxsize[0],
            refine_factors, nthreads, autocorr=True
        )
        DD_sub[idx] = DD_result
        
        # Compute DR (data vs random, using sub_boxsize)
        DR_result = _call_DDsmu(
            data_sub, random, sedges, mubin, with_weight, sub_boxsize[0],
            refine_factors, nthreads, autocorr=False
        )
        DR_sub[idx] = DR_result
        
        # Extract pair counts
        DD = extract_npairs(DD_result, with_weight).reshape(sbin, mubin)
        DR = extract_npairs(DR_result, with_weight).reshape(sbin, mubin)
        
        # Compute normalization factors
        if with_weight:
            nd = np.sum(data_sub[:, 3])
            sum_wd2 = np.sum(data_sub[:, 3]**2)
        else:
            nd = len(data_sub)
            sum_wd2 = nd
        
        norm_DD = nd * nd - sum_wd2
        norm_DR = nd * nr
        
        # Remove self-pairs from s=0, mu=0 bin
        DD_corrected = DD.copy()
        if sedges[0] == 0.0:
            DD_corrected[0, 0] -= nd
        
        # Create xismu object
        xismu_sub[idx] = create_xismu_from_pairs(
            DD_corrected, DR, RR_corrected, norm_DD, norm_DR, norm_RR, sedges, muedges
        )
    
    if verbose:
        print("Done!")
    
    if full_output:
        return {
            'xismu_sub': xismu_sub,
            'sedges': sedges,
            'muedges': muedges,
            'DD_sub': DD_sub,
            'DR_sub': DR_sub,
            'RR_result': RR_result,
        }
    else:
        return {
            'xismu_sub': xismu_sub,
            'RR_result': RR_result,
        }


def check_and_shift_data(data_list, boxsize, n, index_list=None):
    """
    Check if data points are in correct subvolume positions.
    If not, shift them to the correct position based on the provided index_list.
    
    Parameters
    ----------
    data_list : list of ndarray
        List of subvolume data arrays. Each array has shape (N, 3) or (N, 4).
    boxsize : float
        Total box size.
    n : int
        Number of divisions along each axis (n^3 total subvolumes).
    index_list : list of tuple, optional
        List of (ix, iy, iz) indices for each data in data_list.
        If provided, use these indices to determine the correct position.
        If None, assume data_list is ordered as [sub_000, sub_001, ..., sub_00n-1, sub_010, ...].
    
    Returns
    -------
    data_list_shifted : list of ndarray
        Data list with coordinates shifted to correct positions if needed.
    """
    sub_size = boxsize / n
    data_list_shifted = []
    
    for idx, data in enumerate(data_list):
        if data is None or len(data) == 0:
            data_list_shifted.append(data)
            continue
        
        # Get subvolume indices from index_list or compute from idx
        if index_list is not None:
            ix, iy, iz = index_list[idx]
        else:
            ix = idx // (n * n)
            iy = (idx // n) % n
            iz = idx % n
        
        # Expected coordinate range
        x_min, x_max = ix * sub_size, (ix + 1) * sub_size
        y_min, y_max = iy * sub_size, (iy + 1) * sub_size
        z_min, z_max = iz * sub_size, (iz + 1) * sub_size
        
        # Check if data is already in correct position
        x_in_range = np.all((data[:, 0] >= x_min) & (data[:, 0] < x_max))
        y_in_range = np.all((data[:, 1] >= y_min) & (data[:, 1] < y_max))
        z_in_range = np.all((data[:, 2] >= z_min) & (data[:, 2] < z_max))
        
        if x_in_range and y_in_range and z_in_range:
            data_list_shifted.append(data)
        else:
            # Shift data to correct position using index-based offset
            data_shifted = data.copy()
            shift_x = ix * sub_size
            shift_y = iy * sub_size
            shift_z = iz * sub_size
            
            data_shifted[:, 0] += shift_x
            data_shifted[:, 1] += shift_y
            data_shifted[:, 2] += shift_z
            
            data_list_shifted.append(data_shifted)
    
    return data_list_shifted


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


def _call_DDsmu(data1, data2, sedges, mubin, with_weight, boxsize, 
                refine_factors, nthreads, autocorr, verbose=False):
    """
    Wrapper for DDsmu call.
    """
    x_refine_factor, y_refine_factor, z_refine_factor = refine_factors
    
    if with_weight:
        if autocorr:
            result = DDsmu(
                autocorr, nthreads=nthreads, binfile=sedges, 
                mu_max=1.0, nmu_bins=mubin,
                X1=data1[:, 0], Y1=data1[:, 1], Z1=data1[:, 2],
                weights1=data1[:, 3], weight_type="pair_product",
                verbose=verbose, periodic=False, boxsize=boxsize,
                xbin_refine_factor=x_refine_factor,
                ybin_refine_factor=y_refine_factor,
                zbin_refine_factor=z_refine_factor
            )
        else:
            result = DDsmu(
                autocorr, nthreads=nthreads, binfile=sedges,
                mu_max=1.0, nmu_bins=mubin,
                X1=data1[:, 0], Y1=data1[:, 1], Z1=data1[:, 2],
                weights1=data1[:, 3], weight_type="pair_product",
                X2=data2[:, 0], Y2=data2[:, 1], Z2=data2[:, 2],
                weights2=data2[:, 3],
                verbose=verbose, periodic=False, boxsize=boxsize,
                xbin_refine_factor=x_refine_factor,
                ybin_refine_factor=y_refine_factor,
                zbin_refine_factor=z_refine_factor
            )
    else:
        if autocorr:
            result = DDsmu(
                autocorr, nthreads=nthreads, binfile=sedges,
                mu_max=1.0, nmu_bins=mubin,
                X1=data1[:, 0], Y1=data1[:, 1], Z1=data1[:, 2],
                verbose=verbose, periodic=False, boxsize=boxsize,
                xbin_refine_factor=x_refine_factor,
                ybin_refine_factor=y_refine_factor,
                zbin_refine_factor=z_refine_factor
            )
        else:
            result = DDsmu(
                autocorr, nthreads=nthreads, binfile=sedges,
                mu_max=1.0, nmu_bins=mubin,
                X1=data1[:, 0], Y1=data1[:, 1], Z1=data1[:, 2],
                X2=data2[:, 0], Y2=data2[:, 1], Z2=data2[:, 2],
                verbose=verbose, periodic=False, boxsize=boxsize,
                xbin_refine_factor=x_refine_factor,
                ybin_refine_factor=y_refine_factor,
                zbin_refine_factor=z_refine_factor
            )
    
    return result


def compute_DD(data_list, sedges, mubin, with_weight, boxsize,
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
    n_cubes = len(data_list)
    n = ngrids[0]  # Assume cubic grid for neighbor calculation
    
    if verbose:
        print("Computing DD internal...")
    
    # Step 1: Compute DD internal for each subvolume
    DD_internal = {}
    iterator = tqdm(enumerate(data_list), total=n_cubes, desc="DD internal", disable=not verbose)
    for idx, data in iterator:
        if data is None or len(data) < 2:
            DD_internal[idx] = None
            continue
        
        result = _call_DDsmu(
            data, None, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=True
        )
        DD_internal[idx] = result
    
    if verbose:
        print("Computing DD cross...")
    
    # Step 2: Compute DD cross between neighboring subvolumes
    DD_cross = {}
    computed_pairs = set()
    pairs_to_compute = []
    
    for idx in range(n_cubes):
        data1 = data_list[idx]
        if data1 is None or len(data1) == 0:
            continue
        
        neighbors = get_neighbors(idx, n)
        
        for neighbor_idx in neighbors:
            pair = (min(idx, neighbor_idx), max(idx, neighbor_idx))
            if pair in computed_pairs:
                continue
            computed_pairs.add(pair)
            
            data2 = data_list[neighbor_idx]
            if data2 is None or len(data2) == 0:
                continue
            
            pairs_to_compute.append((pair, data1, data2))
    
    iterator = tqdm(pairs_to_compute, desc="DD cross", disable=not verbose)
    for pair, data1, data2 in iterator:
        result = _call_DDsmu(
            data1, data2, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=False
        )
        result["npairs"] *= 2  # To be consistent with the number of pairs in the full catalog
        DD_cross[pair] = result
    
    return DD_internal, DD_cross


def compute_DR(data_list, random_list, sedges, mubin, with_weight, boxsize,
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
        Only includes pairs where idx1 < idx2.
    """
    n_cubes = len(data_list)
    n = ngrids[0]  # Assume cubic grid for neighbor calculation
    
    # Step 1: Compute DR internal (same position: data_i vs random_i)
    DR_internal = {}
    iterator = tqdm(enumerate(zip(data_list, random_list)), total=n_cubes, desc="DR internal", disable=not verbose)
    for idx, (data, random_sub) in iterator:
        if data is None or len(data) == 0:
            DR_internal[idx] = None
            continue
        if random_sub is None or len(random_sub) == 0:
            DR_internal[idx] = None
            continue
        
        result = _call_DDsmu(
            data, random_sub, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=False
        )
        DR_internal[idx] = result
    
    # Step 2: Compute DR cross between neighboring subvolumes (data_i vs random_j)
    DR_cross = {}
    computed_pairs = set()
    pairs_to_compute = []
    
    for idx in range(n_cubes):
        data = data_list[idx]
        if data is None or len(data) == 0:
            continue
        
        neighbors = get_neighbors(idx, n)
        
        for neighbor_idx in neighbors:
            pair = (min(idx, neighbor_idx), max(idx, neighbor_idx))
            if pair in computed_pairs:
                continue
            computed_pairs.add(pair)
            
            random_neighbor = random_list[neighbor_idx]
            if random_neighbor is None or len(random_neighbor) == 0:
                continue
            
            # data_i vs random_j (where j is neighbor of i)
            pairs_to_compute.append((pair, data, random_neighbor))
    
    iterator = tqdm(pairs_to_compute, desc="DR cross", disable=not verbose)
    for pair, data, random_sub in iterator:
        result = _call_DDsmu(
            data, random_sub, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=False
        )
        DR_cross[pair] = result
    
    return DR_internal, DR_cross


def compute_RR(random_list, sedges, mubin, with_weight, boxsize,
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
    RR_result : structured array
        Combined RR result (sum of internal + cross pairs).
    """
    n_cubes = len(random_list)
    n = ngrids[0]  # Assume cubic grid for neighbor calculation
    sbin = len(sedges) - 1
    
    if verbose:
        print("Computing RR internal...")
    
    # Step 1: Compute RR internal for each subvolume
    RR_internal = {}
    iterator = tqdm(enumerate(random_list), total=n_cubes, desc="RR internal", disable=not verbose)
    for idx, random_sub in iterator:
        if random_sub is None or len(random_sub) < 2:
            RR_internal[idx] = None
            continue
        
        result = _call_DDsmu(
            random_sub, None, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=True
        )
        RR_internal[idx] = result
    
    if verbose:
        print("Computing RR cross...")
    
    # Step 2: Compute RR cross between neighboring subvolumes
    RR_cross = {}
    computed_pairs = set()
    pairs_to_compute = []
    
    for idx in range(n_cubes):
        random1 = random_list[idx]
        if random1 is None or len(random1) == 0:
            continue
        
        neighbors = get_neighbors(idx, n)
        
        for neighbor_idx in neighbors:
            pair = (min(idx, neighbor_idx), max(idx, neighbor_idx))
            if pair in computed_pairs:
                continue
            computed_pairs.add(pair)
            
            random2 = random_list[neighbor_idx]
            if random2 is None or len(random2) == 0:
                continue
            
            pairs_to_compute.append((pair, random1, random2))
    
    iterator = tqdm(pairs_to_compute, desc="RR cross", disable=not verbose)
    for pair, random1, random2 in iterator:
        result = _call_DDsmu(
            random1, random2, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=False
        )
        result["npairs"] *= 2  # To be consistent with the number of pairs in the full catalog
        RR_cross[pair] = result
    
    # Step 3: Sum all RR pairs
    RR_total = np.zeros((sbin, mubin), dtype=np.float64)
    
    for idx, result in RR_internal.items():
        if result is not None:
            RR_total += extract_npairs(result, with_weight).reshape(sbin, mubin)
    
    for pair, result in RR_cross.items():
        if result is not None:
            RR_total += extract_npairs(result, with_weight).reshape(sbin, mubin)
    
    # Create a structured array to return (mimicking DDsmu output format)
    # We need to construct the proper output format
    mumax_str = "mumax" if hasattr(RR_internal.get(0, {}), 'dtype') and RR_internal.get(0) is not None and "mumax" in RR_internal[0].dtype.names else "mu_max"
    
    # Get mu edges from first non-None result
    muedges = None
    for idx, result in RR_internal.items():
        if result is not None:
            muedges = np.append([0], result[mumax_str].reshape(sbin, mubin)[0])
            break
    
    if muedges is None:
        muedges = np.linspace(0, 1, mubin + 1)
    
    # Construct output format
    # Create a mock structured array with the same format as DDsmu
    npairs_flat = RR_total.flatten()
    weightavg_flat = np.ones_like(npairs_flat)  # Placeholder
    
    # Create structured array
    dtype = np.dtype([
        ('smin', np.float64),
        ('smax', np.float64),
        (mumax_str, np.float64),
        ('npairs', np.float64),
        ('weightavg', np.float64)
    ])
    
    RR_result = np.zeros(sbin * mubin, dtype=dtype)
    
    # Fill in values
    s_centers = (sedges[:-1] + sedges[1:]) / 2.0
    mu_centers = (muedges[:-1] + muedges[1:]) / 2.0
    
    idx = 0
    for i in range(sbin):
        for j in range(mubin):
            RR_result[idx]['smin'] = sedges[i]
            RR_result[idx]['smax'] = sedges[i+1]
            RR_result[idx][mumax_str] = muedges[j+1]
            RR_result[idx]['npairs'] = RR_total[i, j]
            RR_result[idx]['weightavg'] = weightavg_flat[idx]
            idx += 1
    
    return RR_result


def extract_npairs(result, with_weight):
    """
    Extract npairs and weighted pairs from DDsmu result.
    
    Parameters
    ----------
    result : structured array
        DDsmu result.
    with_weight : bool
        Whether to use weights.
    
    Returns
    -------
    npairs : ndarray
        Number of pairs (or weighted pairs).
    """
    if result is None:
        return None
    
    npairs = result['npairs']
    
    if with_weight:
        weightavg = result['weightavg']
        if not np.all(weightavg == 0):
            npairs = npairs * weightavg
    
    return npairs


def create_xismu_from_pairs(DD, DR, RR, norm_DD, norm_DR, norm_RR, sedges, muedges):
    """
    Create xismu object from pair counts.
    
    Parameters
    ----------
    DD : ndarray
        DD pair counts (sbin, mubin).
    DR : ndarray
        DR pair counts (sbin, mubin).
    RR : ndarray
        RR pair counts (sbin, mubin).
    norm_DD : float
        Normalization factor for DD.
    norm_DR : float
        Normalization factor for DR.
    norm_RR : float
        Normalization factor for RR.
    sedges : array-like
        Separation bin edges.
    muedges : array-like
        Mu bin edges.
    
    Returns
    -------
    xismu_obj : xismu
        xismu object containing the correlation function.
    """
    sbin = len(sedges) - 1
    mubin = len(muedges) - 1
    smax = sedges[-1]
    
    # Create S and Mu meshgrid
    s_array = (sedges[1:] + sedges[:-1]) / 2.0
    mu_array = (muedges[1:] + muedges[:-1]) / 2.0
    S, Mu = np.meshgrid(s_array, mu_array, indexing="ij")
    
    # Create xismu object
    xismu_obj = xismu(
        smax=smax, sbin=sbin, mubin=mubin,
        DDnorm=norm_DD, DRnorm=norm_DR, RRnorm=norm_RR,
        S=S, Mu=Mu,
        DD=DD, DR=DR, RR=RR,
        set_data=True
    )
    
    return xismu_obj


def compute_jackknife_xi(DD_internal, DD_cross, DR_internal, DR_cross, RR_result,
                         data_list, random_list, sedges, mubin, with_weight):
    """
    Compute two-point correlation function for each Jackknife sample.
    
    Parameters
    ----------
    DD_internal : dict
        DD results for internal pairs of each subvolume.
    DD_cross : dict
        DD results for cross-pairs between neighboring subvolumes.
    DR_internal : dict
        DR results for same-position pairs (data_i vs random_i).
    DR_cross : dict
        DR results for cross-pairs between neighboring subvolumes (data_i vs random_j).
    RR_result : structured array
        RR result (combined internal + cross pairs).
    data_list : list of ndarray
        List of n^3 subvolume data arrays.
    random_list : list of ndarray
        List of n^3 subvolume random arrays.
    sedges : array-like
        Separation bin edges.
    mubin : int
        Number of mu bins.
    with_weight : bool
        Whether to use weights.
    
    Returns
    -------
    xi_results : dict
        Dictionary with:
        - 'xismu_jk': ndarray of xismu objects (n_cubes,) - xismu for each JK sample
        - 'xismu_full': xismu object for full sample
        - 'sedges': separation bin edges
        - 'muedges': mu bin edges
        - 'DD_internal': dict of internal DD results
        - 'DD_cross': dict of cross DD results
        - 'DR_internal': dict of internal DR results
        - 'DR_cross': dict of cross DR results
        - 'RR_result': RR result
    """
    n_cubes = len(data_list)
    ngrids_1d = round(n_cubes ** (1/3))
    sbin = len(sedges) - 1
    
    # Extract RR
    RR = extract_npairs(RR_result, with_weight).reshape(sbin, mubin)
    # Get mu edges from result
    mumax_str = "mumax" if "mumax" in RR_result.dtype.names else "mu_max"
    muedges = np.append([0], RR_result[mumax_str].reshape(sbin, mubin)[0])
    
    # Compute total DD, DR
    DD_total = np.zeros((sbin, mubin))
    DR_total = np.zeros((sbin, mubin))
    
    # Sum all internal DD
    for idx, result in DD_internal.items():
        if result is not None:
            DD_total += extract_npairs(result, with_weight).reshape(sbin, mubin)
    
    # Sum all cross DD
    for pair, result in DD_cross.items():
        if result is not None:
            DD_total += extract_npairs(result, with_weight).reshape(sbin, mubin)
    
    # Sum all internal DR
    for idx, result in DR_internal.items():
        if result is not None:
            DR_total += extract_npairs(result, with_weight).reshape(sbin, mubin)
    
    # Sum all cross DR
    for pair, result in DR_cross.items():
        if result is not None:
            DR_total += extract_npairs(result, with_weight).reshape(sbin, mubin)
    
    # Compute normalization factors
    if with_weight:
        total_data_weight = sum(np.sum(d[:, 3]) for d in data_list if d is not None and len(d) > 0)
        total_random_weight = sum(np.sum(r[:, 3]) for r in random_list if r is not None and len(r) > 0)
        sum_wd2 = sum(np.sum(d[:, 3]**2) for d in data_list if d is not None and len(d) > 0)
        sum_wr2 = sum(np.sum(r[:, 3]**2) for r in random_list if r is not None and len(r) > 0)
    else:
        total_data_weight = sum(len(d) for d in data_list if d is not None)
        total_random_weight = sum(len(r) for r in random_list if r is not None)
        sum_wd2 = total_data_weight
        sum_wr2 = total_random_weight
    
    norm_DD = total_data_weight * total_data_weight - sum_wd2
    norm_DR = total_data_weight * total_random_weight
    norm_RR = total_random_weight * total_random_weight - sum_wr2
    
    # Remove self-pairs from s=0, mu=0 bin (Corrfunc autocorr=True includes them)
    # This is consistent with cal_tpCF_from_pairs in LSS_python
    if sedges[0] == 0.0:
        DD_total[0, 0] -= total_data_weight
        RR[0, 0] -= total_random_weight
    
    # Create xismu for full sample
    xismu_full = create_xismu_from_pairs(
        DD_total, DR_total, RR, norm_DD, norm_DR, norm_RR, sedges, muedges
    )
    
    # Jackknife samples
    xismu_jk = np.empty(n_cubes, dtype=object)
    
    for jk_idx in range(n_cubes):
        # DD for this JK sample (exclude subvolume jk_idx)
        DD_jk = DD_total.copy()
        
        # Subtract internal DD of excluded subvolume
        if DD_internal.get(jk_idx) is not None:
            DD_jk -= extract_npairs(DD_internal[jk_idx], with_weight).reshape(sbin, mubin)
        
        # Subtract cross DD with neighbors
        neighbors = get_neighbors(jk_idx, ngrids_1d)
        for neighbor_idx in neighbors:
            pair = (min(jk_idx, neighbor_idx), max(jk_idx, neighbor_idx))
            if DD_cross.get(pair) is not None:
                DD_jk -= extract_npairs(DD_cross[pair], with_weight).reshape(sbin, mubin)
        
        # DR for this JK sample
        DR_jk = DR_total.copy()
        
        # Subtract internal DR of excluded subvolume
        if DR_internal.get(jk_idx) is not None:
            DR_jk -= extract_npairs(DR_internal[jk_idx], with_weight).reshape(sbin, mubin)
        
        # Subtract cross DR with neighbors
        for neighbor_idx in neighbors:
            pair = (min(jk_idx, neighbor_idx), max(jk_idx, neighbor_idx))
            if DR_cross.get(pair) is not None:
                DR_jk -= extract_npairs(DR_cross[pair], with_weight).reshape(sbin, mubin)
        
        # Recompute normalization for JK sample
        if with_weight:
            excluded_weight = np.sum(data_list[jk_idx][:, 3]) if data_list[jk_idx] is not None and len(data_list[jk_idx]) > 0 else 0
            jk_data_weight = total_data_weight - excluded_weight
            jk_sum_wd2 = sum_wd2 - (np.sum(data_list[jk_idx][:, 3]**2) if data_list[jk_idx] is not None and len(data_list[jk_idx]) > 0 else 0)
        else:
            excluded_count = len(data_list[jk_idx]) if data_list[jk_idx] is not None else 0
            jk_data_weight = total_data_weight - excluded_count
            jk_sum_wd2 = sum_wd2 - excluded_count
        
        norm_DD_jk = jk_data_weight * jk_data_weight - jk_sum_wd2
        norm_DR_jk = jk_data_weight * total_random_weight
        
        # Correct s=0, mu=0 bin for Jackknife sample
        # DD_jk already has total_data_weight subtracted (from DD_total)
        # but should have jk_data_weight subtracted
        if sedges[0] == 0.0:
            DD_jk[0, 0] += (total_data_weight - jk_data_weight)
        
        # Create xismu for this JK sample
        xismu_jk[jk_idx] = create_xismu_from_pairs(
            DD_jk, DR_jk, RR, norm_DD_jk, norm_DR_jk, norm_RR, sedges, muedges
        )
    
    return {
        'xismu_jk': xismu_jk,
        'xismu_full': xismu_full,
        'sedges': sedges,
        'muedges': muedges,
        'DD_internal': DD_internal,
        'DD_cross': DD_cross,
        'DR_internal': DR_internal,
        'DR_cross': DR_cross,
        'RR_result': RR_result,
    }
