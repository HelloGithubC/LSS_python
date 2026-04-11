import numpy as np 
from numba import njit
from .corrfunc import call_DDsmu
from .tpcf import xismu
from tqdm import tqdm
import time
import os
import joblib

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
                       boxsize, ngrids, refine_factors=(2, 2, 1), rr_filename=None, force_rr=False, nthreads=1, verbose=False,
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
    rr_filename : str or None
        Path to save/load RR computation results via joblib.
        If None, compute RR every time.
        If file exists and force_rr=False, load from file.
        If file doesn't exist or force_rr=True, compute and save to file.
    force_rr : bool
        If True, force recompute RR even if file exists.
    nthreads : int
        Number of threads for parallel computation.
    verbose : bool
        Whether to print progress information.
    full_output : bool
        If True, return all intermediate results.
    
    Returns
    -------
    result : dict or ndarray (xismu_jk)
        Dictionary containing (when full_output=False):
        - 'xismu_jk': ndarray of xismu objects (n_cubes,) - xismu for each JK sample
        
        Additional fields (when full_output=True):
        - 'xismu_full': xismu object for full sample
        - 'sedges': separation bin edges
        - 'muedges': mu bin edges
        - 'DD_internal': dict of internal DD results
        - 'DD_cross': dict of cross DD results
        - 'DR_internal': dict of internal DR results (data_i vs random_i)
        - 'DR_cross': dict of cross DR results (data_i vs random_j, j is neighbor of i)
        - 'RR_internal': dict of internal RR results
        - 'RR_cross': dict of cross RR results
        - 'RR_full': RR full result
    """
    # Robustness: ensure data and random have consistent dtypes
    # Use the higher precision dtype
    if data.dtype != random.dtype:
        # Determine higher precision dtype
        if data.dtype.itemsize > random.dtype.itemsize:
            higher_dtype = data.dtype
            lower_name = 'random'
        else:
            higher_dtype = random.dtype
            lower_name = 'data'
        
        if verbose:
            print(f"Warning: data dtype ({data.dtype}) and random dtype ({random.dtype}) differ. "
                  f"Converting both to higher precision dtype: {higher_dtype}")
        
        data = data.astype(higher_dtype)
        random = random.astype(higher_dtype)
    
    # Robustness: add weight column if with_weight=True but data/random have < 4 columns
    # Preserve original dtype to avoid implicit type promotion
    if with_weight:
        if data.shape[1] < 4:
            if verbose:
                print(f"Warning: with_weight=True but data has {data.shape[1]} columns. "
                      f"Adding weight column with value 1.0 (dtype: {data.dtype})")
            data = np.column_stack([data, np.ones(len(data), dtype=data.dtype)])
        if random.shape[1] < 4:
            if verbose:
                print(f"Warning: with_weight=True but random has {random.shape[1]} columns. "
                      f"Adding weight column with value 1.0 (dtype: {random.dtype})")
            random = np.column_stack([random, np.ones(len(random), dtype=random.dtype)])
    
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
    
    # Step 1: Compute RR (internal + cross between neighboring subvolumes)
    # RR is the most time-consuming part, so we put it first and support caching
    RR_internal = None
    RR_cross = None
    RR_full = None

    # Initialize muedges (will be extracted from RR_full or use default)
    computed_muedges = None

    if rr_filename is not None and os.path.exists(rr_filename) and not force_rr:
        if verbose:
            print(f"Loading RR results from {rr_filename}...")
        RR_full_dict = joblib.load(rr_filename)
        RR_internal = RR_full_dict['RR_internal']
        RR_cross = RR_full_dict['RR_cross']
        RR_full = RR_full_dict['RR_full']
        # Load sedges and muedges from saved file
        saved_sedges = RR_full_dict.get('sedges', None)
        saved_muedges = RR_full_dict.get('muedges', None)
        saved_with_weight = RR_full_dict.get('with_weight', with_weight)
        
        # Check if parameters match - give warning for with_weight difference, error for others
        if saved_sedges is None or saved_muedges is None:
            raise ValueError("Cached RR file is corrupted: missing sedges or muedges.")
            
        if not (len(saved_sedges) == len(sedges) and np.allclose(saved_sedges, sedges)):
            raise ValueError(f"Cached RR sedges do not match current sedges. "
                           f"Cache file may be from different configuration.")
            
        # Check muedges consistency - compare with default generation from mubin
        expected_muedges = np.linspace(0, 1, mubin + 1)
        if not np.allclose(saved_muedges, expected_muedges):
            raise ValueError(f"Cached RR muedges do not match expected muedges for mubin={mubin}. "
                           f"Cache file may be from different configuration.")
        
        # Only give warning for with_weight difference (RR weight is usually 1 anyway)
        if saved_with_weight != with_weight:
            if verbose:
                print(f"Warning: loaded RR was computed with with_weight={saved_with_weight}, "
                      f"but current with_weight={with_weight}. Proceeding with cached result.")
        
        computed_muedges = saved_muedges
        sedges = saved_sedges
        
    elif rr_filename is not None and os.path.exists(rr_filename) and force_rr:
        # Force recompute - will overwrite existing cache
        if verbose:
            print(f"Force flag set, recomputing RR and overwriting {rr_filename}...")
        computed_muedges = None
    else:
        # No cache file or force_rr=False but file doesn't exist
        computed_muedges = None
    
    # Only recompute if forced or no valid cache
    if force_rr or computed_muedges is None:
        if verbose:
            if rr_filename is not None and os.path.exists(rr_filename) and force_rr:
                print(f"Force flag set, recomputing RR and saving to {rr_filename}...")
            elif rr_filename is not None:
                print(f"RR file not found, computing RR and saving to {rr_filename}...")
            else:
                print("Computing RR...")
        
        RR_internal, RR_cross, RR_full = compute_RR_JK(
            random_list, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, verbose, ngrids
        )

        # Generate muedges based on mubin (standard linear spacing)
        computed_muedges = np.linspace(0, 1, mubin + 1)

        if rr_filename is not None:
            if verbose:
                print(f"Saving RR results to {rr_filename}...")
            # Save only weighted npairs to reduce storage size
            RR_full_dict = {
                'RR_internal': extract_npairs(RR_internal, with_weight),
                'RR_cross': extract_npairs(RR_cross, with_weight),
                'RR_full': extract_npairs(RR_full, with_weight),
                'sedges': sedges,
                'muedges': computed_muedges,
                'with_weight': with_weight,
            }
            # Ensure directory exists
            os.makedirs(os.path.dirname(rr_filename), exist_ok=True)
            joblib.dump(RR_full_dict, rr_filename)
    
    # Step 2: Compute DD (internal + cross between neighboring subvolumes)
    if verbose:
        print("Computing DD...")
    DD_internal, DD_cross = compute_DD_JK(
        data_list, sedges, mubin, with_weight, boxsize,
        refine_factors, nthreads, verbose, ngrids
    )
    
    # Step 3: Compute DR (internal + cross between neighboring subvolumes)
    if verbose:
        print("Computing DR...")
    DR_internal, DR_cross = compute_DR_JK(
        data_list, random_list, sedges, mubin, with_weight, boxsize,
        refine_factors, nthreads, verbose, ngrids
    )
    
    if verbose:
        print("Computing Jackknife correlation functions...")
    
    # Step 4: Compute Jackknife correlation functions
    xi_results = compute_jackknife_xi(
        DD_internal, DD_cross, DR_internal, DR_cross, RR_internal, RR_cross,
        data_list, random_list, sedges, mubin, with_weight, muedges=computed_muedges
    )
    
    if verbose:
        print("Done!")
    
    if full_output:
        # Return only simplified arrays (weighted npairs) to reduce storage size
        # Keep xismu objects and edges, convert pair counts to simple arrays
        # The arrays about RR have been stored in the cache file, so we can skip them here
        xi_results_filtered = {
            'xismu_jk': xi_results['xismu_jk'],
            'xismu_full': xi_results['xismu_full'],
            'sedges': xi_results['sedges'],
            'muedges': xi_results['muedges'],
            'DD_internal': extract_npairs(DD_internal, with_weight),
            'DD_cross': extract_npairs(DD_cross, with_weight),
            'DR_internal': extract_npairs(DR_internal, with_weight),
            'DR_cross': extract_npairs(DR_cross, with_weight),
        }
        return xi_results_filtered
    else:
        return xi_results['xismu_jk']

def run_subsample_tpCF(data, random, sedges, mubin, with_weight,
                           boxsize, ngrids, refine_factors=(2, 2, 1), 
                           rr_filename=None, force_rr=False, nthreads=1, verbose=False, full_output=False):
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
    rr_filename : str or None
        Path to save/load RR computation results via joblib.
        If None, compute RR every time.
        If file exists and force_rr=False, load from file.
        If file doesn't exist or force_rr=True, compute and save to file.
    force_rr : bool
        If True, force recompute RR even if file exists.
    nthreads : int
        Number of threads for parallel computation.
    verbose : bool
        Whether to print progress information.
    full_output : bool
        If True, return full output with DD_sub, DR_sub.
    
    Returns
    -------
    result : dict or ndarray (xismu_sub)
        Dictionary containing:
        - 'xismu_sub': ndarray of xismu objects (n_cubes,) - xismu for each subvolume
        - 'sedges': separation bin edges (if full_output)
        - 'muedges': mu bin edges (if full_output)
        - 'DD_sub': dict of DD results for each subvolume (if full_output)
        - 'DR_sub': dict of DR results for each subvolume (if full_output)
    """
    # Robustness: ensure data and random have consistent dtypes
    # Use the higher precision dtype
    if data.dtype != random.dtype:
        # Determine higher precision dtype
        if data.dtype.itemsize > random.dtype.itemsize:
            higher_dtype = data.dtype
            lower_name = 'random'
        else:
            higher_dtype = random.dtype
            lower_name = 'data'
        
        if verbose:
            print(f"Warning: data dtype ({data.dtype}) and random dtype ({random.dtype}) differ. "
                  f"Converting both to higher precision dtype: {higher_dtype}")
        
        data = data.astype(higher_dtype)
        random = random.astype(higher_dtype)
    
    # Robustness: add weight column if with_weight=True but data/random have < 4 columns
    # Preserve original dtype to avoid implicit type promotion
    if with_weight:
        if data.shape[1] < 4:
            if verbose:
                print(f"Warning: with_weight=True but data has {data.shape[1]} columns. "
                      f"Adding weight column with value 1.0 (dtype: {data.dtype})")
            data = np.column_stack([data, np.ones(len(data), dtype=data.dtype)])
        if random.shape[1] < 4:
            if verbose:
                print(f"Warning: with_weight=True but random has {random.shape[1]} columns. "
                      f"Adding weight column with value 1.0 (dtype: {random.dtype})")
            random = np.column_stack([random, np.ones(len(random), dtype=random.dtype)])
    
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
    
    # Compute normalization for RR (same for all subvolumes)
    if with_weight:
        nr = np.sum(random[:, 3])
        sum_wr2 = np.sum(random[:, 3]**2)
    else:
        nr = len(random)
        sum_wr2 = nr
    norm_RR = nr * nr - sum_wr2
    
    # Initialize muedges (will be extracted from RR_result or use default)
    computed_muedges = None
    RR_result = None
    rr_from_cache = False

    # Try to load RR from cache
    if rr_filename is not None and os.path.exists(rr_filename) and not force_rr:
        if verbose:
            print(f"Loading RR results from {rr_filename}...")
        RR_cache_dict = joblib.load(rr_filename)
        RR_result = RR_cache_dict['RR_result']
        
        # Load sedges and muedges from saved file
        saved_sedges = RR_cache_dict.get('sedges', None)
        saved_muedges = RR_cache_dict.get('muedges', None)
        saved_with_weight = RR_cache_dict.get('with_weight', with_weight)
        saved_sub_boxsize = RR_cache_dict.get('sub_boxsize', None)
        
        # Check if parameters match
        if saved_sedges is None or saved_muedges is None:
            raise ValueError("Cached RR file is corrupted: missing sedges or muedges.")
            
        if not (len(saved_sedges) == len(sedges) and np.allclose(saved_sedges, sedges)):
            raise ValueError(f"Cached RR sedges do not match current sedges. "
                           f"Cache file may be from different configuration.")
            
        # Check muedges consistency
        expected_muedges = np.linspace(0, 1, mubin + 1)
        if not np.allclose(saved_muedges, expected_muedges):
            raise ValueError(f"Cached RR muedges do not match expected muedges for mubin={mubin}. "
                           f"Cache file may be from different configuration.")
        
        # Check sub_boxsize consistency
        if saved_sub_boxsize is not None and not np.allclose(saved_sub_boxsize, sub_boxsize):
            raise ValueError(f"Cached RR sub_boxsize does not match current sub_boxsize. "
                           f"Cache file may be from different configuration.")
        
        # Only give warning for with_weight difference
        if saved_with_weight != with_weight:
            if verbose:
                print(f"Warning: loaded RR was computed with with_weight={saved_with_weight}, "
                      f"but current with_weight={with_weight}. Proceeding with cached result.")
        
        computed_muedges = saved_muedges
        sedges = saved_sedges
        rr_from_cache = True
        
    elif rr_filename is not None and os.path.exists(rr_filename) and force_rr:
        # Force recompute - will overwrite existing cache
        if verbose:
            print(f"Force flag set, recomputing RR and overwriting {rr_filename}...")
        computed_muedges = None
    else:
        # No cache file or force_rr=False but file doesn't exist
        computed_muedges = None
    
    # Only recompute if forced or no valid cache
    if force_rr or not rr_from_cache:
        if verbose:
            if rr_filename is not None and os.path.exists(rr_filename) and force_rr:
                print(f"Force flag set, recomputing RR and saving to {rr_filename}...")
            elif rr_filename is not None:
                print(f"RR file not found, computing RR and saving to {rr_filename}...")
            else:
                print("Computing RR...")
            start_time = time.time()
        
        RR_result = call_DDsmu(
            random, None, sedges, mubin, with_weight, sub_boxsize[0],
            refine_factors, nthreads, autocorr=True
        )
        
        # Generate muedges based on mubin
        computed_muedges = np.linspace(0, 1, mubin + 1)
        
        if verbose:
            end_time = time.time()
            print(f"Done! Time elapsed: {end_time - start_time:.2f} seconds")

        if rr_filename is not None:
            if verbose:
                print(f"Saving RR results to {rr_filename}...")
            # Save complete RR_result (structured array) with self-pair correction applied
            RR_result_corrected = RR_result.copy()
            if sedges[0] == 0.0:
                # Apply self-pair correction before saving
                npairs = RR_result_corrected['npairs'].copy()
                npairs[0] -= nr
                RR_result_corrected['npairs'] = npairs
            
            RR_cache_dict = {
                'RR_result': RR_result_corrected,
                'sedges': sedges,
                'muedges': computed_muedges,
                'with_weight': with_weight,
                'sub_boxsize': sub_boxsize,
            }
            # Ensure directory exists
            os.makedirs(os.path.dirname(rr_filename), exist_ok=True)
            joblib.dump(RR_cache_dict, rr_filename)

    # Extract RR pair counts (same for all subvolumes)
    RR = extract_npairs(RR_result, with_weight).reshape(sbin, mubin)

    # Get mu edges
    mumax_str = "mumax" if "mumax" in RR_result.dtype.names else "mu_max"
    muedges = np.append([0], RR_result[mumax_str].reshape(sbin, mubin)[0])
    
    # RR self-pair correction
    RR_corrected = RR.copy()
    if sedges[0] == 0.0:
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
        DD_result = call_DDsmu(
            data_sub, None, sedges, mubin, with_weight, sub_boxsize[0],
            refine_factors, nthreads, autocorr=True
        )
        DD_sub[idx] = DD_result
        
        # Compute DR (data vs random, using sub_boxsize)
        DR_result = call_DDsmu(
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
        }
    else:
        xismu_sub


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
        
        result = call_DDsmu(
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
        result = call_DDsmu(
            data1, data2, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=False
        )
        result["npairs"] *= 2  # To be consistent with the number of pairs in the full catalog
        DD_cross[pair] = result
    
    return DD_internal, DD_cross


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
        
        result = call_DDsmu(
            data, random_sub, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=False
        )
        DR_internal[idx] = result
    
    # Step 2: Compute DR cross between neighboring subvolumes
    # DR is a cross-correlation, so (data_i, random_j) and (data_j, random_i) are different pairs
    # With autocorr=False, Corrfunc computes both directions, so we need to include both
    DR_cross = {}
    pairs_to_compute = []
    
    for idx in range(n_cubes):
        data = data_list[idx]
        if data is None or len(data) == 0:
            continue
        
        neighbors = get_neighbors(idx, n)
        
        for neighbor_idx in neighbors:
            random_neighbor = random_list[neighbor_idx]
            if random_neighbor is None or len(random_neighbor) == 0:
                continue
            
            # Direction 1: data_i vs random_j (where j is neighbor of i)
            pair_ij = (idx, neighbor_idx)
            pairs_to_compute.append((pair_ij, data, random_neighbor))
    
    iterator = tqdm(pairs_to_compute, desc="DR cross", disable=not verbose)
    for pair, data, random_sub in iterator:
        result = call_DDsmu(
            data, random_sub, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=False
        )
        DR_cross[pair] = result
    
    return DR_internal, DR_cross


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
        
        result = call_DDsmu(
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
        result = call_DDsmu(
            random1, random2, sedges, mubin, with_weight, boxsize,
            refine_factors, nthreads, autocorr=False
        )
        result["npairs"] *= 2  # To be consistent with the number of pairs in the full catalog
        RR_cross[pair] = result
    
    # Step 3: Sum all RR pairs - this is our final RR_full
    RR_full = np.zeros((sbin, mubin), dtype=np.float64)
    
    for idx, result in RR_internal.items():
        if result is not None:
            RR_full += extract_npairs(result, with_weight).reshape(sbin, mubin)
    
    for pair, result in RR_cross.items():
        if result is not None:
            RR_full += extract_npairs(result, with_weight).reshape(sbin, mubin)
    
    # Flatten to match extract_npairs output format
    RR_full = RR_full.flatten()
    
    return RR_internal, RR_cross, RR_full

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


def compute_total_pairs_and_norm(DD_internal=None, DD_cross=None, DR_internal=None, DR_cross=None, 
                                 RR_internal=None, RR_cross=None,
                                 data_list=None, random_list=None, sedges=None, mubin=None, 
                                 with_weight=False, muedges=None):
    """
    Compute total pair counts and normalization factors for full sample.
    Inputs should be already weighted logs from extract_npairs.
    
    Parameters
    ----------
    DD_internal : dict or ndarray, optional
        DD results for internal pairs (already processed by extract_npairs).
        If dict: mapping subvolume index to weighted pair counts.
        If ndarray: weighted pair counts array (sbin, mubin).
    DD_cross : dict or ndarray, optional  
        DD results for cross-pairs (already processed by extract_npairs).
    DR_internal : dict or ndarray, optional
        DR results for same-position pairs (already processed by extract_npairs).
    DR_cross : dict or ndarray, optional
        DR results for cross-pairs (already processed by extract_npairs).
    RR_internal : dict or ndarray, optional
        RR results for internal pairs (already processed by extract_npairs).
    RR_cross : dict or ndarray, optional
        RR results for cross-pairs (already processed by extract_npairs).
    data_list : list of ndarray, optional
        List of subvolume data arrays. Required for normalization computation.
    random_list : list of ndarray, optional
        List of subvolume random arrays. Required for normalization computation.
    sedges : array-like, optional
        Separation bin edges. Required for sbin.
    mubin : int, optional
        Number of mu bins. Required for pair computation.
    with_weight : bool
        Whether to use weights.
    muedges : array-like, optional
        Mu bin edges. Will be returned if provided.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'DD_total': ndarray - total DD pair counts (sbin, mubin)
        - 'DR_total': ndarray - total DR pair counts (sbin, mubin) 
        - 'RR_total': ndarray - total RR pair counts (sbin, mubin)
        - 'norm_DD': float - DD normalization factor (requires data_list)
        - 'norm_DR': float - DR normalization factor (requires data_list and random_list)
        - 'norm_RR': float - RR normalization factor (requires random_list)
        - 'data_weight': float - total data weight/count
        - 'random_weight': float - total random weight/count
        - 'data_weight_sq': float - sum of squared data weights
        - 'random_weight_sq': float - sum of squared random weights
        - 'muedges': array-like - mu bin edges (if provided)
        
    Raises
    ------
    ValueError
        If none of DD_internal/DD_cross, DR_internal/DR_cross, RR_internal/RR_cross are provided.
        If sedges or mubin is None when pair computation is requested.
    """
    import warnings
    
    # Determine sbin
    sbin = len(sedges) - 1 if sedges is not None else None
    
    if sbin is None and (DD_internal is not None or DD_cross is not None or 
                         DR_internal is not None or DR_cross is not None or
                         RR_internal is not None or RR_cross is not None):
        raise ValueError("sedges must be provided when computing pair totals")
    
    if mubin is None and (DD_internal is not None or DD_cross is not None or 
                          DR_internal is not None or DR_cross is not None or
                          RR_internal is not None or RR_cross is not None):
        raise ValueError("mubin must be provided when computing pair totals")
    
    # Check input availability and issue warnings
    dd_inputs_provided = DD_internal is not None or DD_cross is not None
    dr_inputs_provided = DR_internal is not None or DR_cross is not None
    rr_inputs_provided = RR_internal is not None or RR_cross is not None
    
    if not (dd_inputs_provided or dr_inputs_provided or rr_inputs_provided):
        raise ValueError("At least one pair of inputs (DD_internal/DD_cross, DR_internal/DR_cross, RR_internal/RR_cross) must be provided")
    
    if dd_inputs_provided and not (dr_inputs_provided and rr_inputs_provided):
        warnings.warn("Only DD_internal and/or DD_cross provided. DR and RR inputs are missing.")
    elif dr_inputs_provided and not (dd_inputs_provided and rr_inputs_provided):
        warnings.warn("Only DR_internal and/or DR_cross provided. DD and RR inputs are missing.")
    elif rr_inputs_provided and not (dd_inputs_provided and dr_inputs_provided):
        warnings.warn("Only RR_internal and/or RR_cross provided. DD and DR inputs are missing.")
    elif (dd_inputs_provided ^ dr_inputs_provided) or (dd_inputs_provided ^ rr_inputs_provided) or (dr_inputs_provided ^ rr_inputs_provided):
        # XOR condition - partial inputs
        warnings.warn("Partial inputs detected. Some pair types are missing.")
    
    result = {
        'DD_total': None,
        'DR_total': None, 
        'RR_total': None,
        'norm_DD': None,
        'norm_DR': None,
        'norm_RR': None,
        'data_weight': None,
        'random_weight': None,
        'data_weight_sq': None,
        'random_weight_sq': None,
        'muedges': muedges
    }
    
    # Helper function to sum contributions
    def _sum_contributions(internal, cross, sbin, mubin):
        total = np.zeros((sbin, mubin))
        
        def _extract_and_add(val, total, sbin, mubin):
            """Extract npairs from value (if structured array) and add to total."""
            if val is None:
                return
            # Check if it's a structured array (DDsmu result)
            if isinstance(val, np.ndarray) and val.dtype.names is not None:
                val = extract_npairs(val, with_weight)
            # Now val should be a simple array
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    total += val.reshape(sbin, mubin)
                else:
                    total += val
        
        if internal is not None:
            if isinstance(internal, dict):
                for idx, val in internal.items():
                    _extract_and_add(val, total, sbin, mubin)
            else:  # ndarray
                _extract_and_add(internal, total, sbin, mubin)
        
        if cross is not None:
            if isinstance(cross, dict):
                for pair, val in cross.items():
                    _extract_and_add(val, total, sbin, mubin)
            else:  # ndarray
                _extract_and_add(cross, total, sbin, mubin)
        
        return total
    
    # Compute DD_total if inputs are provided
    if DD_internal is not None or DD_cross is not None:
        result['DD_total'] = _sum_contributions(DD_internal, DD_cross, sbin, mubin)
    
    # Compute DR_total if inputs are provided
    if DR_internal is not None or DR_cross is not None:
        result['DR_total'] = _sum_contributions(DR_internal, DR_cross, sbin, mubin)
    
    # Compute RR_total if inputs are provided
    if RR_internal is not None or RR_cross is not None:
        result['RR_total'] = _sum_contributions(RR_internal, RR_cross, sbin, mubin)
    
    # Compute normalization factors and weights
    if data_list is not None:
        if with_weight:
            result['data_weight'] = sum(np.sum(d[:, 3]) for d in data_list if d is not None and len(d) > 0)
            result['data_weight_sq'] = sum(np.sum(d[:, 3]**2) for d in data_list if d is not None and len(d) > 0)
        else:
            result['data_weight'] = sum(len(d) for d in data_list if d is not None)
            result['data_weight_sq'] = result['data_weight']
        
        # norm_DD = N_d * N_d - sum(w_d^2)
        result['norm_DD'] = result['data_weight'] * result['data_weight'] - result['data_weight_sq']
    
    if random_list is not None:
        if with_weight:
            result['random_weight'] = sum(np.sum(r[:, 3]) for r in random_list if r is not None and len(r) > 0)
            result['random_weight_sq'] = sum(np.sum(r[:, 3]**2) for r in random_list if r is not None and len(r) > 0)
        else:
            result['random_weight'] = sum(len(r) for r in random_list if r is not None)
            result['random_weight_sq'] = result['random_weight']
        
        # norm_RR = N_r * N_r - sum(w_r^2)
        result['norm_RR'] = result['random_weight'] * result['random_weight'] - result['random_weight_sq']
    
    # norm_DR = N_d * N_r
    if result['data_weight'] is not None and result['random_weight'] is not None:
        result['norm_DR'] = result['data_weight'] * result['random_weight']
    
    return result


def compute_jackknife_deductions(jk_idx, DD_internal=None, DD_cross=None, DR_internal=None, DR_cross=None,
                                RR_internal=None, RR_cross=None, data_list=None, random_list=None,
                                sedges=None, mubin=None, with_weight=False, ngrids_1d=None):
    """
    Compute deductions (pairs and normalization) for excluding a subvolume.
    Returns what needs to be subtracted from totals.
    
    Parameters
    ----------
    jk_idx : int
        Index of subvolume to exclude.
    DD_internal : dict, optional
        DD results for internal pairs (already processed by extract_npairs).
    DD_cross : dict, optional
        DD results for cross pairs (already processed by extract_npairs).
    DR_internal : dict, optional
        DR results for internal pairs (already processed by extract_npairs).
    DR_cross : dict, optional
        DR results for cross pairs (already processed by extract_npairs).
    RR_internal : dict, optional
        RR results for internal pairs (already processed by extract_npairs).
    RR_cross : dict, optional
        RR results for cross pairs (already processed by extract_npairs).
    data_list : list of ndarray, optional
        Data subvolume lists.
    random_list : list of ndarray, optional
        Random subvolume lists.
    sedges : array-like, optional
        Separation bin edges.
    mubin : int, optional
        Number of mu bins.
    with_weight : bool
        Whether to use weights.
    ngrids_1d : int, optional
        Number of divisions along one axis.
    
    Returns
    -------
    result : dict
        Dictionary with keys (values are None if insufficient data):
        - 'DD_deduct': ndarray - DD pairs to deduct (sbin, mubin)
        - 'DR_deduct': ndarray - DR pairs to deduct (sbin, mubin)
        - 'RR_deduct': ndarray - RR pairs to deduct (sbin, mubin)
        - 'excluded_data_weight': float - excluded data weight/count
        - 'excluded_random_weight': float - excluded random weight/count
        - 'excluded_data_weight_sq': float - sum of squared data weights (for norm_DD)
        - 'excluded_random_weight_sq': float - sum of squared random weights (for norm_RR)
        
    Raises
    ------
    ValueError
        If sedges or mubin is None when pair computation is requested.
        If none of DD_internal/DD_cross, DR_internal/DR_cross, RR_internal/RR_cross are provided.
    """
    import warnings
    
    sbin = len(sedges) - 1 if sedges is not None else None
    
    result = {
        'DD_deduct': None,
        'DR_deduct': None,
        'RR_deduct': None,
        'excluded_data_weight': None,
        'excluded_random_weight': None,
        'excluded_data_weight_sq': None,
        'excluded_random_weight_sq': None
    }
    
    # Check input availability
    dd_inputs_provided = DD_internal is not None or DD_cross is not None
    dr_inputs_provided = DR_internal is not None or DR_cross is not None
    rr_inputs_provided = RR_internal is not None or RR_cross is not None
    
    if not (dd_inputs_provided or dr_inputs_provided or rr_inputs_provided):
        raise ValueError("At least one pair of inputs (DD_internal/DD_cross, DR_internal/DR_cross, RR_internal/RR_cross) must be provided")
    
    # Check sbin and mubin for pair computation
    if (dd_inputs_provided or dr_inputs_provided or rr_inputs_provided) and (sbin is None or mubin is None):
        raise ValueError("sedges and mubin must be provided when computing pair deductions")
    
    # Issue warnings for partial inputs
    if dd_inputs_provided and not (dr_inputs_provided and rr_inputs_provided):
        warnings.warn("Only DD_internal and/or DD_cross provided. DR and RR inputs are missing.")
    elif dr_inputs_provided and not (dd_inputs_provided and rr_inputs_provided):
        warnings.warn("Only DR_internal and/or DR_cross provided. DD and RR inputs are missing.")
    elif rr_inputs_provided and not (dd_inputs_provided and dr_inputs_provided):
        warnings.warn("Only RR_internal and/or RR_cross provided. DD and DR inputs are missing.")
    elif (dd_inputs_provided ^ dr_inputs_provided) or (dd_inputs_provided ^ rr_inputs_provided) or (dr_inputs_provided ^ rr_inputs_provided):
        # XOR condition - partial inputs
        warnings.warn("Partial inputs detected. Some pair types are missing.")
    
    # Check ngrids_1d for cross-pair computation
    if ngrids_1d is None and (DD_cross is not None or DR_cross is not None or RR_cross is not None):
        warnings.warn("ngrids_1d is None but cross-pair inputs are provided. Cross-pair deductions will be skipped.")
        
    # Compute DD deduction
    if dd_inputs_provided:
        DD_deduct = np.zeros((sbin, mubin))
        
        # Internal DD of excluded subvolume
        if DD_internal is not None and jk_idx in DD_internal:
            if DD_internal[jk_idx] is not None:
                DD_deduct += extract_npairs(DD_internal[jk_idx], with_weight).reshape(sbin, mubin)
        
        # Cross DD with neighbors
        if DD_cross is not None and ngrids_1d is not None:
            neighbors = get_neighbors(jk_idx, ngrids_1d)
            for neighbor_idx in neighbors:
                pair = (min(jk_idx, neighbor_idx), max(jk_idx, neighbor_idx))
                if pair in DD_cross and DD_cross[pair] is not None:
                    DD_deduct += extract_npairs(DD_cross[pair], with_weight).reshape(sbin, mubin)
                    
        result['DD_deduct'] = DD_deduct
    
    # Compute DR deduction
    if dr_inputs_provided:
        DR_deduct = np.zeros((sbin, mubin))
        
        # Internal DR of excluded subvolume
        if DR_internal is not None and jk_idx in DR_internal:
            if DR_internal[jk_idx] is not None:
                DR_deduct += extract_npairs(DR_internal[jk_idx], with_weight).reshape(sbin, mubin)
        
        # Cross DR with neighbors (both directions)
        if DR_cross is not None and ngrids_1d is not None:
            neighbors = get_neighbors(jk_idx, ngrids_1d)
            for neighbor_idx in neighbors:
                # Direction 1: (jk_idx, neighbor_idx)
                pair_ij = (jk_idx, neighbor_idx)
                if pair_ij in DR_cross and DR_cross[pair_ij] is not None:
                    DR_deduct += extract_npairs(DR_cross[pair_ij], with_weight).reshape(sbin, mubin)
                
                # Direction 2: (neighbor_idx, jk_idx)
                pair_ji = (neighbor_idx, jk_idx)
                if pair_ji in DR_cross and DR_cross[pair_ji] is not None:
                    DR_deduct += extract_npairs(DR_cross[pair_ji], with_weight).reshape(sbin, mubin)
                    
        result['DR_deduct'] = DR_deduct
    
    # Compute RR deduction
    if rr_inputs_provided:
        RR_deduct = np.zeros((sbin, mubin))
        
        # Internal RR of excluded subvolume
        if RR_internal is not None and jk_idx in RR_internal:
            if RR_internal[jk_idx] is not None:
                RR_deduct += extract_npairs(RR_internal[jk_idx], with_weight).reshape(sbin, mubin)
        
        # Cross RR with neighbors
        if RR_cross is not None and ngrids_1d is not None:
            neighbors = get_neighbors(jk_idx, ngrids_1d)
            for neighbor_idx in neighbors:
                pair = (min(jk_idx, neighbor_idx), max(jk_idx, neighbor_idx))
                if pair in RR_cross and RR_cross[pair] is not None:
                    RR_deduct += extract_npairs(RR_cross[pair], with_weight).reshape(sbin, mubin)
                    
        result['RR_deduct'] = RR_deduct
    
    # Compute excluded weights (for external normalization computation)
    if data_list is not None and jk_idx < len(data_list):
        if data_list[jk_idx] is not None and len(data_list[jk_idx]) > 0:
            if with_weight:
                result['excluded_data_weight'] = np.sum(data_list[jk_idx][:, 3])
                result['excluded_data_weight_sq'] = np.sum(data_list[jk_idx][:, 3]**2)
            else:
                result['excluded_data_weight'] = len(data_list[jk_idx])
                result['excluded_data_weight_sq'] = len(data_list[jk_idx])
        else:
            result['excluded_data_weight'] = 0
            result['excluded_data_weight_sq'] = 0
    else:
        if data_list is not None:
            warnings.warn(f"jk_idx {jk_idx} is out of range for data_list (length {len(data_list)})")
    
    if random_list is not None and jk_idx < len(random_list):
        if random_list[jk_idx] is not None and len(random_list[jk_idx]) > 0:
            if with_weight:
                result['excluded_random_weight'] = np.sum(random_list[jk_idx][:, 3])
                result['excluded_random_weight_sq'] = np.sum(random_list[jk_idx][:, 3]**2)
            else:
                result['excluded_random_weight'] = len(random_list[jk_idx])
                result['excluded_random_weight_sq'] = len(random_list[jk_idx])
        else:
            result['excluded_random_weight'] = 0
            result['excluded_random_weight_sq'] = 0
    else:
        if random_list is not None:
            warnings.warn(f"jk_idx {jk_idx} is out of range for random_list (length {len(random_list)})")
    
    return result


def compute_jackknife_xi(DD_internal, DD_cross, DR_internal, DR_cross, RR_internal, RR_cross,
                         data_list, random_list, sedges, mubin, with_weight, muedges=None):
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
        Each value can be a structured array or a simplified ndarray (weighted npairs).
    RR_internal : dict
        RR results for internal pairs of each subvolume.
        Each value can be a structured array or a simplified ndarray.
    RR_cross : dict
        RR results for cross-pairs between neighboring subvolumes.
        Each value can be a structured array or a simplified ndarray.
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
    muedges : array-like, optional
        Mu bin edges. If None, will use default linear spacing from 0 to 1.
    
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
        - 'RR_internal': dict of internal RR results
        - 'RR_cross': dict of cross RR results
    """
    n_cubes = len(data_list)
    ngrids_1d = round(n_cubes ** (1/3))
    
    # Compute total pairs and normalization factors
    total_result = compute_total_pairs_and_norm(
        DD_internal, DD_cross, DR_internal, DR_cross, RR_internal, RR_cross,
        data_list, random_list, sedges, mubin, with_weight, muedges
    )
    
    # Check if we have sufficient data
    if any(v is None for v in [total_result['DD_total'], total_result['DR_total'], total_result['RR_total'], 
                             total_result['norm_DD'], total_result['norm_DR'], total_result['norm_RR']]):
        # Return empty results
        xismu_jk = np.empty(n_cubes, dtype=object)
        for i in range(n_cubes):
            xismu_jk[i] = None
        xismu_full = None
        return {
            'xismu_jk': xismu_jk,
            'xismu_full': xismu_full,
            'sedges': sedges,
            'muedges': total_result['muedges'],
            'DD_internal': DD_internal,
            'DD_cross': DD_cross,
            'DR_internal': DR_internal,
            'DR_cross': DR_cross,
            'RR_internal': RR_internal,
            'RR_cross': RR_cross,
        }
    
    # Create xismu for full sample
    xismu_full = create_xismu_from_pairs(
        total_result['DD_total'], total_result['DR_total'], total_result['RR_total'],
        total_result['norm_DD'], total_result['norm_DR'], total_result['norm_RR'],
        sedges, total_result['muedges']
    )
    
    # Extract total weights for convenience
    total_data_weight = total_result['data_weight']
    total_random_weight = total_result['random_weight']
    total_data_weight_sq = total_result['data_weight_sq']
    total_random_weight_sq = total_result['random_weight_sq']
    
    # Jackknife samples
    xismu_jk = np.empty(n_cubes, dtype=object)
    
    for jk_idx in range(n_cubes):
        # Compute deductions for this Jackknife sample
        deductions = compute_jackknife_deductions(
            jk_idx, DD_internal, DD_cross, DR_internal, DR_cross,
            RR_internal, RR_cross, data_list, random_list,
            sedges, mubin, with_weight, ngrids_1d
        )
        
        # Compute Jackknife sample pairs by applying deductions
        DD_jk = None
        if total_result['DD_total'] is not None and deductions['DD_deduct'] is not None:
            DD_jk = total_result['DD_total'].copy() - deductions['DD_deduct']
        
        DR_jk = None
        if total_result['DR_total'] is not None and deductions['DR_deduct'] is not None:
            DR_jk = total_result['DR_total'].copy() - deductions['DR_deduct']
            
        RR_jk = None
        if total_result['RR_total'] is not None and deductions['RR_deduct'] is not None:
            RR_jk = total_result['RR_total'].copy() - deductions['RR_deduct']
        
        # Compute Jackknife sample normalizations
        # Jackknife weights
        jk_data_weight = total_data_weight - deductions['excluded_data_weight']
        jk_random_weight = total_random_weight - deductions['excluded_random_weight']
        jk_data_weight_sq = total_data_weight_sq - deductions['excluded_data_weight_sq']
        jk_random_weight_sq = total_random_weight_sq - deductions['excluded_random_weight_sq']
        
        # norm_DD_jk = N_d_jk * N_d_jk - sum(w_d_jk^2)
        norm_DD_jk = jk_data_weight * jk_data_weight - jk_data_weight_sq
        
        # norm_DR_jk = N_d_jk * N_r_jk
        norm_DR_jk = jk_data_weight * jk_random_weight
        
        # norm_RR_jk = N_r_jk * N_r_jk - sum(w_r_jk^2)
        norm_RR_jk = jk_random_weight * jk_random_weight - jk_random_weight_sq
        
        # Apply self-pair correction for s=0, mu=0 bin
        # When excluding a subvolume, we need to add back the self-pairs that were removed
        # because the excluded particles no longer contribute to self-pairs
        if sedges is not None and sedges[0] == 0.0:
            if DD_jk is not None and deductions['excluded_data_weight'] is not None:
                DD_jk[0, 0] += deductions['excluded_data_weight']
            if RR_jk is not None and deductions['excluded_random_weight'] is not None:
                RR_jk[0, 0] += deductions['excluded_random_weight']
        
        # Create xismu for this JK sample if we have valid results
        if all(v is not None for v in [DD_jk, DR_jk, RR_jk, norm_DD_jk, norm_DR_jk, norm_RR_jk]):
            xismu_jk[jk_idx] = create_xismu_from_pairs(
                DD_jk, DR_jk, RR_jk, norm_DD_jk, norm_DR_jk, norm_RR_jk,
                sedges, total_result['muedges']
            )
        else:
            xismu_jk[jk_idx] = None
    
    return {
        'xismu_jk': xismu_jk,
        'xismu_full': xismu_full,
        'sedges': sedges,
        'muedges': total_result['muedges'],
        'DD_internal': DD_internal,
        'DD_cross': DD_cross,
        'DR_internal': DR_internal,
        'DR_cross': DR_cross,
        'RR_internal': RR_internal,
        'RR_cross': RR_cross,
    }

def get_cov_matrix(array, cov_type="normal", need_slice=slice(None, -1, None), use_Hartlab=False):
    """ To get the covariance matrix of specific array
    
    Parameters
    ----------
    array : ndarray
        Input array with shape (n_samples, n_features) or (n_samples,)
    cov_type : str, optional
        The type of covariance matrix, support "normal", "jk" and "subsample", by default "normal"
    use_Hartlab : bool, optional
        Whether to apply Hartlab correction for biased estimation,
        only applicable for "subsample" and "jk" types, by default False
    
    Returns
    -------
    ndarray
        The covariance matrix after covariance correction
    """
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError(f"Input array must be 2D, but got {array.ndim}D")
    n_samples, n_features = array.shape

    array = array[:, need_slice]
    
    if cov_type == "normal":
        # Standard sample covariance (unbiased, divides by N-1)
        cov = np.cov(array, rowvar=False, bias=False)
        
    elif cov_type == "subsample":
        # Subsample covariance from run_subsample_tpCF
        # Each subbox is an independent estimate of the full box statistics
        # The samples (n_samples) correspond to the number of subboxes, which 
        # represents the volume ratio between full box and subbox.
        # 
        # Standard covariance estimate from np.cov gives:
        # cov_np = 1/(N-1) * sum((xi - x_mean)^2)
        # 
        # To estimate full box covariance from subbox measurements:
        # cov_full = cov_sub / n_samples (volume correction)
        
        # np.cov with bias=False gives: 1/(N-1) * sum((xi - x_mean)^2)
        cov = np.cov(array, rowvar=False, bias=False)
        
        # Volume correction: divide by number of subboxes
        correction_factor = 1.0 / n_samples
        cov = cov * correction_factor
        
        # Apply Hartlab correction if requested
        if use_Hartlab:
            # Hartlab correction factor for subsample covariance
            hartlab_factor = (n_samples - 1) / (n_samples - n_features - 2)
            cov = cov * hartlab_factor
        
    elif cov_type == "jk":
        # Jackknife covariance from run_jackknife_tpCF
        # 
        # Standard Jackknife covariance formula:
        # cov_jk = (N-1)/N * sum((xi - x_mean)^2)
        # 
        # But np.cov with bias=False gives:
        # cov_np = 1/(N-1) * sum((xi - x_mean)^2)
        # 
        # So the conversion factor is:
        # cov_jk = (N-1)/N * (N-1) * cov_np = (N-1)^2 / N * cov_np
        
        # np.cov with bias=False gives: 1/(N-1) * sum((xi - x_mean)^2)
        cov = np.cov(array, rowvar=False, bias=False)
        
        # Jackknife correction: (N-1)^2 / N
        correction_factor = (n_samples - 1) ** 2 / n_samples
        cov = cov * correction_factor
        
        # Apply Hartlab correction if requested
        if use_Hartlab:
            # Hartlab correction factor for Jackknife covariance
            hartlab_factor = (n_samples - 1) / (n_samples - n_features - 2)
            cov = cov * hartlab_factor
        
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}. Supported types are 'normal', 'subsample', 'jk'")
    
    return cov

def get_std_array_from_cov(cov_matrix):
    return np.sqrt(np.diag(cov_matrix))

def box_shift_perodic(data, shift, boxsize, inplace=False):
    """
    Apply periodic boundary conditions to shift box positions.
    
    Parameters
    ----------
    data : ndarray
        Input data array with shape (N, D), where D >= 3.
        First three columns are x, y, z coordinates.
    shift : array_like
        Three-element sequence (dx, dy, dz) controlling translation distances.
        Can be list, tuple, or numpy array.
    boxsize : float or array_like
        Size of the box. If float, assumes cubic box of that size.
        If array-like, should have 3 elements for (Lx, Ly, Lz).
    inplace : bool, optional
        Whether to modify the input data in-place, by default False
    
    Returns
    -------
    ndarray
        New data array with shifted coordinates under periodic boundary conditions.
        Same shape as input data.
    """
    import numpy as np
    
    if not inplace:
        data = data.copy()
    shift = np.asarray(shift, dtype=float)
    boxsize = np.asarray(boxsize, dtype=float)
    
    if data.ndim != 2:
        raise ValueError("Input data must be 2D array")
    if data.shape[1] < 3:
        raise ValueError("Input data must have at least 3 columns")
    if shift.shape != (3,):
        raise ValueError("Shift must be a 3-element sequence")
    if boxsize.ndim == 0:
        boxsize = np.array([boxsize, boxsize, boxsize])
    elif boxsize.shape != (3,):
        raise ValueError("Boxsize must be scalar or 3-element sequence")
    
    # Apply shift to first three columns (x, y, z)
    data[:, :3] += shift
    
    # Apply periodic boundary conditions with given box size
    data[:, 0] = data[:, 0] % boxsize[0]
    data[:, 1] = data[:, 1] % boxsize[1]
    data[:, 2] = data[:, 2] % boxsize[2]
    
    return data

def cal_Fisher_matrix(func, best_fit, cov_matrix, delta=None, computed_jac=None, return_jac=False):
    """
    Calculate Fisher matrix from covariance matrix and model function.

    The Fisher information matrix is computed using the finite difference
    approximation of the gradient of the model function with respect to
    parameters:

    F = (∂μ/∂θ)^T * C^{-1} * (∂μ/∂θ)

    where μ = func(θ) is the model prediction, C is the covariance matrix,
    and ∂μ/∂θ is the Jacobian matrix.

    Parameters
    ----------
    func : callable
        Model function that takes parameters as individual arguments.
        Example: for 2 parameters, func(x1, x2).
    best_fit : array_like
        Best-fit parameter values.
    cov_matrix : ndarray
        Covariance matrix of the parameters.
    delta : float or array_like, optional
        Finite difference step size. If None, automatically determined
        using optimal step size δ = ε^(1/3) * max(|θ|, 1) for each parameter,
        where ε ≈ 2.22e-16 is machine epsilon. This gives 4th order
        accuracy for central differences. Can also be a single float
        applied to all parameters, or an array matching best_fit length.

    Returns
    -------
    ndarray
        Fisher information matrix with shape (n_params, n_params).

    Notes
    -----
    The automatic step size is based on the optimal step for numerical
    differentiation to minimize truncation and round-off errors:

    δ_optimal ≈ ε^(1/3) * max(|x|, 1)

    where ε is machine epsilon. This balances the truncation error
    (∝ δ²) and round-off error (∝ 1/δ) in central differences.
    """
    import numpy as np

    if computed_jac is None:
        # Convert inputs to numpy arrays
        best_fit = np.atleast_1d(best_fit)
        n_params = len(best_fit)

        # Determine step sizes for each parameter
        if delta is None:
            # Automatic step size: δ = ε^(1/3) * max(|θ|, 1)
            # ε^(1/3) ≈ 6.05e-6 for double precision
            machine_eps = np.finfo(float).eps
            delta_factor = machine_eps ** (1/3)
            delta = delta_factor * np.maximum(np.abs(best_fit), 1.0)
        elif np.isscalar(delta):
            delta = np.full(n_params, delta)
        else:
            delta = np.atleast_1d(delta)
            if len(delta) != n_params:
                raise ValueError(f"delta length {len(delta)} does not match "
                            f"number of parameters {n_params}")

        # Compute inverse covariance matrix
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular, cannot compute inverse")

        # Compute Jacobian: ∂μ/∂θ for each parameter
        # We use central difference: (f(x+δ) - f(x-δ)) / (2δ)
        jacobian = np.zeros(n_params)

        # Evaluate function at best fit point to determine output dimension
        f0 = func(*best_fit)

        # Determine if model output is scalar or vector
        if hasattr(f0, '__len__') and not isinstance(f0, (float, int)):
            f0 = np.asarray(f0)
            n_output = f0.shape[0] if f0.ndim > 0 else 1
        else:
            f0 = float(f0)
            n_output = 1

        # Validate cov_matrix shape
        cov_matrix = np.atleast_2d(cov_matrix)
        if cov_matrix.shape != (n_output, n_output):
            raise ValueError(f"cov_matrix shape {cov_matrix.shape} does not match "
                            f"model output dimension {n_output}")

        # Initialize Jacobian matrix (n_output x n_params)
        jacobian = np.zeros((n_output, n_params))

        # Compute gradient for each parameter using central differences
        for i in range(n_params):
            # Forward point
            theta_plus = best_fit.copy()
            theta_plus[i] += delta[i]
            f_plus = func(*theta_plus)

            # Backward point
            theta_minus = best_fit.copy()
            theta_minus[i] -= delta[i]
            f_minus = func(*theta_minus)

            # Central difference
            if n_output == 1:
                jacobian[0, i] = (f_plus - f_minus) / (2 * delta[i])
            else:
                jacobian[:, i] = (np.asarray(f_plus) - np.asarray(f_minus)) / (2 * delta[i])

    # Compute Fisher matrix: F = J^T * C^{-1} * J
    # jacobian shape: (n_output, n_params)
    # cov_inv shape: (n_params, n_params)
    # Result: (n_params, n_output) @ (n_params, n_params) @ (n_output, n_params)
    #        = (n_params, n_params) @ (n_output, n_params) -> need to transpose jacobian
    else:
        jacobian = computed_jac
        cov_inv = np.linalg.inv(cov_matrix)
    fisher = jacobian.T @ cov_inv @ jacobian

    if return_jac:
        return fisher, jacobian
    else:
        return fisher


def cal_Fisher_matrix_from_precomputed(precomputed_data, best_fit, delta, cov_matrix, return_jac=False):
    """
    Calculate Fisher matrix from precomputed function values at parameter points.

    This function is useful when function evaluations are expensive and have been
    precomputed at points around the best-fit parameters. It computes the Jacobian
    matrix from the precomputed values and then calls cal_Fisher_matrix.

    The Fisher information matrix is computed as:
    F = (∂μ/∂θ)^T * C^{-1} * (∂μ/∂θ)

    where the Jacobian ∂μ/∂θ is computed using central differences from the
    precomputed function values.

    Parameters
    ----------
    precomputed_data : dict
        Dictionary containing precomputed function values. Format:
        {param_index: {'plus': f(theta + delta_i), 'minus': f(theta - delta_i)}}
        
        Example for 2 parameters:
        {
            0: {'plus': array([...]), 'minus': array([...])},  # f(best_fit + delta[0], best_fit[1])
            1: {'plus': array([...]), 'minus': array([...])}   # f(best_fit[0], best_fit[1] + delta[1])
        }
        
        For scalar function outputs, use float values instead of arrays.
    
    best_fit : array_like
        Best-fit parameter values.
    
    delta : float or array_like
        Finite difference step size(s). Can be:
        - A single float applied to all parameters
        - An array matching best_fit length
    
    cov_matrix : ndarray
        Covariance matrix of the model outputs.
    
    return_jac : bool, default False
        If True, return both Fisher matrix and Jacobian matrix.

    Returns
    -------
    fisher : ndarray
        Fisher information matrix with shape (n_params, n_params).
    
    jacobian : ndarray, optional
        Jacobian matrix with shape (n_output, n_params). Only returned if
        return_jac=True.

    Examples
    --------
    >>> # Precompute expensive function evaluations
    >>> best_fit = [1.0, 2.0]
    >>> delta = [0.01, 0.02]
    >>> 
    >>> # Evaluate at perturbed points (can be done in parallel or saved from previous runs)
    >>> precomputed_data = {
    ...     0: {
    ...         'plus': expensive_model(1.01, 2.0),   # best_fit[0] + delta[0]
    ...         'minus': expensive_model(0.99, 2.0)   # best_fit[0] - delta[0]
    ...     },
    ...     1: {
    ...         'plus': expensive_model(1.0, 2.02),   # best_fit[1] + delta[1]
    ...         'minus': expensive_model(1.0, 1.98)   # best_fit[1] - delta[1]
    ...     }
    ... }
    >>> 
    >>> # Compute Fisher matrix from precomputed values
    >>> fisher = cal_Fisher_matrix_from_precomputed(
    ...     precomputed_data, best_fit, delta, cov_matrix
    ... )

    Notes
    -----
    This function avoids redundant function evaluations by using precomputed
    values, which is particularly useful when:
    - Function evaluations are computationally expensive
    - Function values have been computed in parallel
    - Function values are available from previous optimization/sampling runs
    
    The Jacobian is computed using central differences:
    ∂f/∂θ_i ≈ [f(θ + δ_i e_i) - f(θ - δ_i e_i)] / (2 δ_i)
    """
    import numpy as np
    
    # Convert inputs to numpy arrays
    best_fit = np.atleast_1d(best_fit)
    n_params = len(best_fit)
    
    # Validate delta
    if np.isscalar(delta):
        delta = np.full(n_params, delta)
    else:
        delta = np.atleast_1d(delta)
        if len(delta) != n_params:
            raise ValueError(f"delta length {len(delta)} does not match "
                           f"number of parameters {n_params}")
    
    # Validate precomputed_data
    if not isinstance(precomputed_data, dict):
        raise TypeError("precomputed_data must be a dictionary")
    
    if set(precomputed_data.keys()) != set(range(n_params)):
        raise ValueError(f"precomputed_data must contain keys 0, 1, ..., {n_params-1}, "
                        f"but got keys {sorted(precomputed_data.keys())}")
    
    # Check structure of each entry
    for i in range(n_params):
        if 'plus' not in precomputed_data[i] or 'minus' not in precomputed_data[i]:
            raise ValueError(f"precomputed_data[{i}] must contain 'plus' and 'minus' keys")
    
    # Get the first function value to determine output dimension
    first_value = precomputed_data[0]['plus']
    if hasattr(first_value, '__len__') and not isinstance(first_value, (float, int)):
        first_value = np.asarray(first_value)
        n_output = first_value.shape[0] if first_value.ndim > 0 else 1
    else:
        n_output = 1
    
    # Initialize Jacobian matrix (n_output x n_params)
    jacobian = np.zeros((n_output, n_params))
    
    # Compute Jacobian from precomputed values
    for i in range(n_params):
        f_plus = precomputed_data[i]['plus']
        f_minus = precomputed_data[i]['minus']
        
        # Convert to numpy arrays if needed
        if n_output > 1:
            f_plus = np.asarray(f_plus)
            f_minus = np.asarray(f_minus)
        
        # Central difference
        if n_output == 1:
            jacobian[0, i] = (f_plus - f_minus) / (2 * delta[i])
        else:
            jacobian[:, i] = (f_plus - f_minus) / (2 * delta[i])
    
    # Call cal_Fisher_matrix with computed Jacobian
    return cal_Fisher_matrix(
        func=None,  # Not needed when computed_jac is provided
        best_fit=best_fit,
        cov_matrix=cov_matrix,
        computed_jac=jacobian,
        return_jac=return_jac
    )


def _compute_ellipse_params_from_fisher(fisher, confidence_level=0.683):
    """
    Compute ellipse parameters from Fisher matrix (internal helper function).

    Parameters
    ----------
    fisher : ndarray
        Fisher information matrix, must be 2x2.
    confidence_level : float, default 0.683
        Confidence level for the ellipse (0 < confidence_level < 1).

    Returns
    -------
    dict
        Dictionary containing:
        - 'semi_minor' : float - semi-minor axis length (smaller)
        - 'semi_major' : float - semi-major axis length (larger)
        - 'angle_rad' : float - rotation angle in radians
        - 'angle_deg' : float - rotation angle in degrees
        - 'eigenvals' : ndarray - eigenvalues [λ_small, λ_large]
        - 'eigenvecs' : ndarray - eigenvectors as columns
        - 'delta_chi2' : float - chi-squared critical value

    Raises
    ------
    ValueError
        If fisher is not a 2x2 matrix or not positive definite.
    """
    import numpy as np
    from scipy.stats import chi2

    # Validate fisher matrix dimensions
    fisher = np.atleast_2d(fisher)
    if fisher.shape != (2, 2):
        raise ValueError(f"Fisher matrix must be 2x2 for ellipse computation, "
                        f"got shape {fisher.shape}")

    # Validate confidence level
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")

    # Chi-squared critical value for 2 degrees of freedom
    delta_chi2 = chi2.ppf(confidence_level, df=2)

    # Eigen-decomposition of Fisher matrix
    # Fisher matrix F is the inverse of parameter covariance matrix C: F = C^{-1}
    # The error ellipse satisfies: (θ - θ₀)^T · F · (θ - θ₀) = Δχ²
    #
    # Eigen-decomposition: F = Q · Λ · Q^T
    #   where Λ = diag(λ₁, λ₂) with λ₁ ≥ λ₂ > 0 (eigenvalues)
    #   Q is orthogonal matrix (rotation) from eigenvectors
    #
    # In eigenvector coordinates: λ₁ u₁² + λ₂ u₂² = Δχ²
    #   → u₁²/(Δχ²/λ₁) + u₂²/(Δχ²/λ₂) = 1
    #   → semi-axis lengths: a = sqrt(Δχ²/λ₁), b = sqrt(Δχ²/λ₂)
    #
    # Note: λ₁ is the larger eigenvalue → a is the semi-minor (smaller error)
    #       λ₂ is the smaller eigenvalue → b is the semi-major (larger error)

    eigenvals, eigenvecs = np.linalg.eigh(fisher)
    # eigh returns sorted ascending: λ₂ ≤ λ₁
    lambda_small, lambda_large = eigenvals  # λ₂ (small), λ₁ (large)
    # eigenvectors are columns: v₁ (for λ₁), v₂ (for λ₂)

    # Check for positive definiteness (both eigenvalues > 0)
    if lambda_small <= 0:
        raise ValueError(f"Fisher matrix is not positive definite. "
                        f"Eigenvalues: {eigenvals}. "
                        f"The Fisher matrix must be positive definite for ellipse computation. "
                        f"This may indicate parameter degeneracy or a poorly constrained model.")

    # Semi-axis lengths (before rotation)
    # a = sqrt(Δχ² / λ_large)  (semi-minor, smaller)
    # b = sqrt(Δχ² / λ_small)  (semi-major, larger)
    semi_minor = np.sqrt(delta_chi2 / lambda_large)
    semi_major = np.sqrt(delta_chi2 / lambda_small)

    # Angle of the ellipse (rotation from eigenvector of larger eigenvalue)
    # The eigenvector corresponding to the larger eigenvalue (λ₁) gives
    # the direction of the semi-minor axis (smaller uncertainty).
    v_major = eigenvecs[:, 1]  # eigenvector for λ₁ (larger eigenvalue)
    angle_rad = np.arctan2(v_major[1], v_major[0])
    angle_deg = np.degrees(angle_rad)

    return {
        'semi_minor': semi_minor,
        'semi_major': semi_major,
        'angle_rad': angle_rad,
        'angle_deg': angle_deg,
        'eigenvals': eigenvals,
        'eigenvecs': eigenvecs,
        'delta_chi2': delta_chi2
    }


def cal_ellipse_from_fisher(fisher, confidence_level=0.683, full_output=False):
    """
    Calculate ellipse area and parameters from Fisher matrix.

    Parameters
    ----------
    fisher : ndarray
        Fisher information matrix, must be 2x2.
    confidence_level : float, default 0.683
        Confidence level for the ellipse (0 < confidence_level < 1).
        For 1σ Gaussian: 0.683, for 2σ: 0.954, for 3σ: 0.997.
    full_output : bool, default False
        If True, return additional ellipse parameters.

    Returns
    -------
    area : float
        Area of the error ellipse (π * a * b).
    params : dict, optional
        Only returned if full_output=True. Dictionary containing:
        - 'semi_minor' : float - semi-minor axis length (smaller)
        - 'semi_major' : float - semi-major axis length (larger)
        - 'minor_axis_slope' : float - slope of the semi-minor axis
        - 'major_axis_slope' : float - slope of the semi-major axis
        - 'angle_rad' : float - rotation angle in radians
        - 'angle_deg' : float - rotation angle in degrees
        - 'eigenvals' : ndarray - eigenvalues [λ_small, λ_large]
        - 'eigenvecs' : ndarray - eigenvectors as columns
        - 'delta_chi2' : float - chi-squared critical value

    Raises
    ------
    ValueError
        If fisher is not a 2x2 matrix or not positive definite.

    Notes
    -----
    The ellipse area is calculated as: Area = π * a * b
    where a is the semi-minor axis and b is the semi-major axis.

    The axis slopes are calculated from the eigenvectors of the Fisher matrix:
    - The semi-minor axis aligns with the eigenvector of the larger eigenvalue
    - The semi-major axis aligns with the eigenvector of the smaller eigenvalue
    """
    import numpy as np

    # Compute ellipse parameters using helper function
    params = _compute_ellipse_params_from_fisher(fisher, confidence_level)

    # Calculate area: π * a * b
    area = np.pi * params['semi_minor'] * params['semi_major']

    if full_output:
        # Calculate axis slopes from eigenvectors
        # eigenvectors are columns: v_small (for λ_small), v_large (for λ_large)
        # semi-minor axis aligns with v_large (eigenvector for larger eigenvalue)
        # semi-major axis aligns with v_small (eigenvector for smaller eigenvalue)
        v_minor = params['eigenvecs'][:, 1]  # eigenvector for λ_large (semi-minor)
        v_major = params['eigenvecs'][:, 0]  # eigenvector for λ_small (semi-major)

        # Slope = y/x (be careful with vertical lines where x ≈ 0)
        # For near-vertical lines, slope approaches infinity
        minor_axis_slope = v_minor[1] / v_minor[0] if np.abs(v_minor[0]) > 1e-10 else np.inf
        major_axis_slope = v_major[1] / v_major[0] if np.abs(v_major[0]) > 1e-10 else np.inf

        # Add slopes to params
        params['minor_axis_slope'] = minor_axis_slope
        params['major_axis_slope'] = major_axis_slope

        return area, params
    else:
        return area


def plot_ellipse_from_fisher(fisher, best_fit, ax=None, **kwargs):
    """
    Plot error ellipse from Fisher matrix.

    The error ellipse represents the 1\sigma (68.3%) confidence region
    for two parameters, derived from the Fisher information matrix.

    Parameters
    ----------
    fisher : ndarray
        Fisher information matrix, must be 2x2.
    best_fit : array_like
        Best-fit parameter values [p1, p2] for the ellipse center.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, creates a new figure and axis.
    **kwargs : dict
        Additional keyword arguments for customization. Supported keys:

        **Ellipse properties** (passed to matplotlib.patches.Ellipse):
        - ellipse_color / color : str, default 'C0'
            Color of the ellipse edge.
        - ellipse_alpha / alpha : float, default 0.5
            Transparency of the ellipse fill.
        - ellipse_facecolor / facecolor : str, optional
            Fill color of the ellipse. If None, uses ellipse_color with alpha.
        - ellipse_edgecolor / edgecolor : str, optional
            Edge color of the ellipse. If None, uses ellipse_color.
        - ellipse_linewidth / linewidth : float, default 1.5
            Width of the ellipse edge.
        - ellipse_linestyle / linestyle : str, default '-'
            Style of the ellipse edge.
        - ellipse_fill / fill : bool, default True
            Whether to fill the ellipse.
        - ellipse_zorder : float, optional
            Z-order for the ellipse patch.

        **Confidence level**:
        - confidence_level : float, default 0.683
            Confidence level for the ellipse (0 < confidence_level < 1).
            For 1\sigma Gaussian: 0.683, for 2\sigma: 0.954, for 3\sigma: 0.997.
            The ellipse size scales with sqrt(χ² quantile).

        **Center marker**:
        - show_center : bool, default True
            Whether to show a marker at the best-fit center point.
        - center_marker : str, default 'x'
            Marker style for the center point.
        - center_color / center_markercolor : str, default 'C3'
            Color of the center marker.
        - center_size / markersize : float, default 8
            Size of the center marker.
        - center_zorder : float, optional
            Z-order for the center marker.

        **Axis limits**:
        - xlim : tuple, optional
            (xmin, xmax) for the axis. Auto-determined if not provided.
        - ylim : tuple, optional
            (ymin, ymax) for the axis. Auto-determined if not provided.
        - padding : float, default 0.1
            Fractional padding for auto axis limits (10% of ellipse extent).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axis with the ellipse plotted.

    Raises
    ------
    ValueError
        If fisher is not a 2x2 matrix.
    ImportError
        If matplotlib is not installed.

    Notes
    -----
    The ellipse represents the contour of constant likelihood deviation:
        (θ - θ₀)^T · F · (θ - θ₀) = Δχ²

    For a 1\sigma confidence region in 2D, Δχ² = χ²_{2, 0.683} ≈ 2.28.
    The semi-axis lengths are: a = sqrt(Δχ² / λ₁), b = sqrt(Δχ² / λ₂)
    where λ₁, λ₂ are the eigenvalues of the Fisher matrix.
    """
    import numpy as np

    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError:
        raise ImportError("matplotlib is required for plotting. "
                         "Install it with: pip install matplotlib")

    # Validate fisher matrix dimensions
    fisher = np.atleast_2d(fisher)
    if fisher.shape != (2, 2):
        raise ValueError(f"Fisher matrix must be 2x2 for ellipse plotting, "
                        f"got shape {fisher.shape}")

    # Validate best_fit
    best_fit = np.atleast_1d(best_fit)
    if len(best_fit) != 2:
        raise ValueError(f"best_fit must have exactly 2 elements, got {len(best_fit)}")

    # Create axis if not provided
    if ax is None:
        ax = plt.gca()

    # Extract kwargs with defaults
    # Ellipse appearance
    ellipse_color = kwargs.get('ellipse_color', kwargs.get('color', 'C0'))
    ellipse_alpha = kwargs.get('ellipse_alpha', kwargs.get('alpha', 0.5))
    ellipse_facecolor = kwargs.get('ellipse_facecolor', kwargs.get('facecolor', None))
    ellipse_edgecolor = kwargs.get('ellipse_edgecolor', kwargs.get('edgecolor', None))
    ellipse_linewidth = kwargs.get('ellipse_linewidth', kwargs.get('linewidth', 1.5))
    ellipse_linestyle = kwargs.get('ellipse_linestyle', kwargs.get('linestyle', '-'))
    ellipse_fill = kwargs.get('ellipse_fill', kwargs.get('fill', True))
    ellipse_zorder = kwargs.get('ellipse_zorder', kwargs.get('zorder', 1))

    # Confidence level
    confidence_level = kwargs.get('confidence_level', 0.683)

    # Center marker
    show_center = kwargs.get('show_center', True)
    center_marker = kwargs.get('center_marker', 'x')
    center_color = kwargs.get('center_color', kwargs.get('center_markercolor', 'C3'))
    center_size = kwargs.get('center_size', kwargs.get('markersize', 8))
    center_zorder = kwargs.get('center_zorder', 2)

    # Axis limits
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    padding = kwargs.get('padding', 0.1)

    # Compute ellipse parameters using helper function
    ellipse_params = _compute_ellipse_params_from_fisher(fisher, confidence_level)
    
    # Extract parameters
    a = ellipse_params['semi_minor']  # semi-minor axis
    b = ellipse_params['semi_major']  # semi-major axis
    angle_rad = ellipse_params['angle_rad']
    angle_deg = ellipse_params['angle_deg']
    eigenvecs = ellipse_params['eigenvecs']

    # Set facecolor default to ellipse_color with alpha if not specified
    if ellipse_facecolor is None:
        ellipse_facecolor = ellipse_color
    if ellipse_edgecolor is None:
        ellipse_edgecolor = ellipse_color
    
    # Handle fill=False: set facecolor to 'none' instead of using alpha=0
    # This ensures the edge remains visible
    if not ellipse_fill:
        ellipse_facecolor = 'none'

    # Create and add ellipse patch
    ellipse = Ellipse(
        xy=best_fit,
        width=2 * a,      # full width (2 * semi-axis)
        height=2 * b,     # full height (2 * semi-axis)
        angle=angle_deg,
        facecolor=ellipse_facecolor,
        edgecolor=ellipse_edgecolor,
        alpha=ellipse_alpha,
        linewidth=ellipse_linewidth,
        linestyle=ellipse_linestyle,
        zorder=ellipse_zorder
    )
    ax.add_patch(ellipse)

    # Plot center point if requested
    if show_center:
        ax.plot(best_fit[0], best_fit[1],
                marker=center_marker,
                color=center_color,
                markersize=center_size,
                zorder=center_zorder,
                linestyle='None')

    # Set axis limits if not provided, auto-scale based on ellipse
    if xlim is None or ylim is None:
        # Compute bounding box of ellipse correctly
        # For a rotated ellipse, the bounding box extent is:
        # width = 2 * sqrt((a*cos(θ))² + (b*sin(θ))²)
        # height = 2 * sqrt((a*sin(θ))² + (b*cos(θ))²)
        # where θ is the rotation angle, a and b are semi-axes
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        half_width = np.sqrt((a * cos_angle)**2 + (b * sin_angle)**2)
        half_height = np.sqrt((a * sin_angle)**2 + (b * cos_angle)**2)
        
        x_min = best_fit[0] - half_width
        x_max = best_fit[0] + half_width
        y_min = best_fit[1] - half_height
        y_max = best_fit[1] + half_height

        # Apply padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = padding * max(x_range, 1e-8)  # avoid division by zero
        y_pad = padding * max(y_range, 1e-8)

        if xlim is None:
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
        if ylim is None:
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Set labels if not already set
    if ax.get_xlabel() == '':
        ax.set_xlabel('Parameter 1')
    if ax.get_ylabel() == '':
        ax.set_ylabel('Parameter 2')

    return ax