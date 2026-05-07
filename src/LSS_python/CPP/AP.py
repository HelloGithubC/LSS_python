import numpy as np

from .lib.AP_pybind import mapping_smudata_dense_cpp_float, mapping_smudata_dense_cpp_double  # type: ignore


def mapping_smudata_dense_cpp(
    smutabstd,
    cosmos_tuple,
    smu_bin_tuple,
    smu_all_bound_tuple,
    smu_mapping_bound_tuple,
    nthreads=1
):
    """
    C++ version of mapping_smudata_dense using pybind11.
    
    Maps dense (s, mu) grid data to sparse grid with AP effect conversion.
    
    Parameters
    ----------
    smutabstd : numpy.ndarray
        Input data array of shape (sbin_dense, mubin_dense, element_size).
    cosmos_tuple : tuple
        Cosmological parameters: (Hz_f, Hz_m, DA_f, DA_m)
    smu_bin_tuple : tuple
        Bin parameters: (sbin_dense, sbin_sparse, mubin_dense, mubin_sparse)
    smu_all_bound_tuple : tuple
        Full grid bounds: (smin_all, smax_all, mumin_all, mumax_all)
    smu_mapping_bound_tuple : tuple
        Mapping bounds: (smin_mapping, smax_mapping, mumin_mapping, mumax_mapping)
        Note: mumin_mapping and mumax_mapping are typically 0.0 and 1.0
    nthreads : int, optional
        Number of OpenMP threads. Default 1.
    
    Returns
    -------
    numpy.ndarray
        Mapped data array of shape (sbin_sparse, mubin_sparse, element_size).
    """
    # Validate input
    if not isinstance(smutabstd, np.ndarray):
        raise TypeError("smutabstd must be a numpy array")
    
    if smutabstd.ndim != 3:
        raise ValueError(f"smutabstd must be 3-dimensional, got {smutabstd.ndim}")
    
    # Ensure contiguous array
    smutabstd = np.ascontiguousarray(smutabstd)
    
    # Unpack tuples
    Hz_f, Hz_m, DA_f, DA_m = cosmos_tuple
    sbin_dense, sbin_sparse, mubin_dense, mubin_sparse = smu_bin_tuple
    smin_all, smax_all, mumin_all, mumax_all = smu_all_bound_tuple
    smin_mapping, smax_mapping, _, _ = smu_mapping_bound_tuple
    
    # Convert to Python native types if needed
    sbin_dense = int(sbin_dense)
    mubin_dense = int(mubin_dense)
    sbin_sparse = int(sbin_sparse)
    mubin_sparse = int(mubin_sparse)
    
    # Select appropriate precision version
    if smutabstd.dtype == np.float32:
        result = mapping_smudata_dense_cpp_float(
            smutabstd,
            sbin_dense, mubin_dense,
            sbin_sparse, mubin_sparse,
            float(Hz_f), float(Hz_m), float(DA_f), float(DA_m),
            float(smin_all), float(smax_all), float(mumin_all), float(mumax_all),
            float(smin_mapping), float(smax_mapping),
            nthreads
        )
    elif smutabstd.dtype == np.float64:
        result = mapping_smudata_dense_cpp_double(
            smutabstd,
            sbin_dense, mubin_dense,
            sbin_sparse, mubin_sparse,
            Hz_f, Hz_m, DA_f, DA_m,
            smin_all, smax_all, mumin_all, mumax_all,
            smin_mapping, smax_mapping,
            nthreads
        )
    else:
        raise ValueError(f"Unsupported dtype: {smutabstd.dtype}. Must be float32 or float64.")
    
    return result
