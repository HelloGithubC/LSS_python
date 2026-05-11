import os 

import numpy as np 
import ctypes 

from .base import c_double_complex, c_float_complex

script_dir = os.path.dirname(os.path.abspath(__file__))
fftpower_lib = ctypes.CDLL(os.path.join(script_dir + "/lib", "fftpower_ctype.so"))

def deal_ps_3d_c_api(complex_field, ps_3d=None, kernel=None, ps_3d_factor=1.0, shotnoise=0.0, nthreads=1):
    """
    Calculate 3D power spectrum from complex field using C API.
    
    Args:
        complex_field: Input complex field array (complex64 or complex128)
        ps_3d: Optional pre-allocated array for output. If provided, it will be 
               COMPLETELY OVERWRITTEN with new results. Use np.empty() for best 
               performance (no initialization needed). If None, a new array will 
               be created automatically. Useful for memory reuse in repeated calls.
        kernel: Optional kernel array (must match complex_field precision)
        ps_3d_factor: Factor for power spectrum normalization
        shotnoise: Shot noise to subtract
        nthreads: Number of threads for parallel computation
    
    Returns:
        ps_3d: 3D power spectrum array (same object if ps_3d was provided)
    """
    if complex_field.dtype != np.complex128 and complex_field.dtype != np.complex64:
        print(f"Warning: complex_field({complex_field.dtype}) not complex128 or complex64. Trying to convert to complex64")
        complex_field = complex_field.astype(np.complex64)
    
    # Determine expected dtype for ps_3d
    np_real_dtype = np.float64 if complex_field.dtype == np.complex128 else np.float32
    real_dtype = ctypes.c_double if complex_field.dtype == np.complex128 else ctypes.c_float
    
    # Handle ps_3d parameter
    if ps_3d is None:
        # Create new array (backward compatible behavior)
        ps_3d = np.empty(complex_field.shape, dtype=np_real_dtype)
    else:
        # Validate provided ps_3d array
        if not isinstance(ps_3d, np.ndarray):
            raise TypeError(f"ps_3d must be a numpy.ndarray or None, got {type(ps_3d)}")
        if ps_3d.dtype != np_real_dtype:
            raise TypeError(f"ps_3d dtype must be {np_real_dtype} for complex_field dtype {complex_field.dtype}, got {ps_3d.dtype}")
        if ps_3d.shape != complex_field.shape:
            raise ValueError(f"ps_3d shape {ps_3d.shape} must match complex_field shape {complex_field.shape}")
    
    if kernel is not None:
        # Extract real part of kernel (kernel should be real-valued)
        kernel = kernel.real
        # Match precision: complex64 -> float32, complex128 -> float64
        expected_dtype = np.float32 if complex_field.dtype == np.complex64 else np.float64
        if kernel.dtype != expected_dtype:
            print(f"Warning: the dtype of kernel ({kernel.dtype}) does not match expected ({expected_dtype}). Converting kernel")
            kernel = kernel.astype(expected_dtype)

    ngrids = np.array(complex_field.shape, dtype=np.uint64)
    
    complex_field_ptr = complex_field.ctypes.data_as(ctypes.POINTER(c_double_complex if complex_field.dtype == np.complex128 else c_float_complex))
    kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(real_dtype)) if kernel is not None else ctypes.POINTER(real_dtype)()
    ps_3d_ptr = ps_3d.ctypes.data_as(ctypes.POINTER(real_dtype))
    ngrids_ptr = ngrids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    nthreads = ctypes.c_int(nthreads)
    if complex_field.dtype == np.complex128:
        ps_3d_factor = ctypes.c_double(ps_3d_factor)
        shotnoise = ctypes.c_double(shotnoise)
        fftpower_lib.DealPS3D_double(complex_field_ptr, kernel_ptr, ps_3d_ptr, ngrids_ptr, ps_3d_factor, shotnoise, nthreads)
    else:
        ps_3d_factor = ctypes.c_float(ps_3d_factor)
        shotnoise = ctypes.c_float(shotnoise)
        fftpower_lib.DealPS3D_float(complex_field_ptr, kernel_ptr, ps_3d_ptr, ngrids_ptr, ps_3d_factor, shotnoise, nthreads)
    
    return ps_3d
    

def cal_ps_c_api(ps_3d, k_arrays_list, k_array, mu_array = None, k_logarithmic=False, nthreads=1):
    # ps_3d is now a real array (float32 or float64)
    if ps_3d.dtype != np.float64 and ps_3d.dtype != np.float32:
        print("Warning: ps_3d not float64 or float32. Trying to convert to float32")
        try:
            ps_3d = ps_3d.astype(np.float32)
        except ValueError:
            raise ValueError("ps_3d must be float32 or float64, or convertible to float32")
    
    kx_array, ky_array, kz_array = k_arrays_list
    kx_array = kx_array.astype(np.float64, copy=False)
    ky_array = ky_array.astype(np.float64, copy=False)
    kz_array = kz_array.astype(np.float64, copy=False)

    if k_array[0] > k_array[1]:
        k_array = np.copy(k_array[::-1])
    k_array = k_array.astype(np.float64, copy=False)
    kbin = k_array.shape[0] - 1
    if mu_array is None:
        use_mu = False
    else:
        use_mu = True
        if mu_array[0] > mu_array[1]:
            mu_array = np.copy(mu_array[::-1])
    mu_array = mu_array.astype(np.float64, copy=False) if use_mu else None
    mubin = mu_array.shape[0] - 1 if use_mu else 1

    # Use float64 for power (real power spectrum output)
    power = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    power_k = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    power_mu = np.zeros(shape=(kbin, mubin), dtype=np.float64) if use_mu else None
    power_modes = np.zeros(shape=(kbin, mubin), dtype=np.uint64)
    ngrids = np.array(ps_3d.shape, dtype=np.uint64)

    real_dtype = ctypes.c_double if ps_3d.dtype == np.float64 else ctypes.c_float
    ps_3d_ptr = ps_3d.ctypes.data_as(ctypes.POINTER(real_dtype))
    ngrids_ptr = ngrids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    kx_array_ptr = kx_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ky_array_ptr = ky_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    kz_array_ptr = kz_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    k_array_ptr = k_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    mu_array_ptr = mu_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if use_mu else ctypes.POINTER(ctypes.c_double)()
    k_bin = ctypes.c_uint32(kbin)
    mu_bin = ctypes.c_uint32(mubin)

    power_ptr = power.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    power_k_ptr = power_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    power_mu_ptr = power_mu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if use_mu else ctypes.POINTER(ctypes.c_double)()
    power_modes_ptr = power_modes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    nthreads = ctypes.c_int(nthreads)
    k_logarithmic = ctypes.c_bool(k_logarithmic)

    if ps_3d.dtype == np.float64:
        cal_func = fftpower_lib.CalculatePS_double
    else:
        cal_func = fftpower_lib.CalculatePS_float
    cal_func(ps_3d_ptr, ngrids_ptr, kx_array_ptr, ky_array_ptr, kz_array_ptr, k_array_ptr, mu_array_ptr, k_bin, mu_bin, power_ptr, power_modes_ptr, power_k_ptr, power_mu_ptr, nthreads, k_logarithmic)
    return power_k, power_mu, power, power_modes