import os 

import numpy as np 
import ctypes 

from .base import c_double_complex, c_float_complex

script_dir = os.path.dirname(os.path.abspath(__file__))
fftpower_lib = ctypes.CDLL(os.path.join(script_dir + "/lib", "fftpower.so"))

def deal_ps_3d_c_api(complex_field, kernel=None, ps_3d_factor=1.0, shotnoise=0.0, nthreads=1):
    if complex_field.dtype != np.complex128 and complex_field.dtype != np.complex64:
        print(f"Warning: complex_field({complex_field.dtype}) not complex128 or complex64. Tring to convert to complex64")
        complex_field = complex_field.astype(np.complex64)
    if kernel is not None:
        if kernel.dtype != complex_field.dtype:
            print("Warning: the dtype of kernel does not match the dtype of complex_field. Trying to convert kernel to match complex_field")
            kernel = kernel.astype(complex_field.dtype)

    ngrids = np.array(complex_field.shape, dtype=np.uint64)

    complex_dtype = c_double_complex if complex_field.dtype == np.complex128 else c_float_complex
    complex_field_ptr = complex_field.ctypes.data_as(ctypes.POINTER(complex_dtype))
    kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(complex_dtype)) if kernel is not None else ctypes.POINTER(complex_dtype)()
    ngrids_ptr = ngrids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    nthreads = ctypes.c_int(nthreads)
    if complex_field.dtype == np.complex128:
        ps_3d_factor = ctypes.c_double(ps_3d_factor)
        shotnoise = ctypes.c_double(shotnoise)
        fftpower_lib.DealPS3D_double(complex_field_ptr, kernel_ptr, ngrids_ptr, ps_3d_factor, shotnoise, nthreads)
    else:
        ps_3d_factor = ctypes.c_float(ps_3d_factor)
        shotnoise = ctypes.c_float(shotnoise)
        fftpower_lib.DealPS3D_float(complex_field_ptr, kernel_ptr, ngrids_ptr, ps_3d_factor, shotnoise, nthreads)
    

def cal_ps_c_api(ps_3d, k_arrays_list, k_array, mu_array = None, k_logarithmic=False, nthreads=1):
    if ps_3d.dtype != np.complex128 and ps_3d.dtype != np.complex64:
        print("Warning: ps_3d not complex128 or complex64. Tring to convert to complex64")
        try:
            ps_3d = ps_3d.astype(np.complex64)
        except ValueError:
            raise ValueError("ps_3d must be complex64 or complex128, or convertable to complex64")
    
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

    power = np.zeros(shape=(kbin, mubin), dtype=np.complex128)
    power_k = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    power_mu = np.zeros(shape=(kbin, mubin), dtype=np.float64) if use_mu else None
    power_modes = np.zeros(shape=(kbin, mubin), dtype=np.uint64)
    ngrids = np.array(ps_3d.shape, dtype=np.uint64)

    complex_dtype = c_double_complex if ps_3d.dtype == np.complex128 else c_float_complex
    ps_3d_ptr = ps_3d.ctypes.data_as(ctypes.POINTER(complex_dtype))
    ngrids_ptr = ngrids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    kx_array_ptr = kx_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ky_array_ptr = ky_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    kz_array_ptr = kz_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    k_array_ptr = k_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    mu_array_ptr = mu_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if use_mu else ctypes.POINTER(ctypes.c_double)()
    k_bin = ctypes.c_uint32(kbin)
    mu_bin = ctypes.c_uint32(mubin)

    power_ptr = power.ctypes.data_as(ctypes.POINTER(complex_dtype))
    power_k_ptr = power_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    power_mu_ptr = power_mu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if use_mu else ctypes.POINTER(ctypes.c_double)()
    power_modes_ptr = power_modes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    nthreads = ctypes.c_int(nthreads)
    k_logarithmic = ctypes.c_bool(k_logarithmic)

    if ps_3d.dtype == np.complex128:
        cal_func = fftpower_lib.CalculatePS_double
    else:
        cal_func = fftpower_lib.CalculatePS_float
    cal_func(ps_3d_ptr, ngrids_ptr, kx_array_ptr, ky_array_ptr, kz_array_ptr, k_array_ptr, mu_array_ptr, k_bin, mu_bin, power_ptr, power_modes_ptr, power_k_ptr, power_mu_ptr, nthreads, k_logarithmic)
    return power_k, power_mu, power, power_modes