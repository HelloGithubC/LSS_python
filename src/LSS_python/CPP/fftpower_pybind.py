import numpy as np 

from .lib.fftpower_pybind import deal_ps_3d_float, deal_ps_3d_double # type: ignore
from .lib.fftpower_pybind import cal_ps_float, cal_ps_double # type: ignore
from .lib.fftpower_pybind import cal_ps_2d_from_mesh_float, cal_ps_2d_from_mesh_double # type: ignore
from .lib.fftpower_pybind import cal_ps_from_ps_2d_float, cal_ps_from_ps_2d_double # type: ignore

def deal_ps_3d_pybind(ps_3d, kernel=None, ps_3d_factor=1.0, shotnoise=0.0, nthreads=1):
    if not ps_3d.flags.contiguous:
        raise ValueError("ps_3d must be contiguous")
    
    if ps_3d.dtype == np.complex64:
        deal_ps_3d_float(ps_3d, kernel, ps_3d_factor, shotnoise, nthreads)
    else:
        deal_ps_3d_double(ps_3d, kernel, ps_3d_factor, shotnoise, nthreads)

def cal_ps_pybind(ps_3d, k_arrays_list, k_array, mu_array = None, k_logarithmic=False, nthreads=1):
    if not ps_3d.flags.contiguous:
        raise ValueError("ps_3d must be contiguous")
    
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
        mubin = 1
        mu_array = None
    else:
        use_mu = True
        if mu_array[0] > mu_array[1]:
            mu_array = np.copy(mu_array[::-1])
        mu_array = mu_array.astype(np.float64, copy=False)
        mubin = mu_array.shape[0] - 1 

    power = np.zeros(shape=(kbin, mubin), dtype=np.complex128)
    power_k = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    power_mu = np.zeros(shape=(kbin, mubin), dtype=np.float64) if use_mu else None
    power_modes = np.zeros(shape=(kbin, mubin), dtype=np.uint64)
    
    if ps_3d.dtype == np.complex64:
        cal_ps_float(ps_3d, kx_array, ky_array, kz_array, k_array, mu_array, power, power_modes, power_k, power_mu, nthreads, k_logarithmic)
    else:
        cal_ps_double(ps_3d, kx_array, ky_array, kz_array, k_array, mu_array, power, power_modes, power_k, power_mu, nthreads, k_logarithmic)

    return power_k, power_mu, power, power_modes

def cal_ps_2d_from_mesh(mesh, mesh_kernel, k_arrays, ps_factor, shotnoise, nthreads=1, dk=None):
    if mesh.complex_field is None:
        raise ValueError("Complex field is not set.")
    complex_field = mesh.complex_field
    kernel = mesh_kernel.complex_field if mesh_kernel is not None else None
    if k_arrays is None:
        BoxSize = mesh.attrs["BoxSize"]
        Nmesh = mesh.attrs["Nmesh"]
        k_x_array = np.fft.fftfreq(Nmesh[0], d=BoxSize[0] / Nmesh[0]) * 2.0 * np.pi
        k_y_array = np.fft.fftfreq(Nmesh[1], d=BoxSize[1] / Nmesh[1]) * 2.0 * np.pi
        k_z_array = np.fft.rfftfreq(Nmesh[2], d=BoxSize[2] / Nmesh[2]) * 2.0 * np.pi
    else:
        k_x_array, k_y_array, k_z_array = k_arrays

    if dk is None:
        dk = (k_z_array[1] - k_z_array[0]) * 2.0

    k_perp_source = np.sqrt(k_x_array**2 + k_y_array**2)
    k_perp_min = np.min(k_perp_source)
    k_perp_max = np.max(k_perp_source)
    k_perp_edge = np.arange(k_perp_min, k_perp_max + dk, dk)
    k_perp_bin = len(k_perp_edge) - 1 

    k_parallel_edge = np.arange(k_z_array[0], k_z_array[-1] + dk, dk)
    k_parallel_bin = len(k_parallel_edge) - 1
    k_2d = np.zeros(shape=(k_perp_bin, k_parallel_bin, 2), dtype=np.float64)

    ps_dtype = np.complex64 if complex_field.dtype == np.complex64 else np.complex128
    ps_2d = np.zeros(shape=(k_perp_bin, k_parallel_bin), dtype=ps_dtype)
    modes_2d = np.zeros(shape=(k_perp_bin, k_parallel_bin), dtype=np.uint64)

    if complex_field.dtype == np.complex64:
        cal_ps_2d_from_mesh_float(complex_field, kernel, ps_2d, k_2d, modes_2d, k_perp_edge, k_parallel_edge, k_x_array, k_y_array, k_z_array, ps_factor, shotnoise, nthreads)
    else:
        cal_ps_2d_from_mesh_double(complex_field, kernel, ps_2d, k_2d, modes_2d, k_perp_edge, k_parallel_edge, k_x_array, k_y_array, k_z_array, ps_factor, shotnoise, nthreads)

    index_zero = (modes_2d == 0)
    k_perp_temp = k_2d[:, :, 0]
    k_perp_temp[index_zero] = np.nan 
    ps_2d[index_zero] = np.nan
    index_zero_not = np.logical_not(index_zero)
    k_perp_temp[index_zero_not] = k_perp_temp[index_zero_not] / modes_2d[index_zero_not]
    ps_2d[index_zero_not] = ps_2d[index_zero_not] / modes_2d[index_zero_not]

    return k_2d, ps_2d, modes_2d

def cal_pkmu_from_ps_2d(ps_2d, k_2d, k_edge, mu_edge, k_logarithmic=False, nthreads=1):
    if not ps_2d.flags.contiguous:
        raise ValueError("ps_2d must be contiguous")
    if not k_2d.flags.contiguous:
        raise ValueError("k_2d must be contiguous")

    k_2d = k_2d.astype(np.float64, copy=False)
    k_edge = k_edge.astype(np.float64, copy=False)

    kbin = len(k_edge) - 1

    if mu_edge is not None and len(mu_edge) > 2:
        mu_edge = mu_edge.astype(np.float64, copy=False)
        mubin = len(mu_edge) - 1
        use_mu = True
    else:
        mu_edge = None
        mubin = 1
        use_mu = False

    ps_dtype = np.complex64 if ps_2d.dtype == np.complex64 else np.complex128
    k_out_2d = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    mu_out_2d = np.zeros(shape=(kbin, mubin), dtype=np.float64) if use_mu else None
    ps_kmu = np.zeros(shape=(kbin, mubin), dtype=ps_dtype)
    modes = np.zeros(shape=(kbin, mubin), dtype=np.uint64)

    if ps_2d.dtype == np.complex64:
        cal_ps_from_ps_2d_float(ps_2d, k_2d, k_out_2d, mu_out_2d, ps_kmu, modes, k_edge, mu_edge, nthreads)
    else:
        cal_ps_from_ps_2d_double(ps_2d, k_2d, k_out_2d, mu_out_2d, ps_kmu, modes, k_edge, mu_edge, nthreads)

    return k_out_2d, mu_out_2d, ps_kmu, modes
