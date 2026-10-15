import numpy as np 

from .lib.fftpower_pybind import deal_ps_3d_float, deal_ps_3d_double # type: ignore
from .lib.fftpower_pybind import cal_ps_float, cal_ps_double # type: ignore
from .lib.fftpower_pybind import cal_ps_2d_from_mesh_float, cal_ps_2d_from_mesh_double # type: ignore
from .lib.fftpower_pybind import cal_ps_from_ps_2d # type: ignore

def deal_ps_3d_pybind(complex_field, ps_3d=None, kernel=None, ps_3d_factor=1.0, shotnoise=0.0, nthreads=1):
    """
    Calculate 3D power spectrum from complex field.
    
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
    
    Example:
        # Memory reuse for repeated calculations
        ps_3d = np.empty(complex_field.shape, dtype=np.float32)
        for i in range(100):
            ps_3d = deal_ps_3d_pybind(complex_field, ps_3d=ps_3d, ...)
    """
    # Type checks for external inputs
    if not isinstance(complex_field, np.ndarray):
        raise TypeError(f"complex_field must be a numpy.ndarray, got {type(complex_field)}")
    if not complex_field.flags.contiguous:
        raise ValueError("complex_field must be contiguous")
    if complex_field.dtype not in [np.complex64, np.complex128]:
        raise TypeError(f"complex_field must be complex64 or complex128, got {complex_field.dtype}")
    
    # Determine expected dtype for ps_3d
    expected_dtype = np.float32 if complex_field.dtype == np.complex64 else np.float64
    
    # Handle ps_3d parameter
    if ps_3d is None:
        # Create new array (backward compatible behavior)
        ps_3d = np.empty(complex_field.shape, dtype=expected_dtype)
    else:
        # Validate provided ps_3d array
        if not isinstance(ps_3d, np.ndarray):
            raise TypeError(f"ps_3d must be a numpy.ndarray or None, got {type(ps_3d)}")
        if not ps_3d.flags.contiguous:
            raise ValueError("ps_3d must be contiguous")
        if ps_3d.dtype != expected_dtype:
            raise TypeError(f"ps_3d dtype must be {expected_dtype} for complex_field dtype {complex_field.dtype}, got {ps_3d.dtype}")
        if ps_3d.shape != complex_field.shape:
            raise ValueError(f"ps_3d shape {ps_3d.shape} must match complex_field shape {complex_field.shape}")
    
    # Check kernel if provided
    if kernel is not None:
        if not isinstance(kernel, np.ndarray):
            raise TypeError(f"kernel must be a numpy.ndarray or None, got {type(kernel)}")
        if not kernel.flags.contiguous:
            raise ValueError("kernel must be contiguous")
    
    # Type checks for scalar parameters
    if not isinstance(ps_3d_factor, (int, float, np.number)):
        raise TypeError(f"ps_3d_factor must be a number, got {type(ps_3d_factor)}")
    if not isinstance(shotnoise, (int, float, np.number)):
        raise TypeError(f"shotnoise must be a number, got {type(shotnoise)}")
    if not isinstance(nthreads, (int, np.integer)):
        raise TypeError(f"nthreads must be an integer, got {type(nthreads)}")
    
    if complex_field.dtype == np.complex64:
        deal_ps_3d_float(complex_field, ps_3d, kernel, ps_3d_factor, shotnoise, nthreads)
    else:
        deal_ps_3d_double(complex_field, ps_3d, kernel, ps_3d_factor, shotnoise, nthreads)
    
    return ps_3d

def cal_ps_pybind(ps_3d, k_arrays_list, k_array, mu_array = None, k_logarithmic=False, nthreads=1):
    # Type check for ps_3d (now real array)
    if not isinstance(ps_3d, np.ndarray):
        raise TypeError(f"ps_3d must be a numpy.ndarray, got {type(ps_3d)}")
    if not ps_3d.flags.contiguous:
        raise ValueError("ps_3d must be contiguous")
    if ps_3d.dtype not in [np.float32, np.float64]:
        raise TypeError(f"ps_3d must be float32 or float64, got {ps_3d.dtype}")
    
    # Type check for k_arrays_list
    if not isinstance(k_arrays_list, (list, tuple)):
        raise TypeError(f"k_arrays_list must be a list or tuple, got {type(k_arrays_list)}")
    if len(k_arrays_list) != 3:
        raise ValueError(f"k_arrays_list must have exactly 3 elements, got {len(k_arrays_list)}")
    kx_array, ky_array, kz_array = k_arrays_list
    for name, arr in [("kx_array", kx_array), ("ky_array", ky_array), ("kz_array", kz_array)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
        if not arr.flags.contiguous:
            raise ValueError(f"{name} must be contiguous")
    
    kx_array = kx_array.astype(np.float64, copy=False)
    ky_array = ky_array.astype(np.float64, copy=False)
    kz_array = kz_array.astype(np.float64, copy=False)

    # Type check for k_array
    if not isinstance(k_array, np.ndarray):
        raise TypeError(f"k_array must be a numpy.ndarray, got {type(k_array)}")
    if not k_array.flags.contiguous:
        raise ValueError("k_array must be contiguous")
    if k_array.dtype not in [np.float32, np.float64]:
        raise TypeError(f"k_array must be float32 or float64, got {k_array.dtype}")
    
    if k_array[0] > k_array[1]:
        k_array = np.copy(k_array[::-1])
    k_array = k_array.astype(np.float64, copy=False)
    kbin = k_array.shape[0] - 1
    
    # Type check for mu_array
    if mu_array is not None:
        if not isinstance(mu_array, np.ndarray):
            raise TypeError(f"mu_array must be a numpy.ndarray or None, got {type(mu_array)}")
        if not mu_array.flags.contiguous:
            raise ValueError("mu_array must be contiguous")
        if mu_array.dtype not in [np.float32, np.float64]:
            raise TypeError(f"mu_array must be float32 or float64, got {mu_array.dtype}")
        use_mu = True
        if mu_array[0] > mu_array[1]:
            mu_array = np.copy(mu_array[::-1])
        mu_array = mu_array.astype(np.float64, copy=False)
        mubin = mu_array.shape[0] - 1 
    else:
        use_mu = False
        mubin = 1
        mu_array = None

    # Use float64 for power (real power spectrum output)
    power = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    power_k = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    power_mu = np.zeros(shape=(kbin, mubin), dtype=np.float64) if use_mu else None
    power_modes = np.zeros(shape=(kbin, mubin), dtype=np.uint64)
    
    # Type checks for scalar parameters
    if not isinstance(k_logarithmic, (bool, np.bool_)):
        raise TypeError(f"k_logarithmic must be a boolean, got {type(k_logarithmic)}")
    if not isinstance(nthreads, (int, np.integer)):
        raise TypeError(f"nthreads must be an integer, got {type(nthreads)}")
    
    if ps_3d.dtype == np.float32:
        cal_ps_float(ps_3d, kx_array, ky_array, kz_array, k_array, mu_array, power, power_modes, power_k, power_mu, nthreads, k_logarithmic)
    else:
        cal_ps_double(ps_3d, kx_array, ky_array, kz_array, k_array, mu_array, power, power_modes, power_k, power_mu, nthreads, k_logarithmic)

    return power_k, power_mu, power, power_modes

def cal_ps_2d_from_mesh(mesh, mesh_kernel, ps_factor, shotnoise, nthreads=1, dk=None):
    # Type check for mesh
    if not hasattr(mesh, 'complex_field') or not hasattr(mesh, 'attrs'):
        raise TypeError(f"mesh must have 'complex_field' and 'attrs' attributes, got {type(mesh)}")
    if mesh.complex_field is None:
        raise ValueError("Complex field is not set.")
    complex_field = mesh.complex_field
    if not isinstance(complex_field, np.ndarray):
        raise TypeError(f"mesh.complex_field must be a numpy.ndarray, got {type(complex_field)}")
    if not complex_field.flags.contiguous:
        raise ValueError("mesh.complex_field must be contiguous")
    if complex_field.dtype not in [np.complex64, np.complex128]:
        raise TypeError(f"mesh.complex_field must be complex64 or complex128, got {complex_field.dtype}")
    
    # Extract real part of kernel (kernel should be real-valued, stored in complex array)
    kernel = None
    if mesh_kernel is not None:
        if not hasattr(mesh_kernel, 'complex_field'):
            raise TypeError(f"mesh_kernel must have 'complex_field' attribute, got {type(mesh_kernel)}")
        kernel = mesh_kernel.complex_field.real
        if not isinstance(kernel, np.ndarray):
            raise TypeError(f"mesh_kernel.complex_field must be a numpy.ndarray, got {type(kernel)}")
        if not kernel.flags.contiguous:
            raise ValueError("mesh_kernel.complex_field must be contiguous")
    
    # Type check for BoxSize and Nmesh from mesh.attrs
    if "BoxSize" not in mesh.attrs or "Nmesh" not in mesh.attrs:
        raise ValueError("mesh.attrs must contain 'BoxSize' and 'Nmesh'")
    BoxSize = mesh.attrs["BoxSize"]
    Nmesh = mesh.attrs["Nmesh"]
    if not isinstance(BoxSize, (list, tuple, np.ndarray)) or len(BoxSize) != 3:
        raise TypeError(f"BoxSize must be a sequence of 3 numbers, got {BoxSize}")
    if not isinstance(Nmesh, (list, tuple, np.ndarray)) or len(Nmesh) != 3:
        raise TypeError(f"Nmesh must be a sequence of 3 integers, got {Nmesh}")
    
    k_x_array = np.fft.fftfreq(Nmesh[0], d=BoxSize[0] / Nmesh[0]) * 2.0 * np.pi
    k_y_array = np.fft.fftfreq(Nmesh[1], d=BoxSize[1] / Nmesh[1]) * 2.0 * np.pi
    k_z_array = np.fft.rfftfreq(Nmesh[2], d=BoxSize[2] / Nmesh[2]) * 2.0 * np.pi

    if dk is None:
        dk = k_z_array[1] - k_z_array[0]
    else:
        if not isinstance(dk, (int, float, np.number)):
            raise TypeError(f"dk must be a number, got {type(dk)}")

    k_perp_source = np.sqrt(k_x_array**2 + k_y_array**2)
    k_perp_min = np.min(k_perp_source)
    k_perp_max = np.max(k_perp_source)
    k_perp_edge = np.arange(k_perp_min, k_perp_max + dk, dk)
    k_perp_bin = len(k_perp_edge) - 1 

    k_parallel_edge = np.arange(k_z_array[0], k_z_array[-1] + dk, dk)
    k_parallel_bin = len(k_parallel_edge) - 1
    k_2d = np.zeros(shape=(k_perp_bin, k_parallel_bin, 2), dtype=np.float64)

    ps_dtype = np.float32 if complex_field.dtype == np.complex64 else np.float64
    ps_2d = np.zeros(shape=(k_perp_bin, k_parallel_bin), dtype=np.float64)
    modes_2d = np.zeros(shape=(k_perp_bin, k_parallel_bin), dtype=np.uint64)

    # Type checks for scalar parameters
    if not isinstance(ps_factor, (int, float, np.number)):
        raise TypeError(f"ps_factor must be a number, got {type(ps_factor)}")
    if not isinstance(shotnoise, (int, float, np.number)):
        raise TypeError(f"shotnoise must be a number, got {type(shotnoise)}")
    if not isinstance(nthreads, (int, np.integer)):
        raise TypeError(f"nthreads must be an integer, got {type(nthreads)}")

    if complex_field.dtype == np.complex64:
        cal_ps_2d_from_mesh_float(complex_field, kernel, ps_2d, k_2d, modes_2d, k_perp_edge, k_parallel_edge, k_x_array, k_y_array, k_z_array, ps_factor, shotnoise, nthreads)
    else:
        cal_ps_2d_from_mesh_double(complex_field, kernel, ps_2d, k_2d, modes_2d, k_perp_edge, k_parallel_edge, k_x_array, k_y_array, k_z_array, ps_factor, shotnoise, nthreads)

    index_zero = (modes_2d == 0)
    k_perp_temp = k_2d[:, :, 0]
    k_paral_temp = k_2d[:, :, 1]
    k_perp_temp[index_zero] = np.nan
    k_paral_temp[index_zero] = np.nan
    ps_2d[index_zero] = np.nan
    index_zero_not = np.logical_not(index_zero)
    k_perp_temp[index_zero_not] = k_perp_temp[index_zero_not] / modes_2d[index_zero_not]
    k_paral_temp[index_zero_not] = k_paral_temp[index_zero_not] / modes_2d[index_zero_not]
    ps_2d[index_zero_not] = ps_2d[index_zero_not] / modes_2d[index_zero_not]

    # Note: paral_factor is already applied in the C++ implementation (CalPS2D function) during mode counting
    # No need to apply it again here to avoid double counting

    return k_2d, ps_2d, modes_2d

def cal_pkmu_from_ps_2d(ps_2d, k_2d, modes_2d, k_edge, mu_edge, k_logarithmic=False, nthreads=1):
    # Type check for ps_2d
    if not isinstance(ps_2d, np.ndarray):
        raise TypeError(f"ps_2d must be a numpy.ndarray, got {type(ps_2d)}")
    if not ps_2d.flags.contiguous:
        raise ValueError("ps_2d must be contiguous")
    if ps_2d.dtype not in [np.float32, np.float64]:
        raise TypeError(f"ps_2d must be float32 or float64, got {ps_2d.dtype}")
    
    # Type check for k_2d
    if not isinstance(k_2d, np.ndarray):
        raise TypeError(f"k_2d must be a numpy.ndarray, got {type(k_2d)}")
    if not k_2d.flags.contiguous:
        raise ValueError("k_2d must be contiguous")
    if k_2d.dtype not in [np.float32, np.float64]:
        raise TypeError(f"k_2d must be float32 or float64, got {k_2d.dtype}")
    
    # Type check for modes_2d
    if not isinstance(modes_2d, np.ndarray):
        raise TypeError(f"modes_2d must be a numpy.ndarray, got {type(modes_2d)}")
    if not modes_2d.flags.contiguous:
        raise ValueError("modes_2d must be contiguous")
    if modes_2d.dtype not in [np.uint32, np.uint64]:
        raise TypeError(f"modes_2d must be uint32 or uint64, got {modes_2d.dtype}")

    k_2d = k_2d.astype(np.float64, copy=False)
    
    # Type check for k_edge
    if not isinstance(k_edge, np.ndarray):
        raise TypeError(f"k_edge must be a numpy.ndarray, got {type(k_edge)}")
    if not k_edge.flags.contiguous:
        raise ValueError("k_edge must be contiguous")
    if k_edge.dtype not in [np.float32, np.float64]:
        raise TypeError(f"k_edge must be float32 or float64, got {k_edge.dtype}")
    k_edge = k_edge.astype(np.float64, copy=False)

    kbin = len(k_edge) - 1

    # Type check for mu_edge
    if mu_edge is not None:
        if not isinstance(mu_edge, np.ndarray):
            raise TypeError(f"mu_edge must be a numpy.ndarray or None, got {type(mu_edge)}")
        if not mu_edge.flags.contiguous:
            raise ValueError("mu_edge must be contiguous")
        if mu_edge.dtype not in [np.float32, np.float64]:
            raise TypeError(f"mu_edge must be float32 or float64, got {mu_edge.dtype}")
        mu_edge = mu_edge.astype(np.float64, copy=False)
        mubin = len(mu_edge) - 1
        use_mu = True
    else:
        mu_edge = None
        mubin = 1
        use_mu = False

    k_out_2d = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    mu_out_2d = np.zeros(shape=(kbin, mubin), dtype=np.float64) if use_mu else None
    ps_kmu = np.zeros(shape=(kbin, mubin), dtype=np.float64)
    modes = np.zeros(shape=(kbin, mubin), dtype=np.uint64)

    # Type checks for scalar parameters
    if not isinstance(k_logarithmic, (bool, np.bool_)):
        raise TypeError(f"k_logarithmic must be a boolean, got {type(k_logarithmic)}")
    if not isinstance(nthreads, (int, np.integer)):
        raise TypeError(f"nthreads must be an integer, got {type(nthreads)}")

    cal_ps_from_ps_2d(ps_2d, k_2d, modes_2d, k_out_2d, mu_out_2d, ps_kmu, modes, k_edge, mu_edge, nthreads)

    return k_out_2d, mu_out_2d, ps_kmu, modes
