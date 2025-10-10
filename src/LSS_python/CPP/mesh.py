import numpy as np 
import os 
import ctypes 

from .base import c_double_complex, c_float_complex

script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_lib = ctypes.CDLL(os.path.join(script_dir + "/lib", "mesh.so"))
def to_mesh_c_api(pos, boxsize, ngrids, field, weights=None, values=None, resampler = "CIC", shift=0.0, nthreads=1):
    global mesh_lib
    dtype = pos.dtype 
    if dtype == np.float32 or dtype == np.float64:
        pass 
    else:
        try:
            pos = pos.astype(np.float32)
            dtype = np.float32
        except:
            raise ValueError("Positions must be float32 or float64, or other types that can be cast to float32")
        
    if resampler == "CIC":
        to_mesh_float = mesh_lib.run_cic_float
        to_mesh_double = mesh_lib.run_cic_double 
    elif resampler == "NGP":
        to_mesh_float = mesh_lib.run_ngp_float
        to_mesh_double = mesh_lib.run_ngp_double 
    elif resampler == "TSC":
        to_mesh_float = mesh_lib.run_tsc_float
        to_mesh_double = mesh_lib.run_tsc_double
    elif resampler == "PCS":
        to_mesh_float = mesh_lib.run_pcs_float
        to_mesh_double = mesh_lib.run_pcs_double 
    else:
        raise ValueError("Method must be CIC, NGP, TSC or PCS")
    
    if weights is not None:
        if weights.shape[0] != pos.shape[0]:
            raise ValueError("Weights must have the same number of particles as positions")
        if weights.dtype != dtype:
            weights = weights.astype(dtype)
    if values is not None:
        if values.shape[0] != pos.shape[0]:
            raise ValueError("Values must have the same number of particles as positions")
        if values.dtype != dtype:
            values = values.astype(dtype)

    try:
        length = len(boxsize)
    except TypeError:
        boxsize_array = np.array([boxsize] * 3, dtype=dtype)
    else:
        boxsize_array = np.array(boxsize, dtype=dtype)
    try: 
        length = len(ngrids)
    except TypeError:
        ngrids_array = np.array([ngrids] * 3, dtype=np.uint64)
    else:
        ngrids_array = np.array(ngrids, dtype=np.uint64)

    ngrids_ptr = ngrids_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    nparticle = pos.shape[0]
    ctypes_ptr = ctypes.c_float if dtype == np.float32 else ctypes.c_double
    
    if field.shape != tuple(ngrids_array):
        raise ValueError("Output array must have the same shape as ngrids")
    if field.dtype != dtype:
        raise ValueError("Output array must have the same dtype as positions")

    pos_ptr = pos.ctypes.data_as(ctypes.POINTER(ctypes_ptr))
    field_ptr = field.ctypes.data_as(ctypes.POINTER(ctypes_ptr))
    boxsize_ptr = boxsize_array.ctypes.data_as(ctypes.POINTER(ctypes_ptr))
    weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes_ptr)) if weights is not None else ctypes.POINTER(ctypes_ptr)()
    values_ptr = values.ctypes.data_as(ctypes.POINTER(ctypes_ptr)) if values is not None else ctypes.POINTER(ctypes_ptr)()

    nparticle = ctypes.c_uint64(nparticle)
    nthreads = ctypes.c_int(nthreads)

    if dtype == np.float32:
        shift = ctypes.c_float(shift)
        to_mesh_float(pos_ptr, field_ptr, nparticle, boxsize_ptr, ngrids_ptr, weights_ptr, values_ptr, shift, nthreads)
    else:
        do_shift = ctypes.c_double(shift)
        to_mesh_double(pos_ptr, field_ptr, nparticle, boxsize_ptr, ngrids_ptr, weights_ptr, values_ptr, do_shift, nthreads)

    return field

def do_compensation_c_api(complex_field, k_arrays=None, resampler="CIC", interlaced=False, nthreads=1, copy=False):
    dtype = complex_field.dtype 
    if dtype == np.complex64 or dtype == np.complex128:
        pass 
    else:
        try:
            complex_field = complex_field.astype(np.complex64)
            dtype = np.complex64
            copy = False
        except:
            raise ValueError("Positions must be complex64 or complex128, or other types that can be cast to complex64")
        
    if copy:
        complex_field = np.copy(complex_field)

    if interlaced:
        do_compensation_float = None 
        if resampler == "NGP":
            return complex_field
    else:    
        if resampler == "CIC":
            do_compensation_float = mesh_lib.do_compensation_cic_float
            do_compensation_double = mesh_lib.do_compensation_cic_double
        elif resampler == "NGP":
            return complex_field 
        elif resampler == "TSC":
            do_compensation_float = mesh_lib.do_compensation_tsc_float
            do_compensation_double = mesh_lib.do_compensation_tsc_double
        elif resampler == "PCS":
            do_compensation_float = mesh_lib.do_compensation_pcs_float
            do_compensation_double = mesh_lib.do_compensation_pcs_double
        else:
            raise ValueError("Method must be CIC, NGP, TSC or PCS")

    kx_array, ky_array, kz_array = k_arrays
    kx_array = kx_array.astype(np.float64, copy=False)
    ky_array = ky_array.astype(np.float64, copy=False)
    kz_array = kz_array.astype(np.float64, copy=False)
    ngrids_array = np.array([kx_array.shape[0], ky_array.shape[0], kz_array.shape[0]], dtype=np.uint64)

    ngrids_ptr = ngrids_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    complex_ctype = c_float_complex if dtype == np.complex64 else c_double_complex
    
    complex_field_ptr = complex_field.ctypes.data_as(ctypes.POINTER(complex_ctype))
    kx_array_ptr = kx_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ky_array_ptr = ky_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    kz_array_ptr = kz_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    nthreads = ctypes.c_int(nthreads)

    if not interlaced:
        if dtype == np.complex64:
            do_compensation_float(complex_field_ptr, ngrids_ptr, kx_array_ptr, ky_array_ptr, kz_array_ptr, nthreads)
        else:
            do_compensation_double(complex_field_ptr, ngrids_ptr, kx_array_ptr, ky_array_ptr, kz_array_ptr, nthreads)
    else:
        if resampler == "CIC":
            p = 2
        elif resampler == "TSC":
            p = 3
        elif resampler == "PCS":
            p = 4
        else:
            raise ValueError("Method must be CIC, NGP, TSC or PCS")
        p = ctypes.c_int(p)
        if dtype == np.complex64:
            mesh_lib.do_compensation_intelaced_float(complex_field_ptr, ngrids_ptr, kx_array_ptr, ky_array_ptr, kz_array_ptr, p, nthreads)
        else:
            mesh_lib.do_compensation_intelaced_double(complex_field_ptr, ngrids_ptr, kx_array_ptr, ky_array_ptr, kz_array_ptr, p, nthreads)

    return complex_field

def do_interlacing_c_api(c1, c2, boxsize, Nmesh, k_arrays, nthreads=1):
    H = boxsize / Nmesh
    try: 
        length = len(H)
    except TypeError:
        H_array = np.array([H] * 3, dtype=np.float64)
    else:
        H_array = np.array(H, dtype=np.float64)
    if k_arrays is None:
        kx_array = (
            np.fft.fftfreq(Nmesh[0], d=1.0)
            * 2.0
            * np.pi
            * Nmesh[0]
            / boxsize[0]
        )
        ky_array = (
            np.fft.fftfreq(Nmesh[1], d=1.0)
            * 2.0
            * np.pi
            * Nmesh[1]
            / boxsize[1]
        )
        kz_array = (
            np.fft.fftfreq(Nmesh[2], d=1.0)
            * 2.0
            * np.pi
            * Nmesh[2]
            / boxsize[2]
        )[: c1.shape[2]]
    else:
        kx_array, ky_array, kz_array = k_arrays
        kx_array = kx_array.astype(np.float64, copy=False)
        ky_array = ky_array.astype(np.float64, copy=False)
        kz_array = kz_array.astype(np.float64, copy=False)
    ngrids_array = np.array([kx_array.shape[0], ky_array.shape[0], kz_array.shape[0]], dtype=np.uint64)

    complex_ctype = c_float_complex if c1.dtype == np.complex64 else c_double_complex
    c1_ptr = c1.ctypes.data_as(ctypes.POINTER(complex_ctype))
    c2_ptr = c2.ctypes.data_as(ctypes.POINTER(complex_ctype))
    H_ptr = H_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ngrids_ptr = ngrids_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    kx_array_ptr = kx_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ky_array_ptr = ky_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    kz_array_ptr = kz_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    nthreads = ctypes.c_int(nthreads)

    if c1.dtype == np.complex64:
        mesh_lib.do_interlacing_float(c1_ptr, c2_ptr, H_ptr, ngrids_ptr, kx_array_ptr, ky_array_ptr, kz_array_ptr, nthreads)
    else:
        mesh_lib.do_interlacing_double(c1_ptr, c2_ptr, H_ptr, ngrids_ptr, kx_array_ptr, ky_array_ptr, kz_array_ptr, nthreads)