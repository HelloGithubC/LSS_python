import numpy as np 

from .lib.mesh_pybind import run_mesh_float, run_mesh_double # type: ignore
from .lib.mesh_pybind import do_compensation_float, do_compensation_double # type: ignore
from .lib.mesh_pybind import do_compensation_interlaced_float, do_compensation_interlaced_double # type: ignore
from .lib.mesh_pybind import do_interlace_float, do_interlace_double # type: ignore

def to_mesh_pybind(pos, boxsize, ngrids, field, weights, values, shift, resampler, nthreads=1):
    pos = np.ascontiguousarray(pos)
    if resampler == "NGP":
        mesh_type = 0
    elif resampler == "CIC":
        mesh_type = 1
    elif resampler == "TSC":
        mesh_type = 2
    elif resampler == "PCS":
        mesh_type = 3
    else:
        raise ValueError(f"Invalid resampler: {resampler}")
    if pos.dtype == np.float32:
        run_mesh_float(pos, field, weights, values, boxsize, ngrids, shift, mesh_type, nthreads)
    else:
        run_mesh_double(pos, field, weights, values, boxsize, ngrids, shift, mesh_type, nthreads)
    
def do_compensation_pybind(complex_field, ngrids, k_arrays, resampler, do_interlaced=False, nthreads=1):
    if not complex_field.flags.contiguous:
        raise ValueError("complex_field must be contiguous")
    if resampler == "CIC":
        mesh_type = 1
        p = 2
    elif resampler == "TSC":
        mesh_type = 2
        p = 3
    elif resampler == "PCS":
        mesh_type = 3
        p = 4
    else:
        raise ValueError(f"Invalid resampler: {resampler}")
    
    kx_array, ky_array, kz_array = k_arrays
    if complex_field.dtype == np.float32:
        if do_interlaced:
            do_compensation_interlaced_float(complex_field, ngrids, kx_array, ky_array, kz_array, p, nthreads)
        else:
            do_compensation_float(complex_field, ngrids, kx_array, ky_array, kz_array, mesh_type, nthreads)
    else:
        if do_interlaced:
            do_compensation_interlaced_double(complex_field, ngrids, kx_array, ky_array, kz_array, p, nthreads)
        else:
            do_compensation_double(complex_field, ngrids, kx_array, ky_array, kz_array, mesh_type, nthreads)
    
def do_interlace_pybind(c1, c2, boxsize, Nmesh, k_arrays, nthreads):
    H = boxsize / Nmesh
    try: 
        length = len(H)
    except TypeError:
        H = np.array([H] * 3, dtype=np.float64)
    else:
        H = np.array(H, dtype=np.float64)
    ngrids = Nmesh

    if not c1.flags.contiguous:
        raise ValueError("c1 must be contiguous")
    c2 = np.ascontiguousarray(c2)
    kx_array, ky_array, kz_array = k_arrays
    if c1.dtype == np.float32:
        do_interlace_float(c1, c2, H, ngrids, kx_array, ky_array, kz_array, nthreads)
    else:
        do_interlace_double(c1, c2, H, ngrids, kx_array, ky_array, kz_array, nthreads)