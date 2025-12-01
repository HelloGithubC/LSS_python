import numpy as np
from numba import njit, prange, get_thread_id, set_num_threads
@njit 
def run_cic_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, shift=0.0):
    NDIM = 3
    nparticle = pos.shape[0]
    pos_i = np.zeros((2, NDIM), dtype=np.uint32)
    for i in range(nparticle):
        sub_BoxSize = BoxSize_array / Nmesh_array
        is_in_box = True
        
        diff_ratio_temp = np.zeros((NDIM,2), dtype=np.float64)
        for j in range(NDIM):
            if pos[i, j] < 0.0 or pos[i, j] > BoxSize_array[j]:
                is_in_box = False
                break 

            pos_i_float = pos[i, j] / sub_BoxSize[j] + shift
            pos_i_temp = np.int64(np.floor(pos_i_float))
            diff_ratio_temp[j, 1] = pos_i_float - pos_i_temp
            diff_ratio_temp[j, 0] = 1.0 - diff_ratio_temp[j, 1]

            pos_i[0, j] = (pos_i_temp + Nmesh_array[j]) % Nmesh_array[j]
            pos_i[1, j] = (pos_i[0, j] + 1) % Nmesh_array[j]
        
        if not is_in_box:
            continue
        weight_temp = weight[i] if weight is not None else 1.0
        value_temp = value[i] if value is not None else 1.0
        field_temp = value_temp * weight_temp

        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    field[pos_i[ix, 0], pos_i[iy, 1], pos_i[iz, 2]] += (diff_ratio_temp[0, ix] * diff_ratio_temp[1, iy] * diff_ratio_temp[2, iz]) * field_temp

@njit 
def run_ngp_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, shift=0.0):
    NDIM = 3
    nparticle = pos.shape[0]
    pos_i = np.zeros(NDIM, dtype=np.uint32)
    pos_i_float = np.zeros(NDIM, dtype=np.float64)
    for i in range(nparticle):
        sub_BoxSize = BoxSize_array / Nmesh_array  
        is_in_box = True
        
        for j in range(NDIM):
            if pos[i, j] < 0.0 or pos[i, j] > BoxSize_array[j]:
                is_in_box = False
                break
            pos_i_float[j] = pos[i, j] / sub_BoxSize[j] + shift
            pos_i_temp = np.int64(np.floor(pos_i_float[j] + 0.5))

            pos_i[j] = (pos_i_temp + Nmesh_array[j]) % Nmesh_array[j]

        if not is_in_box:
            continue
        x_i, y_i, z_i = pos_i 

        weight_temp = weight[i] if weight is not None else 1.0
        value_temp = value[i] if value is not None else 1.0
        field_temp = value_temp * weight_temp


        field[x_i, y_i, z_i] += field_temp

@njit
def run_tsc_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, shift=0.0):
    NDIM = 3
    nparticle = pos.shape[0]
    pos_i = np.zeros((3,NDIM), dtype=np.uint32)
    diff_ratio_temp = np.zeros((NDIM, 3), dtype=np.float64)
    for i in range(nparticle):
        sub_BoxSize = BoxSize_array / Nmesh_array
        is_in_box = True

        for j in range(NDIM):
            if pos[i, j] < 0.0 or pos[i, j] > BoxSize_array[j]:
                is_in_box = False
                break

            pos_i_float = pos[i, j] / sub_BoxSize[j] + shift 
            pos_i_temp = np.int64(np.floor(pos_i_float - 0.5))
            for k in range(3):
                distance_ratio_temp = np.abs(pos_i_temp + k - pos_i_float) 
                if distance_ratio_temp < 0.5:
                    diff_ratio_temp[j, k] = 0.75 - distance_ratio_temp**2 
                elif distance_ratio_temp < 1.5:
                    diff_ratio_temp[j, k] = 0.5 * (1.5 - distance_ratio_temp)**2
                else:
                    diff_ratio_temp[j, k] = 0.0

            pos_i[0, j] = (pos_i_temp + Nmesh_array[j]) % Nmesh_array[j]
            pos_i[1, j] = (pos_i[0,j] + 1) % Nmesh_array[j]
            pos_i[2, j] = (pos_i[1,j] + 1) % Nmesh_array[j]
        
        if not is_in_box:
            continue

        weight_temp = weight[i] if weight is not None else 1.0
        value_temp = value[i] if value is not None else 1.0
        field_temp = value_temp * weight_temp

        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    field[pos_i[ix][0], pos_i[iy][1], pos_i[iz][2]] += (diff_ratio_temp[0,ix] * diff_ratio_temp[1,iy] * diff_ratio_temp[2,iz]) * field_temp

@njit
def run_pcs_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, shift=0.0):
    NDIM = 3
    nparticle = pos.shape[0]
    pos_i = np.zeros((4,NDIM), dtype=np.uint32)
    diff_ratio_temp = np.zeros((NDIM, 4), dtype=np.float64)
    for i in range(nparticle):
        sub_BoxSize = BoxSize_array / Nmesh_array
        is_in_box = True

        for j in range(NDIM):
            if pos[i, j] < 0.0 or pos[i, j] > BoxSize_array[j]:
                is_in_box = False
                break

            pos_i_float = pos[i, j] / sub_BoxSize[j] + shift 
            pos_i_temp = np.int64(np.floor(pos_i_float - 1.0))
            for k in range(4):
                distance_ratio_temp = np.abs(pos_i_temp + k - pos_i_float) 
                if distance_ratio_temp < 1.0:
                    diff_ratio_temp[j, k] = (4.0 - 6.0 * distance_ratio_temp**2 + 3.0 * distance_ratio_temp**3) / 6.0
                elif distance_ratio_temp < 2.0:
                    diff_ratio_temp[j, k] = (2.0 - distance_ratio_temp) ** 3 / 6.0
                else:
                    diff_ratio_temp[j, k] = 0.0

            pos_i[0, j] = (pos_i_temp + Nmesh_array[j]) % Nmesh_array[j]
            pos_i[1, j] = (pos_i[0,j] + 1) % Nmesh_array[j]
            pos_i[2, j] = (pos_i[1,j] + 1) % Nmesh_array[j]
            pos_i[3, j] = (pos_i[2,j] + 1) % Nmesh_array[j]
        
        if not is_in_box:
            continue

        weight_temp = weight[i] if weight is not None else 1.0
        value_temp = value[i] if value is not None else 1.0
        field_temp = value_temp * weight_temp

        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    field[pos_i[ix][0], pos_i[iy][1], pos_i[iz][2]] += (diff_ratio_temp[0,ix] * diff_ratio_temp[1,iy] * diff_ratio_temp[2,iz]) * field_temp

@njit(parallel=True)
def do_compensation_cic(complex_field, k_arrays, nthreads=1):
    set_num_threads(nthreads)
    k_x_array, k_y_array, k_z_array = k_arrays
    for i in prange(complex_field.shape[0]):
        k_x = k_x_array[i]
        w_x = np.sqrt(1.0 - 2.0/3.0 * np.sin(k_x/2.0)**2)
        for j in range(complex_field.shape[1]):
            k_y = k_y_array[j]
            w_y = np.sqrt(1.0 - 2.0/3.0 * np.sin(k_y/2.0)**2)
            for k in range(complex_field.shape[2]):
                k_z = k_z_array[k]
                w_z = np.sqrt(1.0 - 2.0/3.0 * np.sin(k_z/2.0)**2)
                complex_field[i, j, k] /= w_x * w_y * w_z

@njit(parallel=True)
def do_compensation_tsc(complex_field, k_arrays, nthreads=1):
    set_num_threads(nthreads)
    k_x_array, k_y_array, k_z_array = k_arrays
    for i in prange(complex_field.shape[0]):
        k_x = k_x_array[i]
        sinx_temp = np.sin(k_x/2.0)
        w_x = np.sqrt(1.0 - sinx_temp**2 + 2.0/15.0 * sinx_temp**4)
        for j in range(complex_field.shape[1]):
            k_y = k_y_array[j]
            siny_temp = np.sin(k_y/2.0)
            w_y = np.sqrt(1.0 - siny_temp**2 + 2.0/15.0 * siny_temp**4)
            for k in range(complex_field.shape[2]):
                k_z = k_z_array[k]
                sinz_temp = np.sin(k_z/2.0)
                w_z = np.sqrt(1.0 - sinz_temp**2 + 2.0/15.0 * sinz_temp**4)
                complex_field[i, j, k] /= w_x * w_y * w_z

@njit(parallel=True)
def do_compensation_pcs(complex_field, k_arrays, nthreads=1):
    set_num_threads(nthreads)
    k_x_array, k_y_array, k_z_array = k_arrays
    for i in prange(complex_field.shape[0]):
        k_x = k_x_array[i]
        sinx_temp = np.sin(k_x/2.0)
        w_x = np.sqrt(1.0 - 4.0 / 3.0 * sinx_temp**2 + 6.0/15.0 * sinx_temp**4 - 4.0/315.0 * sinx_temp**6)
        for j in range(complex_field.shape[1]):
            k_y = k_y_array[j]
            siny_temp = np.sin(k_y/2.0)
            w_y = np.sqrt(1.0 - 4.0 / 3.0 * siny_temp**2 + 6.0/15.0 * siny_temp **4 - 4.0/315.0 * siny_temp**6)
            for k in range(complex_field.shape[2]):
                k_z = k_z_array[k]
                sinz_temp = np.sin(k_z/2.0)
                w_z = np.sqrt(1.0 - 4.0 / 3.0 * sinz_temp**2 + 6.0/15.0 * sinz_temp**4 - 4.0/315.0 * sinz_temp**6)
                complex_field[i, j, k] /= w_x * w_y * w_z

@njit(parallel=True)
def do_compensation_interlace(complex_field, k_arrays, p=1, nthreads=1):
    k_x_array, k_y_array, k_z_array = k_arrays
    set_num_threads(nthreads)
    for i in prange(complex_field.shape[0]):
        k_x = k_x_array[i]
        w_x = np.sinc(k_x / 2.0 / np.pi) ** p
        w_x = 1.0 if k_x == 0.0 else w_x
        for j in range(complex_field.shape[1]):
            k_y = k_y_array[j]
            w_y = np.sinc(k_y / 2.0 / np.pi) ** p
            w_y = 1.0 if k_y == 0.0 else w_y
            for k in range(complex_field.shape[2]):
                k_z = k_z_array[k]
                w_z = np.sinc(k_z / 2.0 / np.pi) ** p
                w_z = 1.0 if k_z == 0.0 else w_z
                complex_field[i, j, k] /= w_x * w_y * w_z

@njit(parallel=True)
def do_interlacing(c1, c2, H, k_arrays):
    kx_array, ky_array, kz_array = k_arrays
    for ix in prange(kx_array.shape[0]):
        kx = kx_array[ix]
        for iy in range(ky_array.shape[0]):
            ky = ky_array[iy]
            for iz in range(kz_array.shape[0]):
                kz = kz_array[iz]
                kH = kx * H[0] + ky * H[1] + kz * H[2]
                c1[ix, iy, iz] = c1[ix, iy, iz] * 0.5 + c2[ix, iy, iz] * 0.5 * np.exp(0.5 * 1j * kH)

def to_mesh_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, resampler="CIC", shift=0.0):
    if resampler == "CIC":
        run_cic_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, shift)
    elif resampler == "TSC":
        run_tsc_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, shift)
    elif resampler == "PCS":
        run_pcs_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, shift)
    elif resampler == "NGP":
        run_ngp_numba(pos, weight, value, field, BoxSize_array, Nmesh_array, shift)
    else:
        raise ValueError(f"resampler must be CIC, NGP, TSC or PCS. Now is {resampler}")
                
def do_compensation_from_numba(complex_field, k_arrays, resampler="CIC", interlace=False, nthreads=1):
    if interlace:
        if resampler == "CIC":
            do_compensation_interlace(complex_field, k_arrays, p=2, nthreads=nthreads)
        elif resampler == "TSC":
            do_compensation_interlace(complex_field, k_arrays, p=3, nthreads=nthreads)
        elif resampler == "PCS":
            do_compensation_interlace(complex_field, k_arrays, p=4, nthreads=nthreads)
        elif resampler == "NGP":
            return
        else:
            raise ValueError(f"resampler must be CIC, NGP, TSC or PCS. Now is {resampler}")   
    else: 
        if resampler == "CIC":
            do_compensation_cic(complex_field, k_arrays, nthreads)
        elif resampler == "TSC":
            do_compensation_tsc(complex_field, k_arrays, nthreads)
        elif resampler == "PCS":
            do_compensation_pcs(complex_field, k_arrays, nthreads)
        elif resampler == "NGP":
            return
        else:
            raise ValueError(f"resampler must be CIC, NGP, TSC or PCS. Now is {resampler}")
    
def do_interlacing_from_numba(complex_field, complex_field_shift, BoxSize, Nmesh, k_arrays):
    H = BoxSize / Nmesh 
    try:
        length = len(H)
    except TypeError:
        H = np.array([H] * 3)
    if k_arrays is None:
        k_x_array = (
            np.fft.fftfreq(Nmesh[0], d=1.0)
            * 2.0
            * np.pi
            * Nmesh[0]
            / BoxSize[0]
        )
        k_y_array = (
            np.fft.fftfreq(Nmesh[1], d=1.0)
            * 2.0
            * np.pi
            * Nmesh[1]
            / BoxSize[1]
        )
        k_z_array = (
            np.fft.fftfreq(Nmesh[2], d=1.0)
            * 2.0
            * np.pi
            * Nmesh[2]
            / BoxSize[2]
        )[: complex_field.shape[2]]
        k_arrays = (k_x_array, k_y_array, k_z_array)
    do_interlacing(complex_field, complex_field_shift, H, k_arrays)