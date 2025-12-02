import numpy as np 
from numba import njit

def get_sub_box_shift(position, boxsize, ngrids, max_points_default=None, perodic=False):
    if isinstance(boxsize, float) or isinstance(boxsize, int):
        boxsize = np.array([boxsize, boxsize, boxsize])
    if isinstance(ngrids, float) or isinstance(ngrids, int):
        ngrids = np.array([ngrids, ngrids, ngrids], dtype=np.int32)
    return get_sub_box_shift_core(position, boxsize, ngrids, max_points_default, perodic)

@njit
def get_sub_box_shift_core(data, boxsize, ngrids, max_points_default=None, perodic=False):
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