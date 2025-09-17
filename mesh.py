import numpy as np 
from numba import njit

@njit 
def run_cic(pos, field, boxsize_array, ngrids_array):
    nparticle = pos.shape[0]
    pos_i = np.zeros(3, dtype=np.uint32)
    for i in range(nparticle):
        sub_boxsize = boxsize_array / ngrids_array
        pos_temp = np.copy(pos[i])
        
        diff_ratio_temp = np.zeros((3,2), dtype=np.float64)
        for j in range(3):
            if pos_temp[j] < 0.0 or pos_temp[j] > boxsize_array[j]:
                continue
            pos_i[j] = np.uint32(pos_temp[j] / sub_boxsize[j])
            if pos_i[j] == ngrids_array[j]:
                pos_i[j] = 0
                pos_temp[j] = 0.0 
            
            diff_ratio_temp[j,0] = (pos_temp[j] - pos_i[j] * sub_boxsize[j]) / sub_boxsize[j]
            diff_ratio_temp[j,1] = 1.0 - diff_ratio_temp[j,0]

        x_i, y_i, z_i = pos_i 
        x_i_next = x_i + 1 if x_i < ngrids_array[0] - 1 else 0
        y_i_next = y_i + 1 if y_i < ngrids_array[1] - 1 else 0
        z_i_next = z_i + 1 if z_i < ngrids_array[2] - 1 else 0

        field[x_i, y_i, z_i] += diff_ratio_temp[0,1] * diff_ratio_temp[1,1] * diff_ratio_temp[2,1]
        field[x_i, y_i, z_i_next] += diff_ratio_temp[0,1] * diff_ratio_temp[1,1] * diff_ratio_temp[2,0]
        field[x_i, y_i_next, z_i] += diff_ratio_temp[0,1] * diff_ratio_temp[1,0] * diff_ratio_temp[2,1]
        field[x_i, y_i_next, z_i_next] += diff_ratio_temp[0,1] * diff_ratio_temp[1,0] * diff_ratio_temp[2,0]

        field[x_i_next, y_i, z_i] += diff_ratio_temp[0,0] * diff_ratio_temp[1,1] * diff_ratio_temp[2,1]
        field[x_i_next, y_i, z_i_next] += diff_ratio_temp[0,0] * diff_ratio_temp[1,1] * diff_ratio_temp[2,0]
        field[x_i_next, y_i_next, z_i] += diff_ratio_temp[0,0] * diff_ratio_temp[1,0] * diff_ratio_temp[2,1]
        field[x_i_next, y_i_next, z_i_next] += diff_ratio_temp[0,0] * diff_ratio_temp[1,0] * diff_ratio_temp[2,0]

def run_cic_from_cuda(pos, boxsize, ngrids):
    import cupy as cp 
    from .cuda.mesh import run_cic_core_kernel
    if isinstance(boxsize, float):
        boxsize = cp.array([boxsize, boxsize, boxsize])
    else:
        boxsize = cp.array(boxsize)
    if isinstance(ngrids, int):
        ngrids = cp.array([ngrids, ngrids, ngrids])
    else:
        ngrids = cp.array(ngrids)
    
    nparticle = pos.shape[0]
    nthreads = nparticle if nparticle < 1024 else 1024 
    ngridx = nparticle // nthreads + 1

    field = cp.zeros((ngrids[0].item(), ngrids[1].item(), ngrids[2].item()), dtype=cp.float32)
    run_cic_core_kernel((ngridx, ), (nthreads, ), (pos, nparticle, field, boxsize, ngrids))

    return cp.asnumpy(field)