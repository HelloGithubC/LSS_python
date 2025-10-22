import numpy as np 
from numba import njit 
from scipy.spatial import KDTree

@njit 
def w_kernel_jit(d_array, h):
    w_array = np.zeros(d_array.shape, dtype=np.float64)
    for i in range(d_array.shape[0]):
        sigma = 1.0 / (np.pi * h**3) 
        q = d_array[i] / h
        if q < 0.0:
            w_array[i] = 0.0
        elif q <= 1.0:
            w_array[i] = sigma * (0.25 *(2.0-q)**3 - (1.0-q)**3)
        elif q < 2.0:
            w_array[i] = sigma * 0.25 *(2.0-q)**3
        else:
            w_array[i] = 0.0
    return w_array

@njit
def dw_kernel_jit(d_array, h):
    dw_array = np.zeros(d_array.shape, dtype=np.float64)
    for i in range(d_array.shape[0]):
        sigma = 1.0 / (np.pi * h**4) 
        q = d_array[i] / h
        if q < 0.0:
            dw_array[i] = 0.0
        elif q <= 1.0:
            dw_array[i] = sigma * (0.25*3.0 * (2.0-q)**2 - 3.0*(1.0-q)**2)
        elif q < 2.0:
            dw_array[i] = sigma * 0.25 * 3.0 * (2.0-q)**2
        else:
            dw_array[i] = 0.0
    return dw_array

@njit 
def cal_rho_array(distance_array, h, use_max_distance_as_h=False):
    """ cal rho to get MCF
    Note:
        distance_array: [n_galaxy, n_neighbor]
    """
    rho_array = np.zeros(distance_array.shape[0], dtype=np.float64)
    for i in range(distance_array.shape[0]):
        if use_max_distance_as_h:
            h = np.max(distance_array[i]) / 2.0
        rho_temp = w_kernel_jit(distance_array[i], h)
        rho_array[i] = np.sum(rho_temp)
    return rho_array

@njit 
def cal_drho_array(distance_array, x0, h, use_max_distance_as_h=False):
    """ cal drho 
    Note:
        distance_array: [n_galaxy, n_neighbor]
    """
    raise NotImplementedError

def w_kernel(d, h):
    sigma = 1.0 / (np.pi * h**3)
    q = d / h
    if q < 0.0:
        return 0.0
    elif q <= 1.0:
        return sigma * (0.25 *(2.0-q)**3 - (1.0-q)**3)
    elif q < 2.0:
        return sigma * 0.25 *(2.0-q)**3
    else:
        return 0.0
    
def cal_rho(distance_array, h, use_max_distance_as_h=False):
    """
    Note:
        distance_array: [n_neighbor]
    """
    if use_max_distance_as_h:
        h = np.max(distance_array) / 2.0
    return np.sum(w_kernel(distance_array, h))

def create_rho(data, boxsize, k=30, nthreads=1):
    kdtree = KDTree(data, boxsize=boxsize + 1e-5)
    distance_array, _ = kdtree.query(data, k=k, workers=nthreads)
    rho_array = cal_rho_array(distance_array, h=None, use_max_distance_as_h=True)
    data_new = np.concatenate([data, rho_array[:, None]], axis=1)
    return data_new
