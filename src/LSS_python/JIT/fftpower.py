import numpy as np 
from numba import njit, set_num_threads, prange, get_thread_id

@njit(parallel=True)
def deal_ps_3d_multithreads(ps_3d, ps_3d_kernel=None, ps_3d_factor=1.0, shotnoise=0.0, nthreads=1):
    set_num_threads(nthreads)
    for ix in prange(ps_3d.shape[0]):
        for iy in prange(ps_3d.shape[1]):
            for iz in prange(ps_3d.shape[2]):
                kernel_element = ps_3d_kernel[ix, iy, iz] if ps_3d_kernel is not None else 1.0+0.0j
                ps_3d[ix, iy, iz] = (ps_3d[ix, iy, iz] * np.conj(ps_3d[ix, iy, iz]) * ps_3d_factor - shotnoise) * kernel_element
    return

@njit
def deal_ps_3d_single(ps_3d, ps_3d_kernel=None, ps_3d_factor=1.0, shotnoise=0.0):
    for ix in range(ps_3d.shape[0]):
        for iy in range(ps_3d.shape[1]):
            for iz in range(ps_3d.shape[2]):
                kernel_element = ps_3d_kernel[ix, iy, iz] if ps_3d_kernel is not None else 1.0+0.0j
                ps_3d[ix, iy, iz] = (ps_3d[ix, iy, iz] * np.conj(ps_3d[ix, iy, iz]) * ps_3d_factor - shotnoise) * kernel_element
    return

@njit(parallel=True)
def cal_ps_from_numba(ps_3d, k_arrays_list, k_array, mu_array, k_logarithmic=False, nthreads=1):
    set_num_threads(nthreads)
    kx_array, ky_array, kz_array = k_arrays_list
    kbin = k_array.shape[0] - 1 
    mubin = mu_array.shape[0] - 1
    k_min = k_array[0]
    k_max = k_array[-1]
    if k_logarithmic:
        k_diff = np.log(k_array[1]) - np.log(k_array[0])
        k_min_log = np.log(k_min)
    else:
        k_diff = k_array[1] - k_array[0]
        k_min_log = 0.0
    if mubin <= 1:
        use_mu = False
        mu_diff = 1.0
        mu_min = mu_array[0]
        mu_max = mu_array[-1]
    else:
        use_mu = True
        mu_diff = mu_array[1] - mu_array[0]
        mu_min = mu_array[0]
        mu_max = mu_array[-1]
    
    Pkmu_threads = np.zeros((nthreads, kbin, mubin), dtype=np.complex128)
    count_threads = np.zeros((nthreads, kbin, mubin), dtype=np.uint32)
    k_mesh_threads = np.zeros((nthreads, kbin, mubin), dtype=np.float64)
    mu_mesh_threads = np.zeros((nthreads, kbin, mubin), dtype=np.float64)

    kx_array = np.abs(kx_array)
    ky_array = np.abs(ky_array)
    for ix in prange(len(kx_array)):
        kx = kx_array[ix]
        for iy in prange(len(ky_array)):
            ky = ky_array[iy]
            for iz in prange(len(kz_array)):
                kz = kz_array[iz]
                if (kx <= k_min / 2.0 and  ky <= k_min / 2.0 and kz <= k_min / 2.0) or \
                (kx > k_max or ky > k_max or kz > k_max):
                    continue
                mode = 1 if iz == 0 else 2
                k = np.sqrt(kx**2 + ky**2 + kz**2)
                if k < k_min or k > k_max:
                    continue
                if k_logarithmic:
                    k_i = int((np.log(k) - k_min_log) / k_diff)
                else:
                    k_i = int((k - k_min) / k_diff)
                if k_i == kbin:
                    k_i -= 1 

                if use_mu:
                    mu = kz / k 
                    if mu < mu_min or mu > mu_max:
                        continue
                    mu_i = int((mu - mu_min) / mu_diff)
                    if mu_i == mubin:
                        mu_i -= 1
                else:
                    mu = 0.0
                    mu_i = 0 
                thread_id = get_thread_id()
                Pkmu_threads[thread_id, k_i, mu_i] += ps_3d[ix, iy, iz] * mode
                k_mesh_threads[thread_id, k_i, mu_i] += k * mode
                mu_mesh_threads[thread_id, k_i, mu_i] += mu * mode
                count_threads[thread_id, k_i, mu_i] += mode
    Pkmu = np.sum(Pkmu_threads, axis=0)
    k_mesh = np.sum(k_mesh_threads, axis=0)
    mu_mesh = np.sum(mu_mesh_threads, axis=0)
    count = np.sum(count_threads, axis=0)
    return k_mesh, mu_mesh, Pkmu, count 