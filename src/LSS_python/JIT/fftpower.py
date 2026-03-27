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

def cal_ps_2d_from_mesh(mesh, mesh_kernel=None, k_arrays=None, ps_factor=1.0, shotnoise=0.0, nthreads=1, return_modes=False):
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
    
    k_perp_source = np.sqrt(k_x_array**2 + k_y_array**2)
    k_perp_min = np.min(k_perp_source)
    k_perp_max = np.max(k_perp_source)
    dk_perp = k_z_array[1] - k_z_array[0]
    k_perp_edge = np.arange(k_perp_min, k_perp_max + dk_perp, dk_perp)

    k_2d, ps_2d, modes_2d = cal_ps_2d_core(complex_field, kernel, [k_x_array, k_y_array, k_z_array], k_perp_edge, ps_factor, shotnoise, nthreads)

    if return_modes:
        return k_2d, ps_2d, modes_2d
    else:
        return k_2d, ps_2d

@njit(parallel=True)
def cal_ps_2d_core(complex_field, kernel, k_arrays, k_perp_edge, ps_factor, shotnoise, nthreads=1):
    k_x_array, k_y_array, k_z_array = k_arrays
    k_parallel_size = len(k_z_array)
    k_perp_size = len(k_perp_edge) - 1
    dk_perp = k_perp_edge[1] - k_perp_edge[0]

    k_2d = np.zeros((k_perp_size, k_parallel_size, 2), dtype=np.float32)
    ps_2d = np.zeros((k_perp_size, k_parallel_size), dtype=np.complex64)
    modes_2d = np.zeros((k_perp_size, k_parallel_size), dtype=np.uint64)

    set_num_threads(nthreads)
    for i_z in prange(len(k_z_array)):
        k_z = k_z_array[i_z]
        modes = np.zeros(k_perp_size, dtype=np.int32)
        k_2d[:, i_z, 1] = k_z
        for i_x in range(len(k_x_array)):
            k_x = k_x_array[i_x]
            for i_y in range(len(k_y_array)):
                k_y = k_y_array[i_y]
                k_perp = np.sqrt(k_x ** 2 + k_y ** 2)
                if k_perp < k_perp_edge[0] or k_perp > k_perp_edge[-1]:
                    continue
                k_perp_index = int((k_perp - k_perp_edge[0]) / dk_perp)
                if k_perp_index == k_perp_size:
                    k_perp_index -= 1
                else:
                    kernel_value = 1.0 if kernel is None else kernel[i_x, i_y, i_z]
                    modes[k_perp_index] += 1
                    k_2d[k_perp_index, i_z, 0] += k_perp
                    ps_2d[k_perp_index, i_z] += ((complex_field[i_x, i_y, i_z] * np.conjugate(complex_field[i_x, i_y, i_z]) * ps_factor) - shotnoise) * kernel_value
        for i in range(k_perp_size):
            if modes[i] > 0:
                k_2d[i, i_z, 0] /= modes[i]
                ps_2d[i, i_z] /= modes[i]
            else:
                k_2d[i, i_z, 0] = np.nan 
                ps_2d[i, i_z] = np.nan
        modes_2d[:,i_z] = modes
    return k_2d, ps_2d, modes_2d

@njit(parallel=True)
def cal_pkmu_from_ps_2d(ps_2d, k_2d, k_edge, mu_edge, k_logarithmic=False, nthreads=1):
    """
    Calculate power spectrum from pre-computed 2D power spectrum.
    
    Args:
        ps_2d: 2D power spectrum array
        k_2d: 2D k-space coordinates array with shape (k_perp_bin, k_parallel_bin, 2)
        k_edge: k bin edges array
        mu_edge: mu bin edges array
        k_logarithmic: Whether k bins are logarithmic
        nthreads: Number of threads to use
        
    Returns:
        k_mesh: Average k values per bin
        mu_mesh: Average mu values per bin
        Pkmu: Power spectrum values
        count: Mode counts
    """
    set_num_threads(nthreads)
    
    kbin = k_edge.shape[0] - 1
    mubin = mu_edge.shape[0] - 1
    
    # 根据 mubin 判断是否使用 mu 分箱
    use_mu = (mubin > 1)
    
    if k_logarithmic:
        k_diff = np.log(k_edge[1]) - np.log(k_edge[0])
        k_min_log = np.log(k_edge[0])
    else:
        k_diff = k_edge[1] - k_edge[0]
        k_min_log = 0.0
    
    if use_mu:
        mu_diff = mu_edge[1] - mu_edge[0]
        mu_min = mu_edge[0]
        mu_max = mu_edge[-1]
    else:
        mu_diff = 1.0
        mu_min = mu_edge[0]
        mu_max = mu_edge[-1]
    
    k_min = k_edge[0]
    k_max = k_edge[-1]
    
    # 初始化线程安全的数组
    Pkmu_threads = np.zeros((nthreads, kbin, mubin), dtype=np.complex128)
    count_threads = np.zeros((nthreads, kbin, mubin), dtype=np.uint32)
    k_mesh_threads = np.zeros((nthreads, kbin, mubin), dtype=np.float64)
    mu_mesh_threads = np.zeros((nthreads, kbin, mubin), dtype=np.float64)
    
    k_perp_size = k_2d.shape[0]
    k_parallel_size = k_2d.shape[1]
    
    # 外层循环使用并行
    for i_paral in prange(k_parallel_size):
        if i_paral == 0:
            paral_factor = 1
        else:
            paral_factor = 2
            
        # 内层循环不需要并行，因为外层已经并行
        for i_perp in range(k_perp_size):
            k_perp = k_2d[i_perp, i_paral, 0]
            k_parallel = k_2d[i_perp, i_paral, 1]
            
            # 跳过无效的 k 值
            if np.isnan(k_perp) or np.isnan(k_parallel):
                continue
                
            # 计算总波数 k 和 mu
            k = np.sqrt(k_perp ** 2 + k_parallel ** 2)
            
            # 跳过无效的 k 值
            if k < k_min or k > k_max:
                continue
                
            # 计算 k bin 索引
            if k_logarithmic:
                k_i = int((np.log(k) - k_min_log) / k_diff)
            else:
                k_i = int((k - k_min) / k_diff)
            
            # 处理边界情况
            if k_i == kbin:
                k_i -= 1
                
            # 计算 mu bin 索引
            if use_mu:
                mu = k_parallel / k
                if mu < mu_min or mu > mu_max:
                    continue
                mu_i = int((mu - mu_min) / mu_diff)
                if mu_i == mubin:
                    mu_i -= 1
            else:
                mu = 0.0
                mu_i = 0
                
            # 使用线程安全的索引计算
            thread_id = get_thread_id()
            if thread_id < nthreads:
                Pkmu_threads[thread_id, k_i, mu_i] += ps_2d[i_perp, i_paral] * paral_factor
                k_mesh_threads[thread_id, k_i, mu_i] += k * paral_factor
                mu_mesh_threads[thread_id, k_i, mu_i] += mu * paral_factor
                count_threads[thread_id, k_i, mu_i] += paral_factor
    
    # 合并线程结果
    Pkmu = np.sum(Pkmu_threads, axis=0)
    k_mesh = np.sum(k_mesh_threads, axis=0)
    mu_mesh = np.sum(mu_mesh_threads, axis=0)
    count = np.sum(count_threads, axis=0)
    
    return k_mesh, mu_mesh, Pkmu, count

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
                    mu_i = int((mu - mu_min) // mu_diff)
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