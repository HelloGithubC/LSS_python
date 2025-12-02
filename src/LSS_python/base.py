import numpy as np 
from numba import njit

CONST_C = 299792.458
# CONST_h = 0.667
VOLUME_FULL = 41252.96
@njit
def Hz_jit(z, omega_m, w=-1.0):
    return 100 * np.sqrt(omega_m*(1+z)**3 + (1-omega_m)*(1+z)**(3*(1+w)))

@njit
def DA_jit(z, omega_m, w=-1.0):
    return comov_dist_jit(z, omega_m, w) / (1.0 + z)

@njit
def comov_dist_jit(z, omega_m, w=-1.0, z_start=0.0, z_point=1000):
    z = np.linspace(z_start,z,z_point)
    dz = np.diff(z)
    H_inv_array = np.empty(len(z), dtype=np.float64)
    for i in range(z_point):
        H_inv_array[i] = 1.0 / Hz_jit(z[i], omega_m, w)
    return 0.5 * CONST_C * np.sum(dz * (H_inv_array[1:] + H_inv_array[:-1]))
@njit
def comov_dist_array_jit(z_array, omega_m, w=-1.0, z_start=0.0, z_point=1000):
    comov_dist_array = np.empty(len(z_array), dtype=z_array.dtype)
    for j, z in enumerate(z_array):
        z_temp = np.linspace(z_start,z,z_point)
        dz = np.diff(z_temp)
        H_inv_array = np.empty(len(z_temp), dtype=np.float64)
        for i in range(z_point):
            H_inv_array[i] = 1.0 / Hz_jit(z_temp[i], omega_m, w)
        comov_dist_array[j] = 0.5 * CONST_C * np.sum(dz * (H_inv_array[1:] + H_inv_array[:-1]))
    return comov_dist_array

def Hz(z, omega_m, w=-1.0):
    return 100 * np.sqrt(omega_m*(1+z)**3 + (1-omega_m)*(1+z)**(3*(1+w)))

def DA(z, omega_m, w=-1.0):
    return comov_dist(z, omega_m, w) / (1.0 + z)

def comov_dist(z, omega_m, w=-1.0, z_start=0.0, z_point=1000):
    z = np.linspace(z_start,z,z_point)
    dz = np.diff(z)
    H_inv_array = np.empty(len(z), dtype=np.float64)
    for i in range(len(z)):
        H_inv_array[i] = 1.0 / Hz(z[i], omega_m, w)
    return 0.5 * CONST_C * np.sum(dz * (H_inv_array[1:] + H_inv_array[:-1]))

def cal_HI_factor(redshift, omega_m, BoxSize, Nmesh, h=0.677, omega_b=0.049):
    rho_c = (2.7752e11) * h**2
    rho_b = rho_c * omega_b
    if isinstance(BoxSize, float):
        BoxSize = np.array([BoxSize] * 3)
    if isinstance(Nmesh, int):
        Nmesh = np.array([Nmesh] * 3) 
    if len(BoxSize) != 3 and len(Nmesh) != 3:
        raise ValueError("BoxSize and Nmesh must be 3-dimension")
    Vcell = np.prod(BoxSize / Nmesh)
    HI_factor = (
        (1.0 / Vcell)
        / (rho_b * 0.76)
        * 23
        * ((0.15 / (omega_m - omega_b)) * (1 + redshift) / 10.0) ** (0.5)
        * (omega_b * h / 0.02)
    )
    return HI_factor * 1e10

def traz(V_array, x_array, y_array=None):
    """ A simple function to calculate the trapezoidal integration. Only support mesh when y_array is set.
    
    V_array: Must be the same shape with x_array(1D) or np.meshgrid(x_array, y_array, "ij")
    y_array: If set, then calculate the 2D trapezoidal integration

    """
    if y_array is None:
        use_1D = True 
        use_2D = False
    else:
        use_1D = False
        use_2D = True 

    if use_1D:
        delta_x_array = x_array[1:] - x_array[:-1]
        total_V = 0.5 * np.sum(delta_x_array * (V_array[1:] + V_array[:-1]))
        return total_V 
    
    if use_2D:
        X_mesh, Y_mesh = np.meshgrid(x_array, y_array, indexing="ij")
        delta_X_mesh = X_mesh[1:, 1:] - X_mesh[:-1, :-1]
        delta_Y_mesh = Y_mesh[1:, 1:] - Y_mesh[:-1, :-1]
        total_V = 0.25 * np.sum(delta_X_mesh * delta_Y_mesh * (V_array[1:,1:] + V_array[:-1,:-1] + V_array[1:,:-1] + V_array[:-1,1:]))
        return total_V

def get_chi2(snap_ids_pair, P_ap_dict, P_sys_dict=None, sys_array=None, cov_source_dict=None, need_slice = slice(None, -1, None), do_pinv=False, pinv_rcond=1e-5, cov_shift=5, cov_matrix_inv=None):
    snap1, snap2 = snap_ids_pair

    P_ap_size = P_ap_dict[snap1][need_slice].shape[-1]
    P_ap_num = np.prod(P_ap_dict[snap1].shape[:-1])

    if cov_matrix_inv is None:
        if cov_source_dict is None:
            raise ValueError("cov_matrix_inv must be set when cov_source_dict is None")
        
        cov_source_size = cov_source_dict[snap1][need_slice].shape[-1]
        cov_source_num = np.prod(cov_source_dict[snap1].shape[:-1])

        if P_ap_size != cov_source_size:
            raise ValueError(f"The size of P_source({P_ap_size:d}) and cov_source({cov_source_size:d}) must be the same")
        
        cov_source_diff_array = np.zeros(shape=(cov_source_num, cov_source_size))

        cov_source_array_1 = cov_source_dict[snap1].reshape(-1,cov_source_size)
        cov_source_array_2 = cov_source_dict[snap2].reshape(-1,cov_source_size)

        for i_cov in range(cov_source_num):
            i_cov_shift = i_cov + cov_shift
            if i_cov_shift >= cov_source_num:
                i_cov_shift -= cov_source_num
            cov_source_diff_array[i_cov] = (cov_source_array_1[i_cov_shift] - cov_source_array_2[i_cov])[need_slice]
        cov_matrix = np.cov(cov_source_diff_array, rowvar=False)
        cov_matrix_inv = np.linalg.pinv(cov_matrix, rcond=pinv_rcond) if do_pinv else np.linalg.inv(cov_matrix)
    else:
        cov_matrix_size = cov_matrix_inv.shape[0]
        if P_ap_size != cov_matrix_size:
            raise ValueError(f"The size of P_source({P_ap_size:d}) and cov_matrix_inv({cov_matrix_size:d}) must be the same")
        
    chi2_array = np.zeros(shape=(P_ap_num,))
    P_ap_array_1 = P_ap_dict[snap1].reshape(-1,P_ap_size)
    P_ap_array_2 = P_ap_dict[snap2].reshape(-1,P_ap_size)
    if sys_array is None:
        sys_array = P_sys_dict[snap1] - P_sys_dict[snap2]
    
    for i_ap in range(P_ap_num):
        P_temp = ((P_ap_array_1[i_ap] - P_ap_array_2[i_ap] - sys_array).reshape(-1,1))[need_slice]
        chi2_array[i_ap] = cal_chi2_core(P_temp, cov_matrix_inv)

    return chi2_array

def cal_chi2_core(P, cov_matrix_inv):
    return P.T @ cov_matrix_inv @ P
    
    
def get_level(chi2_array, smooth=False, smooth_sigma=0.5, return_number=False, return_chi2_smoothed=False):
    threashold = 0.68
    if smooth:
        from scipy.ndimage import gaussian_filter
        chi2_array_smoothed = gaussian_filter(chi2_array, sigma=smooth_sigma)
    else:
        chi2_array_smoothed = chi2_array

    chi2_array_1d = chi2_array_smoothed.ravel()
    chi2_array_distribution = np.sort(np.exp(-chi2_array_1d * 0.5))[::-1]
    chi2_array_distr_sum = np.sum(chi2_array_distribution)
    chi2_array_distr_ratio = chi2_array_distribution / chi2_array_distr_sum

    chi2_distr_temp = 0.0
    for chi2_distr_ratio in chi2_array_distr_ratio:
        chi2_distr_temp += chi2_distr_ratio
        if chi2_distr_temp >= threashold:
            chi2_temp = np.log(chi2_distr_ratio * chi2_array_distr_sum) * -2.0
            level = chi2_temp
            if return_number:
                number = np.sum(chi2_array_1d <= chi2_temp)
            break
    if return_number or return_chi2_smoothed:
        result_list = [level,]
        if return_number:
            result_list.append(number)
        if return_chi2_smoothed:
            result_list.append(chi2_array_smoothed)
        return result_list
    else:
        return level
    
def get_chain(
    backend_filename, is_CPL=False, remove_exception=False, thin=5, discard_min=500, return_loglikes=False, flatten=True
):
    from emcee.backends import HDFBackend
    dim = 3 if is_CPL else 2
    exception_factor = 2
    backend = HDFBackend(backend_filename, read_only=True)
    autocorr_time = int(np.max(backend.get_autocorr_time(tol=0)))
    if remove_exception:
        chains = backend.get_chain(
            thin=thin, discard=max(discard_min, autocorr_time), flat=False
        )
        except_index = []
        for i in range(chains.shape[1]):
            chain_temp = np.delete(chains, i, axis=1)
            mean_values = np.mean(chain_temp.reshape(-1,2), axis=0)
            std_values = np.std(chain_temp.reshape(-1,2), axis=0)
            mean = np.mean(chains[:, i], axis=0)
            if (mean > (mean_values + exception_factor * std_values)).any() or (
                mean < (mean_values - exception_factor * std_values)
            ).any():
                except_index.append(i)
        chain = np.delete(chains, except_index, axis=1)

    else:
        chain = backend.get_chain(
            thin=thin, discard=max(discard_min, autocorr_time), flat=False
        )
    if flatten:
        chain = chain.reshape(-1, dim)
    if return_loglikes:
        loglikes = backend.get_log_prob(thin=thin, discard=max(discard_min, autocorr_time), flat=False)
        if remove_exception:
            loglikes = np.delete(loglikes, except_index, axis=1)
        if flatten:
            loglikes = loglikes.ravel()
        return chain, loglikes
    else:
        return chain
    

def get_need_index(x, y, X_mesh, Y_mesh):
    """ Get the index of x and y in X_mesh and Y_mesh
        x, y: element
        X_mesh, Y_mesh: ndarray, with the same shape
    """
    index_source = np.argmin(np.abs(X_mesh - x) + np.abs(Y_mesh - y))
    return np.unravel_index(index_source, X_mesh.shape)