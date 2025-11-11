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