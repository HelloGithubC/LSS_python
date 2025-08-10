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
    for i in range(len(z)):
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