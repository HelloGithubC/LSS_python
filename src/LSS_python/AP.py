import numpy as np 
from numba import njit
import math

from .base import Hz, DA, Hz_jit, DA_jit, cal_HI_factor

def tpcf_convert_main(xismu, omega_mf, w_f, omega_mm, w_m, redshift, convert_method="dense", assis_xismu=None):
    sbin = xismu.xis.shape[0]
    mubin = xismu.xis.shape[1]

    if convert_method == "dense":
        if sbin < 300 and mubin < 240:
            raise RuntimeError(f"Dense conversion is not available for {sbin:d} sbins and {mubin:d} mubins")
        if assis_xismu is None:
            raise RuntimeError("Assis xismu is required for dense conversion")
        xismu_new = xismu.cosmo_conv_DenseToSparse(
            omstd = omega_mf, 
            wstd = w_f, 
            omwrong = omega_mm, 
            wwrong = w_m, 
            redshift = redshift, 
        )
        xismu_new.s_array = assis_xismu.S[:,0]
        xismu_new.mu_array = assis_xismu.Mu[0]
        return xismu_new
    elif convert_method == "simple":
        if sbin > 300 and mubin > 240:
            raise RuntimeError(f"Simple conversion is not available for {sbin:d} sbins and {mubin:d} mubins")
        return xismu.cosmo_conv_simple(
            omstd = omega_mf, 
            wstd = w_f, 
            omwrong = omega_mm, 
            wwrong = w_m, 
            redshift = redshift,
        )
    else:
        raise ValueError("convert_method must be 'simple' or 'dense'")

def ps_convert_main(ps_3d, omega_mf, w_f, omega_mm, w_m, redshift, boxsize, device_id=-1, **kargs):
    """
    ps_3d: The 3d PS after removing the shot noise and including kernel
    boxsize: The boxsize of the simulation. float or ndarray is OK.

    kargs:
        Nmesh: Default 1024
        k_min: Default 0.01
        k_max: Default 3.0
        dk: Default 0.01
        Nmu: Default 30
        mode: Default '2d'
        nthreads: Default 1
        add_HI: Default False.
        device_id: If >= 0, use GPU. Default -1.
    """
    from .fftpower import FFTPower
    z = redshift
    Hz_f, Hz_m = Hz(z, omega_mf, w_f), Hz(z, omega_mm, w_m)
    DA_f, DA_m = DA(z, omega_mf, w_f), DA(z, omega_mm, w_m)
    perp_convert_factor = DA_m / DA_f
    parallel_convert_factor = Hz_f / Hz_m
    convert_array = np.array(
        [perp_convert_factor, perp_convert_factor, parallel_convert_factor]
    )
    boxsize_array = boxsize * convert_array

    Nmesh = kargs.get("Nmesh", 1024)
    k_min = kargs.get("k_min", 0.01)
    k_max = kargs.get("k_max", 3.0)
    dk = kargs.get("dk", 0.01)
    Nmu = kargs.get("Nmu", 100)
    mode = kargs.get("mode", "2d")
    nthreads = kargs.get("nthreads", 1)
    add_HI = kargs.get("add_HI", False)

    fftpower_new = FFTPower(Nmesh=Nmesh, BoxSize=boxsize_array, shotnoise=0.0)
    fftpower_new.is_run_ps_3d = True
    _ = fftpower_new.cal_ps_from_3d(
            ps_3d,
            k_min,
            k_max,
            dk,
            Nmu=Nmu,
            mode=mode,
            k_logarithmic=False,
            nthreads=nthreads,
            device_id=device_id,
            c_api=True
    )

    if add_HI:
        HI_factor = cal_HI_factor(redshift, omega_mm, boxsize_array, Nmesh)
        fftpower_new.attrs["HI_factor"] = HI_factor
        if mode == "1d":
            fftpower_new.power["Pk"] *= HI_factor ** 2 * np.prod(convert_array)
        else:
            fftpower_new.power["Pkmu"] *= HI_factor ** 2 * np.prod(convert_array)

    return fftpower_new

def snap_box_convert_main(position, omega_mf, w_f, omega_mm, w_m, redshift, boxsize_old, los_axis=2, inplace=False):
    """
    position: The position of the particles. ndarray with shape (N, 3)
    boxsize: The boxsize of the simulation. float or ndarray is OK.

    Return:
        The converted position. ndarray with shape (N, 3)
    """
    if not inplace:
        position = np.copy(position)
    Hz_f, Hz_m = Hz(redshift, omega_mf, w_f), Hz(redshift, omega_mm, w_m)
    DA_f, DA_m = DA(redshift, omega_mf, w_f), DA(redshift, omega_mm, w_m)
    perp_convert_factor = DA_m / DA_f
    parallel_convert_factor = Hz_f / Hz_m
    convert_array = np.array(
        [perp_convert_factor, perp_convert_factor, parallel_convert_factor]
    )
    if los_axis != 2:
        convert_array[los_axis] = parallel_convert_factor
        convert_array[2] = perp_convert_factor
    position = position * convert_array
    return position, boxsize_old * convert_array

def get_convert_array(omega_mf, w_f, omega_mm, w_m, redshift, los_axis=2):
    Hz_f, Hz_m = Hz(redshift, omega_mf, w_f), Hz(redshift, omega_mm, w_m)
    DA_f, DA_m = DA(redshift, omega_mf, w_f), DA(redshift, omega_mm, w_m)
    perp_convert_factor = DA_m / DA_f
    parallel_convert_factor = Hz_f / Hz_m
    convert_array = np.array(
        [perp_convert_factor, perp_convert_factor, parallel_convert_factor]
    )
    if los_axis != 2:
        convert_array[los_axis] = parallel_convert_factor
        convert_array[2] = perp_convert_factor
    return convert_array