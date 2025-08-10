import numpy as np 
from numba import njit
import math

from .tpcf import xismu
from .base import Hz, DA, cal_HI_factor

def tpcf_convert_main(xismu:xismu, omega_mf, w_f, omega_mm, w_m, redshift, convert_method="dense", assis_xismu=None):
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
    
def ps_convert_main(ps_3d, omega_mf, w_f, omega_mm, w_m, redshift, boxsize, **kargs):
    """
    ps_3d: The 3d PS after removing the shot noise and includes kernel. Not including HI_factor
    boxsize: The boxsize of the simulation. float or ndarray is OK.

    kargs:
        Nmesh: Default 1024
        k_min: Default 0.01
        k_max: Default 3.0
        dk: Default 0.01
        Nmu: Default 30
        mode: Default '2d'
        nthreads: Default 1
    """
    from my_fft import FFTPower
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

    fftpower_new = FFTPower(Nmesh=Nmesh, BoxSize=boxsize_array, shotnoise=0.0)
    fftpower_new.is_run_ps_3d = True
    _ = fftpower_new.run(
        ps_3d,
        k_min,
        k_max,
        dk,
        Nmu=Nmu,
        mode=mode,
        linear=True,
        nthreads=nthreads,
        run_ps_3d=False
    )

    HI_factor = cal_HI_factor(redshift, omega_mm, boxsize_array, Nmesh)
    if mode == "1d":
        fftpower_new.power["Pk"] *= HI_factor ** 2 * np.prod(convert_array)
    else:
        fftpower_new.power["Pkmu"] *= HI_factor ** 2 * np.prod(convert_array)

    return fftpower_new