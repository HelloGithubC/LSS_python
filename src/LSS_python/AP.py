import numpy as np 
from numba import njit
import math

from .base import Hz, DA, cal_HI_factor
from .tpcf import xismu

def tpcf_convert_main(xismu, omega_mf, w_f, omega_mm, w_m, redshift, convert_method="dense", assis_xismu=None, wa_f=0.0, wa_m=0.0,smin_mapping=3.0, smax_mapping=60.0) -> xismu | None:
    sbin = xismu.xis.shape[0]
    mubin = xismu.xis.shape[1]

    if abs(omega_mm - omega_mf) < 1e-8 and abs(w_m - w_f) < 1e-8 and abs(wa_m - wa_f) < 1e-8:
        return assis_xismu
    if redshift < 1e-5:
        return assis_xismu

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
            wastd = wa_f, 
            wawrong = wa_m,
            smin_mapping=smin_mapping, 
            smax_mapping=smax_mapping,
        )
        xismu_new.s_array = np.nanmean(assis_xismu.S, axis=1)
        xismu_new.mu_array = np.nanmean(assis_xismu.Mu, axis=0)
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
            wastd = wa_f, 
            wawrong = wa_m,
            smin_mapping=smin_mapping, 
            smax_mapping=smax_mapping,
        )
    else:
        raise ValueError("convert_method must be 'simple' or 'dense'")

def ps_convert_main(ps_3d, omega_mf, w_f, omega_mm, w_m, redshift, boxsize, ps_rescale=False, mesh_done_norm=True, device_id=-1, pybind=False, **kargs):
    """
    ps_3d: The 3d PS after removing the shot noise and including kernel
    boxsize: The boxsize of the simulation. float or ndarray is OK.

    kargs:
        use_new_kernel: Default False.
        ps_3d_kernel: 
        Nmesh: Default 1024
        k_min: Default 0.01
        k_max: Default 3.0
        dk: Default 0.01
        Nmu: Default 30
        mode: Default '2d'
        nthreads: Default 1
        add_HI: Default False.
        device_id: If >= 0, use GPU. Default -1.
        shotnoise: Only add to fftpower.attrs but not to deduct. 
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
    boxsize_array = boxsize * np.ones(3)
    boxsize_convert_array = boxsize * convert_array

    Nmesh = kargs.get("Nmesh", 1024)

    Nmesh_array = Nmesh * np.ones(3)
    k_min = kargs.get("k_min", 0.01)
    k_max = kargs.get("k_max", 3.0)
    dk = kargs.get("dk", 0.01)
    Nmu = kargs.get("Nmu", 30)
    mode = kargs.get("mode", "2d")
    nthreads = kargs.get("nthreads", 1)
    add_HI = kargs.get("add_HI", False)

    if ps_rescale:
        from ._AP_core import rescale_ps
        kx = np.fft.fftfreq(n=Nmesh, d=boxsize_array[0]/Nmesh_array[0]) * np.pi * 2 
        ky = np.fft.fftfreq(n=Nmesh, d=boxsize_array[1]/Nmesh_array[1]) * np.pi * 2
        kz = np.fft.rfftfreq(n=Nmesh, d=boxsize_array[2]/Nmesh_array[2]) * np.pi * 2
        ps_3d_need = rescale_ps(ps_3d, kx, ky, kz, perp_convert_factor, parallel_convert_factor, nthreads)
        fftpower_new = FFTPower(Nmesh=Nmesh_array, BoxSize=boxsize_array)
    else:
        ps_3d_need = ps_3d
        fftpower_new = FFTPower(Nmesh=Nmesh_array, BoxSize=boxsize_convert_array)
    fftpower_new.is_run_ps_3d = True
    _ = fftpower_new.cal_ps_from_3d(
            ps_3d_need,
            k_min,
            k_max,
            dk,
            Nmu=Nmu,
            mode=mode,
            k_logarithmic=False,
            nthreads=nthreads,
            device_id=device_id,
            c_api=True,
            pybind=pybind
    )

    if mode == "1d":
        key_str = "Pk"
    else:
        key_str = "Pkmu"
    
    if mesh_done_norm:
        fftpower_new.attrs["mesh_done_norm"] = True
        fftpower_new.power[key_str] *= np.prod(convert_array)
    else:
        fftpower_new.attrs["mesh_done_norm"] = False
        fftpower_new.power[key_str] *= 1.0 / np.prod(convert_array)

    if add_HI:
        HI_factor = cal_HI_factor(redshift, omega_mf, V_cell=1.0) # The relation of convertion should not be changed in AP test
        fftpower_new.attrs["HI_factor"] = HI_factor
    else:
        HI_factor = 1.0

    fftpower_new.power[key_str] *= HI_factor**2

    return fftpower_new

def ps_2d_convert_main(ps_2d, k_2d, omega_mf, w_f, omega_mm, w_m, redshift, boxsize, mesh_done_norm=True, w_af=None, w_am=None, **kargs):
    """
    ps_2d: The 2D PS array from cal_ps_2d_from_mesh.
    k_2d: 2D k-space coordinates with shape (k_perp_bin, k_parallel_bin, 2),
          where k_2d[..., 0] is k_perp and k_2d[..., 1] is k_parallel.
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
        c_api: Default False.
        shotnoise: Only add to fftpower.attrs but not to deduct.
    """
    from .fftpower import FFTPower
    z = redshift
    if w_af is None and w_am is None:
        Hz_f, Hz_m = Hz(z, omega_mf, w_f), Hz(z, omega_mm, w_m)
        DA_f, DA_m = DA(z, omega_mf, w_f), DA(z, omega_mm, w_m)
    else:
        from .base import Hz_w0wa
        Hz_f, Hz_m = Hz_w0wa(z, omega_mf, w_f, w_af), Hz_w0wa(z, omega_mm, w_m, w_am)
        DA_f, DA_m = DA(z, omega_mf, w_f, w_af), DA(z, omega_mm, w_m, w_am)
    perp_convert_factor = DA_m / DA_f
    parallel_convert_factor = Hz_f / Hz_m

    k_min = kargs.get("k_min", 0.01)
    k_max = kargs.get("k_max", 3.0)
    dk = kargs.get("dk", 0.01)
    Nmu = kargs.get("Nmu", 30)
    mode = kargs.get("mode", "2d")
    nthreads = kargs.get("nthreads", 1)
    add_HI = kargs.get("add_HI", False)
    c_api = kargs.get("c_api", False)

    # Transform k_2d coordinates for AP effect
    k_2d_converted = np.copy(k_2d)
    k_2d_converted[..., 0] *= 1.0 / perp_convert_factor   # k_perp
    k_2d_converted[..., 1] *= 1.0 / parallel_convert_factor  # k_parallel

    boxsize_array = boxsize * np.ones(3) if isinstance(boxsize, (float, int)) else np.array(boxsize, dtype=float)
    Nmesh = kargs.get("Nmesh", 1024)
    Nmesh_array = Nmesh * np.ones(3) if isinstance(Nmesh, (int, float)) else np.array(Nmesh, dtype=np.int32)

    fftpower_new = FFTPower(Nmesh=Nmesh_array, BoxSize=boxsize_array)
    fftpower_new.is_run_ps_3d = True
    _ = fftpower_new.cal_pkmu_from_ps_2d(
        ps_2d,
        k_2d_converted,
        k_min,
        k_max,
        dk,
        Nmu=Nmu,
        mode=mode,
        k_logarithmic=False,
        nthreads=nthreads,
        c_api=c_api,
    )

    if mode == "1d":
        key_str = "Pk"
    else:
        key_str = "Pkmu"

    convert_prod = perp_convert_factor**2 * parallel_convert_factor
    if mesh_done_norm:
        fftpower_new.attrs["mesh_done_norm"] = True
        fftpower_new.power[key_str] *= convert_prod
    else:
        fftpower_new.attrs["mesh_done_norm"] = False
        fftpower_new.power[key_str] *= 1.0 / convert_prod

    if add_HI:
        HI_factor = cal_HI_factor(redshift, omega_mf, V_cell=1.0)
        fftpower_new.attrs["HI_factor"] = HI_factor
    else:
        HI_factor = 1.0

    fftpower_new.power[key_str] *= HI_factor**2

    return fftpower_new

def snap_box_convert_main(position, omega_mf, w_f, omega_mm, w_m, redshift, boxsize_old, los_axis=2, inplace=False, return_boxsize_new=False):
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

    if return_boxsize_new:
        return position, boxsize_old * convert_array
    else:
        return position

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

def degree_AP(parameters_f, parameters_m, redshift_pair, return_ratio_list=False):
    if len(parameters_f) != len(parameters_m):
        raise ValueError("The number of parameters_f and parameters_m must be the same.")
    if len(parameters_f) == 3:
        use_CPL = True 
    elif len(parameters_f) == 2:
        use_CPL = False
    else:
        raise ValueError("The number of parameters_f and parameters_m must be 2 or 3.")

    if len(redshift_pair) != 2:
        raise ValueError("The number of redshift_pair must be 2.")
    if use_CPL:
        from .base import Hz_w0wa
        omega_mf, w_f, wa_f = parameters_f
        omega_mm, w_m, wa_m = parameters_m
        ratio_list = []
        for redshift in redshift_pair:
            Hz_f, Hz_m = Hz_w0wa(redshift, omega_mf, w_f, wa_f), Hz_w0wa(redshift, omega_mm, w_m, wa_m)
            DA_f, DA_m = DA(redshift, omega_mf, w_f, wa_f), DA(redshift, omega_mm, w_m, wa_m)
            ratio_list.append(DA_m * Hz_m / DA_f / Hz_f)
    else:
        omega_mf, w_f = parameters_f
        omega_mm, w_m = parameters_m
        ratio_list = []
        for redshift in redshift_pair:
            Hz_f, Hz_m = Hz(redshift, omega_mf, w_f), Hz(redshift, omega_mm, w_m)
            DA_f, DA_m = DA(redshift, omega_mf, w_f), DA(redshift, omega_mm, w_m)
            ratio_list.append(DA_m * Hz_m / DA_f / Hz_f)
    if return_ratio_list:
        return ratio_list[0]/ratio_list[1], ratio_list 
    else:
        return ratio_list[0]/ratio_list[1]