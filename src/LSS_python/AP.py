import numpy as np
from numba import njit
import math

from .base import Hz, DA, Hz_w0wa
from .tpcf import xismu
from .HI import cal_HI_factor


def _ap_convert_factors(redshift, omega_mf, w_f, omega_mm, w_m, wa_f=None, wa_m=None):
    if wa_f is None or wa_m is None:
        Hz_f, Hz_m = Hz(redshift, omega_mf, w_f), Hz(redshift, omega_mm, w_m)
        DA_f, DA_m = DA(redshift, omega_mf, w_f), DA(redshift, omega_mm, w_m)
    else:
        Hz_f, Hz_m = Hz_w0wa(redshift, omega_mf, w_f, wa_f), Hz_w0wa(redshift, omega_mm, w_m, wa_m)
        DA_f, DA_m = DA(redshift, omega_mf, w_f, wa_f), DA(redshift, omega_mm, w_m, wa_m)
    return DA_m / DA_f, Hz_f / Hz_m


def _should_convert(redshift, omega_mf, w_f, omega_mm, w_m, wa_f=None, wa_m=None, ap_tol=1e-5):
    """Decide by the Euclidean distance of (perp-1, parallel-1) instead of
    comparing each cosmological parameter independently."""
    perp, parallel = _ap_convert_factors(redshift, omega_mf, w_f, omega_mm, w_m, wa_f, wa_m)
    ap_distance = math.sqrt((perp - 1.0) ** 2 + (parallel - 1.0) ** 2)
    return ap_distance >= ap_tol, perp, parallel


def tpcf_convert_main(xismu: xismu, omega_mf, w_f, omega_mm, w_m, redshift, convert_method="dense", assis_xismu=None, wa_f=0.0, wa_m=0.0, smin_mapping=3.0, smax_mapping=60.0, c_api=True, ap_tol=1e-5) -> xismu | None:
    if xismu.xis is not None:
        sbin = xismu.xis.shape[0]
        mubin = xismu.xis.shape[1]
    else:
        raise ValueError("xismu.xis is None")

    should_convert, _, _ = _should_convert(
        redshift, omega_mf, w_f, omega_mm, w_m, wa_f, wa_m, ap_tol=ap_tol
    )
    if not should_convert:
        print(f"Warning: AP effect is negligible (AP distance < {ap_tol:g}), return xismu directly")
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
            c_api = c_api
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

def ps_convert_main(ps_3d, omega_mf, w_f, omega_mm, w_m, redshift, boxsize, mesh_done_norm=True, device_id=-1, pybind=True, **kargs):
    """
    ps_3d: The 3d PS after removing the shot noise and including kernel
    boxsize: The boxsize of the simulation. float or ndarray is OK.

    kargs:
        use_new_kernel: Default False.
        ps_3d_kernel: 
        Nmesh: Default 1024
        kmin: Default 0.01
        kmax: Default 3.0
        dk: Default 0.01
        Nmu: Default 30
        mode: Default '2d'
        nthreads: Default 1
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
    boxsize_convert_array = boxsize_array * convert_array

    Nmesh = kargs.get("Nmesh", 1024)

    Nmesh_array = Nmesh * np.ones(3)
    kmin = kargs.get("kmin", 0.01)
    kmax = kargs.get("kmax", 3.0)
    dk = kargs.get("dk", 0.01)
    Nmu = kargs.get("Nmu", 30)
    mode = kargs.get("mode", "2d")
    nthreads = kargs.get("nthreads", 1)

    ps_3d_need = ps_3d
    fftpower_new = FFTPower(Nmesh=Nmesh_array, BoxSize=boxsize_convert_array)
    fftpower_new.is_run_ps_3d = True
    _ = fftpower_new.cal_ps_from_3d(
            ps_3d_need,
            kmin,
            kmax,
            dk,
            Nmu=Nmu,
            mode=mode,
            k_logarithmic=False,
            nthreads=nthreads,
            device_id=device_id,
            c_api=True,
            pybind=pybind
    )
    if fftpower_new.power is None:
        raise RuntimeError("FFTPower.cal_ps_from_3d failed")

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

    return fftpower_new

def _ps_2d_convert_fallback(fftpower_2d, omega_mf, w_f, omega_mm, w_m, redshift, mesh_done_norm=True, w_af=None, w_am=None):
    """
    fftpower_2d: The FFTPower2D object
    """
    from .fftpower import FFTPower2D
    z = redshift
    perp_convert_factor, parallel_convert_factor = _ap_convert_factors(z, omega_mf, w_f, omega_mm, w_m, w_af, w_am)

    k_2d = fftpower_2d.k_2d
    boxsize = fftpower_2d.attrs["BoxSize"]
    Nmesh = fftpower_2d.attrs["Nmesh"]

    # Transform k_2d coordinates for AP effect
    k_2d_converted = np.copy(k_2d)
    k_2d_converted[..., 0] *= 1.0 / perp_convert_factor   # k_perp
    k_2d_converted[..., 1] *= 1.0 / parallel_convert_factor  # k_parallel

    boxsize_array = boxsize * np.array([perp_convert_factor, perp_convert_factor, parallel_convert_factor])
    fftpower_new = FFTPower2D(Nmesh=Nmesh, BoxSize=boxsize_array)
    fftpower_new.k_2d = k_2d_converted 
    fftpower_new.ps_2d = np.copy(fftpower_2d.ps_2d)
    fftpower_new.modes_2d = fftpower_2d.modes_2d
    if fftpower_2d.kperp_edges is not None:
        fftpower_new.kperp_edges = np.copy(fftpower_2d.kperp_edges) / perp_convert_factor
    if fftpower_2d.kparallel_edges is not None:
        fftpower_new.kparallel_edges = np.copy(fftpower_2d.kparallel_edges) / parallel_convert_factor
    fftpower_new.removed_shotnoise = fftpower_2d.removed_shotnoise
    fftpower_new.attrs = dict(fftpower_2d.attrs)
    fftpower_new.attrs["BoxSize"] = boxsize_array

    convert_prod = perp_convert_factor**2 * parallel_convert_factor
    if mesh_done_norm:
        fftpower_new.attrs["mesh_done_norm"] = True
        fftpower_new.ps_2d *= convert_prod
    else:
        fftpower_new.attrs["mesh_done_norm"] = False
        fftpower_new.ps_2d *= 1.0 / convert_prod

    return fftpower_new

def ps_2d_convert_main(
    fftpower_2d, omega_mf, w_f, omega_mm, w_m, redshift,
    mesh_done_norm=True, w_af=None, w_am=None,
    kmin=0.1, kmax=2.1, dk=0.01, Nmu=20,
    mode="2d", k_logarithmic=False, nthreads=1, subcell_n=1, c_api=True,
    ap_tol=1e-5,
):
    """Apply the 2D AP transformation and return a binned :class:`FFTPower`.

    ``subcell_n=1`` is the exact compatibility path: it first performs the
    established coordinate transformation and then invokes
    :meth:`FFTPower2D.cal_pkmu_from_ps_2d`.  The same standard rebinning is
    used whenever no AP transformation is needed, irrespective of
    ``subcell_n``. Larger values otherwise use the direct, conservative AP
    rebinning implementation, which samples each original 2D cell into
    ``subcell_n`` by ``subcell_n`` subcells. Both the constant-w and CPL
    ``w0wa`` backgrounds are supported; pass ``w_af`` and ``w_am`` together
    to select the latter.

    Parameters controlling the output P(k, mu) bins are forwarded to the
    selected rebinning implementation.  The return value is always an
    :class:`FFTPower`; callers must not apply a second P(k, mu) conversion.
    """
    if isinstance(subcell_n, (bool, np.bool_)) or not isinstance(
        subcell_n, (int, np.integer)
    ) or subcell_n < 1:
        raise ValueError("subcell_n must be a positive integer.")

    should_convert, _, _ = _should_convert(
        redshift, omega_mf, w_f, omega_mm, w_m, w_af, w_am, ap_tol=ap_tol
    )
    no_ap_transformation = not should_convert

    if subcell_n == 1 or no_ap_transformation:
        converted = _ps_2d_convert_fallback(
            fftpower_2d, omega_mf, w_f, omega_mm, w_m, redshift,
            mesh_done_norm=mesh_done_norm, w_af=w_af, w_am=w_am,
        )
        return converted.cal_pkmu_from_ps_2d(
            kmin=kmin,
            kmax=kmax,
            dk=dk,
            Nmu=Nmu,
            mode=mode,
            k_logarithmic=k_logarithmic,
            nthreads=nthreads,
            c_api=c_api,
        )

    return fftpower_2d.cal_pkmu_from_ps_2d_ap(
        omega_mf,
        w_f,
        omega_mm,
        w_m,
        redshift,
        kmin=kmin,
        kmax=kmax,
        dk=dk,
        Nmu=Nmu,
        mode=mode,
        k_logarithmic=k_logarithmic,
        mesh_done_norm=mesh_done_norm,
        subcell_n=subcell_n,
        nthreads=nthreads,
        c_api=c_api,
        wa_f=w_af,
        wa_m=w_am,
    )

def snap_box_convert_main(position, omega_mf, w_f, omega_mm, w_m, redshift, boxsize_old, wa_f=0.0, wa_m=None, los_axis=2, inplace=False, return_boxsize_new=False, ap_tol=1e-5):
    """
    position: The position of the particles. ndarray with shape (N, 3)
    boxsize: The boxsize of the simulation. float or ndarray is OK.

    Return:
        The converted position. ndarray with shape (N, 3)
    """
    if not inplace:
        position = np.copy(position)

    should_convert, _, _ = _should_convert(
        redshift, omega_mf, w_f, omega_mm, w_m, wa_f, wa_m, ap_tol=ap_tol
    )
    if not should_convert:
        print(f"Warning: AP effect is negligible (AP distance < {ap_tol:g}), return position directly")
        if return_boxsize_new:
            return position, boxsize_old
        else:
            return position
    perp_convert_factor, parallel_convert_factor = _ap_convert_factors(
        redshift, omega_mf, w_f, omega_mm, w_m, wa_f, wa_m
    )
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
