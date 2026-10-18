from numba import njit, prange, set_num_threads
import numpy as np 
import math

class LoopStopException(Exception):
    pass


@njit
def LinearInterpolation_2d(data, x1y1, x2y2, x3y3):
    x1, y1 = x1y1
    x2, y2 = x2y2
    x3, y3 = x3y3

    frac_x = (x3 - x1) / (x2 - x1)
    frac_y = (y3 - y1) / (y2 - y1)
    f_x3y1 = data[x1, y1] + (data[x2, y1] - data[x1, y1]) * frac_x
    f_x3y2 = data[x1, y2] + (data[x2, y2] - data[x1, y2]) * frac_x
    fun_x3y3 = f_x3y1 + (f_x3y2 - f_x3y1) * frac_y
    return fun_x3y3


@njit
def set2d_array_values(source, source_index, dest, dest_index):
    s_i1, s_i2 = source_index
    d_i1, d_i2 = dest_index
    for i1 in range(s_i1.shape[0]):
        for i2 in range(s_i1.shape[1]):
            dest[d_i1[i1, i2], d_i2[i1, i2]] = source[s_i1[i1, i2], s_i2[i1, i2]]

@njit
def smu_cosmo_convert(s_f, mu_f, DA_f, DA_t, H_f, H_t):
    """s1: angular direction; s2: LOS direction
    f: fiducial; t: task
    """
    s_r = s_f * mu_f
    s_p = s_f * np.sqrt(1 - mu_f**2)
    alpha_p = DA_t / DA_f
    alpha_r = H_f / H_t
    s_t = np.sqrt((alpha_p * s_p) ** 2 + (alpha_r * s_r) ** 2)
    mu_t = alpha_r * s_r / (s_t + 1e-15)
    return s_t, mu_t

def mapping_smudata_to_another_cosmology_simple(
    smutabstd,
    DAstd,
    DAnew,
    Hstd,
    Hnew,
    deltamu=1.0 / 120.0,
    max_mubin=120,
    smin_mapping=3.0,
    smax_mapping=60.0,
):
    assert smin_mapping >= 0 and smin_mapping < smutabstd.shape[0]
    assert smax_mapping > smin_mapping and smax_mapping <= smutabstd.shape[0]
    assert max_mubin == smutabstd.shape[1]

    smutab = np.copy(smutabstd)
    s2_array = np.arange(smin_mapping, smax_mapping, 1, dtype=np.float64)
    s2_index_array = s2_array.astype(np.int64)
    mu2_index_array = np.arange(max_mubin)
    mu2_array = mu2_index_array * deltamu
    convert_tunple = (DAnew, DAstd, Hnew, Hstd)

    simple_core(
        smutab,
        smutabstd,
        s2_index_array,
        s2_array,
        mu2_index_array,
        mu2_array,
        deltamu,
        max_mubin,
        convert_tunple,
    )
    return smutab


def mapping_smudata_dense(smutabstd, cosmos_tuple, smu_bin_tuple,
                          smu_all_bound_tuple, smu_mapping_bound_tuple, *,
                          convert_algorithm="polygon"):
    """Compatibility adapter for the serial Numba dense AP kernels."""
    from ._ap_dense import CONVERT_ALGORITHMS, validate_mapping
    from ._ap_polygon import map_counts

    values = validate_mapping(smutabstd, cosmos_tuple, smu_bin_tuple,
                              smu_all_bound_tuple, smu_mapping_bound_tuple, convert_algorithm)
    return map_counts(*values, CONVERT_ALGORITHMS.index(convert_algorithm))


@njit
def simple_core(
    smutab,
    smutabstd,
    s2_index_array,
    s2_array,
    mu2_index_array,
    mu2_array,
    deltamu,
    max_mubin,
    convert_tunple,
):
    DAnew, DAstd, Hnew, Hstd = convert_tunple
    for s_i in range(len(s2_index_array)):
        s2, s2_index = s2_array[s_i], s2_index_array[s_i]
        for mu_i in range(len(mu2_index_array)):
            mu2, mu2_index = mu2_array[mu_i], mu2_index_array[mu_i]
            s1, mu1 = smu_cosmo_convert(s2, mu2, DAnew, DAstd, Hnew, Hstd)
            mu1 /= deltamu

            s1_int, mu1_int = int(s1), int(mu1)
            if mu1_int == max_mubin - 1:
                mu1_int -= 1
            s1_int_add = s1_int + 1
            mu1_int_add = mu1_int + 1
            smutab[s2_index, mu2_index] = LinearInterpolation_2d(
                smutabstd, (s1_int, mu1_int), (s1_int_add, mu1_int_add), (s1, mu1)
            )


@njit(parallel=True, fastmath=True)
def _trilinear_interp_extrap(
    P, kx, ky, kz,
    kx_new, ky_new, kz_new,
    nthreads=1
):
    nx, ny, nz = len(kx), len(ky), len(kz)
    out = np.empty((len(kx_new), len(ky_new), len(kz_new)), dtype=P.dtype)

    set_num_threads(nthreads)
    for i in prange(len(kx_new)):
        x = kx_new[i]
        ix = np.searchsorted(kx, x) - 1
        ix = max(0, min(ix, nx - 2))
        x0, x1 = kx[ix], kx[ix + 1]
        tx = (x - x0) / (x1 - x0)

        for j in range(len(ky_new)):
            y = ky_new[j]
            iy = np.searchsorted(ky, y) - 1
            iy = max(0, min(iy, ny - 2))
            y0, y1 = ky[iy], ky[iy + 1]
            ty = (y - y0) / (y1 - y0)

            for k in range(len(kz_new)):
                z = kz_new[k]
                iz = np.searchsorted(kz, z) - 1
                iz = max(0, min(iz, nz - 2))
                z0, z1 = kz[iz], kz[iz + 1]
                tz = (z - z0) / (z1 - z0)

                c000 = P[ix,   iy,   iz]
                c100 = P[ix+1, iy,   iz]
                c010 = P[ix,   iy+1, iz]
                c110 = P[ix+1, iy+1, iz]
                c001 = P[ix,   iy,   iz+1]
                c101 = P[ix+1, iy,   iz+1]
                c011 = P[ix,   iy+1, iz+1]
                c111 = P[ix+1, iy+1, iz+1]

                out[i, j, k] = (
                    c000 * (1-tx)*(1-ty)*(1-tz) +
                    c100 * tx*(1-ty)*(1-tz) +
                    c010 * (1-tx)*ty*(1-tz) +
                    c110 * tx*ty*(1-tz) +
                    c001 * (1-tx)*(1-ty)*tz +
                    c101 * tx*(1-ty)*tz +
                    c011 * (1-tx)*ty*tz +
                    c111 * tx*ty*tz
                )
    return out

def rescale_ps(
    P, kx, ky, kz,
    alpha_perp, alpha_para,
    nthreads=1
):
    """
    P.shape = (Nx, Ny, Nz)
    kx, ky from fftfreq
    kz from rfftfreq
    """

    # ---- 排序 kx, ky（fftfreq 非单调）----
    ix = np.argsort(kx)
    iy = np.argsort(ky)

    kx_s = kx[ix]
    ky_s = ky[iy]
    kz_s = kz  # rfftfreq 已单调

    P_s = P[ix][:, iy]

    # ---- 新网格（等距）----
    kx_new =  kx_s
    ky_new =  ky_s
    kz_new =  kz_s

    # ---- 反向映射 ----
    kx_old = kx_new / alpha_perp
    ky_old = ky_new / alpha_perp
    kz_old = kz_new / alpha_para

    # ---- 插值 + 外插 ----
    P_new_sorted = _trilinear_interp_extrap(
        P_s, kx_s, ky_s, kz_s,
        kx_old, ky_old, kz_old,
        nthreads=nthreads
    )

    # ---- 恢复原 fftfreq 顺序 ----
    ix_inv = np.empty_like(ix)
    iy_inv = np.empty_like(iy)
    ix_inv[ix] = np.arange(len(ix))
    iy_inv[iy] = np.arange(len(iy))

    P_new = P_new_sorted[ix_inv][:, iy_inv]

    return P_new
