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
    smin_mapping=3,
    smax_mapping=60,
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


def mapping_smudata_to_another_cosmology_DenseToSparse(
    smutabstd,
    DAstd,
    DAnew,
    Hstd,
    Hnew,
    deltas1=150.0 / 750.0,
    deltamu1=1.0 / 600.0,
    deltas2=150.0 / 150.0,
    deltamu2=1.0 / 120.0,
    smax=150.0,
    smin_mapping=3,
    smax_mapping=60,
    compute_rows=[4, 5, 6, 9],
    save_counts=False,
):
    # Check dim of smutabstd
    if len(smutabstd.shape) == 2:
        smutab1 = smutabstd[:, :, np.newaxis]
    elif len(smutabstd.shape) == 3:
        smutab1 = smutabstd[:, :, compute_rows]
    else:
        raise ValueError("smutabstd only support dim=2 or 3")
    elementSize = smutab1.shape[2]

    # Set Alias
    global smu_cosmo_convert
    Convert = smu_cosmo_convert
    MUMIN, MUMAX = 0.0, 1.0

    # Find std bound
    s1bound_min = min(
        Convert(0.0, MUMIN, DAnew, DAstd, Hnew, Hstd)[0],
        Convert(0.0, MUMAX, DAnew, DAstd, Hnew, Hstd)[0],
    )
    s1bound_max = max(
        Convert(smax, MUMIN, DAnew, DAstd, Hnew, Hstd)[0],
        Convert(smax, MUMAX, DAnew, DAstd, Hnew, Hstd)[0],
    )
    nums1 = math.floor(min(s1bound_max, smax) / deltas1)
    maxs1 = math.floor(smax / deltas1)
    nummu1 = math.floor(MUMAX / deltamu1)
    mins1 = int(max(s1bound_min, 0.0) / deltas1)

    # Find converted bound
    s2bound_min = int(smin_mapping / deltas2)
    s2bound_max = int(smax_mapping / deltas2)

    # First define smutab2
    nums2 = math.floor(smax / deltas2)
    nummu2 = math.floor(MUMAX / deltamu2)
    smutab2 = np.zeros((nums2, nummu2, elementSize))

    save_count_array = np.zeros(smutab2.shape[0:2])

    # some preparation
    # order: lower left, lower right, upper left, upper right
    ipositions = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    convert_tuple = (DAstd, DAnew, Hstd, Hnew, deltas1, deltamu1, deltas2, deltamu2)
    bound_index_tuple = (
        mins1,
        maxs1,
        nums1,
        nummu1,
        nums2,
        nummu2,
        s2bound_min,
        s2bound_max,
    )

    try:
        mapping_dense_core(
            smutab1,
            smutab2,
            ipositions,
            convert_tuple,
            bound_index_tuple,
            save_count_array,
            save_counts,
        )
    except LoopStopException as e:
        print(
            f"DAstd: {DAstd:.5f}, DAnew: {DAnew:.5f}, Hstd: {Hstd:.5f}, Hnew: {Hnew:.5f}"
        )
        raise LoopStopException(e)
    if save_counts:
        return smutab2, save_count_array
    else:
        return smutab2
    
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

@njit(parallel=True)
def convert_dense_core(pairs_dense_mesh, dense_mesh, cosmology_tuple, bound_tuple, smu_bin_tuple, dsmu_sparse_tuple, scope_tunple=(1, 10), nthreads=1):
    Hz_f, Hz_m, DA_f, DA_m = cosmology_tuple
    smin, smax, mumin, mumax = bound_tuple
    sbin_dense, sbin_sparse, mubin_dense, mubin_sparse = smu_bin_tuple
    ds_sparse, dmu_sparse = dsmu_sparse_tuple
    s_scope, mu_scope = scope_tunple

    dense_mesh_converted = np.empty(shape=dense_mesh.shape)
    masked_mesh = np.zeros(shape=dense_mesh.shape[:2], dtype=np.bool_)
    
    set_num_threads(nthreads)
    for i_s in prange(sbin_dense):
        for i_mu in range(mubin_dense):
            for i_p in range(4):
                s_f, mu_f = dense_mesh[i_s, i_mu, i_p]
                
                s_t, mu_t = smu_cosmo_convert(s_f, mu_f, DA_f, DA_m, Hz_f, Hz_m)
                if np.isnan(s_t):
                    s_t = s_f 
                if np.isnan(mu_t):
                    mu_t = mu_f
                if s_t < smin or s_t > smax or mu_t < mumin or mu_t > mumax:
                    dense_mesh_converted[i_s, i_mu, :, 0] = np.nan
                    dense_mesh_converted[i_s, i_mu, :, 1] = np.nan
                    masked_mesh[i_s, i_mu] = True
                    break 
                dense_mesh_converted[i_s, i_mu, i_p] = (s_t, mu_t)
    scope_mesh = np.zeros(shape=dense_mesh.shape[:2], dtype=np.int16)
    outpoint_mesh = np.zeros(shape=dense_mesh.shape[:2], dtype=np.int16)
    pairs_sparse_mesh = np.zeros(shape=(sbin_sparse, mubin_sparse, )+ pairs_dense_mesh.shape[2:], dtype=np.float64)

    for i_s in prange(sbin_dense):
        for i_mu in range(mubin_dense):
            if masked_mesh[i_s, i_mu]:
                scope_mesh[i_s, i_mu] = -99
                outpoint_mesh[i_s, i_mu] = -99
                continue
            scope_mesh[i_s, i_mu] = 0
            s_sparse_0_temp, mu_sparse_0_temp = dense_mesh_converted[i_s, i_mu, 0]
            s_bin_sparse_0_temp = np.int32(s_sparse_0_temp / ds_sparse)
            mu_bin_sparse_0_temp = np.int32(mu_sparse_0_temp / dmu_sparse)
            for i_p in range(1,4):
                s_sparse_temp, mu_sparse_temp = dense_mesh_converted[i_s, i_mu, i_p]
                s_bin_sparse_temp = np.int32(s_sparse_temp / ds_sparse)
                mu_bin_sparse_temp = np.int32(mu_sparse_temp / dmu_sparse)
                add_scope_temp = (s_bin_sparse_temp - s_bin_sparse_0_temp) * 1 + (mu_bin_sparse_temp - mu_bin_sparse_0_temp) * 2
                scope_mesh[i_s, i_mu] += add_scope_temp
                if add_scope_temp != 0:
                    outpoint_mesh[i_s, i_mu] += 1
            scope_temp = scope_mesh[i_s, i_mu]
            outpoint_temp = outpoint_mesh[i_s, i_mu]
            mesh_0 = pairs_sparse_mesh[s_bin_sparse_0_temp, mu_bin_sparse_0_temp]
            if (scope_temp == 0 and outpoint_temp == 0) or (scope_temp == -s_scope and outpoint_temp == 1) or (scope_temp == s_scope and outpoint_temp == 1):
                rate_0 = 1.0
                rate_1 = 0.0
                mesh_0 += pairs_dense_mesh[i_s, i_mu] * rate_0
            elif scope_temp == 2*s_scope and outpoint_temp == 2:
                s_sparse_1_temp = dense_mesh_converted[i_s, i_mu, 3, 0]
                s_diff_rate_temp = ((s_bin_sparse_0_temp + 1) * ds_sparse - s_sparse_0_temp) / (s_sparse_1_temp - s_sparse_0_temp)
                mu_diff_rate_temp = 1.0
                rate_0 = s_diff_rate_temp * mu_diff_rate_temp
                rate_1 = 1.0 - rate_0 
                mesh_1 = pairs_sparse_mesh[s_bin_sparse_0_temp + 1, mu_bin_sparse_0_temp]
                mesh_0 += pairs_dense_mesh[i_s, i_mu] * rate_0
                mesh_1 += pairs_dense_mesh[i_s, i_mu] * rate_1
            elif (scope_temp == mu_scope + (mu_scope - s_scope) and outpoint_temp == 2) or (scope_temp == mu_scope*2 and outpoint_temp == 2) or (scope_temp == mu_scope + (mu_scope + s_scope) and outpoint_temp == 2) or (scope_temp == mu_scope + (mu_scope + s_scope) and outpoint_temp == 3):
                mu_sparse_1_temp = dense_mesh_converted[i_s, i_mu, 3, 1]
                mu_diff_rate_temp = ((mu_bin_sparse_0_temp + 1) * dmu_sparse - mu_sparse_0_temp) / (mu_sparse_1_temp - mu_sparse_0_temp)
                s_diff_rate_temp = 1.0
                rate_0 = mu_diff_rate_temp * s_diff_rate_temp
                rate_1 = 1.0 - rate_0
                mesh_1 = pairs_sparse_mesh[s_bin_sparse_0_temp, mu_bin_sparse_0_temp + 1]
                mesh_0 += pairs_dense_mesh[i_s, i_mu] * rate_0
                mesh_1 += pairs_dense_mesh[i_s, i_mu] * rate_1
            elif scope_temp == 3*s_scope and outpoint_temp == 3:
                rate_0 = 0.0 
                rate_1 = 1.0 
                mesh_1 = pairs_sparse_mesh[s_bin_sparse_0_temp + 1, mu_bin_sparse_0_temp]
                mesh_1 += pairs_dense_mesh[i_s, i_mu] * rate_1
            elif scope_temp == 2 * s_scope + 2 * mu_scope:
                s_sparse_1_temp, mu_sparse_1_temp = dense_mesh_converted[i_s, i_mu, 3]
                s_diff_rate_temp = ((s_bin_sparse_0_temp + 1) * ds_sparse - s_sparse_0_temp) / (s_sparse_1_temp - s_sparse_0_temp)
                mu_diff_rate_temp = ((mu_bin_sparse_0_temp + 1) * dmu_sparse - mu_sparse_0_temp) / (mu_sparse_1_temp - mu_sparse_0_temp)
                rate_0 = s_diff_rate_temp * mu_diff_rate_temp
                rate_1 = (1.0 - s_diff_rate_temp) * mu_diff_rate_temp
                rate_2 = s_diff_rate_temp * (1.0 - mu_diff_rate_temp)
                rate_3 = (1.0 - s_diff_rate_temp) * (1.0 - mu_diff_rate_temp)
                mesh_1 = pairs_sparse_mesh[s_bin_sparse_0_temp + 1, mu_bin_sparse_0_temp]
                mesh_2 = pairs_sparse_mesh[s_bin_sparse_0_temp, mu_bin_sparse_0_temp + 1]
                mesh_3 = pairs_sparse_mesh[s_bin_sparse_0_temp + 1, mu_bin_sparse_0_temp + 1]
                mesh_0 += pairs_dense_mesh[i_s, i_mu] * rate_0
                mesh_1 += pairs_dense_mesh[i_s, i_mu] * rate_1
                mesh_2 += pairs_dense_mesh[i_s, i_mu] * rate_2
                mesh_3 += pairs_dense_mesh[i_s, i_mu] * rate_3
            elif scope_temp == s_scope + 2*(s_scope + mu_scope) and outpoint_temp == 3:
                mu_sparse_1_temp = dense_mesh_converted[i_s, i_mu, 3, 1]
                s_diff_rate_temp = 1.0 
                mu_diff_rate_temp = ((mu_bin_sparse_0_temp + 1) * dmu_sparse - mu_sparse_0_temp) / (mu_sparse_1_temp - mu_sparse_0_temp)
                rate_0 = 0.0
                rate_1 = s_diff_rate_temp * mu_diff_rate_temp
                rate_2 = 1.0 - rate_1
                mesh_1 = pairs_sparse_mesh[s_bin_sparse_0_temp, mu_bin_sparse_0_temp + 1]
                mesh_2 = pairs_sparse_mesh[s_bin_sparse_0_temp, mu_bin_sparse_0_temp]
                mesh_0 += pairs_dense_mesh[i_s, i_mu] * rate_0
                mesh_1 += pairs_dense_mesh[i_s, i_mu] * rate_1
                mesh_2 += pairs_dense_mesh[i_s, i_mu] * rate_2
            else:
                ## nopython
                raise ValueError(f"scope and outpoint is not supported")
                ## python
                # raise ValueError(f"scope({scope_temp}) at ({i_s:d}, {i_mu:d}) is not supported")
    return pairs_sparse_mesh, dense_mesh_converted, scope_mesh, outpoint_mesh

@njit(parallel=True)
def convert_dense_core_test(dense_mesh, cosmology_tuple, bound_tuple, smu_bin_tuple, dsmu_sparse_tuple, scope_tunple, nthreads=1):
    Hz_f, Hz_m, DA_f, DA_m = cosmology_tuple
    smin, smax, mumin, mumax = bound_tuple
    sbin_dense, mubin_dense = smu_bin_tuple
    ds_sparse, dmu_sparse = dsmu_sparse_tuple
    s_scope, mu_scope = scope_tunple

    dense_mesh_converted = np.empty(shape=dense_mesh.shape)
    masked_mesh = np.zeros(shape=dense_mesh.shape[:2], dtype=np.bool_)

    set_num_threads(nthreads)
    for i_s in prange(sbin_dense):
        for i_mu in range(mubin_dense):
            for i_p in range(4):
                s_f, mu_f = dense_mesh[i_s, i_mu, i_p]
                
                s_t, mu_t = smu_cosmo_convert(s_f, mu_f, DA_f, DA_m, Hz_f, Hz_m)
                if np.isnan(s_t):
                    s_t = s_f 
                if np.isnan(mu_t):
                    mu_t = mu_f
                if s_t < smin or s_t > smax or mu_t < mumin or mu_t > mumax:
                    dense_mesh_converted[i_s, i_mu, :, 0] = np.nan
                    dense_mesh_converted[i_s, i_mu, :, 1] = np.nan
                    masked_mesh[i_s, i_mu] = True
                    break 
                dense_mesh_converted[i_s, i_mu, i_p] = (s_t, mu_t)
    scope_mesh = np.zeros(shape=dense_mesh.shape[:2], dtype=np.int16)
    outpoint_mesh = np.zeros(shape=dense_mesh.shape[:2], dtype=np.int16)

    for i_s in prange(sbin_dense):
        for i_mu in range(mubin_dense):
            if masked_mesh[i_s, i_mu]:
                scope_mesh[i_s, i_mu] = -99
                outpoint_mesh[i_s, i_mu] = -99
                continue
            scope_mesh[i_s, i_mu] = 0
            s_sparse_0_temp, mu_sparse_0_temp = dense_mesh_converted[i_s, i_mu, 0]
            s_bin_sparse_0_temp = np.int32(s_sparse_0_temp / ds_sparse)
            mu_bin_sparse_0_temp = np.int32(mu_sparse_0_temp / dmu_sparse)
            for i_p in range(1,4):
                s_sparse_temp, mu_sparse_temp = dense_mesh_converted[i_s, i_mu, i_p]
                s_bin_sparse_temp = np.int32(s_sparse_temp / ds_sparse)
                mu_bin_sparse_temp = np.int32(mu_sparse_temp / dmu_sparse)
                add_scope_temp = (s_bin_sparse_temp - s_bin_sparse_0_temp) * s_scope + (mu_bin_sparse_temp - mu_bin_sparse_0_temp) * mu_scope
                scope_mesh[i_s, i_mu] += add_scope_temp
                if add_scope_temp != 0:
                    outpoint_mesh[i_s, i_mu] += 1
            
    return dense_mesh_converted, scope_mesh, outpoint_mesh

@njit
def mapping_dense_core(
    smutab1,
    smutab2,
    ipositions,
    convert_tuple,
    bound_index_tuple,
    save_count_array,
    save_counts,
):
    """
    convert_list: [DAstd, DAnew, Hstd, Hnew, deltas1, deltamu1, deltas2, deltamu2]
    bound_index_list: [mins1, maxs1, nums1, nummu1, s2bound_min, s2bound_max]
    save_cound_list: [save_counts, save_count_array]
    """
    Convert = smu_cosmo_convert
    DAstd, DAnew, Hstd, Hnew, deltas1, deltamu1, deltas2, deltamu2 = convert_tuple
    (
        mins1,
        maxs1,
        nums1,
        nummu1,
        nums2,
        nummu2,
        s2bound_min,
        s2bound_max,
    ) = bound_index_tuple
    elementSize = smutab1.shape[2]
    deltasmu_1 = np.array([deltas1, deltamu1], dtype=np.float64)
    deltasmu_2 = np.array([deltas2, deltamu2], dtype=np.float64)
    for is2 in range(nums2):
        for imu2 in range(nummu2):
            positions2_sparse = (
                ipositions + np.array([is2, imu2], dtype=np.int64)
            ) * deltasmu_2
            positions2_s_sparse, positions2_mu_sparse = (
                positions2_sparse[:, 0],
                positions2_sparse[:, 1],
            )
            if (
                positions2_s_sparse.min() >= s2bound_max
                or positions2_s_sparse.max() <= s2bound_min
            ):
                positions1_s_sparse, positions1_mu_sparse = (
                    positions2_s_sparse,
                    positions2_mu_sparse,
                )
                need_convert = False
            else:
                positions1_s_sparse = np.empty(positions2_s_sparse.shape)
                positions1_mu_sparse = np.empty(positions2_mu_sparse.shape)
                for i in range(len(positions1_s_sparse)):
                    positions1_s_sparse[i], positions1_mu_sparse[i] = Convert(
                        positions2_s_sparse[i],
                        positions2_mu_sparse[i],
                        DAnew,
                        DAstd,
                        Hnew,
                        Hstd,
                    )
                need_convert = True
            sub_s1_bound_index = (positions1_s_sparse / deltas1).astype(np.int64)
            sub_s1bound_min = sub_s1_bound_index.min()
            sub_s1bound_max = sub_s1_bound_index.max()
            sub_mu1_bound_index = (positions1_mu_sparse / deltamu1).astype(np.int64)
            sub_mu1bound_min = sub_mu1_bound_index.min()
            sub_mu1bound_max = sub_mu1_bound_index.max()
            if need_convert:
                sub_s1bound_min, sub_s1bound_max = max(sub_s1bound_min, mins1), min(
                    sub_s1bound_max, nums1
                )
                sub_mu1bound_min, sub_mu1bound_max = max(sub_mu1bound_min, 0), min(
                    sub_mu1bound_max, nummu1
                )
            if sub_s1bound_max < sub_s1bound_min:
                print(is2, imu2, sub_s1bound_max, sub_s1bound_min)
                raise ValueError("Subbound error: s")
            if sub_mu1bound_max < sub_mu1bound_min:
                print((is2, imu2, sub_mu1bound_max, sub_mu1bound_min))
                raise ValueError("Subbound error: mu")

            if sub_s1bound_max == maxs1:
                sub_s1bound_max -= 1
            if sub_mu1bound_max == nummu1:
                sub_mu1bound_max -= 1

            if not need_convert:
                for i in range(elementSize):
                    need_data = smutab1[
                        sub_s1bound_min : sub_s1bound_max + 1,
                        sub_mu1bound_min : sub_mu1bound_max + 1,
                        i,
                    ]
                    smutab2[is2, imu2, i] += np.sum(need_data)
                if save_counts:
                    save_count_array[is2, imu2] += (
                        need_data.shape[0] * need_data.shape[1]
                    )
            else:
                s_up_index = (is2 + 1) - 1e-10
                s_low_index = is2 - 1e-10
                mu_left_index = imu2 - 1e-10
                mu_right_index = (imu2 + 1) - 1e-10
                for is1 in range(sub_s1bound_min, sub_s1bound_max + 1):
                    for imu1 in range(sub_mu1bound_min, sub_mu1bound_max + 1):
                        s_low_bound = (
                            s_up_bound
                        ) = mu_left_bound = mu_right_bound = False
                        sboundFlag = muboundFlag = False
                        rate = 0.5
                        positions1_dense = (
                            ipositions + np.array([is1, imu1], dtype=np.int64)
                        ) * deltasmu_1
                        positions1_s_dense, positions1_mu_dense = (
                            positions1_dense[:, 0],
                            positions1_dense[:, 1],
                        )
                        positions2_s_dense = np.empty(positions1_s_dense.shape)
                        positions2_mu_dense = np.empty(positions1_mu_dense.shape)
                        for i in range(len(positions1_s_sparse)):
                            positions2_s_dense[i], positions2_mu_dense[i] = Convert(
                                positions1_s_dense[i],
                                positions1_mu_dense[i],
                                DAstd,
                                DAnew,
                                Hstd,
                                Hnew,
                            )
                        positions2_s_dense_index_source = positions2_s_dense / deltas2
                        positions2_mu_dense_index_source = (
                            positions2_mu_dense / deltamu2
                        )
                        positions2_s_dense_index = (
                            positions2_s_dense_index_source
                        ).astype(np.int64)
                        positions2_mu_dense_index = (
                            positions2_mu_dense_index_source
                        ).astype(np.int64)
                        points_inward_outward = np.logical_and(
                            positions2_s_dense_index == is2,
                            positions2_mu_dense_index == imu2,
                        )
                        in_sparse_points_count = np.sum(points_inward_outward)
                        if in_sparse_points_count == 4:
                            rate = 1
                        # temp_s_distance and temp_mu_distance are both index in sparse grid
                        else:
                            square_temp = (
                                positions2_s_dense_index_source[2]
                                - positions2_s_dense_index_source[0]
                            ) * (
                                positions2_mu_dense_index_source[1]
                                - positions2_mu_dense_index_source[0]
                            )
                            if in_sparse_points_count == 1:
                                if points_inward_outward[0]:  # lower left
                                    temp_s_distance = (
                                        positions2_s_dense_index_source[0] - s_up_index
                                    )
                                    temp_mu_distance = (
                                        positions2_mu_dense_index_source[0]
                                        - mu_right_index
                                    )
                                    if (
                                        temp_s_distance
                                        * (
                                            positions2_s_dense_index_source[2]
                                            - s_up_index
                                        )
                                        < 0
                                    ):
                                        s_up_bound = True
                                    if (
                                        temp_mu_distance
                                        * (
                                            positions2_mu_dense_index_source[1]
                                            - mu_right_index
                                        )
                                        < 0
                                    ):
                                        mu_right_bound = True
                                if points_inward_outward[1]:  # lower right
                                    temp_s_distance = (
                                        positions2_s_dense_index_source[1] - s_up_index
                                    )
                                    temp_mu_distance = (
                                        positions2_mu_dense_index_source[1]
                                        - mu_left_index
                                    )
                                    if (
                                        temp_s_distance
                                        * (
                                            positions2_s_dense_index_source[3]
                                            - s_up_index
                                        )
                                        < 0
                                    ):
                                        s_up_bound = True
                                    if (
                                        temp_mu_distance
                                        * (
                                            positions2_mu_dense_index_source[0]
                                            - mu_left_index
                                        )
                                        < 0
                                    ):
                                        mu_left_bound = True
                                if points_inward_outward[2]:  # upper left
                                    temp_s_distance = (
                                        positions2_s_dense_index_source[2] - s_low_index
                                    )
                                    temp_mu_distance = (
                                        positions2_mu_dense_index_source[2]
                                        - mu_right_index
                                    )
                                    if (
                                        temp_s_distance
                                        * (
                                            positions2_s_dense_index_source[0]
                                            - s_low_index
                                        )
                                        < 0
                                    ):
                                        s_low_bound = True
                                    if (
                                        temp_mu_distance
                                        * (
                                            positions2_mu_dense_index_source[3]
                                            - mu_right_index
                                        )
                                        < 0
                                    ):
                                        mu_right_bound = True
                                if points_inward_outward[3]:  # upper right
                                    temp_s_distance = (
                                        positions2_s_dense_index_source[3] - s_low_index
                                    )
                                    temp_mu_distance = (
                                        positions2_mu_dense_index_source[3]
                                        - mu_left_index
                                    )
                                    if (
                                        temp_s_distance
                                        * (
                                            positions2_s_dense_index_source[1]
                                            - s_low_index
                                        )
                                        < 0
                                    ):
                                        s_low_bound = True
                                    if (
                                        temp_mu_distance
                                        * (
                                            positions2_mu_dense_index_source[2]
                                            - mu_left_index
                                        )
                                        < 0
                                    ):
                                        mu_left_bound = True
                                if s_low_bound or s_up_bound:
                                    sboundFlag = True
                                if mu_left_bound or mu_right_bound:
                                    muboundFlag = True
                                ################ test ###################
                                if not sboundFlag and not muboundFlag:
                                    # print(is1,imu1, is2, imu2)
                                    # raise LoopStopException("No bound at point = 1")
                                    rate = 0.0
                                ################ test ###################
                                if (
                                    sboundFlag and muboundFlag
                                ):  # Near angle, use rectangle approximation
                                    s_distance = abs(temp_s_distance)
                                    mu_distance = abs(temp_mu_distance)
                                    rate = (s_distance * mu_distance) / square_temp
                                elif sboundFlag:
                                    s_distance = abs(temp_s_distance)
                                    temp_delta_mu = (
                                        positions2_mu_dense_index_source[1]
                                        - positions2_mu_dense_index_source[0]
                                    )
                                    temp_delta_s = (
                                        positions2_s_dense_index_source[2]
                                        - positions2_s_dense_index_source[0]
                                    )
                                    rate = (
                                        0.5
                                        * s_distance
                                        * s_distance
                                        * temp_delta_mu
                                        / temp_delta_s
                                    ) / square_temp
                                else:  # muboudFlag
                                    mu_distance = abs(temp_mu_distance)
                                    temp_delta_mu = (
                                        positions2_mu_dense_index_source[1]
                                        - positions2_mu_dense_index_source[0]
                                    )
                                    temp_delta_s = (
                                        positions2_s_dense_index_source[2]
                                        - positions2_s_dense_index_source[0]
                                    )
                                    rate = (
                                        0.5
                                        * mu_distance
                                        * mu_distance
                                        * temp_delta_s
                                        / temp_delta_mu
                                    ) / square_temp
                            elif in_sparse_points_count == 2:
                                if (
                                    points_inward_outward[0]
                                    and points_inward_outward[1]
                                ):
                                    s_up_bound = True
                                    s_low_bound = False
                                if (
                                    points_inward_outward[0]
                                    and points_inward_outward[2]
                                ):
                                    mu_right_bound = True
                                    mu_left_bound = False
                                if (
                                    points_inward_outward[1]
                                    and points_inward_outward[3]
                                ):
                                    mu_left_bound = True
                                    mu_right_bound = False
                                if (
                                    points_inward_outward[2]
                                    and points_inward_outward[3]
                                ):
                                    s_low_bound = True
                                    s_up_bound = False
                                if s_low_bound or s_up_bound:
                                    sboundFlag = True
                                if mu_left_bound or mu_right_bound:
                                    muboundFlag = True
                                ################ test ###################
                                if not sboundFlag and not muboundFlag:
                                    # print(is1,imu1, is2, imu2)
                                    # raise LoopStopException("No bound at point = 2")
                                    rate = 0.0
                                ################ test ###################
                                if sboundFlag and muboundFlag:
                                    print(is1, imu1, is2, imu2)
                                    raise LoopStopException(
                                        "Two bounds at point = 2. NB!"
                                    )
                                if sboundFlag:
                                    s_distance = (
                                        s_up_index - positions2_s_dense_index_source[0]
                                    ) * s_up_bound + (
                                        positions2_s_dense_index_source[2] - s_low_index
                                    ) * s_low_bound
                                    temp_delta_mu = (
                                        positions2_mu_dense_index_source[1]
                                        - positions2_mu_dense_index_source[0]
                                    )
                                    rate = (s_distance * temp_delta_mu) / square_temp
                                else:  # muboundFlag
                                    mu_distance = (
                                        mu_right_index
                                        - positions2_mu_dense_index_source[0]
                                    ) * mu_right_bound + (
                                        positions2_mu_dense_index_source[1]
                                        - mu_left_index
                                    ) * mu_left_bound
                                    temp_delta_s = (
                                        positions2_s_dense_index_source[2]
                                        - positions2_s_dense_index_source[0]
                                    )
                                    rate = (mu_distance * temp_delta_s) / square_temp
                            elif in_sparse_points_count == 3:
                                if not points_inward_outward[0]:  # lower left
                                    temp_s_distance = (
                                        positions2_s_dense_index_source[0] - s_low_index
                                    )
                                    temp_mu_distance = (
                                        positions2_mu_dense_index_source[0]
                                        - mu_left_index
                                    )
                                    if (
                                        temp_s_distance
                                        * (
                                            positions2_s_dense_index_source[2]
                                            - s_low_index
                                        )
                                        < 0
                                    ):
                                        s_low_bound = True
                                    if (
                                        temp_mu_distance
                                        * (
                                            positions2_mu_dense_index_source[1]
                                            - mu_left_index
                                        )
                                        < 0
                                    ):
                                        mu_left_bound = True
                                if not points_inward_outward[1]:  # lower right
                                    temp_s_distance = (
                                        positions2_s_dense_index_source[1] - s_low_index
                                    )
                                    temp_mu_distance = (
                                        positions2_mu_dense_index_source[1]
                                        - mu_right_index
                                    )
                                    if (
                                        temp_s_distance
                                        * (
                                            positions2_s_dense_index_source[3]
                                            - s_low_index
                                        )
                                        < 0
                                    ):
                                        s_low_bound = True
                                    if (
                                        temp_mu_distance
                                        * (
                                            positions2_mu_dense_index_source[0]
                                            - mu_right_index
                                        )
                                        < 0
                                    ):
                                        mu_right_bound = True
                                if not points_inward_outward[2]:  # upper left
                                    temp_s_distance = (
                                        positions2_s_dense_index_source[2] - s_up_index
                                    )
                                    temp_mu_distance = (
                                        positions2_mu_dense_index_source[2]
                                        - mu_left_index
                                    )
                                    if (
                                        temp_s_distance
                                        * (
                                            positions2_s_dense_index_source[0]
                                            - s_up_index
                                        )
                                        < 0
                                    ):
                                        s_up_bound = True
                                    if (
                                        temp_mu_distance
                                        * (
                                            positions2_mu_dense_index_source[3]
                                            - mu_left_index
                                        )
                                        < 0
                                    ):
                                        mu_left_bound = True
                                if not points_inward_outward[3]:  # upper right
                                    temp_s_distance = (
                                        positions2_s_dense_index_source[3] - s_up_index
                                    )
                                    temp_mu_distance = (
                                        positions2_mu_dense_index_source[3]
                                        - mu_right_index
                                    )
                                    if (
                                        temp_s_distance
                                        * (
                                            positions2_s_dense_index_source[1]
                                            - s_up_index
                                        )
                                        < 0
                                    ):
                                        s_up_bound = True
                                    if (
                                        temp_mu_distance
                                        * (
                                            positions2_mu_dense_index_source[2]
                                            - mu_right_index
                                        )
                                        < 0
                                    ):
                                        mu_right_bound = True
                                if s_low_bound or s_up_bound:
                                    sboundFlag = True
                                if mu_left_bound or mu_right_bound:
                                    muboundFlag = True
                                ################ test ###################
                                if not sboundFlag and not muboundFlag:
                                    # print(is1,imu1, is2, imu2)
                                    # raise LoopStopException("No bound at point = 3")
                                    rate = 1.0
                                ################ test ###################
                                if (
                                    sboundFlag and muboundFlag
                                ):  # Near angle, use rectangle approximation
                                    s_distance = abs(temp_s_distance)
                                    mu_distance = abs(temp_mu_distance)
                                    # Take the complement
                                    rate = (
                                        square_temp - s_distance * mu_distance
                                    ) / square_temp
                                elif sboundFlag:
                                    s_distance = abs(temp_s_distance)
                                    temp_delta_mu = (
                                        positions2_mu_dense_index_source[1]
                                        - positions2_mu_dense_index_source[0]
                                    )
                                    temp_delta_s = (
                                        positions2_s_dense_index_source[2]
                                        - positions2_s_dense_index_source[0]
                                    )
                                    rate = (
                                        square_temp
                                        - 0.5
                                        * s_distance
                                        * s_distance
                                        * temp_delta_mu
                                        / temp_delta_s
                                    ) / square_temp
                                else:  # muboudFlag
                                    mu_distance = abs(temp_mu_distance)
                                    temp_delta_mu = (
                                        positions2_mu_dense_index_source[1]
                                        - positions2_mu_dense_index_source[0]
                                    )
                                    temp_delta_s = (
                                        positions2_s_dense_index_source[2]
                                        - positions2_s_dense_index_source[0]
                                    )
                                    rate = (
                                        square_temp
                                        - 0.5
                                        * mu_distance
                                        * mu_distance
                                        * temp_delta_s
                                        / temp_delta_mu
                                    ) / square_temp
                            else:
                                rate = 0.0

                        if rate > 1:
                            rate = 0.5
                        for i in range(elementSize):
                            smutab2[is2, imu2, i] += smutab1[is1, imu1, i] * rate
                        if save_counts:
                            save_count_array[is2, imu2] += rate