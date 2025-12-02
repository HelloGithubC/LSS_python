import numpy as np 
from numba import njit 

from ._AP_core import smu_cosmo_convert 

@njit 
def convert_dense_core_test(i_smu_sparse, cosmo_tuple, smu_all_bound_tuple, smu_bound_tuple, smu_bin_tuple, dsmu_tuple, scope_tuple=(1,5)):
    """
    Args:
        i_smu_sparse: (i_s, i_mu)
        cosmo_tuple: (Hz_f, Hz_m, DA_f, DA_m)
        smu_all_bound_tuple: (smin_all, smax_all, mumin_all, mumax_all)
        smu_bound_tuple: (smin, smax, mumin, mumax)
        smu_bin_tuple: (sbin_dense, mubin_dense)
        dsmu_tuple: (ds_sparse, ds_dense, dmu_sparse, dmu_dense)
        scope_tuple: (s_scope, mu_scope)
    """
    i_s_sparse, i_mu_sparse = i_smu_sparse
    Hz_f, Hz_m, DA_f, DA_m = cosmo_tuple
    sbin_dense, mubin_dense = smu_bin_tuple
    smin_all, smax_all, mumin_all, mumax_all = smu_all_bound_tuple
    smin, smax, mumin, mumax = smu_bound_tuple
    ds_sparse, ds_dense, dmu_sparse, dmu_dense = dsmu_tuple

    s_sparse_2 = smin_all + i_s_sparse * ds_sparse
    mu_sparse_2 = mumin_all + i_mu_sparse * dmu_sparse 
    s_scope, mu_scope = scope_tuple

    any_points_in_bound = False
    s_points_sparse_2 = np.empty((2, 2))
    mu_points_sparse_2 = np.empty((2, 2))
    for s_add in (0,1):
        s_sparse_2_temp = s_sparse_2 + s_add * ds_sparse
        for mu_add in (0,1):
            mu_sparse_2_temp = mu_sparse_2 + mu_add * dmu_sparse
            s_points_sparse_2[s_add, mu_add] = s_sparse_2_temp
            mu_points_sparse_2[s_add, mu_add] = mu_sparse_2_temp
            if any_points_in_bound:
                continue
            if s_sparse_2_temp >= smin and s_sparse_2_temp <= smax and mu_sparse_2_temp >= mumin and mu_sparse_2_temp <= mumax:
                any_points_in_bound = True
                
    if any_points_in_bound:
        need_convert = True
        s_points_sparse_1, mu_points_sparse_1 = smu_cosmo_convert(
            s_points_sparse_2, mu_points_sparse_2, DA_m, DA_f, Hz_m, Hz_f
        )
        s_sparse_1_min = np.min(s_points_sparse_1)
        mu_sparse_1_min = np.min(mu_points_sparse_1)
        if s_sparse_1_min < smin_all:
            s_sparse_1_min = smin_all
        if mu_sparse_1_min < mumin_all:
            mu_sparse_1_min = mumin_all
        s_sparse_1_max = np.max(s_points_sparse_1)
        mu_sparse_1_max = np.max(mu_points_sparse_1)
        if s_sparse_1_max > smax_all:
            s_sparse_1_max = smax_all
        if mu_sparse_1_max > mumax_all:
            mu_sparse_1_max = mumax_all

        i_s_dense_1_min = np.int32((s_sparse_1_min - smin_all) / ds_dense)
        i_mu_dense_1_min = np.int32((mu_sparse_1_min - mumin_all) / dmu_dense)
        i_s_dense_1_max = np.int32((s_sparse_1_max - smin_all) / ds_dense)
        i_mu_dense_1_max = np.int32((mu_sparse_1_max - mumin_all) / dmu_dense)

        if i_s_dense_1_max < sbin_dense:
            i_s_add_temp = 2 
        else:
            i_s_add_temp = 1
        if i_mu_dense_1_max < mubin_dense:
            i_mu_add_temp = 2 
        else:
            i_mu_add_temp = 1
        smu_points_dense_2 = np.empty((i_s_dense_1_max - i_s_dense_1_min + i_s_add_temp, i_mu_dense_1_max - i_mu_dense_1_min + i_mu_add_temp, 2))
        for is_dense_1 in range(i_s_dense_1_min, i_s_dense_1_max + i_s_add_temp):
            for imu_dense_1 in range(i_mu_dense_1_min, i_mu_dense_1_max + i_mu_add_temp):
                s_dense_1_temp = smin_all + is_dense_1 * ds_dense
                mu_dense_1_temp = mumin_all + imu_dense_1 * dmu_dense
                s_dense_2_temp, mu_dense_2_temp = smu_cosmo_convert(
                    s_dense_1_temp, mu_dense_1_temp, DA_f, DA_m, Hz_f, Hz_m
                )
                if s_dense_2_temp >= smax_all:
                    s_dense_2_temp = smax_all - 1e-8
                if s_dense_2_temp < smin_all:
                    s_dense_2_temp = smin_all + 1e-8
                if mu_dense_2_temp >= mumax_all:
                    mu_dense_2_temp = mumax_all - 1e-8
                if mu_dense_2_temp < mumin_all:
                    mu_dense_2_temp = mumin_all + 1e-8
                is_temp = is_dense_1 - i_s_dense_1_min
                imu_temp = imu_dense_1 - i_mu_dense_1_min
                smu_points_dense_2[is_temp, imu_temp, 0] = s_dense_2_temp
                smu_points_dense_2[is_temp, imu_temp, 1] = mu_dense_2_temp
        
        s_cubes_dense_size = smu_points_dense_2.shape[0] - 1 
        mu_cubes_dense_size = smu_points_dense_2.shape[1] - 1
        scope_dense = np.zeros(shape=(s_cubes_dense_size, mu_cubes_dense_size), dtype=np.int32)
        outpoint_dense = np.zeros(shape=(s_cubes_dense_size, mu_cubes_dense_size), dtype=np.int16)
        
        for i_s_cube_dense in range(s_cubes_dense_size):
            for i_mu_cube_dense in range(mu_cubes_dense_size):
                for s_add in (0,1):
                    for mu_add in (0,1):
                        s_point_temp = smu_points_dense_2[i_s_cube_dense + s_add, i_mu_cube_dense + mu_add, 0]
                        mu_point_temp = smu_points_dense_2[i_s_cube_dense + s_add, i_mu_cube_dense + mu_add, 1]
                        i_s_sparse_2_temp = np.int32((s_point_temp - smin_all) / ds_sparse)
                        i_mu_sparse_2_temp = np.int32((mu_point_temp - mumin_all) / dmu_sparse)
                        scope_s_add_temp = (i_s_sparse_2_temp - i_s_sparse) * s_scope
                        scope_mu_add_temp = (i_mu_sparse_2_temp - i_mu_sparse) * mu_scope
                        scope_dense[i_s_cube_dense, i_mu_cube_dense] += scope_s_add_temp + scope_mu_add_temp
                        if scope_s_add_temp != 0 or scope_mu_add_temp != 0:
                            outpoint_dense[i_s_cube_dense, i_mu_cube_dense] += 1
                    
    else:
        need_convert = False
        s_sparse_2_min = np.min(s_points_sparse_2)
        mu_sparse_2_min = np.min(mu_points_sparse_2)
        if s_sparse_2_min < smin_all:
            s_sparse_2_min = smin_all
        if mu_sparse_2_min < mumin_all:
            mu_sparse_2_min = mumin_all
        s_sparse_2_max = np.max(s_points_sparse_2)
        mu_sparse_2_max = np.max(mu_points_sparse_2)
        if s_sparse_2_max > smax_all:
            s_sparse_2_max = smax_all
        if mu_sparse_2_max > mumax_all:
            mu_sparse_2_max = mumax_all

        is_dense_2_min = np.int32((s_sparse_2_min - smin_all) / ds_dense)
        imu_dense_2_min = np.int32((mu_sparse_2_min - mumin_all) / dmu_dense)
        is_dense_2_max = np.int32((s_sparse_2_max - smin_all) / ds_dense)
        imu_dense_2_max = np.int32((mu_sparse_2_max - mumin_all) / dmu_dense)

        smu_points_dense_2 = np.empty(shape=(is_dense_2_max - is_dense_2_min + 1, imu_dense_2_max - imu_dense_2_min + 1, 2))
        for is_dense_2 in range(is_dense_2_min, is_dense_2_max + 1):
            for imu_dense_2 in range(imu_dense_2_min, imu_dense_2_max + 1):
                s_dense_2_temp = smin_all + is_dense_2 * ds_dense
                mu_dense_2_temp = mumin_all + imu_dense_2 * dmu_dense 
                is_temp = is_dense_2 - is_dense_2_min
                imu_temp = imu_dense_2 - imu_dense_2_min
                smu_points_dense_2[is_temp, imu_temp, 0] = s_dense_2_temp
                smu_points_dense_2[is_temp, imu_temp, 0] = mu_dense_2_temp

        s_cubes_dense_size = smu_points_dense_2.shape[0] - 1 
        mu_cubes_dense_size = smu_points_dense_2.shape[1] - 1
        scope_dense = np.zeros(shape=(s_cubes_dense_size, mu_cubes_dense_size), dtype=np.int32)
        outpoint_dense = np.zeros(shape=(s_cubes_dense_size, mu_cubes_dense_size), dtype=np.int16)

    return need_convert, smu_points_dense_2, scope_dense, outpoint_dense