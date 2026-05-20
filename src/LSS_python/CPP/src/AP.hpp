#ifndef AP_HPP
#define AP_HPP

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <omp.h>
#include <iostream>

/**
 * @brief Perform Alcock-Paczynski cosmological conversion on (s, mu) coordinates
 */
inline void smu_cosmo_convert(double s_f, double mu_f, double DA_f, double DA_t, double H_f, double H_t, double& s_t, double& mu_t) {
    double s_r = s_f * mu_f;
    double s_p = s_f * std::sqrt(1.0 - mu_f * mu_f);
    double alpha_p = DA_t / DA_f;
    double alpha_r = H_f / H_t;
    s_t = std::sqrt((alpha_p * s_p) * (alpha_p * s_p) + (alpha_r * s_r) * (alpha_r * s_r));
    mu_t = alpha_r * s_r / (s_t + 1e-15);
}

/**
 * @brief Find dense grid index bounds from sparse grid points
 */
inline void find_dense_in_sparse(
    const double s_points_sparse[2], const double mu_points_sparse[2],
    double smin_all, double smax_all, double mumin_all, double mumax_all,
    double ds_dense, double dmu_dense,
    int& i_s_min, int& i_s_max, int& i_mu_min, int& i_mu_max
) {
    double s_min = std::min(s_points_sparse[0], s_points_sparse[1]);
    double mu_min = std::min(mu_points_sparse[0], mu_points_sparse[1]);
    double s_max = std::max(s_points_sparse[0], s_points_sparse[1]);
    double mu_max = std::max(mu_points_sparse[0], mu_points_sparse[1]);
    
    if (s_min < smin_all) s_min = smin_all;
    if (mu_min < mumin_all) mu_min = mumin_all;
    if (s_max > smax_all) s_max = smax_all;
    if (mu_max > mumax_all) mu_max = mumax_all;
    
    i_s_min = static_cast<int>(std::floor((s_min - smin_all) / ds_dense));
    i_mu_min = static_cast<int>(std::floor((mu_min - mumin_all) / dmu_dense));
    i_s_max = static_cast<int>(std::floor((s_max - smin_all) / ds_dense));
    i_mu_max = static_cast<int>(std::floor((mu_max - mumin_all) / dmu_dense));
}

/**
 * @brief Calculate the overlap rate between a dense cell and sparse cell
 */
inline double get_rate(
    int outpoint, int scope,
    int s_scope, int mu_scope,
    double s_sparse_low, double s_sparse_high, double mu_sparse_low, double mu_sparse_high,
    const double smu_dense_points[4][2]  // [point_index][s_or_mu], order: (0,0), (1,0), (0,1), (1,1)
) {
    if (outpoint == 0) return 1.0;
    if (outpoint == 4) return 0.0;
    
    double dense_s = smu_dense_points[1][0] - smu_dense_points[0][0];
    double dense_mu = smu_dense_points[2][1] - smu_dense_points[0][1];
    double dense_area = dense_s * dense_mu;
    
    if (outpoint == 2) {
        if (scope == s_scope * 2) {
            double in_area = (2.0 * s_sparse_high - smu_dense_points[0][0] - smu_dense_points[2][0]) 
                           * (smu_dense_points[2][1] - smu_dense_points[0][1]) / 2.0;
            return in_area / dense_area;
        } else if (scope == -s_scope * 2) {
            double in_area = (smu_dense_points[1][0] + smu_dense_points[3][0] - 2.0 * s_sparse_low) 
                           * (smu_dense_points[3][1] - smu_dense_points[1][1]) / 2.0;
            return in_area / dense_area;
        } else if (scope == mu_scope * 2 || scope == mu_scope * 2 - s_scope || scope == mu_scope * 2 + s_scope) {
            return (mu_sparse_high - smu_dense_points[0][1]) / dense_mu;
        } else if (scope == -mu_scope * 2 || scope == -mu_scope * 2 + s_scope || scope == -mu_scope * 2 - s_scope) {
            return (smu_dense_points[2][1] - mu_sparse_low) / dense_mu;
        } else {
            return 0.0;
        }
    }
    
    if (outpoint == 3) {
        if (scope == -s_scope * 3 || scope == s_scope * 3) {
            return 0.0;
        }
        if (scope == -(s_scope + mu_scope) * 2 || scope == -(s_scope + mu_scope) * 2 + s_scope || scope == -(s_scope + mu_scope) * 2 - s_scope) {
            double high_temp = smu_dense_points[3][1] - mu_sparse_low;
            double top_line = smu_dense_points[3][0] - s_sparse_low;
            double middle_line = top_line + (smu_dense_points[1][0] - smu_dense_points[3][0]) * high_temp / dense_mu;
            double in_area = (top_line + middle_line) * high_temp / 2.0;
            return in_area / dense_area;
        }
        if (scope == 2 * (mu_scope - s_scope) || scope == 2 * (mu_scope - s_scope) - s_scope || scope == 2 * (mu_scope - s_scope) + s_scope) {
            double high_temp = mu_sparse_high - smu_dense_points[1][1];
            double tail_line = smu_dense_points[1][0] - s_sparse_low;
            double middle_line = tail_line + (smu_dense_points[3][0] - smu_dense_points[1][0]) * high_temp / dense_mu;
            double in_area = (tail_line + middle_line) * high_temp / 2.0;
            return in_area / dense_area;
        }
        if (scope == (s_scope + mu_scope) * 2 || scope == (s_scope + mu_scope) * 2 - s_scope || scope == (s_scope + mu_scope) * 2 + s_scope) {
            double high_temp = mu_sparse_high - smu_dense_points[0][1];
            double tail_line = s_sparse_high - smu_dense_points[0][0];
            double middle_line = tail_line + (smu_dense_points[2][0] - smu_dense_points[0][0]) * high_temp / dense_mu;
            double in_area = (tail_line + middle_line) * high_temp / 2.0;
            return in_area / dense_area;
        }
        if (scope == 2 * (s_scope - mu_scope) || scope == 2 * (s_scope - mu_scope) - s_scope || scope == 2 * (s_scope - mu_scope) + s_scope) {
            double high_temp = smu_dense_points[2][1] - mu_sparse_low;
            double top_line = s_sparse_high - smu_dense_points[2][0];
            double middle_line = top_line + (smu_dense_points[0][0] - smu_dense_points[2][0]) * high_temp / dense_mu;
            double in_area = (top_line + middle_line) * high_temp / 2.0;
            return in_area / dense_area;
        }
        return 0.0;
    }
    
    if (outpoint == 1) {
        if (scope == s_scope || scope == -s_scope) {
            return 1.0;
        }
    }
    
    return 0.0;
}

/**
 * @brief Main function: mapping_smudata_dense implementation
 * Mirrors the Python JIT implementation in _AP_core.py exactly
 */
inline std::vector<double> mapping_smudata_dense(
    const std::vector<double>& smutabstd,  // shape: (sbin_dense, mubin_dense, element_size)
    size_t sbin_dense, size_t mubin_dense,
    size_t sbin_sparse, size_t mubin_sparse,
    double Hz_f, double Hz_m, double DA_f, double DA_m,
    double smin_all, double smax_all, double mumin_all, double mumax_all,
    double smin_mapping, double smax_mapping,
    int nthreads = 1
) {
    // Calculate bin sizes
    double delta_s = smax_all - smin_all;
    double delta_mu = mumax_all - mumin_all;
    
    double ds_dense = delta_s / sbin_dense;
    double ds_sparse = delta_s / sbin_sparse;
    double dmu_dense = delta_mu / mubin_dense;
    double dmu_sparse = delta_mu / mubin_sparse;
    
    // Element size (number of data columns)
    size_t element_size = smutabstd.size() / (sbin_dense * mubin_dense);
    
    // Allocate output array: (sbin_sparse, mubin_sparse, element_size)
    std::vector<double> smutab2(sbin_sparse * mubin_sparse * element_size, 0.0);
    
    // Scope parameters (from smu_scope_tuple=(1,5) in Python)
    const int s_scope = 1;
    const int mu_scope = 5;
    
    omp_set_num_threads(nthreads);
    #pragma omp parallel for schedule(static)
    for (size_t i_s_sparse = 0; i_s_sparse < sbin_sparse; ++i_s_sparse) {
        for (size_t i_mu_sparse = 0; i_mu_sparse < mubin_sparse; ++i_mu_sparse) {
            // Step 1: Get sparse grid point coordinates (lower-left corner of sparse cell)
            double s_sparse_2 = smin_all + i_s_sparse * ds_sparse;
            double mu_sparse_2 = mumin_all + i_mu_sparse * dmu_sparse;
            
            // Step 2: Check if any of the 4 corner points are within mapping bounds
            // The sparse cell has 4 corners: (s_sparse_2, mu_sparse_2) and offsets
            bool any_points_in_bound = false;
            double s_points_sparse_2[2], mu_points_sparse_2[2];  // Only need 2 points: min and max s
            
            for (int s_add = 0; s_add <= 1; ++s_add) {
                for (int mu_add = 0; mu_add <= 1; ++mu_add) {
                    double s_temp = s_sparse_2 + s_add * ds_sparse;
                    double mu_temp = mu_sparse_2 + mu_add * dmu_sparse;
                    
                    // Store unique s values
                    if (s_add == 0) {
                        s_points_sparse_2[0] = s_temp;
                        mu_points_sparse_2[0] = mu_temp;
                    } else if (s_add == 1 && mu_add == 0) {
                        s_points_sparse_2[1] = s_temp;
                        mu_points_sparse_2[1] = mu_temp;
                    }
                    
                    if (s_temp >= smin_mapping && s_temp <= smax_mapping && 
                        mu_temp >= mumin_all && mu_temp <= mumax_all) {
                        any_points_in_bound = true;
                    }
                }
            }
            
            // Step 3: Determine if conversion is needed and get dense bounds
            bool need_convert = any_points_in_bound;
            int i_s_dense_min, i_s_dense_max, i_mu_dense_min, i_mu_dense_max;
            
            if (need_convert) {
                // Convert sparse points to dense coordinates (inverse transform)
                double s_points_dense[2], mu_points_dense[2];
                smu_cosmo_convert(s_points_sparse_2[0], mu_points_sparse_2[0], DA_m, DA_f, Hz_m, Hz_f, 
                                 s_points_dense[0], mu_points_dense[0]);
                smu_cosmo_convert(s_points_sparse_2[1], mu_points_sparse_2[1], DA_m, DA_f, Hz_m, Hz_f, 
                                 s_points_dense[1], mu_points_dense[1]);
                
                find_dense_in_sparse(s_points_dense, mu_points_dense,
                                   smin_all, smax_all, mumin_all, mumax_all,
                                   ds_dense, dmu_dense,
                                   i_s_dense_min, i_s_dense_max, i_mu_dense_min, i_mu_dense_max);
            } else {
                // No conversion: use sparse points directly
                find_dense_in_sparse(s_points_sparse_2, mu_points_sparse_2,
                                   smin_all, smax_all, mumin_all, mumax_all,
                                   ds_dense, dmu_dense,
                                   i_s_dense_min, i_s_dense_max, i_mu_dense_min, i_mu_dense_max);
            }
            
            // Step 4: Calculate i_s_add_temp and i_mu_add_temp
            int i_s_add_temp = (i_s_dense_max < static_cast<int>(sbin_dense)) ? 2 : 1;
            int i_mu_add_temp = (i_mu_dense_max < static_cast<int>(mubin_dense)) ? 2 : 1;
            
            // Step 5: Build smu_points_dense_2 array
            int s_cubes_size = i_s_dense_max - i_s_dense_min + i_s_add_temp;
            int mu_cubes_size = i_mu_dense_max - i_mu_dense_min + i_mu_add_temp;
            
            // Temporary storage for dense points: [i_s][i_mu][s_or_mu]
            std::vector<double> smu_points_dense_2(s_cubes_size * mu_cubes_size * 2, 0.0);
            
            for (int is_dense = i_s_dense_min; is_dense < i_s_dense_max + i_s_add_temp; ++is_dense) {
                for (int imu_dense = i_mu_dense_min; imu_dense < i_mu_dense_max + i_mu_add_temp; ++imu_dense) {
                    double s_dense_1 = smin_all + is_dense * ds_dense;
                    double mu_dense_1 = mumin_all + imu_dense * dmu_dense;
                    
                    double s_dense_2, mu_dense_2;
                    if (need_convert) {
                        // Forward transform: dense_1 -> dense_2
                        smu_cosmo_convert(s_dense_1, mu_dense_1, DA_f, DA_m, Hz_f, Hz_m, s_dense_2, mu_dense_2);
                    } else {
                        s_dense_2 = s_dense_1;
                        mu_dense_2 = mu_dense_1;
                    }
                    
                    // Clamp to bounds
                    if (s_dense_2 >= smax_all) s_dense_2 = smax_all - 1e-8;
                    if (s_dense_2 < smin_all) s_dense_2 = smin_all + 1e-8;
                    if (mu_dense_2 >= mumax_all) mu_dense_2 = mumax_all - 1e-8;
                    if (mu_dense_2 < mumin_all) mu_dense_2 = mumin_all + 1e-8;
                    
                    int is_temp = is_dense - i_s_dense_min;
                    int imu_temp = imu_dense - i_mu_dense_min;
                    smu_points_dense_2[(is_temp * mu_cubes_size + imu_temp) * 2 + 0] = s_dense_2;
                    smu_points_dense_2[(is_temp * mu_cubes_size + imu_temp) * 2 + 1] = mu_dense_2;
                }
            }
            
            // Step 6: Calculate rates_dense
            int s_cubes_dense_size = s_cubes_size - 1;
            int mu_cubes_dense_size = mu_cubes_size - 1;
            
            double s_sparse = ds_sparse * i_s_sparse;
            double mu_sparse = dmu_sparse * i_mu_sparse;
            double s_sparse_low = s_sparse;
            double s_sparse_high = s_sparse + ds_sparse;
            double mu_sparse_low = mu_sparse;
            double mu_sparse_high = mu_sparse + dmu_sparse;
            
            std::vector<double> rates_dense(s_cubes_dense_size * mu_cubes_dense_size, 0.0);
            
            if (need_convert) {
                for (int i_s_cube = 0; i_s_cube < s_cubes_dense_size; ++i_s_cube) {
                    for (int i_mu_cube = 0; i_mu_cube < mu_cubes_dense_size; ++i_mu_cube) {
                        int scope_temp = 0;
                        int outpoint_temp = 0;
                        double smu_dense_points[4][2];  // Order: (0,0), (1,0), (0,1), (1,1)
                        
                        // Get the 4 corner points of the dense cell
                        int point_idx = 0;
                        for (int mu_add = 0; mu_add <= 1; ++mu_add) {
                            for (int s_add = 0; s_add <= 1; ++s_add) {
                                int idx = (i_s_cube + s_add) * mu_cubes_size + (i_mu_cube + mu_add);
                                smu_dense_points[point_idx][0] = smu_points_dense_2[idx * 2 + 0];
                                smu_dense_points[point_idx][1] = smu_points_dense_2[idx * 2 + 1];
                                
                                int i_s_sparse_2 = static_cast<int>(std::floor((smu_dense_points[point_idx][0] - smin_all) / ds_sparse));
                                int i_mu_sparse_2 = static_cast<int>(std::floor((smu_dense_points[point_idx][1] - mumin_all) / dmu_sparse));
                                
                                int scope_s = (i_s_sparse_2 - static_cast<int>(i_s_sparse)) * s_scope;
                                int scope_mu = (i_mu_sparse_2 - static_cast<int>(i_mu_sparse)) * mu_scope;
                                scope_temp += scope_s + scope_mu;
                                
                                if (scope_s != 0 || scope_mu != 0) {
                                    outpoint_temp++;
                                }
                                point_idx++;
                            }
                        }
                        
                        rates_dense[i_s_cube * mu_cubes_dense_size + i_mu_cube] = 
                            get_rate(outpoint_temp, scope_temp, s_scope, mu_scope,
                                   s_sparse_low, s_sparse_high, mu_sparse_low, mu_sparse_high,
                                   smu_dense_points);
                    }
                }
            } else {
                for (int i_s_cube = 0; i_s_cube < s_cubes_dense_size; ++i_s_cube) {
                    for (int i_mu_cube = 0; i_mu_cube < mu_cubes_dense_size; ++i_mu_cube) {
                        int idx = i_s_cube * mu_cubes_size + i_mu_cube;
                        double s_point = smu_points_dense_2[idx * 2 + 0] + 0.5 * ds_dense;
                        // Note: Use dmu_sparse here, not dmu_dense (matches Python line 1002)
                        double mu_point = smu_points_dense_2[idx * 2 + 1] + 0.5 * dmu_sparse;
                        
                        int i_s_sparse_2 = static_cast<int>(std::floor((s_point - smin_all) / ds_sparse));
                        int i_mu_sparse_2 = static_cast<int>(std::floor((mu_point - mumin_all) / dmu_sparse));
                        
                        if (i_s_sparse_2 == static_cast<int>(i_s_sparse) && i_mu_sparse_2 == static_cast<int>(i_mu_sparse)) {
                            rates_dense[i_s_cube * mu_cubes_dense_size + i_mu_cube] = 1.0;
                        } else {
                            rates_dense[i_s_cube * mu_cubes_dense_size + i_mu_cube] = 0.0;
                        }
                    }
                }
            }
            
            // Step 7: Accumulate weighted sum into output
            // smutab2[i_s_sparse, i_mu_sparse, i_element] = sum(rates_array * smutab1[i_s_dense:i_s_dense+size_s, i_mu_dense:i_mu_dense+size_mu, i_element])
            for (int i_s_cube = 0; i_s_cube < s_cubes_dense_size; ++i_s_cube) {
                for (int i_mu_cube = 0; i_mu_cube < mu_cubes_dense_size; ++i_mu_cube) {
                    double rate = rates_dense[i_s_cube * mu_cubes_dense_size + i_mu_cube];
                    if (rate > 1e-10) {
                        int i_s_dense_start = i_s_dense_min + i_s_cube;
                        int i_mu_dense_start = i_mu_dense_min + i_mu_cube;
                        
                        for (size_t i_element = 0; i_element < element_size; ++i_element) {
                            // Input layout: (sbin_dense, mubin_dense, element_size)
                            // Flattened: [i_s * mubin_dense + i_mu] * element_size + i_element
                            size_t in_idx = (i_s_dense_start * mubin_dense + i_mu_dense_start) * element_size + i_element;
                            double val = smutabstd[in_idx];
                            
                            // Output layout: (sbin_sparse, mubin_sparse, element_size)
                            size_t out_idx = (i_s_sparse * mubin_sparse + i_mu_sparse) * element_size + i_element;
                            smutab2[out_idx] += rate * val;
                        }
                    }
                }
            }
        }
    }
    
    return smutab2;
}

#endif // AP_HPP
