#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <omp.h>

template <typename T>
inline size_t GetBin(const T value, const T *array, const T array_diff, bool is_logarithmic)
{
    if (is_logarithmic)
    {
        return static_cast<size_t>(std::floor((std::log(value) - std::log(array[0])) / array_diff));
    }
    else 
    {
        return static_cast<size_t>(std::floor((value - array[0]) / array_diff));
    }
}

template <typename T>
void CalPSFromPS2D(const std::complex<T> *ps_2d, const double *k_2d, 
    double *k_out_2d, double *mu_out_2d, std::complex<T> *ps_kmu, size_t *modes,
    const double k_min, const double k_max, const double dk, const uint32_t Nmu,
    const size_t k_perp_bin, const size_t k_parallel_bin, int nthreads)
{
    uint32_t kbin = static_cast<uint32_t>((k_max - k_min) / dk);
    uint32_t mubin = Nmu;
    double mu_diff = 1.0 / Nmu;

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        std::vector<std::vector<std::complex<T>>> ps_kmu_threads(kbin);
        std::vector<std::vector<size_t>> modes_threads(kbin);
        std::vector<std::vector<double>> k_out_2d_threads(kbin);
        std::vector<std::vector<double>> mu_out_2d_threads(kbin);

        for (uint32_t k_i = 0; k_i < kbin; k_i++)
        {
            ps_kmu_threads[k_i].resize(mubin);
            std::fill(ps_kmu_threads[k_i].begin(), ps_kmu_threads[k_i].end(), std::complex<T>(0.0, 0.0));
            modes_threads[k_i].resize(mubin);
            std::fill(modes_threads[k_i].begin(), modes_threads[k_i].end(), 0uL);
            k_out_2d_threads[k_i].resize(mubin);
            std::fill(k_out_2d_threads[k_i].begin(), k_out_2d_threads[k_i].end(), 0.0);
            mu_out_2d_threads[k_i].resize(mubin);
            std::fill(mu_out_2d_threads[k_i].begin(), mu_out_2d_threads[k_i].end(), 0.0);
        }

        #pragma omp for schedule(static)
        for (size_t i_paral = 0; i_paral < k_parallel_bin; i_paral++)
        {
            size_t paral_factor = (i_paral == 0) ? 1uL : 2uL;

            for (size_t i_perp = 0; i_perp < k_perp_bin; i_perp++)
            {
                size_t index_2d = i_paral + k_parallel_bin * i_perp;
                double k_perp = k_2d[0 + i_paral * 2 + i_perp * 2 * k_parallel_bin];
                double k_parallel = k_2d[1 + i_paral * 2 + i_perp * 2 * k_parallel_bin];
                double k = std::sqrt(k_perp * k_perp + k_parallel * k_parallel);
                
                size_t k_index = static_cast<size_t>((k - k_min) / dk);
                if (k_index >= kbin)
                {
                    continue;
                }

                double mu = k_parallel / k;
                size_t mu_index = static_cast<size_t>(mu / mu_diff);
                if (mu_index >= mubin)
                {
                    continue;
                }

                modes_threads[k_index][mu_index] += paral_factor;
                ps_kmu_threads[k_index][mu_index] += ps_2d[index_2d] * static_cast<T>(paral_factor);
                k_out_2d_threads[k_index][mu_index] += k * static_cast<double>(paral_factor);
                mu_out_2d_threads[k_index][mu_index] += mu * static_cast<double>(paral_factor);
            }
        }

        for (uint32_t k_i = 0; k_i < kbin; k_i++)
        {
            for (uint32_t mu_i = 0; mu_i < mubin; mu_i++)
            {
                #pragma omp atomic
                modes[mu_i + k_i * mubin] += modes_threads[k_i][mu_i];
                #pragma omp critical
                ps_kmu[mu_i + k_i * mubin] += ps_kmu_threads[k_i][mu_i];
                #pragma omp atomic
                k_out_2d[mu_i + k_i * mubin] += k_out_2d_threads[k_i][mu_i];
                #pragma omp atomic
                mu_out_2d[mu_i + k_i * mubin] += mu_out_2d_threads[k_i][mu_i];
            }
        }
    }

    for (uint32_t i = 0; i < kbin; i++)
    {
        for (uint32_t j = 0; j < mubin; j++)
        {
            size_t index = j + i * mubin;
            if (modes[index] > 0)
            {
                ps_kmu[index] /= static_cast<T>(modes[index]);
                k_out_2d[index] /= static_cast<double>(modes[index]);
                mu_out_2d[index] /= static_cast<double>(modes[index]);
            }
            else
            {
                ps_kmu[index] = std::complex<T>(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
                k_out_2d[index] = std::numeric_limits<double>::quiet_NaN();
                mu_out_2d[index] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
}


template <typename T>
void CalPS2D(std::complex<T> *complex_field, std::complex<T> *kernel, 
    std::complex<T> *ps_2d, double* k_2d, size_t* modes_2d, 
    double* k_perp_edge, size_t k_perp_bin, 
    const double *kx_array, const double *ky_array, const double *kz_array, size_t *ngrids, 
    T ps_3d_factor, T shotnoise, int nthreads)
{
    size_t nx = ngrids[0];
    size_t ny = ngrids[1];
    size_t nz = ngrids[2];

    double k_perp_diff = k_perp_edge[1] - k_perp_edge[0];

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        size_t index_3d = 0uL;
        size_t index_2d = 0uL;
        std::complex<T> kernel_temp;
        std::complex<T> value_temp;
        double k_perp = 0.0;
        size_t ik_perp = 0uL;
        #pragma omp for schedule(static)
        for (size_t iz = 0; iz < nz; iz++)
        {
            for (size_t i_perp = 0; i_perp < k_perp_bin; i_perp++)
            {
                k_2d[1 + iz * 2 + i_perp * 2 * nz] = kz_array[iz];
            }
            for (size_t ix = 0; ix < nx; ix++)
            {
                for (size_t iy = 0; iy < ny; iy++ )
                {
                    index_3d = iz + nz * iy + nz * ny * ix;
                    kernel_temp = (kernel != nullptr)? kernel[index_3d] : std::complex<T>(1.0, 0.0);
                    value_temp = (complex_field[index_3d] * std::conj(complex_field[index_3d]) * ps_3d_factor - shotnoise) * kernel_temp;
                    
                    k_perp = std::sqrt(kx_array[ix] * kx_array[ix] + ky_array[iy] * ky_array[iy]);
                    if (k_perp < k_perp_edge[0] || k_perp > k_perp_edge[k_perp_bin])
                    {
                        continue;
                    }
                    ik_perp = GetBin(k_perp, k_perp_edge, k_perp_diff, false);
                    if (ik_perp == k_perp_bin)
                    {
                        ik_perp -= 1uL;
                    }

                    index_2d = iz + nz * ik_perp;
                    k_2d[0 + iz * 2 + ik_perp * 2 * nz] += k_perp;
                    ps_2d[index_2d] += value_temp;

                    modes_2d[index_2d] += 1uL;
                } 
            }
        }
    }
}

template <typename T>
void CalPSFromPS2D(const std::complex<T> *ps_2d, const double *k_2d, 
    double *k_out_2d, double *mu_out_2d, std::complex<T> *ps_kmu, size_t *modes,
    const double* k_edge, const double* mu_edge, const size_t kbin, const size_t mubin,
    const size_t k_perp_bin, const size_t k_parallel_bin, int nthreads)
{
    double dk = k_edge[1] - k_edge[0];

    bool use_mu = false;
    double mu_diff = 0.0;
    if (mu_edge != nullptr)
    {
        use_mu = true;
        mu_diff = mu_edge[1] - mu_edge[0];
    }

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        std::vector<std::vector<std::complex<T>>> ps_kmu_threads(kbin);
        std::vector<std::vector<size_t>> modes_threads(kbin);
        std::vector<std::vector<double>> k_out_2d_threads(kbin);
        std::vector<std::vector<double>> mu_out_2d_threads(kbin);

        for (uint32_t k_i = 0; k_i < kbin; k_i++)
        {
            ps_kmu_threads[k_i].resize(mubin);
            std::fill(ps_kmu_threads[k_i].begin(), ps_kmu_threads[k_i].end(), std::complex<T>(0.0, 0.0));
            modes_threads[k_i].resize(mubin);
            std::fill(modes_threads[k_i].begin(), modes_threads[k_i].end(), 0uL);
            k_out_2d_threads[k_i].resize(mubin);
            std::fill(k_out_2d_threads[k_i].begin(), k_out_2d_threads[k_i].end(), 0.0);
            
            if (use_mu)
            {
                mu_out_2d_threads[k_i].resize(mubin);
                std::fill(mu_out_2d_threads[k_i].begin(), mu_out_2d_threads[k_i].end(), 0.0);
            }
        }
    
        size_t mu_index = 0uL;
        #pragma omp for schedule(static)
        for (size_t i_paral = 0; i_paral < k_parallel_bin; i_paral++)
        {
            size_t paral_factor = (i_paral == 0) ? 1uL : 2uL;

            for (size_t i_perp = 0; i_perp < k_perp_bin; i_perp++)
            {
                size_t index_2d = i_paral + k_parallel_bin * i_perp;
                double k_perp = k_2d[0 + i_paral * 2 + i_perp * 2 * k_parallel_bin];
                double k_parallel = k_2d[1 + i_paral * 2 + i_perp * 2 * k_parallel_bin];
                
                // 跳过无效的 k 值
                if (std::isnan(k_perp) || std::isnan(k_parallel))
                {
                    continue;
                }
                
                double k = std::sqrt(k_perp * k_perp + k_parallel * k_parallel);
                
                if (k < k_edge[0] || k > k_edge[kbin])
                {
                    continue;
                }
                size_t k_index = GetBin(k, k_edge, dk, false);
                if (k_index == kbin)
                {
                    k_index -= 1uL;
                }

                if (use_mu)
                {
                    double mu = k_parallel / k;
                    if (mu < mu_edge[0] || mu > mu_edge[mubin])
                    {
                        continue;
                    }
                    mu_index = static_cast<size_t>(mu / mu_diff);
                    if (mu_index == mubin)
                    {
                        mu_index -= 1uL;
                    }

                    mu_out_2d_threads[k_index][mu_index] += mu * static_cast<double>(paral_factor);
                }
                else 
                {
                    mu_index = 0uL;
                }
                modes_threads[k_index][mu_index] += paral_factor;
                ps_kmu_threads[k_index][mu_index] += ps_2d[index_2d] * static_cast<T>(paral_factor);
                k_out_2d_threads[k_index][mu_index] += k * static_cast<double>(paral_factor);
            }
        }

        for (uint32_t k_i = 0; k_i < kbin; k_i++)
        {
            for (uint32_t mu_i = 0; mu_i < mubin; mu_i++)
            {
                #pragma omp atomic
                modes[mu_i + k_i * mubin] += modes_threads[k_i][mu_i];
                #pragma omp critical
                ps_kmu[mu_i + k_i * mubin] += ps_kmu_threads[k_i][mu_i];
                #pragma omp atomic
                k_out_2d[mu_i + k_i * mubin] += k_out_2d_threads[k_i][mu_i];
                if (use_mu)
                {
                    #pragma omp atomic
                    mu_out_2d[mu_i + k_i * mubin] += mu_out_2d_threads[k_i][mu_i];
                }
            }
        }
    }

}

template <typename T>
void DealPS3D(std::complex<T> *complex_field, std::complex<T> *kernel, size_t *ngrids, T ps_3d_factor, T shotnoise, int nthreads)
{
    size_t nx = ngrids[0];
    size_t ny = ngrids[1];
    size_t nz = ngrids[2];

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        size_t index = 0uL;
        std::complex<T> kernel_temp;
        #pragma omp for schedule(static) collapse(3)
        for (size_t ix = 0; ix < nx; ix++)
        {
            for (size_t iy = 0; iy < ny; iy++)
            {
                for (size_t iz = 0; iz < nz; iz++)
                {
                    index = iz + nz * iy + nz * ny * ix;
                    kernel_temp = (kernel != nullptr)? kernel[index] : std::complex<T>(1.0, 0.0);
                    complex_field[index] = (complex_field[index] * std::conj(complex_field[index]) * ps_3d_factor - shotnoise) * kernel_temp;
                } 
            } 
        }
    }
}

template <typename T>
void CalculatePS(const std::complex<T> *ps_3d, const size_t *ngrids, const double *kx_array, const double *ky_array, const double *kz_array, const double *k_array, const double *mu_array, uint32_t kbin, uint32_t mubin, std::complex<double> *Pkmu, size_t *count, double *k_mesh, double *mu_mesh, int nthreads, bool k_logarithmic)
{
    size_t nx = ngrids[0];
    size_t ny = ngrids[1];
    size_t nz = ngrids[2];

    bool use_mu = (mu_array != nullptr);
    if (!use_mu)
    {
        mubin = 1u;
    }

    double k_diff = k_array[1] - k_array[0];
    double mu_diff = (use_mu)? mu_array[1] - mu_array[0]: 1.0;
    
    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        double kx, ky, kz, k, mu;
        mu = 0.0;
        size_t k_i, mu_i;
        T mode=0.0;
        size_t mode_uint = 0uL;
        std::vector<std::vector<std::complex<double>>> Pkmu_threads(kbin);
        std::vector<std::vector<size_t>> count_threads(kbin);
        std::vector<std::vector<double>> k_mesh_threads(kbin);
        std::vector<std::vector<double>> mu_mesh_threads(kbin);
        for (size_t k_i = 0; k_i < kbin; k_i++)
        {
            Pkmu_threads[k_i].resize(mubin);
            std::fill(Pkmu_threads[k_i].begin(), Pkmu_threads[k_i].end(), std::complex<double>(0.0, 0.0));
            count_threads[k_i].resize(mubin);
            std::fill(count_threads[k_i].begin(), count_threads[k_i].end(), 0uL);
            k_mesh_threads[k_i].resize(mubin);
            std::fill(k_mesh_threads[k_i].begin(), k_mesh_threads[k_i].end(), 0.0);
            if (use_mu)
            {
                mu_mesh_threads[k_i].resize(mubin);
                std::fill(mu_mesh_threads[k_i].begin(), mu_mesh_threads[k_i].end(), 0.0);
            }
        } 

        #pragma omp for schedule(static)
        for (size_t ix = 0; ix < nx; ix++)
        {
            kx = std::abs(kx_array[ix]);
            if (kx > k_array[kbin])
            {
                continue;
            }
            for (size_t iy = 0; iy < ny; iy++)
            {
                ky = std::abs(ky_array[iy]);
                if (ky > k_array[kbin])
                {
                    continue;
                }
                for (size_t iz = 0; iz < nz; iz++)
                {
                    mode = (iz == 0uL)? 1.0: 2.0;
                    mode_uint = (iz == 0uL)? 1uL: 2uL;
                    kz = kz_array[iz]; 
                    if (kz > k_array[kbin])
                    {
                        continue;
                    }
                    if (kx <= k_array[0]/2.0 && ky <= k_array[0]/2.0 && kz <= k_array[0]/2.0)
                    {
                        continue;
                    }
                    k = std::sqrt(kx * kx + ky * ky + kz * kz);
                    if (k < k_array[0] || k > k_array[kbin])
                    {
                        continue;
                    }
                    k_i = GetBin(k, k_array, k_diff, k_logarithmic);
                    if (k_i == kbin)
                    {
                        k_i -= 1uL;
                    }
                    if (k_i > kbin)
                    {
                        std:: printf("Error: k_i > kbin\n k = %.2f, k_i = %ld, k_logarithmic = %d \n", k, k_i, k_logarithmic);
                    }

                    if (use_mu)
                    {
                        mu = kz / k;
                        if (mu < mu_array[0] || mu > mu_array[mubin])
                        {
                            continue;
                        }
                        mu_i = GetBin(mu, mu_array, mu_diff, false);
                        if (mu_i == mubin)
                        {
                            mu_i -= 1uL;
                        }
                    }
                    else 
                    {
                        mu_i = 0u;
                    }

                    Pkmu_threads[k_i][mu_i] += ps_3d[iz + iy * nz + ix * nz * ny] * mode;
                    count_threads[k_i][mu_i] += 1uL * mode_uint;
                    k_mesh_threads[k_i][mu_i] += k * mode;
                    if (use_mu)
                        mu_mesh_threads[k_i][mu_i] += mu * mode;
                } 
                
            }
        }

        for (size_t k_i = 0; k_i < kbin; k_i++)
        {
            for (size_t mu_i = 0; mu_i < mubin; mu_i++)
            {
                #pragma omp critical
                Pkmu[mu_i + k_i * mubin] += Pkmu_threads[k_i][mu_i];
                #pragma omp atomic
                count[mu_i + k_i * mubin] += count_threads[k_i][mu_i];
                #pragma omp atomic 
                k_mesh[mu_i + k_i * mubin] += k_mesh_threads[k_i][mu_i];
                if (use_mu)
                {
                    #pragma omp atomic
                    mu_mesh[mu_i + k_i * mubin] += mu_mesh_threads[k_i][mu_i];
                }
            }
        }
    }
}

