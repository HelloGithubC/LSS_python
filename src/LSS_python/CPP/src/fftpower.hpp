#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <omp.h>

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
inline size_t GetBin(const T value, const T *array, const T array_diff, bool is_logarithmic)
{
    if (is_logarithmic)
    {
        return static_cast<size_t>((std::log(value) - std::log(array[0])) / array_diff);
    }
    else 
    {
        return static_cast<size_t>((value - array[0]) / array_diff);
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
