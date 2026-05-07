#include "fftpower.hpp"

extern "C" {
    void DealPS3D_float(const std::complex<float> *complex_field, const float *kernel, float *ps_3d, size_t* ngrids, float ps_3d_factor, float shotnoise, int nthreads)
    {
        DealPS3D<float>(complex_field, kernel, ps_3d, ngrids, ps_3d_factor, shotnoise, nthreads);
    }

    void DealPS3D_double(const std::complex<double> *complex_field, const double *kernel, double *ps_3d, size_t* ngrids, double ps_3d_factor, double shotnoise, int nthreads)
    {
        DealPS3D<double>(complex_field, kernel, ps_3d, ngrids, ps_3d_factor, shotnoise, nthreads);
    }

    void CalculatePS_double(const double *ps_3d, const size_t *ngrids, const double *kx_array, const double *ky_array, const double *kz_array, const double *k_array, const double *mu_array, uint32_t kbin, uint32_t mubin, double *Pkmu, size_t *count, double *k_mesh, double *mu_mesh, int nthreads, bool k_logarithmic)
    {
        CalculatePS<double>(ps_3d, ngrids, kx_array, ky_array, kz_array, k_array, mu_array, kbin, mubin, Pkmu, count, k_mesh, mu_mesh, nthreads, k_logarithmic);
    }

    void CalculatePS_float(const float *ps_3d, const size_t *ngrids, const double *kx_array, const double *ky_array, const double *kz_array, const double *k_array, const double *mu_array, uint32_t kbin, uint32_t mubin, double *Pkmu, size_t *count, double *k_mesh, double *mu_mesh, int nthreads, bool k_logarithmic)
    {
        CalculatePS<float>(ps_3d, ngrids, kx_array, ky_array, kz_array, k_array, mu_array, kbin, mubin, Pkmu, count, k_mesh, mu_mesh, nthreads, k_logarithmic);
    }

    void GetBin_float(const float value, const float *array, const float array_diff, bool is_logarithmic)
    {
        GetBin<float>(value, array, array_diff, is_logarithmic);
    }

    void GetBin_double(const double value, const double *array, const double array_diff, bool is_logarithmic)
    {
        GetBin<double>(value, array, array_diff, is_logarithmic);
    }

    void GetBin_int(const int value, const int *array, const int array_diff, bool is_logarithmic)
    {
        GetBin<int>(value, array, array_diff, is_logarithmic);
    }

}