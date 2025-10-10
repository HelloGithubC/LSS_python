#include <omp.h>
#include <stdlib.h>
#include <cmath>
#include <array>
#include <stdio.h>
#include <complex>

const int ndim = 3;

template <typename T>
void run_cic(const T *pos, T *field,  size_t np, T boxSize[ndim], size_t ngrids[ndim], const T *weights, const T *values, const T shift, int nthreads)
{
    double sub_boxsize[ndim];
    size_t ny = ngrids[1];
    size_t nz = ngrids[2];

    for (int j = 0; j < ndim; j++) 
    {
        sub_boxsize[j] = boxSize[j] / ngrids[j];
    }
    
    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared) 
    {
        size_t pos_i[2][ndim];
        long long pos_i_temp = 0uL;
        double pos_i_float = 0.0;
        double diff_ratio_temp[ndim][2];

        bool is_in_box = true;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < np; i++) 
        { 
            is_in_box = true;

            double weight = (weights == nullptr)? 1.0 : weights[i];
            double value = (values == nullptr)? 1.0 : values[i];

            for (int j = 0; j < ndim; j++) 
            {
                if (pos[i*ndim+j] < 0.0 || pos[i*ndim+j] > boxSize[j])
                {
                    is_in_box = false;
                    break;
                }
                pos_i_float = pos[i*ndim + j] / sub_boxsize[j] + shift;
                pos_i_temp = static_cast<long long>(std::floor(pos_i_float));

                diff_ratio_temp[j][1] = pos_i_float - pos_i_temp;
                diff_ratio_temp[j][0] = 1.0 - diff_ratio_temp[j][1];

                pos_i[0][j] = static_cast<size_t>((pos_i_temp + ngrids[j]) % ngrids[j]);
                pos_i[1][j] = (pos_i[0][j] + 1u) % ngrids[j];
            } 
            if (!is_in_box)
            {
                continue;
            }
            
            double mass_temp = weight * value;
            
            std::array<u_char,2> delta_ixyz = {0u, 1u};
            for (auto& dix: delta_ixyz)
            {
                for (auto& diy: delta_ixyz)
                {
                    for (auto& diz: delta_ixyz)
                    {
                        #pragma omp atomic
                        field[pos_i[diz][2] + pos_i[diy][1] * nz + pos_i[dix][0] * nz * ny] += static_cast<T>(diff_ratio_temp[2][diz] * diff_ratio_temp[1][diy] * diff_ratio_temp[0][dix] * mass_temp);
                    } 
                }
            }
        }
    }
}

template <typename T>
void run_ngp(const T *pos, T *field,  size_t np, T boxSize[ndim], size_t ngrids[ndim], const T *weights, const T *values, const T shift, int nthreads)
{
    double sub_boxsize[ndim];
    size_t ny = ngrids[1];
    size_t nz = ngrids[2];

    for (int j = 0; j < ndim; j++) 
    {
        sub_boxsize[j] = boxSize[j] / ngrids[j];
    }
    
    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared) 
    {
        size_t pos_i[ndim];
        double pos_i_float = 0.0;
        bool is_in_box = true;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < np; i++) 
        { 
            is_in_box = true;
            double weight = (weights == nullptr)? 1.0 : weights[i];
            double value = (values == nullptr)? 1.0 : values[i];

            for (int j = 0; j < ndim; j++) 
            {
                if (pos[i*ndim+j] < 0.0 || pos[i*ndim+j] > boxSize[j])
                {
                    is_in_box = false;
                    break;
                }
                pos_i_float = pos[i*ndim + j] / sub_boxsize[j] + shift;
                pos_i[j] = (static_cast<size_t>(std::floor(pos_i_float + 0.5)) + ngrids[j]) % ngrids[j];
            } 

            if (!is_in_box)
            {
                continue;
            }
            
            double mass_temp = weight * value;

            #pragma omp atomic
            field[pos_i[2] + pos_i[1] * nz + pos_i[0] * nz * ny] += mass_temp;
        }
    }
}

template <typename T>
void run_tsc(const T *pos, T *field,  size_t np, T boxSize[ndim], size_t ngrids[ndim], const T *weights, const T *values, const T shift, int nthreads)
{
    double sub_boxsize[ndim];
    size_t ny = ngrids[1];
    size_t nz = ngrids[2];

    for (int j = 0; j < ndim; j++) 
    {
        sub_boxsize[j] = boxSize[j] / ngrids[j];
    }
    
    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared) 
    {
        size_t pos_i[3][ndim];
        long long pos_i_temp = 0uL;
        double pos_i_float = 0.0;
        double diff_ratio_temp[ndim][3];
        bool is_in_box = true;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < np; i++) 
        { 
            is_in_box = true;

            double weight = (weights == nullptr)? 1.0 : weights[i];
            double value = (values == nullptr)? 1.0 : values[i];

            for (int j = 0; j < ndim; j++) 
            {
                if (pos[i*ndim+j] < 0.0 || pos[i*ndim+j] > boxSize[j])
                {
                    is_in_box = false;
                    break;
                }
                pos_i_float = pos[i*ndim + j] / sub_boxsize[j] + shift;
                pos_i_temp = static_cast<long long>(std::floor(pos_i_float - 0.5));

                for (int k = 0; k < 3; k++)
                {
                    double distance_ratio = std::fabs(pos_i_temp + k - pos_i_float);
                    if (distance_ratio < 0.5)
                    {
                        diff_ratio_temp[j][k] = 0.75 - distance_ratio * distance_ratio;
                    }
                    else if (distance_ratio < 1.5)
                    {
                        diff_ratio_temp[j][k] = 0.5 * (1.5 - distance_ratio) * (1.5 - distance_ratio);
                    }
                    else 
                    {
                        diff_ratio_temp[j][k] = 0.0;
                    }
                }

                pos_i[0][j] = (pos_i_temp + ngrids[j]) % ngrids[j];
                pos_i[1][j] = (pos_i[0][j] + 1uL) % ngrids[j];
                pos_i[2][j] = (pos_i[1][j] + 1uL) % ngrids[j];
            } 

            if (!is_in_box)
            {
                continue;
            }
            
            double mass_temp = weight * value;

            std::array<u_char,3> delta_ixyz = {0u, 1u, 2u};
            for (auto& dix: delta_ixyz)
            {
                for (auto& diy: delta_ixyz)
                {
                    for (auto& diz: delta_ixyz)
                    {
                        #pragma omp atomic
                        field[pos_i[diz][2] + pos_i[diy][1] * nz + pos_i[dix][0] * nz * ny] += static_cast<T>(diff_ratio_temp[2][diz] * diff_ratio_temp[1][diy] * diff_ratio_temp[0][dix] * mass_temp);
                    } 
                }
            }
        }
    }
}

template <typename T>
void run_pcs(const T *pos, T *field,  size_t np, T boxSize[ndim], size_t ngrids[ndim], const T *weights, const T *values, const T shift, int nthreads)
{
    double sub_boxsize[ndim];
    size_t ny = ngrids[1];
    size_t nz = ngrids[2];

    for (int j = 0; j < ndim; j++) 
    {
        sub_boxsize[j] = boxSize[j] / ngrids[j];
    }
    
    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared) 
    {
        size_t pos_i[4][ndim];
        long long pos_i_temp = 0uL;
        double pos_i_float = 0.0;
        double diff_ratio_temp[ndim][4];
        bool is_in_box = true;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < np; i++) 
        { 
            is_in_box = true;

            double weight = (weights == nullptr)? 1.0 : weights[i];
            double value = (values == nullptr)? 1.0 : values[i];

            for (int j = 0; j < ndim; j++) 
            {
                if (pos[i*ndim+j] < 0.0 || pos[i*ndim+j] > boxSize[j])
                {
                    is_in_box = false;
                    break;
                }
                pos_i_float = pos[i*ndim + j] / sub_boxsize[j] + shift;
                pos_i_temp = static_cast<long long>(std::floor(pos_i_float - 1.0));

                for (int k = 0; k < 4; k++)
                {
                    double distance_ratio = std::fabs(pos_i_temp + k - pos_i_float);
                    if (distance_ratio < 1.0)
                    {
                        diff_ratio_temp[j][k] = (4.0 - 6.0 * distance_ratio * distance_ratio + 3.0 * distance_ratio * distance_ratio * distance_ratio) / 6.0;
                    }
                    else if (distance_ratio < 2.0)
                    {
                        diff_ratio_temp[j][k] = (2.0 - distance_ratio) * (2.0 - distance_ratio) * (2.0 - distance_ratio) / 6.0;
                    }
                    else 
                    {
                        diff_ratio_temp[j][k] = 0.0;
                    }
                }

                pos_i[0][j] = (pos_i_temp + ngrids[j]) % ngrids[j];
                pos_i[1][j] = (pos_i[0][j] + 1uL) % ngrids[j];
                pos_i[2][j] = (pos_i[1][j] + 1uL) % ngrids[j];
                pos_i[3][j] = (pos_i[2][j] + 1uL) % ngrids[j];
            } 

            if (!is_in_box)
            {
                continue;
            }
            
            double mass_temp = weight * value;

            std::array<u_char,4> delta_ixyz = {0u, 1u, 2u, 3u};
            for (auto& dix: delta_ixyz)
            {
                for (auto& diy: delta_ixyz)
                {
                    for (auto& diz: delta_ixyz)
                    {
                        #pragma omp atomic
                        field[pos_i[diz][2] + pos_i[diy][1] * nz + pos_i[dix][0] * nz * ny] += static_cast<T>(diff_ratio_temp[2][diz] * diff_ratio_temp[1][diy] * diff_ratio_temp[0][dix] * mass_temp);
                    } 
                }
            }
        }
    }
}

template <typename T>
void DoCompensationCIC(std::complex<T>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads)
{
    size_t nx, ny, nz;
    nx = ngrids[0];
    ny = ngrids[1];
    nz = ngrids[2];

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        double kx, ky, kz;
        double w[3];
        double sinx, siny, sinz;
        #pragma omp for schedule(static)
        for (size_t ix = 0; ix < nx; ix++)
        {
            kx = kx_array[ix];
            sinx = std::sin(kx/2.0);
            w[0] = std::sqrt(1.0 - 2.0/3.0 * sinx * sinx);
            for (size_t iy = 0; iy < ny; iy++)
            {
                ky = ky_array[iy];
                siny = std::sin(ky/2.0);
                w[1] = std::sqrt(1.0 - 2.0/3.0 * siny * siny);
                for (size_t iz = 0; iz < nz; iz++)
                {
                    kz = kz_array[iz];
                    sinz = std::sin(kz/2.0);
                    w[2] = std::sqrt(1.0 - 2.0/3.0 * sinz * sinz);

                    complex_field[iz + iy*nz + ix*nz*ny] /= w[0] * w[1] * w[2];
                } 
            } 
        }
    }
}

template <typename T>
void DoCompensationTSC(std::complex<T>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads)
{
    size_t nx, ny, nz;
    nx = ngrids[0];
    ny = ngrids[1];
    nz = ngrids[2];

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        double kx, ky, kz;
        double w[3];
        double sinx, siny, sinz;
        #pragma omp for schedule(static)
        for (size_t ix = 0; ix < nx; ix++)
        {
            kx = kx_array[ix];
            sinx = std::sin(kx/2.0);
            w[0] = std::sqrt(1.0 - sinx * sinx + 2.0/15.0 * sinx * sinx * sinx * sinx);
            for (size_t iy = 0; iy < ny; iy++)
            {
                ky = ky_array[iy];
                siny = std::sin(ky/2.0);
                w[1] = std::sqrt(1.0 - siny * siny + 2.0/15.0 * siny * siny * siny * siny);
                for (size_t iz = 0; iz < nz; iz++)
                {
                    kz = kz_array[iz];
                    sinz = std::sin(kz/2.0);
                    w[2] = std::sqrt(1.0 - sinz * sinz + 2.0/15.0 * sinz * sinz * sinz * sinz);

                    complex_field[iz + iy*nz + ix*nz*ny] /= w[0] * w[1] * w[2];
                } 
            } 
        }
    }
}

template <typename T>
void DoCompensationPCS(std::complex<T>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads)
{
    size_t nx, ny, nz;
    nx = ngrids[0];
    ny = ngrids[1];
    nz = ngrids[2];

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        double kx, ky, kz;
        double w[3];
        double sinx, siny, sinz;
        #pragma omp for schedule(static)
        for (size_t ix = 0; ix < nx; ix++)
        {
            kx = kx_array[ix];
            sinx = std::sin(kx/2.0);
            w[0] = std::sqrt(1.0 - 4.0 / 3.0 * sinx * sinx + 6.0/15.0 * sinx * sinx * sinx * sinx - 4.0 / 315.0 * sinx * sinx * sinx * sinx * sinx * sinx);
            for (size_t iy = 0; iy < ny; iy++)
            {
                ky = ky_array[iy];
                siny = std::sin(ky/2.0);
                w[1] = std::sqrt(1.0 - 4.0 / 3.0 * siny * siny + 6.0/15.0 * siny * siny * siny * siny - 4.0 / 315.0 * siny * siny * siny * siny * siny * siny);
                for (size_t iz = 0; iz < nz; iz++)
                {
                    kz = kz_array[iz];
                    sinz = std::sin(kz/2.0);
                    w[2] = std::sqrt(1.0 - 4.0 / 3.0 * sinz * sinz + 6.0/15.0 * sinz * sinz * sinz * sinz - 4.0 / 315.0 * sinz * sinz * sinz * sinz * sinz * sinz);

                    complex_field[iz + iy*nz + ix*nz*ny] /= w[0] * w[1] * w[2];
                } 
            } 
        }
    }
}

template <typename T>
void DoCompensationInterlaced(std::complex<T>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int p, int nthreads)
{
    size_t nx, ny, nz;
    nx = ngrids[0];
    ny = ngrids[1];
    nz = ngrids[2];

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        double kx, ky, kz;
        double w[3];
        double sinc_x, sinc_y, sinc_z;
        #pragma omp for schedule(static)
        for (size_t ix = 0; ix < nx; ix++)
        {
            kx = kx_array[ix];
            sinc_x = std::sin(kx/2.0) / (kx/2.0);
            for (int i = 1; i < p; i++)
            {
                sinc_x *= sinc_x;
            }
            w[0] = (kx == 0.0)? 1.0 : sinc_x;
            for (size_t iy = 0; iy < ny; iy++)
            {
                ky = ky_array[iy];
                sinc_y = std::sin(ky/2.0) / (ky/2.0);
                for (int i = 1; i < p; i++)
                {
                    sinc_y *= sinc_y;
                }
                w[1] = (ky == 0.0)? 1.0 : sinc_y;
                for (size_t iz = 0; iz < nz; iz++)
                {
                    kz = kz_array[iz];
                    sinc_z = std::sin(kz/2.0) / (kz/2.0);
                    for (int i = 1; i < p; i++)
                    {
                        sinc_z *= sinc_z;
                    }
                    w[2] = (kz == 0.0)? 1.0 : sinc_z;

                    complex_field[iz + iy*nz + ix*nz*ny] /= w[0] * w[1] * w[2];
                } 
            } 
        }
    }
}

template <typename T>
void DoInterlace(std::complex<T>* c1, std::complex<T>* c2, double* H, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads)
{
    size_t nx, ny, nz;
    nx = ngrids[0];
    ny = ngrids[1];
    nz = ngrids[2];

    omp_set_num_threads(nthreads);
    #pragma omp parallel default(shared)
    {
        T kH = 0.0;
        double kx, ky, kz;
        T factor = static_cast<T>(0.5);
        #pragma omp for schedule(static)
        for (size_t ix = 0; ix < nx; ix++)
        {
            kx = kx_array[ix];
            for (size_t iy = 0; iy < ny; iy++)
            {
                ky = ky_array[iy];
                for (size_t iz = 0; iz < nz; iz++)
                {
                    kz = kz_array[iz];
                    kH = H[0] * kx + H[1] * ky + H[2] * kz;
                    c1[iz + iy*nz + ix*nz*ny] = c1[iz + iy*nz + ix*nz*ny] * factor + c2[iz + iy*nz + ix*nz*ny] * factor * std::exp(std::complex<T>(0.0, 1.0) * factor * kH);
                }
            } 
        }
    }
}