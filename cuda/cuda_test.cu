#include <complex>
using namespace std;
extern "C" __global__
    void deal_ps_core(complex<float>* ps_3d, const complex<float>* ps_3d_kernel, const int nx, const int ny, const int nz, const double boxsize_prod, const double shotnoise)
    {
        int iz = threadIdx.x;
        int iy = blockIdx.y;
        int ix = blockIdx.x;
        if (iz >= nz || iy >= ny || ix >= nx)
        {
            return;
        }

        unsigned int index = iz + iy * nz + ix * nz * ny;
        ps_3d[index] *= ps_3d_kernel[index];
        ps_3d[index] = ps_3d[index] * conj(ps_3d[index]);
        ps_3d[index].real(ps_3d[index].real() * boxsize_prod);
        ps_3d[index].imag(ps_3d[index].imag() * boxsize_prod);
        ps_3d[index].real(ps_3d[index].real() - shotnoise);
    }

extern "C" __global__
    void run_fftpower_core(const float2* ps_3d, const double* kx_array, const double* ky_array, const double* kz_array, const double* k_array, const double* mu_array, const int nx, const int ny, const int nz, const int nk, const int nmu, const double k_diff, const double mu_diff, double2* Pkmu, unsigned int* count)
    {
        bool use_nmu = true;
        if (nmu == 1)
        {
            bool use_nmu = false;
        }

        int iz = threadIdx.x;
        int iy = blockIdx.y;
        int ix = blockIdx.x;

        if (iz >= nz || iy >= ny || ix >= nx)
        {
            return;
        }

        double k = sqrt(kx_array[ix] * kx_array[ix] + ky_array[iy] * ky_array[iy] + kz_array[iz] * kz_array[iz]);
        int k_i, mu_i;
        if (k < k_array[0] || k > k_array[nk])
        {
            return;
        }
        k_i = static_cast<int>((k - k_array[0]) / k_diff);
        if (k_i == nk)
        {
            k_i--;
        }

        double mu;
        if (use_nmu)
        {
            mu = kz_array[iz] / k;
            if (mu < mu_array[0] || mu > mu_array[nmu])
            {
                return;
            }
            mu_i = static_cast<int>((mu - mu_array[0]) / mu_diff);
            if (mu_i == nmu)
            {
                mu_i--;
            }
        }
        else
        {
            mu = -1.0;
            mu_i = 0;
        }

        int mode = 2;
        if (abs(mu - mu_array[0]) < 1e-8)
        {
            mode = 1;
        }

        float2 element = ps_3d[iz + iy * nz + ix * nz * ny];
        atomicAdd(&Pkmu[mu_i + k_i * nmu].x, element.x * mode);
        atomicAdd(&Pkmu[mu_i + k_i * nmu].y, element.y * mode);
        atomicAdd(&count[mu_i + k_i * nmu], 1u * mode);
    }

extern "C" __global__
    void run_cic_core(const float* pos, float* field, const double* boxsize_array, const int* ngrids_array)
    {
        unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;

        const int NDIM = 3;
        double pos_temp[NDIM];
        int pos_i[NDIM];
        int pos_i_next[NDIM];
        double sub_boxsize[NDIM];
        double diff_ratio_temp[NDIM * 2];

        for (int i = 0; i < NDIM; i++)
        {
            pos_temp[i] = pos[index * NDIM + i];
            if (pos_temp[i] < 0.0 || pos_temp[i] > boxsize_array[i])
            {
                return;
            }
            sub_boxsize[i] = boxsize_array[i] / ngrids_array[i];
            pos_i[i] = static_cast<int>(pos_temp[i] / sub_boxsize[i]);
            pos_i_next[i] = (pos_i[i] == ngrids_array[i] - 1)? 0 : pos_i[i] + 1;
            if (pos_i[i] == ngrids_array[i])
            {
                pos_i[i]--;
                pos_temp[i] = 0.0;
            }
            diff_ratio_temp[i*2 + 0] = (pos_temp[i] - pos_i[i] * sub_boxsize[i]) / sub_boxsize[i];
            diff_ratio_temp[i*2 + 1] = 1.0 - diff_ratio_temp[i*2 + 0];
        }
        
        int nx = ngrids_array[0];
        int ny = ngrids_array[1];
        int nz = ngrids_array[2];

        int ix = pos_i[0];
        int iy = pos_i[1];
        int iz = pos_i[2];

        int ix_next = pos_i_next[0];
        int iy_next = pos_i_next[1];
        int iz_next = pos_i_next[2];

        atomicAdd(&field[iz + iy * nz + ix * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 1]);
        atomicAdd(&field[iz_next + iy * nz + ix * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 1]);
        atomicAdd(&field[iz + iy_next * nz + ix * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 1]);
        atomicAdd(&field[iz_next + iy_next * nz + ix * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 1]);

        atomicAdd(&field[iz + iy * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 0]);
        atomicAdd(&field[iz_next + iy * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 0]);
        atomicAdd(&field[iz + iy_next * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 0]);
        atomicAdd(&field[iz_next + iy_next * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 0]);
    }