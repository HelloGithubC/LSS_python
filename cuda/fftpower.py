import numpy as np 
import cupy as cp 

def get_deal_ps_core_kernel():
    kernel_code = r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void deal_ps_core(complex<float>* ps_3d, complex<float>* ps_3d_kernel, const int nx, const int ny, const int nz, const double boxsize_prod, const double shotnoise, const bool use_kernel)
    {
        int iz = threadIdx.x;
        int iy = blockIdx.y;
        int ix = blockIdx.x;
        if (iz >= nz || iy >= ny || ix >= nx)
        {
            return;
        }

        unsigned int index = iz + iy * nz + ix * nz * ny;

        complex<float> kernel_element;
        if (use_kernel)
        {
            kernel_element = ps_3d_kernel[index];
        }
        else 
        {
            kernel_element = 1.0;
        }
        ps_3d[index] *= kernel_element;
        ps_3d[index] = ps_3d[index] * conj(ps_3d[index]);
        ps_3d[index].real(ps_3d[index].real() * boxsize_prod);
        ps_3d[index].imag(ps_3d[index].imag() * boxsize_prod);
        ps_3d[index].real(ps_3d[index].real() - shotnoise);
        return;
    }
    '''
    return cp.RawKernel(kernel_code, 'deal_ps_core')

def get_run_fftpower_core_kernel():
    kernel_code = r'''
    extern "C" __global__
    void run_fftpower_core(const float2* ps_3d, const double* kx_array, const double* ky_array, const double* kz_array, const double* k_array, const double* mu_array, const int nx, const int ny, const int nz, const int nk, const int nmu, const double k_diff, const double mu_diff, double2* Pkmu, double* k_mesh, double* mu_mesh, unsigned int* count)
    {
        bool use_nmu = true;
        if (nmu == 1)
        {
            use_nmu = false;
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

        unsigned int mode = 2u;
        if (abs(mu - mu_array[0]) < 1e-8)
        {
            mode = 1u;
        }

        float2 element = ps_3d[iz + iy * nz + ix * nz * ny];
        atomicAdd(&Pkmu[mu_i + k_i * nmu].x, element.x * mode);
        atomicAdd(&Pkmu[mu_i + k_i * nmu].y, element.y * mode);
        atomicAdd(&k_mesh[mu_i + k_i * nmu], k * mode);
        atomicAdd(&mu_mesh[mu_i + k_i * nmu], mu * mode);
        atomicAdd(&count[mu_i + k_i * nmu], mode);
    }
    '''
    return cp.RawKernel(kernel_code, 'run_fftpower_core')

deal_ps_core_kernel = get_deal_ps_core_kernel()
run_fftpower_core_kernel = get_run_fftpower_core_kernel()