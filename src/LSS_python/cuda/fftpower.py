import numpy as np 
import cupy as cp 

kernel_code = r'''
#include <cupy/complex.cuh>
extern "C" 
{
    __global__  void deal_ps_core(const complex<float>* complex_field, const float* ps_3d_kernel, float* ps_3d, const int nx, const int ny, const int nz, const double boxsize_prod, const double shotnoise)
    {
        int iz = threadIdx.x;
        int iy = blockIdx.y;
        int ix = blockIdx.x;
        if (iz >= nz || iy >= ny || ix >= nx)
        {
            return;
        }

        bool use_kernel;
        if (ps_3d_kernel == nullptr)
        {
            use_kernel = false;
        }
        else 
        {
            use_kernel = true;
        }

        unsigned int index = iz + iy * nz + ix * nz * ny;

        float kernel_element;
        if (use_kernel)
        {
            kernel_element = ps_3d_kernel[index];
        }
        else 
        {
            kernel_element = 1.0f;
        }
        // Compute power spectrum (real): |delta(k)|^2 * factor - shotnoise
        float power_real = (complex_field[index].real() * complex_field[index].real() + 
                           complex_field[index].imag() * complex_field[index].imag()) * boxsize_prod - shotnoise;
        // Store to real output array
        ps_3d[index] = power_real * kernel_element;
        return;
    }

    __global__  void cal_ps_3d_core(const float* ps_3d, const double* kx_array, const double* ky_array, const double* kz_array, const double* k_array, const double* mu_array, const int nx, const int ny, const int nz, const int nk, const int nmu, const double k_diff, const double mu_diff, double* Pkmu, double* k_mesh, double* mu_mesh, unsigned long long* count)
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

        unsigned int mode = 2uL;
        if (abs(mu - mu_array[0]) < 1e-8)
        {
            mode = 1uL;
        }

        // ps_3d is now a real array
        float power = ps_3d[iz + iy * nz + ix * nz * ny];
        atomicAdd(&Pkmu[mu_i + k_i * nmu], power * mode);
        atomicAdd(&k_mesh[mu_i + k_i * nmu], k * mode);
        atomicAdd(&mu_mesh[mu_i + k_i * nmu], mu * mode);
        atomicAdd(&count[mu_i + k_i * nmu], mode);
    }
}

'''

kernel_module = cp.RawModule(code=kernel_code)

deal_ps_core_kernel = kernel_module.get_function('deal_ps_core')
cal_ps_3d_core_kernel = kernel_module.get_function('cal_ps_3d_core')

def deal_ps_3d_from_cuda(complex_field_gpu, ps_3d_gpu=None, ps_kernel_3d_gpu=None, ps_3d_factor=1.0, shotnoise=0.0):
    """
    Calculate 3D power spectrum from complex field on GPU.
    
    Args:
        complex_field_gpu: Input complex field array on GPU (complex64 or complex128)
        ps_3d_gpu: Optional pre-allocated array on GPU for output. If provided, it will be 
                   COMPLETELY OVERWRITTEN with new results. If None, a new array will be created.
        ps_kernel_3d_gpu: Optional kernel array on GPU (must match complex_field precision)
        ps_3d_factor: Factor for power spectrum normalization
        shotnoise: Shot noise to subtract
    
    Returns:
        ps_3d_gpu: 3D power spectrum array on GPU (same object if ps_3d_gpu was provided)
    """
    if ps_kernel_3d_gpu is None:
        ps_kernel_3d_gpu = 0
    nx, ny, nz = complex_field_gpu.shape
    
    # Determine expected dtype for ps_3d_gpu
    expected_dtype = cp.float32 if complex_field_gpu.dtype == np.complex64 else cp.float64
    
    # Handle ps_3d_gpu parameter
    if ps_3d_gpu is None:
        # Create new array (backward compatible behavior)
        ps_3d_gpu = cp.zeros((nx, ny, nz), dtype=expected_dtype)
    else:
        # Validate provided ps_3d_gpu array
        if not isinstance(ps_3d_gpu, cp.ndarray):
            raise TypeError(f"ps_3d_gpu must be a cupy.ndarray or None, got {type(ps_3d_gpu)}")
        if ps_3d_gpu.dtype != expected_dtype:
            raise TypeError(f"ps_3d_gpu dtype must be {expected_dtype} for complex_field_gpu dtype {complex_field_gpu.dtype}, got {ps_3d_gpu.dtype}")
        if ps_3d_gpu.shape != complex_field_gpu.shape:
            raise ValueError(f"ps_3d_gpu shape {ps_3d_gpu.shape} must match complex_field_gpu shape {complex_field_gpu.shape}")
    
    deal_ps_core_kernel((nx,ny), (nz,), (complex_field_gpu, ps_kernel_3d_gpu, ps_3d_gpu, nx, ny, nz, ps_3d_factor, shotnoise))
    return ps_3d_gpu

def cal_ps_from_cuda(ps_3d_gpu, k_arrays, k_bin_array, mu_bin_array):
    nx, ny, nz = ps_3d_gpu.shape
    k_x_array, k_y_array, k_z_array = k_arrays
    nk = k_bin_array.shape[0] - 1
    nmu = mu_bin_array.shape[0] - 1

    k_diff = (k_bin_array[1] - k_bin_array[0]).item()
    mu_diff = (mu_bin_array[1] - mu_bin_array[0]).item()

    # Use float64 for Pkmu (real power spectrum output)
    Pkmu = cp.zeros((nk, nmu), dtype=cp.float64)
    power_k = cp.zeros((nk, nmu), dtype=cp.float64)
    power_mu = cp.zeros((nk, nmu), dtype=cp.float64)
    modes = cp.zeros((nk, nmu), dtype = cp.uint64)

    cal_ps_3d_core_kernel(
            (nx, ny), 
            (nz,),
            (ps_3d_gpu,
            k_x_array, k_y_array, k_z_array,
            k_bin_array, mu_bin_array,
            nx, ny, nz,
            nk, nmu, 
            k_diff, mu_diff,
            Pkmu, power_k, power_mu, modes)
        )
    
    return power_k, power_mu, Pkmu, modes