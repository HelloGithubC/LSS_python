import numpy as np 
import cupy as cp 

to_mesh_cuda_code = r'''
extern "C" {
    __global__ void run_cic_core(const float* pos, const float* weights, const float* values, const unsigned long long nparticle, float* field, const double* boxsize_array, const unsigned int* ngrids_array, double shift)
    {
        unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= nparticle)
        {   
            return;
        }

        float weight_temp = (weights == nullptr)? 1.0f: weights[index];
        float value_temp = (values == nullptr)? 1.0f: values[index];

        const int NDIM = 3;

        double pos_i_float = 0.0;
        long long pos_i_temp = 0uL;
        unsigned long pos_i[2][NDIM];
        double sub_boxsize[NDIM];
        double diff_ratio_temp[NDIM][2];
        bool is_in_box = true;

        for (int i = 0; i < NDIM; i++)
        {
            if (pos[index * NDIM + i] < 0.0 || pos[index * NDIM + i] > boxsize_array[i])
            {
                is_in_box = false;
                break;
            }
            sub_boxsize[i] = boxsize_array[i] / ngrids_array[i];
            pos_i_float = pos[index * NDIM + i] / sub_boxsize[i] + shift;
            pos_i_temp = static_cast<long long>(floor(pos_i_float));

            diff_ratio_temp[i][1] = pos_i_float - pos_i_temp;
            diff_ratio_temp[i][0] = 1.0 - diff_ratio_temp[i][1];

            pos_i[0][i] = static_cast<unsigned int>(pos_i_temp + ngrids_array[i]) % ngrids_array[i];
            pos_i[1][i] = (pos_i[0][i] + 1) % ngrids_array[i];
        }

        if (!is_in_box)
        {
            return;
        }

        float field_temp = weight_temp * value_temp;

        unsigned int ny = ngrids_array[1];
        unsigned int nz = ngrids_array[2];

        for (unsigned int dix=0u; dix < 2u; dix++)
        {
            for (unsigned int diy=0u; diy < 2u; diy++)
            {
                for (unsigned int diz=0u; diz < 2u; diz++)
                {
                    atomicAdd(&field[pos_i[diz][2] + pos_i[diy][1] * nz + pos_i[dix][0] * nz * ny], diff_ratio_temp[2][diz] * diff_ratio_temp[1][diy] * diff_ratio_temp[0][dix]  * field_temp);
                }
            }
        }
    }

    __global__ void run_ngp_core(const float* pos, const float* weights, const float* values, const unsigned long long nparticle, float* field, const double* boxsize_array, const unsigned int* ngrids_array, double shift)
    {
        unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= nparticle)
        {   
            return;
        }

        float weight_temp = (weights == nullptr)? 1.0f: weights[index];
        float value_temp = (values == nullptr)? 1.0f: values[index];

        const int NDIM = 3;

        double pos_i_float = 0.0;
        unsigned long pos_i[NDIM];
        double sub_boxsize[NDIM];
        bool is_in_box = true;

        for (int i = 0; i < NDIM; i++)
        {
            if (pos[index * NDIM + i] < 0.0 || pos[index * NDIM + i] > boxsize_array[i])
            {
                is_in_box = false;
                break;
            }
            sub_boxsize[i] = boxsize_array[i] / ngrids_array[i];
            pos_i_float = pos[index * NDIM + i] / sub_boxsize[i] + shift;

            pos_i[i] = (static_cast<unsigned int>(floor(pos_i_float + 0.5)) + ngrids_array[i]) % ngrids_array[i];
        }

        if (!is_in_box)
        {
            return;
        }

        unsigned int ny = ngrids_array[1];
        unsigned int nz = ngrids_array[2];


        float field_temp = weight_temp * value_temp;


        atomicAdd(&field[pos_i[2] + pos_i[1] * nz + pos_i[0] * nz * ny], field_temp);
    }

    __global__ void run_tsc_core(const float* pos, const float* weights, const float* values, const unsigned long long nparticle, float* field, const double* boxsize_array, const unsigned int* ngrids_array, double shift)
    {
        unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= nparticle)
        {   
            return;
        }

        float weight_temp = (weights == nullptr)? 1.0f: weights[index];
        float value_temp = (values == nullptr)? 1.0f: values[index];

        const int NDIM = 3;

        double pos_i_float = 0.0;
        long long pos_i_temp = 0uL;
        unsigned long pos_i[3][NDIM];
        double sub_boxsize[NDIM];
        double diff_ratio_temp[NDIM][3];
        bool is_in_box = true;

        for (int i = 0; i < NDIM; i++)
        {
            if (pos[index * NDIM + i] < 0.0 || pos[index * NDIM + i] > boxsize_array[i])
            {
                is_in_box = false;
                break;
            }
            sub_boxsize[i] = boxsize_array[i] / ngrids_array[i];
            pos_i_float = pos[index * NDIM + i] / sub_boxsize[i] + shift;
            pos_i_temp = static_cast<long long>(floor(pos_i_float - 0.5));

            for (int j = 0; j < 3; j++)
            {
                double distance_ratio = fabs(pos_i_temp + j - pos_i_float);
                if (distance_ratio < 0.5)
                {
                    diff_ratio_temp[i][j] = 0.75 - distance_ratio * distance_ratio;
                }
                else if (distance_ratio < 1.5)
                {
                    diff_ratio_temp[i][j] = 0.5 * (1.5 - distance_ratio) * (1.5 - distance_ratio);
                }
                else
                {
                    diff_ratio_temp[i][j] = 0.0;
                }
            }

            pos_i[0][i] = static_cast<unsigned int>(pos_i_temp + ngrids_array[i]) % ngrids_array[i];
            pos_i[1][i] = (pos_i[0][i] + 1) % ngrids_array[i];
            pos_i[2][i] = (pos_i[1][i] + 1) % ngrids_array[i];
        }

        if (!is_in_box)
        {
            return;
        }

        float field_temp = weight_temp * value_temp;

        unsigned int ny = ngrids_array[1];
        unsigned int nz = ngrids_array[2];

        for (unsigned int dix=0u; dix < 3u; dix++)
        {
            for (unsigned int diy=0u; diy < 3u; diy++)
            {
                for (unsigned int diz=0u; diz < 3u; diz++)
                {
                    atomicAdd(&field[pos_i[diz][2] + pos_i[diy][1] * nz + pos_i[dix][0] * nz * ny], diff_ratio_temp[2][diz] * diff_ratio_temp[1][diy] * diff_ratio_temp[0][dix]  * field_temp);
                }
            }
        }
    }

    __global__ void run_pcs_core(const float* pos, const float* weights, const float* values, const unsigned long long nparticle, float* field, const double* boxsize_array, const unsigned int* ngrids_array, double shift)
    {
        unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= nparticle)
        {   
            return;
        }

        float weight_temp = (weights == nullptr)? 1.0f: weights[index];
        float value_temp = (values == nullptr)? 1.0f: values[index];

        const int NDIM = 3;

        double pos_i_float = 0.0;
        long long pos_i_temp = 0uL;
        unsigned long pos_i[4][NDIM];
        double sub_boxsize[NDIM];
        double diff_ratio_temp[NDIM][4];
        bool is_in_box = true;

        for (int i = 0; i < NDIM; i++)
        {
            if (pos[index * NDIM + i] < 0.0 || pos[index * NDIM + i] > boxsize_array[i])
            {
                is_in_box = false;
                break;
            }
            sub_boxsize[i] = boxsize_array[i] / ngrids_array[i];
            pos_i_float = pos[index * NDIM + i] / sub_boxsize[i] + shift;
            pos_i_temp = static_cast<long long>(floor(pos_i_float - 1.0));

            for (int j = 0; j < 4; j++)
            {
                double distance_ratio = fabs(pos_i_temp + j - pos_i_float);
                if (distance_ratio < 1.0)
                {
                    diff_ratio_temp[i][j] = (4.0 - 6.0 * distance_ratio * distance_ratio + 3.0 * distance_ratio * distance_ratio * distance_ratio) / 6.0;
                }
                else if (distance_ratio < 2.0)
                {
                    diff_ratio_temp[i][j] = (2.0 - distance_ratio) * (2.0 - distance_ratio) * (2.0 - distance_ratio) / 6.0;
                }
                else
                {
                    diff_ratio_temp[i][j] = 0.0;
                }
            }

            pos_i[0][i] = static_cast<unsigned int>(pos_i_temp + ngrids_array[i]) % ngrids_array[i];
            pos_i[1][i] = (pos_i[0][i] + 1) % ngrids_array[i];
            pos_i[2][i] = (pos_i[1][i] + 1) % ngrids_array[i];
            pos_i[3][i] = (pos_i[2][i] + 1) % ngrids_array[i];
        }

        if (!is_in_box)
        {
            return;
        }

        float field_temp = weight_temp * value_temp;

        unsigned int ny = ngrids_array[1];
        unsigned int nz = ngrids_array[2];

        for (unsigned int dix=0u; dix < 4u; dix++)
        {
            for (unsigned int diy=0u; diy < 4u; diy++)
            {
                for (unsigned int diz=0u; diz < 4u; diz++)
                {
                    atomicAdd(&field[pos_i[diz][2] + pos_i[diy][1] * nz + pos_i[dix][0] * nz * ny], diff_ratio_temp[2][diz] * diff_ratio_temp[1][diy] * diff_ratio_temp[0][dix]  * field_temp);
                }
            }
        }
    }
}
    '''
to_mesh_kernel = cp.RawModule(code=to_mesh_cuda_code)
run_cic_core_kernel = to_mesh_kernel.get_function("run_cic_core")
run_ngp_core_kernel = to_mesh_kernel.get_function("run_ngp_core")
run_tsc_core_kernel = to_mesh_kernel.get_function("run_tsc_core")
run_pcs_core_kernel = to_mesh_kernel.get_function("run_pcs_core")

do_interlacing_cuda_code = r'''
#include <cupy/complex.cuh>
extern "C"
__global__ void do_interlacing(complex<float> *c1, const complex<float> *c2, const float *H, const float *kx_array, const float *ky_array, const float *kz_array, const int nx, const int ny, const int nz)
{
    int iz = threadIdx.x;
    int iy = blockIdx.y;
    int ix = blockIdx.x;
    if (iz >= nz || iy >= ny || ix >= nx)
    {
        return;
    }

    float kx = kx_array[ix];
    float ky = ky_array[iy];
    float kz = kz_array[iz];
    float kH = H[0] * kx + H[1] * ky + H[2] * kz;
    float factor = 0.5f;

    complex<float> phase = exp(complex<float>(0.0f, factor * kH));
    
    long long index = static_cast<long long>(iz + iy * nz + ix * nz * ny);
    c1[index] = c1[index] * factor + c2[index] * factor * phase;
}
'''
do_interlacing_kernel = cp.RawKernel(code=do_interlacing_cuda_code, name="do_interlacing")
do_interlacing_kernel.compile()

do_compensation_cuda_code = r'''
extern "C"
{
    __global__ void do_compensation_CIC(float2* field, const float* k_x_array, const float* k_y_array, const float* k_z_array, const int nx, const int ny, const int nz)
    {
        const int NDIM = 3;
    
        int iz = threadIdx.x;
        int iy = blockIdx.y;
        int ix = blockIdx.x;
        if (iz >= nz || iy >= ny || ix >= nx)
        {
            return;
        }

        float k_x = k_x_array[ix];
        float k_y = k_y_array[iy];
        float k_z = k_z_array[iz];
        float k_temp[NDIM] = {k_x, k_y, k_z};
        float w_temp[NDIM];

        for (int i = 0; i < NDIM; i++)
        {
            float sin_temp = sin(k_temp[i]/2.0f);
            w_temp[i] = sqrt(1.0f - 2.0f/3.0f *  sin_temp * sin_temp);
        }

        field[iz + iy * nz + ix * nz * ny].x /= w_temp[0] * w_temp[1] * w_temp[2];
        field[iz + iy * nz + ix * nz * ny].y /= w_temp[0] * w_temp[1] * w_temp[2];
    }

    __global__ void do_compensation_TSC(float2* field, const float* k_x_array, const float* k_y_array, const float* k_z_array, const int nx, const int ny, const int nz)
    {
        const int NDIM = 3;
    
        int iz = threadIdx.x;
        int iy = blockIdx.y;
        int ix = blockIdx.x;
        if (iz >= nz || iy >= ny || ix >= nx)
        {
            return;
        }

        float k_x = k_x_array[ix];
        float k_y = k_y_array[iy];
        float k_z = k_z_array[iz];
        float k_temp[NDIM] = {k_x, k_y, k_z};
        float w_temp[NDIM];

        for (int i = 0; i < NDIM; i++)
        {
            float sin_temp = sin(k_temp[i]/2.0f);
            w_temp[i] = sqrt(1.0f - sin_temp * sin_temp + 2.0/15.0 * sin_temp * sin_temp * sin_temp * sin_temp);
        }

        field[iz + iy * nz + ix * nz * ny].x /= w_temp[0] * w_temp[1] * w_temp[2];
        field[iz + iy * nz + ix * nz * ny].y /= w_temp[0] * w_temp[1] * w_temp[2];
    }

    __global__ void do_compensation_PCS(float2* field, const float* k_x_array, const float* k_y_array, const float* k_z_array, const int nx, const int ny, const int nz)
    {
        const int NDIM = 3;
    
        int iz = threadIdx.x;
        int iy = blockIdx.y;
        int ix = blockIdx.x;
        if (iz >= nz || iy >= ny || ix >= nx)
        {
            return;
        }

        float k_x = k_x_array[ix];
        float k_y = k_y_array[iy];
        float k_z = k_z_array[iz];
        float k_temp[NDIM] = {k_x, k_y, k_z};
        float w_temp[NDIM];

        for (int i = 0; i < NDIM; i++)
        {
            float sin_temp = sin(k_temp[i]/2.0f);
            w_temp[i] = sqrt(1.0f - 4.0f / 3.0f * sin_temp * sin_temp + 6.0/15.0 * sin_temp * sin_temp * sin_temp * sin_temp - 4.0f/315.0f * sin_temp * sin_temp * sin_temp * sin_temp * sin_temp * sin_temp);
        }

        field[iz + iy * nz + ix * nz * ny].x /= w_temp[0] * w_temp[1] * w_temp[2];
        field[iz + iy * nz + ix * nz * ny].y /= w_temp[0] * w_temp[1] * w_temp[2];
    }

    __global__ void do_compensation_interlace(float2* field, const float* k_x_array, const float* k_y_array, const float* k_z_array, const int nx, const int ny, const int nz, const unsigned int p)
    {
        const int NDIM = 3;
    
        int iz = threadIdx.x;
        int iy = blockIdx.y;
        int ix = blockIdx.x;
        if (iz >= nz || iy >= ny || ix >= nx)
        {
            return;
        }

        float k_x = k_x_array[ix];
        float k_y = k_y_array[iy];
        float k_z = k_z_array[iz];
        float k_temp[NDIM] = {k_x, k_y, k_z};
        float w_temp[NDIM];

        for (int i = 0; i < NDIM; i++)
        {
            float sinc_temp = sin(k_temp[i]/2.0f) / (k_temp[i]/2.0f);
            for (int j = 1; j < p; j++)
            {
                sinc_temp *= sinc_temp;
            }
            w_temp[i] = (k_temp[i] == 0.0f)? 1.0f : sinc_temp;
        }

        field[iz + iy * nz + ix * nz * ny].x /= w_temp[0] * w_temp[1] * w_temp[2];
        field[iz + iy * nz + ix * nz * ny].y /= w_temp[0] * w_temp[1] * w_temp[2];
    }
} 
'''
do_compensation_kernel = cp.RawModule(code=do_compensation_cuda_code)
do_compensation_CIC_kernel = do_compensation_kernel.get_function("do_compensation_CIC")
do_compensation_TSC_kernel = do_compensation_kernel.get_function("do_compensation_TSC")
do_compensation_PCS_kernel = do_compensation_kernel.get_function("do_compensation_PCS")
do_compensation_interlace_kernel = do_compensation_kernel.get_function("do_compensation_interlace")


def to_mesh_from_cuda(pos, weight, value, field, boxsize, ngrids, resampler="CIC", shift=0.0):
    if pos.dtype != cp.float32:
        pos = pos.astype(cp.float32)
    
    nparticle = cp.uint64(pos.shape[0])
    nthreads = nparticle if nparticle < 512 else 512 
    ngridx = nparticle // nthreads + 1

    weight_need = weight if weight is not None else 0
    value_need = value if value is not None else 0

    if resampler == "CIC":
        run_cic_core_kernel((ngridx, ), (nthreads, ), (pos, weight_need, value_need, nparticle, field, boxsize, ngrids, shift))
    elif resampler == "NGP":
        run_ngp_core_kernel((ngridx, ), (nthreads, ), (pos, weight_need, value_need, nparticle, field, boxsize, ngrids, shift))
    elif resampler == "TSC":
        run_tsc_core_kernel((ngridx, ), (nthreads, ), (pos, weight_need, value_need, nparticle, field, boxsize, ngrids, shift))
    elif resampler == "PCS":
        run_pcs_core_kernel((ngridx, ), (nthreads, ), (pos, weight_need, value_need, nparticle, field, boxsize, ngrids, shift))
    else:
        raise ValueError(f"resampler must be CIC, NGP, TSC or PCS. Now is {resampler}")

def do_compensation_from_cuda(complex_field_gpu, k_arrays_gpu, resampler="CIC", interlaced=False):
    if complex_field_gpu.dtype != cp.complex64:
        complex_field_gpu = complex_field_gpu.astype(cp.complex64)
    
    k_x_array_gpu, k_y_array_gpu, k_z_array_gpu = k_arrays_gpu
    nx, ny, nz = complex_field_gpu.shape

    if not interlaced:
        if resampler == "CIC":
            do_compensation_CIC_kernel((nx, ny), (nz,), (complex_field_gpu, k_x_array_gpu, k_y_array_gpu, k_z_array_gpu, nx, ny, nz))
        elif resampler == "TSC":
            do_compensation_TSC_kernel((nx, ny), (nz,), (complex_field_gpu, k_x_array_gpu, k_y_array_gpu, k_z_array_gpu, nx, ny, nz))
        elif resampler == "PCS":
            do_compensation_PCS_kernel((nx, ny), (nz,), (complex_field_gpu, k_x_array_gpu, k_y_array_gpu, k_z_array_gpu, nx, ny, nz))
        elif resampler == "NGP":
            return 
        else:
            raise ValueError(f"resampler must be CIC, NGP, TSC or PCS. Now is {resampler}")
    else:
        if resampler == "CIC":
            p = cp.uint32(1)
        elif resampler == "TSC":
            p = cp.uint32(2)
        elif resampler == "PCS":
            p = cp.uint32(3)
        elif resampler == "NGP":
            return 
        else:
            raise ValueError(f"resampler must be CIC, NGP, TSC or PCS. Now is {resampler}")
        do_compensation_interlace_kernel((nx, ny), (nz,), (complex_field_gpu, k_x_array_gpu, k_y_array_gpu, k_z_array_gpu, nx, ny, nz, p))

def do_interlacing_from_cuda(c1, c2, boxsize_gpu, Nmesh_gpu, k_arrays_gpu):
    if c1.dtype != cp.complex64:
        c1 = c1.astype(cp.complex64)
    if c2.dtype != cp.complex64:
        c2 = c2.astype(cp.complex64)
    if c1.shape != c2.shape:
        raise ValueError("c1 and c2 must have the same shape")
    H_gpu = boxsize_gpu / Nmesh_gpu
    nx, ny, nz = c1.shape

    if k_arrays_gpu is None:
        kx_array_gpu = (
            cp.fft.fftfreq(Nmesh_gpu[0], d=1.0).astype(cp.float32, copy=False)
            * 2.0
            * np.pi
            * Nmesh_gpu[0]
            / boxsize_gpu[0]
        )
        ky_array_gpu = (
            cp.fft.fftfreq(Nmesh_gpu[1], d=1.0).astype(cp.float32, copy=False)
            * 2.0
            * np.pi
            * Nmesh_gpu[1]
            / boxsize_gpu[1]
        )
        kz_array_gpu = (
            cp.fft.fftfreq(Nmesh_gpu[2], d=1.0).astype(cp.float32, copy=False)
            * 2.0
            * np.pi
            * Nmesh_gpu[2]
            / boxsize_gpu[2]
        )[: c1.shape[2]]
    else:
        kx_array_gpu, ky_array_gpu, kz_array_gpu = k_arrays_gpu
    do_interlacing_kernel((nx, ny), (nz,), (c1, c2, H_gpu, kx_array_gpu, ky_array_gpu, kz_array_gpu, nx, ny, nz))