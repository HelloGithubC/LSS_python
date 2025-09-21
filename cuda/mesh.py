import numpy as np 
import cupy as cp 

def get_run_cic_core_kernel():
    kernel_code = r'''
    extern "C" __global__
    void run_cic_core(const float* pos, const float* weight, const unsigned long long nparticle, float* field, const double* boxsize_array, const int* ngrids_array)
    {
        unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= nparticle)
        {   
            return;
        }

        bool use_weight;
        if (weight == nullptr)
        {
            use_weight = false;
        }
        else 
        {
            use_weight = true;
        }
        float weight_temp = (use_weight)? weight[index] : 1.0f;

        const int NDIM = 3;
        double pos_temp[NDIM];
        int pos_i[NDIM];
        int pos_i_next[NDIM];
        double sub_boxsize[NDIM];
        double diff_ratio_temp[NDIM * 2];

        for (int i = 0; i < NDIM; i++)
        {
            if (index*NDIM + i >= nparticle * NDIM)
            {
                printf("index = %llu, threadIdx.x = %d, blockIdx.x = %d\n", index, threadIdx.x, blockIdx.x);
                return;    
            }
            pos_temp[i] = pos[index * NDIM + i];
            if (pos_temp[i] < 0.0 || pos_temp[i] > boxsize_array[i])
            {
                return;
            }
            sub_boxsize[i] = boxsize_array[i] / ngrids_array[i];
            pos_i[i] = static_cast<int>(pos_temp[i] / sub_boxsize[i]);
            if (pos_i[i] == ngrids_array[i])
            {
                pos_i[i] = 0;
                pos_temp[i] = 0.0;
            }
            pos_i_next[i] = (pos_i[i] == ngrids_array[i] - 1)? 0 : pos_i[i] + 1;
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

        atomicAdd(&field[iz + iy * nz + ix * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 1]  * weight_temp);
        atomicAdd(&field[iz_next + iy * nz + ix * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 1] * weight_temp);
        atomicAdd(&field[iz + iy_next * nz + ix * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 1] * weight_temp);
        atomicAdd(&field[iz_next + iy_next * nz + ix * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 1] * weight_temp);

        atomicAdd(&field[iz + iy * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 0] * weight_temp);
        atomicAdd(&field[iz_next + iy * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 0] * weight_temp);
        atomicAdd(&field[iz + iy_next * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 0] * weight_temp);
        atomicAdd(&field[iz_next + iy_next * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 0] * weight_temp);
    }
    '''
    return cp.RawKernel(kernel_code, 'run_cic_core')

def get_do_compensated_core_kernel():
    kernel_code = r'''
    extern "C" __global__
    void do_compensated_core(float2* field, const float* k_x_array, const float* k_y_array, const float* k_z_array, const int nx, const int ny, const int nz)
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
    '''
    return cp.RawKernel(kernel_code, 'do_compensated_core')

run_cic_core_kernel = get_run_cic_core_kernel()
do_compensated_core_kernel = get_do_compensated_core_kernel()


def run_cic_from_cuda(pos, weight, field, boxsize, ngrids):
    if pos.dtype != cp.float32:
        pos = pos.astype(cp.float32)
    
    nparticle = cp.uint64(pos.shape[0])
    nthreads = nparticle if nparticle < 1024 else 1024 
    ngridx = nparticle // nthreads + 1

    weight_need = weight if weight is not None else 0

    run_cic_core_kernel((ngridx, ), (nthreads, ), (pos, weight_need, nparticle, field, boxsize, ngrids))

def do_compensated_from_cuda(complex_field_gpu, k_arrays_gpu):
    k_x_array_gpu, k_y_array_gpu, k_z_array_gpu = k_arrays_gpu
    nx, ny, nz = complex_field_gpu.shape
    do_compensated_core_kernel((nx, ny), (nz,), (complex_field_gpu, k_x_array_gpu, k_y_array_gpu, k_z_array_gpu, nx, ny, nz))