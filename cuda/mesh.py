import numpy as np 
import cupy as cp 

def get_run_cic_core_kernel():
    kernel_code = r'''
    extern "C" __global__
    void run_cic_core(const float* pos, const unsigned long long nparticle, float* field, const double* boxsize_array, const int* ngrids_array)
    {
        unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= nparticle)
        {   
            return;
        }

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

        atomicAdd(&field[iz + iy * nz + ix * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 1]);
        atomicAdd(&field[iz_next + iy * nz + ix * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 1]);
        atomicAdd(&field[iz + iy_next * nz + ix * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 1]);
        atomicAdd(&field[iz_next + iy_next * nz + ix * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 1]);

        atomicAdd(&field[iz + iy * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 0]);
        atomicAdd(&field[iz_next + iy * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 1] * diff_ratio_temp[0*2 + 0]);
        atomicAdd(&field[iz + iy_next * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 1] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 0]);
        atomicAdd(&field[iz_next + iy_next * nz + ix_next * nz * ny], diff_ratio_temp[2*2 + 0] * diff_ratio_temp[1*2 + 0] * diff_ratio_temp[0*2 + 0]);
    }
    '''
    return cp.RawKernel(kernel_code, 'run_cic_core')

run_cic_core_kernel = get_run_cic_core_kernel()