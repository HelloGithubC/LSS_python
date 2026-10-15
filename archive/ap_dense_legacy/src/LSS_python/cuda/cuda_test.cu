#include <thrust/complex.h>
extern "C"
{
    void do_interlace(complex<float> *c1, const complex<float> *c2, const float *H, const float *kx_array, const float *ky_array, const float *kz_array, const int nx, const int ny, const int nz)
    {
        const int NDIM = 3;
    
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
        
        c1[iz + iy * nz + ix * nz * ny] = c1[iz + iy * nz + ix * nz * ny] * factor + c2[iz + iy * nz + ix * nz * ny] * factor * expf(complex<float>(0.0f, 1.0f) * factor * kH);
    }
} 