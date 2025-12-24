#include "mesh.hpp"

extern "C" {
    void run_cic_float(const float *pos, float *field, size_t np, float boxSize[ndim], size_t ngrids[ndim], const float *weights, const float *values, float shift, int nthreads)
    {
        // size_t ngrids_all = 1uL;
        // for (size_t i = 0; i < ndim; ++i)
        // {
        //     ngrids_all *= ngrids[i];
        // }
        // double* field_double = new double[ngrids_all];
        // for (size_t i = 0; i < ngrids_all; ++i)
        // {
        //     field_double[i] = field[i];
        // }
        run_cic<float>(pos, field, np, boxSize, ngrids, weights, values, shift, nthreads);
        // for (size_t i = 0; i < ngrids_all; ++i)
        // {
        //     field[i] = static_cast<float>(field_double[i]);
        // }
        // delete[] field_double;
    }

    void run_cic_double(const double *pos, double *field, size_t np, double boxSize[ndim], size_t ngrids[ndim], const double *weights, const double *values, double shift, int nthreads)
    {
        run_cic<double>(pos, field, np, boxSize, ngrids, weights, values, shift, nthreads);
    }

    void run_ngp_float(const float *pos, float *field, size_t np, float boxSize[ndim], size_t ngrids[ndim], const float *weights, const float *values, float shift, int nthreads)
    {
        // size_t ngrids_all = 1uL;
        // for (size_t i = 0; i < ndim; ++i)
        // {
        //     ngrids_all *= ngrids[i];
        // }
        // double* field_double = new double[ngrids_all];
        // for (size_t i = 0; i < ngrids_all; ++i)
        // {
        //     field_double[i] = field[i];
        // }
        run_ngp<float>(pos, field, np, boxSize, ngrids, weights, values, shift, nthreads);
        // for (size_t i = 0; i < ngrids_all; ++i)
        // {
        //     field[i] = static_cast<float>(field_double[i]);
        // }
        // delete[] field_double;
    }

    void run_ngp_double(const double *pos, double *field, size_t np, double boxSize[ndim], size_t ngrids[ndim], const double *weights, const double *values, double shift, int nthreads)
    {
        run_ngp<double>(pos, field, np, boxSize, ngrids, weights, values, shift, nthreads);
    }

    void run_tsc_float(const float *pos, float *field, size_t np, float boxSize[ndim], size_t ngrids[ndim], const float *weights, const float *values, float shift, int nthreads)
    {
        // size_t ngrids_all = 1uL;
        // for (size_t i = 0; i < ndim; ++i)
        // {
        //     ngrids_all *= ngrids[i];
        // }
        // double* field_double = new double[ngrids_all];
        // for (size_t i = 0; i < ngrids_all; ++i)
        // {
        //     field_double[i] = field[i];
        // }
        run_tsc<float>(pos, field, np, boxSize, ngrids, weights, values, shift, nthreads);
        // for (size_t i = 0; i < ngrids_all; ++i)
        // {
        //     field[i] = static_cast<float>(field_double[i]);
        // }
        // delete[] field_double;
    }

    void run_tsc_double(const double *pos, double *field, size_t np, double boxSize[ndim], size_t ngrids[ndim], const double *weights, const double *values, double shift, int nthreads)
    {
        run_tsc<double>(pos, field, np, boxSize, ngrids, weights, values, shift, nthreads);
    }

    void run_pcs_float(const float *pos, float *field, size_t np, float boxSize[ndim], size_t ngrids[ndim], const float *weights, const float *values, float shift, int nthreads)
    {
        // size_t ngrids_all = 1uL;
        // for (size_t i = 0; i < ndim; ++i)
        // {
        //     ngrids_all *= ngrids[i];
        // }
        // double* field_double = new double[ngrids_all];
        // for (size_t i = 0; i < ngrids_all; ++i)
        // {
        //     field_double[i] = field[i];
        // }
        run_pcs<float>(pos, field, np, boxSize, ngrids, weights, values, shift, nthreads);
        // for (size_t i = 0; i < ngrids_all; ++i)
        // {
        //     field[i] = static_cast<float>(field_double[i]);
        // }
        // delete[] field_double;
    }

    void run_pcs_double(const double *pos, double *field, size_t np, double boxSize[ndim], size_t ngrids[ndim], const double *weights, const double *values, double shift, int nthreads)
    {
        run_pcs<double>(pos, field, np, boxSize, ngrids, weights, values, shift, nthreads);
    }

    void do_compensation_cic_double(std::complex<double>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads) 
    {
        DoCompensationCIC<double>(complex_field, ngrids, kx_array, ky_array, kz_array, nthreads);
    }

    void do_compensation_cic_float(std::complex<float>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads) 
    {
        DoCompensationCIC<float>(complex_field, ngrids, kx_array, ky_array, kz_array, nthreads);
    }

    void do_compensation_tsc_double(std::complex<double>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads) 
    {
        DoCompensationTSC<double>(complex_field, ngrids, kx_array, ky_array, kz_array, nthreads);
    }

    void do_compensation_tsc_float(std::complex<float>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads) 
    {
        DoCompensationTSC<float>(complex_field, ngrids, kx_array, ky_array, kz_array, nthreads);
    }

    void do_compensation_pcs_double(std::complex<double>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads) 
    {
        DoCompensationPCS<double>(complex_field, ngrids, kx_array, ky_array, kz_array, nthreads);
    }

    void do_compensation_pcs_float(std::complex<float>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads) 
    {
        DoCompensationPCS<float>(complex_field, ngrids, kx_array, ky_array, kz_array, nthreads);
    }

    void do_compensation_intelaced_float(std::complex<float>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int p, int nthreads)
    {
        DoCompensationInterlaced<float>(complex_field, ngrids, kx_array, ky_array, kz_array, p, nthreads);
        // p: 2 for CIC; 3 for TSC; 4 for PCS
    }

    void do_compensation_intelaced_double(std::complex<double>* complex_field, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int p, int nthreads)
    {
        DoCompensationInterlaced<double>(complex_field, ngrids, kx_array, ky_array, kz_array, p, nthreads);
        // p: 2 for CIC; 3 for TSC; 4 for PCS
    }

    void do_interlacing_double(std::complex<double>* c1, std::complex<double>* c2, double* H, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads)
    {
        DoInterlace<double>(c1, c2, H, ngrids, kx_array, ky_array, kz_array, nthreads);
    }

    void do_interlacing_float(std::complex<float>* c1, std::complex<float>* c2, double* H, size_t* ngrids, double* kx_array, double* ky_array, double* kz_array, int nthreads)
    {
        DoInterlace<float>(c1, c2, H, ngrids, kx_array, ky_array, kz_array, nthreads);
    }

}
