#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>
#include <iostream>

#include "mesh.hpp"

namespace py = pybind11;
template <typename T>
void run_mesh(py::array_t<T> pos, 
                     py::array_t<T> field, 
                     py::object weights, 
                     py::object values, 
                     py::array_t<T> boxSize, 
                     py::array_t<size_t> ngrids, 
                     T shift, 
                     int type,
                     int nthreads
                    ) {

    if (!(field.flags() & py::array::c_style) || !field.writeable())  {
        throw std::runtime_error("Field must be C-contiguous and writeable");
    }
    
    // 获取数据指针
    auto pos_buf = pos.request();
    auto field_buf = field.request();
    auto boxSize_buf = boxSize.request();
    auto ngrids_buf = ngrids.request();

    const T* pos_ptr = static_cast<const T*>(pos_buf.ptr);
    T* field_ptr = static_cast<T*>(field_buf.ptr);
    const T* boxSize_ptr = static_cast<const T*>(boxSize_buf.ptr);
    const size_t* ngrids_ptr = static_cast<const size_t*>(ngrids_buf.ptr);

    const T* weights_ptr = nullptr;
    const T* values_ptr = nullptr;

    // 处理 weights
    if (!weights.is_none()) {
        // 确保传入的是正确的 numpy 数组类型
        if (!py::isinstance<py::array_t<T>>(weights)) {
            throw std::runtime_error("weights must be a numpy array of the correct dtype or None");
        }
        auto weights_arr = weights.cast<py::array_t<T>>();
        
        auto weights_buf = weights_arr.request();
        weights_ptr = static_cast<const T*>(weights_buf.ptr);
    }

    // 处理 values
    if (!values.is_none()) {
        if (!py::isinstance<py::array_t<T>>(values)) {
            throw std::runtime_error("values must be a numpy array of the correct dtype or None");
        }
        auto values_arr = values.cast<py::array_t<T>>();

        auto values_buf = values_arr.request();
        values_ptr = static_cast<const T*>(values_buf.ptr);
    }

    size_t np = pos_buf.shape[0];

    // 构造固定大小的数组传给原函数
    T c_boxSize[ndim];
    size_t c_ngrids[ndim];
    
    for(int i=0; i<ndim; ++i) {
        c_boxSize[i] = boxSize_ptr[i];
        c_ngrids[i] = ngrids_ptr[i];
    }

    // 调用原始模板函数
    if (type == 0) run_ngp<T>(pos_ptr, field_ptr, np, c_boxSize, c_ngrids, weights_ptr, values_ptr, shift, nthreads);
    else if (type == 1) run_cic<T>(pos_ptr, field_ptr, np, c_boxSize, c_ngrids, weights_ptr, values_ptr, shift, nthreads);
    else if (type == 2) run_tsc<T>(pos_ptr, field_ptr, np, c_boxSize, c_ngrids, weights_ptr, values_ptr, shift, nthreads);
    else if (type == 3) run_pcs<T>(pos_ptr, field_ptr, np, c_boxSize, c_ngrids, weights_ptr, values_ptr, shift, nthreads);

}

template <typename T>
void do_compensation(py::array_t<std::complex<T>> complex_field,
                    py::array_t<size_t> ngrids,
                    py::array_t<double> kx_array,
                    py::array_t<double> ky_array,
                    py::array_t<double> kz_array,
                    int type,
                    int nthreads)
{
    auto buf_field = complex_field.request();
    auto buf_ngrids = ngrids.request();
    auto buf_kx = kx_array.request();
    auto buf_ky = ky_array.request();
    auto buf_kz = kz_array.request();

    std::complex<T>* ptr_field = static_cast<std::complex<T>*>(buf_field.ptr);
    size_t* ptr_ngrids = static_cast<size_t*>(buf_ngrids.ptr);
    double* ptr_kx = static_cast<double*>(buf_kx.ptr);
    double* ptr_ky = static_cast<double*>(buf_ky.ptr);
    double* ptr_kz = static_cast<double*>(buf_kz.ptr);

    if (type == 1) DoCompensationCIC<T>(ptr_field, ptr_ngrids, ptr_kx, ptr_ky, ptr_kz, nthreads);
    else if (type == 2) DoCompensationTSC<T>(ptr_field, ptr_ngrids, ptr_kx, ptr_ky, ptr_kz, nthreads);
    else if (type == 3) DoCompensationPCS<T>(ptr_field, ptr_ngrids, ptr_kx, ptr_ky, ptr_kz, nthreads);
}

template<typename T>
void do_compensation_interlaced(py::array_t<std::complex<T>> complex_field, 
                               py::array_t<size_t> ngrids,
                               py::array_t<double> kx_array,
                               py::array_t<double> ky_array,
                               py::array_t<double> kz_array,
                               int p,
                               int nthreads)
{
    auto buf_field = complex_field.request();
    auto buf_ngrids = ngrids.request();
    auto buf_kx = kx_array.request();
    auto buf_ky = ky_array.request();
    auto buf_kz = kz_array.request();

    std::complex<T>* ptr_field = static_cast<std::complex<T>*>(buf_field.ptr);
    size_t* ptr_ngrids = static_cast<size_t*>(buf_ngrids.ptr);
    double* ptr_kx = static_cast<double*>(buf_kx.ptr);
    double* ptr_ky = static_cast<double*>(buf_ky.ptr);
    double* ptr_kz = static_cast<double*>(buf_kz.ptr);

    DoCompensationInterlaced<T>(ptr_field, ptr_ngrids, ptr_kx, ptr_ky, ptr_kz, p, nthreads);
}

template<typename T>
void do_interlace(py::array_t<std::complex<T>> c1,
                  py::array_t<std::complex<T>> c2,
                  py::array_t<double> H, 
                  py::array_t<size_t> ngrids,
                  py::array_t<double> kx_array,
                  py::array_t<double> ky_array,
                  py::array_t<double> kz_array,
                  int nthreads)
{
    auto buf_c1 = c1.request();
    auto buf_c2 = c2.request();
    auto buf_H = H.request();
    auto buf_ngrids = ngrids.request();
    auto buf_kx = kx_array.request();
    auto buf_ky = ky_array.request();
    auto buf_kz = kz_array.request();

    std::complex<T>* ptr_c1 = static_cast<std::complex<T>*>(buf_c1.ptr);
    std::complex<T>* ptr_c2 = static_cast<std::complex<T>*>(buf_c2.ptr);
    double* ptr_H = static_cast<double*>(buf_H.ptr);
    size_t* ptr_ngrids = static_cast<size_t*>(buf_ngrids.ptr);
    double* ptr_kx = static_cast<double*>(buf_kx.ptr);
    double* ptr_ky = static_cast<double*>(buf_ky.ptr);
    double* ptr_kz = static_cast<double*>(buf_kz.ptr);

    DoInterlace<T>(ptr_c1, ptr_c2, ptr_H, ptr_ngrids, ptr_kx, ptr_ky, ptr_kz, nthreads);
}


PYBIND11_MODULE(mesh_pybind, m){
    m.doc() = "Mesh based on pybind11";

    // 绑定 float 版本
    m.def("run_mesh_float", &run_mesh<float>, 
          R"pbdoc(
            Run mesh interpolation with float precision.
            
            Parameters
            ----------
            pos : numpy.ndarray (np, 3)
                Particle positions.
            field : numpy.ndarray
                Output field grid (must be contiguous and writable).
            weights : numpy.ndarray
                Particle weights. Default None
            values : numpy.ndarray
                Particle values. Default None
            boxSize : numpy.ndarray (3,)
                Size of the simulation box.
            ngrids : numpy.ndarray (3,)
                Number of grid points in each dimension.
            shift : float
                Shift parameter.
            type: int
                0: NGP, 1: CIC, 2: TSC, 3: PCS
            nthreads : int
                Number of threads to use.
          )pbdoc",
          py::arg("pos"), py::arg("field"), py::arg("weights") = py::none(), py::arg("values") = py::none(),
          py::arg("boxSize"), py::arg("ngrids"), py::arg("shift"), py::arg("type"), py::arg("nthreads"));

    // 绑定 double 版本
    m.def("run_mesh_double", &run_mesh<double>, 
          R"pbdoc(
            Run mesh interpolation with double precision.
            Same arguments as run_mesh_float but with float64 arrays.
          )pbdoc",
          py::arg("pos"), py::arg("field"), py::arg("weights") = py::none(), py::arg("values") = py::none(),
          py::arg("boxSize"), py::arg("ngrids"), py::arg("shift"), py::arg("type"), py::arg("nthreads"));
    
    m.def("do_compensation_float", &do_compensation<float>, 
          R"pbdoc(
            Run compensation mesh interpolation with float precision.
            
            Parameters
            ----------
            complex_field : numpy.ndarray
                Complex field grid (must be contiguous and writable).
            ngrids : numpy.ndarray (3,)
                Number of grid points in each dimension.
            kx_array : numpy.ndarray (N,)
                kx values.
            ky_array : numpy.ndarray (N,)
                ky values.
            kz_array : numpy.ndarray (N,)
                kz values.
            type: int
                0: NGP, 1: CIC, 2: TSC, 3: PCS
            nthreads : int
                Number of threads to use.
          )pbdoc",
          py::arg("complex_field"), py::arg("ngrids"), py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"),
          py::arg("type"), py::arg("nthreads"));
    
    m.def("do_compensation_double", &do_compensation<double>, 
          R"pbdoc(
            Run compensation mesh interpolation with double precision.
            Same arguments as do_compensation_float but with float64 arrays.
          )pbdoc",
          py::arg("complex_field"), py::arg("ngrids"), py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"),
          py::arg("type"), py::arg("nthreads"));

    m.def("do_compensation_interlaced_float", &do_compensation_interlaced<float>, 
          R"pbdoc(
            Run compensation mesh interpolation with float precision.
            
            Parameters
            ----------
            complex_field : numpy.ndarray
                Complex field grid (must be contiguous and writable).
            ngrids : numpy.ndarray (3,)
                Number of grid points in each dimension.
            kx_array : numpy.ndarray (N,)
                kx values.
            ky_array : numpy.ndarray (N,)
                ky values.
            kz_array : numpy.ndarray (N,)
                kz values.
            p : int
                Interpolation order. 2 for CIC; 3 for TSC; 4 for PCS
            nthreads : int
                Number of threads to use.
          )pbdoc",
          py::arg("complex_field"), py::arg("ngrids"), py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"),
          py::arg("p"), py::arg("nthreads"));
    
    m.def("do_compensation_interlaced_double", &do_compensation_interlaced<double>, 
          R"pbdoc(
            Run compensation mesh interpolation with double precision.
            Same arguments as do_compensation_float but with float64 arrays.
          )pbdoc",
          py::arg("complex_field"), py::arg("ngrids"), py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"),
          py::arg("p"), py::arg("nthreads"));

    m.def("do_interlace_float", &do_interlace<float>, 
          R"pbdoc(
            Run interlace mesh interpolation with float precision.
            
            Parameters
            ----------
            c1 : numpy.ndarray
                First complex field grid (must be contiguous and writable).
            c2 : numpy.ndarray
                Second complex field grid (must be contiguous and writable).
            H : numpy.ndarray (3,)
                The size of the grids
            ngrids : numpy.ndarray (3,)
                Number of grid points in each dimension.
            kx_array : numpy.ndarray (N,)
                kx values.
            ky_array : numpy.ndarray (N,)
                ky values.
            kz_array : numpy.ndarray (N,)
                kz values.)
            nthreads : int
                Number of threads to use.
          )pbdoc",
          py::arg("c1"), py::arg("c2"), py::arg("H"), py::arg("ngrids"), py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"),
          py::arg("nthreads"));

    m.def("do_interlace_double", &do_interlace<double>, 
          R"pbdoc(
            Run interlace mesh interpolation with double precision.
            Same arguments as do_interlace_float but with float64 arrays.
          )pbdoc",
          py::arg("c1"), py::arg("c2"), py::arg("H"), py::arg("ngrids"), py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"),
          py::arg("nthreads"));
} 