#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>
#include <iostream>

#include "fftpower.hpp"

namespace py = pybind11;
const int ndim = 3;

template <typename T>
void deal_ps_3d(py::array_t<std::complex<T>> complex_field, 
                py::object kernel,  
                T ps_3d_factor, T shotnoise, int nthreads)
{
    if (!(complex_field.flags() & py::array::c_style) || !complex_field.writeable())  {
        throw std::runtime_error("Field must be C-contiguous and writeable");
    }

    auto complex_field_buf = complex_field.request();
    size_t ngrids[ndim];
    for (int i = 0; i < ndim; i++) {
        ngrids[i] = complex_field_buf.shape[i];
    }
    
    std::complex<T>* complex_field_ptr = static_cast<std::complex<T>*>(complex_field_buf.ptr);
    
    std::complex<T>* kernel_ptr = nullptr;
    if (!kernel.is_none()) {
        if (!py::isinstance<py::array_t<std::complex<T>>>(kernel)) {
            throw std::runtime_error("Kernel must be a numpy complex array");
        }
        auto kernel_array = kernel.cast<py::array_t<std::complex<T>>>();
        auto kernel_buf = kernel_array.request();
        kernel_ptr = static_cast<std::complex<T>*>(kernel_buf.ptr);
    }
    
    DealPS3D(complex_field_ptr, kernel_ptr, ngrids, ps_3d_factor, shotnoise, nthreads);
}

template <typename T>
void cal_ps(py::array_t<std::complex<T>> ps_3d, 
            py::array_t<double> kx_array, 
            py::array_t<double> ky_array, 
            py::array_t<double> kz_array, 
            py::array_t<double> k_array, 
            py::object mu_array, 
            py::array_t<std::complex<double>> Pkmu, 
            py::array_t<size_t> count, 
            py::array_t<double> k_mesh, 
            py::object mu_mesh,
            int nthreads, bool k_logarithmic)
{
    auto ps_3d_buf = ps_3d.request();
    auto kx_array_buf = kx_array.request();
    auto ky_array_buf = ky_array.request();
    auto kz_array_buf = kz_array.request();
    auto k_array_buf = k_array.request();
    auto Pkmu_buf = Pkmu.request();
    auto count_buf = count.request();
    auto k_mesh_buf = k_mesh.request();

    unsigned int kbin = Pkmu_buf.shape[0];
    unsigned int mubin = Pkmu_buf.shape[1];

    const int ndim = 3;
    size_t ngrids[ndim];
    for (int i = 0; i < ndim; i++) {
        ngrids[i] = ps_3d_buf.shape[i];
    }

    double* mu_array_ptr = nullptr;
    if (!mu_array.is_none())
    {
        if (!py::isinstance<py::array_t<double>>(mu_array)) {
            throw std::runtime_error("mu_array must be a numpy double array");
        }
        auto mu_array_need = mu_array.cast<py::array_t<double>>();
        auto mu_array_buf = mu_array_need.request();
        mu_array_ptr = static_cast<double*>(mu_array_buf.ptr);
        mubin = mu_array_buf.shape[0] - 1;
    }
    
    double* mu_mesh_ptr = nullptr;
    if (!mu_mesh.is_none())
    {
        if (!py::isinstance<py::array_t<double>>(mu_mesh)) {
            throw std::runtime_error("mu_mesh must be a numpy double array");
        }
        auto mu_mesh_need = mu_mesh.cast<py::array_t<double>>();
        auto mu_mesh_buf = mu_mesh_need.request();
        mu_mesh_ptr = static_cast<double*>(mu_mesh_buf.ptr);
    }
    
    std::complex<T>* ps_3d_ptr = static_cast<std::complex<T>*>(ps_3d_buf.ptr);
    double* kx_array_ptr = static_cast<double*>(kx_array_buf.ptr);
    double* ky_array_ptr = static_cast<double*>(ky_array_buf.ptr);
    double* kz_array_ptr = static_cast<double*>(kz_array_buf.ptr);
    double* k_array_ptr = static_cast<double*>(k_array_buf.ptr);
    std::complex<double>* Pkmu_ptr = static_cast<std::complex<double>*>(Pkmu_buf.ptr);
    size_t* count_ptr = static_cast<size_t*>(count_buf.ptr);
    double* k_mesh_ptr = static_cast<double*>(k_mesh_buf.ptr);

    CalculatePS<T>(ps_3d_ptr, ngrids, kx_array_ptr, ky_array_ptr, kz_array_ptr, k_array_ptr, mu_array_ptr, kbin, mubin, Pkmu_ptr, count_ptr, k_mesh_ptr, mu_mesh_ptr, nthreads, k_logarithmic);
}

template <typename T>
void cal_ps_2d_from_mesh( 
    py::array_t<std::complex<T>> complex_field, py::object kernel, 
    py::array_t<std::complex<T>> ps_2d, py::array_t<double> k_2d, py::array_t<size_t> modes_2d,
    py::array_t<double> k_perp_edge, py::array_t<double> k_parallel_edge,
    py::array_t<double> kx_array, py::array_t<double> ky_array, py::array_t<double> kz_array,
    T ps_3d_factor, T shotnoise, int nthreads)
{
    // 1. 获取所有数组的 buffer info
    auto buf_complex_field = complex_field.request();
    auto buf_ps_2d = ps_2d.request();
    auto buf_k_2d = k_2d.request();
    auto buf_modes_2d = modes_2d.request();
    auto buf_k_perp_edge = k_perp_edge.request();
    auto buf_k_parallel_edge = k_parallel_edge.request();
    auto buf_kx_array = kx_array.request();
    auto buf_ky_array = ky_array.request();
    auto buf_kz_array = kz_array.request();

    // 2. 提取指针
    std::complex<T>* ptr_complex_field = static_cast<std::complex<T>*>(buf_complex_field.ptr);
    std::complex<T>* ptr_kernel = nullptr; // 初始化为空指针
    if (!kernel.is_none()) {
        // 确保传入的是正确的 numpy 数组类型
        if (!py::isinstance<py::array_t<std::complex<T>>>(kernel)) {
            throw std::runtime_error("kernel must be a numpy array of complex type matching complex_field, or None");
        }
        // 重新获取非空 kernel 的 buffer info 和指针
        ptr_kernel = static_cast<std::complex<T>*>(kernel.cast<py::array_t<std::complex<T>>>().request().ptr);
    }
    std::complex<T>* ptr_ps_2d = static_cast<std::complex<T>*>(buf_ps_2d.ptr);
    double* ptr_k_2d = static_cast<double*>(buf_k_2d.ptr);
    size_t* ptr_modes_2d = static_cast<size_t*>(buf_modes_2d.ptr);
    double* ptr_k_perp_edge = static_cast<double*>(buf_k_perp_edge.ptr);
    double* ptr_k_parallel_edge = static_cast<double*>(buf_k_parallel_edge.ptr);
    const double* ptr_kx_array = static_cast<const double*>(buf_kx_array.ptr);
    const double* ptr_ky_array = static_cast<const double*>(buf_ky_array.ptr);
    const double* ptr_kz_array = static_cast<const double*>(buf_kz_array.ptr);

    size_t k_perp_bin = buf_k_perp_edge.shape[0] - 1uL;
    size_t k_parallel_bin = buf_k_parallel_edge.shape[0] - 1uL;
    size_t ngrids[ndim];
    for (int i = 0; i < ndim; i++) {
        ngrids[i] = buf_complex_field.shape[i];
    }

    // 4. 调用核心 C++ 函数
    CalPS2D<T>(
        ptr_complex_field, ptr_kernel, ptr_ps_2d, ptr_k_2d, ptr_modes_2d,
        ptr_k_perp_edge, ptr_k_parallel_edge, k_perp_bin, k_parallel_bin, ptr_kx_array, ptr_ky_array, ptr_kz_array, ngrids,
        ps_3d_factor, shotnoise, nthreads
    );
}

template <typename T>
void cal_ps_from_ps_2d(
    py::array_t<std::complex<T>> ps_2d, py::array_t<double> k_2d,
    py::array_t<double> k_out_2d, py::object mu_out_2d,
    py::array_t<std::complex<T>> ps_kmu, py::array_t<size_t> modes,
    py::array_t<double> k_edge, py::object mu_edge, int nthreads)
{
    auto buf_ps_2d = ps_2d.request();
    auto buf_k_2d = k_2d.request();
    auto buf_k_out_2d = k_out_2d.request();
    auto buf_ps_kmu = ps_kmu.request();
    auto buf_modes = modes.request();
    auto buf_k_edge = k_edge.request();

    size_t k_perp_bin = buf_ps_2d.shape[0];
    size_t k_parallel_bin = buf_ps_2d.shape[1];
    size_t kbin = buf_k_edge.shape[0] - 1;

    const std::complex<T>* ptr_ps_2d = static_cast<const std::complex<T>*>(buf_ps_2d.ptr);
    const double* ptr_k_2d = static_cast<const double*>(buf_k_2d.ptr);
    double* ptr_k_out_2d = static_cast<double*>(buf_k_out_2d.ptr);
    std::complex<T>* ptr_ps_kmu = static_cast<std::complex<T>*>(buf_ps_kmu.ptr);
    size_t* ptr_modes = static_cast<size_t*>(buf_modes.ptr);
    const double* ptr_k_edge = static_cast<const double*>(buf_k_edge.ptr);

    // Handle mu_out_2d and mu_edge parameters
    double* ptr_mu_out_2d = nullptr;
    size_t mubin = 1;
    const double* ptr_mu_edge = nullptr;

    if (!mu_out_2d.is_none())
    {
        auto buf_mu_out_2d = mu_out_2d.cast<py::array_t<double>>().request();
        ptr_mu_out_2d = static_cast<double*>(buf_mu_out_2d.ptr);
    }

    if (!mu_edge.is_none())
    {
        auto buf_mu_edge = mu_edge.cast<py::array_t<double>>().request();
        ptr_mu_edge = static_cast<const double*>(buf_mu_edge.ptr);
        mubin = buf_mu_edge.shape[0] - 1;
    }

    CalPSFromPS2D<T>(ptr_ps_2d, ptr_k_2d, ptr_k_out_2d, ptr_mu_out_2d, ptr_ps_kmu, ptr_modes,
                     ptr_k_edge, ptr_mu_edge, kbin, mubin, k_perp_bin, k_parallel_bin, nthreads);
}

PYBIND11_MODULE(fftpower_pybind, m){
    m.doc() = "FFTPower based on pybind11";

    m.def("deal_ps_3d_float", &deal_ps_3d<float>, 
    R"pbdoc(
        deal_ps_3d function with float precision

    Parameters
    ----------
    complex_field : numpy.ndarray 
        complex_field
    kernel : numpy.ndarray or None
        kernel
    ps_3d_factor : float
        ps_3d_factor
    shotnoise : float
        shotnoise
    nthreads : int
        nthreads
    )pbdoc",
    py::arg("complex_field"), py::arg("kernel") = py::none(), py::arg("ps_3d_factor"), py::arg("shotnoise"), py::arg("nthreads"));

    m.def("deal_ps_3d_double", &deal_ps_3d<double>, 
    R"pbdoc(
        deal_ps_3d function with double precision
        Same arguments as deal_ps_3d_float but with double precision
    )pbdoc",
    py::arg("complex_field"), py::arg("kernel") = py::none(), py::arg("ps_3d_factor"), py::arg("shotnoise"), py::arg("nthreads"));

    m.def("cal_ps_float", &cal_ps<float>, 
    R"pbdoc(
        cal_ps function with float precision

    Parameters
    ----------
    ps_3d : numpy.ndarray 
        ps_3d
    kx_array : numpy.ndarray 
        kx_array
    ky_array : numpy.ndarray 
        ky_array
    kz_array : numpy.ndarray 
        kz_array
    k_array : numpy.ndarray 
        k_array
    mu_array : numpy.ndarray 
        mu_array. Default None
    Pkmu : numpy.ndarray 
        Pkmu
    count : numpy.ndarray
        count
    k_mesh: numpy.ndarray
        k_mesh
    mu_mesh: numpy.ndarray
        mu_mesh
    nthreads: int
        nthreads
    k_logarithmic: bool
        k_logarithmic
    )pbdoc",
    py::arg("ps_3d"), py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"), py::arg("k_array"), py::arg("mu_array") = py::none(), py::arg("Pkmu"), py::arg("count"), py::arg("k_mesh"), py::arg("mu_mesh") = py::none(), py::arg("nthreads"), py::arg("k_logarithmic"));

    m.def("cal_ps_double", &cal_ps<double>,
    R"pbdoc(
        cal_ps function with float precision
        Same arguments as cal_ps_float but with double precision
    )pbdoc",
    py::arg("ps_3d"), py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"), py::arg("k_array"), py::arg("mu_array") = py::none(), py::arg("Pkmu"), py::arg("count"), py::arg("k_mesh"), py::arg("mu_mesh") = py::none(), py::arg("nthreads"), py::arg("k_logarithmic"));

    m.def("cal_ps_2d_from_mesh_float", &cal_ps_2d_from_mesh<float>,
          R"pbdoc(
            Calculate 2D Power Spectrum from 3D field (float precision).

            Args:
                complex_field (np.ndarray): Input complex field array.
                kernel (np.ndarray, optional): Optional complex kernel array, or None.
                ps_2d (np.ndarray): Output array for 2D power spectrum.
                k_2d (complex): A complex number parameter.
                modes_2d (np.ndarray): Array for mode counts.
                k_perp_edge (np.ndarray): Array for perpendicular k edges.
                k_parallel_edge (np.ndarray): Array for parallel k edges.
                kx_array (np.ndarray): Array of kx values.
                ky_array (np.ndarray): Array of ky values.
                kz_array (np.ndarray): Array of kz values.
                ngrids (np.ndarray): Array of grid sizes.
                ps_3d_factor (float): Factor for 3D power spectrum.
                shotnoise (float): Shot noise value.
                nthreads (int): Number of threads for computation.
                k_perp_logarithmic (bool): Whether k_perp bins are logarithmic.
          )pbdoc",
          py::arg("complex_field"), py::arg("kernel") = py::none(), py::arg("ps_2d"), py::arg("k_2d"),
          py::arg("modes_2d"), py::arg("k_perp_edge"), py::arg("k_parallel_edge"),
          py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"), 
          py::arg("ps_3d_factor"), py::arg("shotnoise"), py::arg("nthreads"));

    m.def("cal_ps_2d_from_mesh_double", &cal_ps_2d_from_mesh<double>,
          R"pbdoc(
            Calculate 2D Power Spectrum from 3D field (double precision).
            Same arguments as cal_ps2d_float but with double precision
          )pbdoc",
          py::arg("complex_field"), py::arg("kernel") = py::none(), py::arg("ps_2d"), py::arg("k_2d"),
          py::arg("modes_2d"), py::arg("k_perp_edge"), py::arg("k_parallel_edge"),
          py::arg("kx_array"), py::arg("ky_array"), py::arg("kz_array"),
          py::arg("ps_3d_factor"), py::arg("shotnoise"), py::arg("nthreads"));

    m.def("cal_ps_from_ps_2d_float", &cal_ps_from_ps_2d<float>,
          R"pbdoc(
            Calculate power spectrum from 2D power spectrum (float precision).

            Args:
                ps_2d (np.ndarray): Input 2D power spectrum array.
                k_2d (np.ndarray): 2D k-value array with shape (k_perp_bin, k_parallel_bin, 2).
                k_out_2d (np.ndarray): Output array for k values.
                mu_out_2d (np.ndarray, optional): Output array for mu values. Default None.
                ps_kmu (np.ndarray): Output array for power spectrum.
                modes (np.ndarray): Output array for mode counts.
                k_edge (np.ndarray): k bin edges array.
                mu_edge (np.ndarray, optional): mu bin edges array. Default None.
                nthreads (int): Number of threads for computation.
          )pbdoc",
          py::arg("ps_2d"), py::arg("k_2d"), py::arg("k_out_2d"), py::arg("mu_out_2d") = py::none(),
          py::arg("ps_kmu"), py::arg("modes"), py::arg("k_edge"), py::arg("mu_edge") = py::none(),
          py::arg("nthreads"));

    m.def("cal_ps_from_ps_2d_double", &cal_ps_from_ps_2d<double>,
          R"pbdoc(
            Calculate power spectrum from 2D power spectrum (double precision).
            Same arguments as cal_ps_from_ps_2d_float but with double precision
          )pbdoc",
          py::arg("ps_2d"), py::arg("k_2d"), py::arg("k_out_2d"), py::arg("mu_out_2d") = py::none(),
          py::arg("ps_kmu"), py::arg("modes"), py::arg("k_edge"), py::arg("mu_edge") = py::none(),
          py::arg("nthreads"));
}
