#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

#include "AP.hpp"

namespace py = pybind11;

py::array_t<double> mapping_smudata_dense_pybind(
    py::array_t<double> smutabstd,
    size_t sbin_dense, size_t mubin_dense,
    size_t sbin_sparse, size_t mubin_sparse,
    double Hz_f, double Hz_m, double DA_f, double DA_m,
    double smin_all, double smax_all, double mumin_all, double mumax_all,
    double smin_mapping, double smax_mapping,
    int nthreads = 1
) {
    // Get buffer info for input array
    auto buf = smutabstd.request();
    
    // Validate input shape
    if (buf.ndim != 3) {
        throw std::runtime_error("Input array must be 3-dimensional");
    }
    if (buf.shape[0] != static_cast<py::ssize_t>(sbin_dense) || 
        buf.shape[1] != static_cast<py::ssize_t>(mubin_dense)) {
        throw std::runtime_error("Input array shape does not match sbin_dense and mubin_dense");
    }
    
    // Get element size from third dimension
    size_t element_size = buf.shape[2];
    
    // Convert to flat vector for C++ function
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<double> smutabstd_vec(ptr, ptr + buf.size);
    
    // Call C++ function
    std::vector<double> result_vec = mapping_smudata_dense(
        smutabstd_vec,
        sbin_dense, mubin_dense,
        sbin_sparse, mubin_sparse,
        Hz_f, Hz_m, DA_f, DA_m,
        smin_all, smax_all, mumin_all, mumax_all,
        smin_mapping, smax_mapping,
        nthreads
    );
    
    // Create output numpy array
    py::array_t<double> result = py::array_t<double>(
        {sbin_sparse, mubin_sparse, element_size},
        result_vec.data()
    );
    
    return result;
}

PYBIND11_MODULE(AP_pybind, m) {
    m.doc() = "AP effect dense-to-sparse conversion based on pybind11";
    
    m.def("mapping_smudata_dense", &mapping_smudata_dense_pybind,
          R"pbdoc(
            Map dense (s, mu) grid data to sparse grid with AP effect conversion.
            
            Parameters
            ----------
            smutabstd : numpy.ndarray (sbin_dense, mubin_dense, element_size)
                Input data array.
            sbin_dense : int
                Number of s bins in dense grid.
            mubin_dense : int
                Number of mu bins in dense grid.
            sbin_sparse : int
                Number of s bins in sparse grid.
            mubin_sparse : int
                Number of mu bins in sparse grid.
            Hz_f : float
                Fiducial Hubble parameter.
            Hz_m : float
                Modified Hubble parameter.
            DA_f : float
                Fiducial angular diameter distance.
            DA_m : float
                Modified angular diameter distance.
            smin_all : float
                Minimum s value in full grid.
            smax_all : float
                Maximum s value in full grid.
            mumin_all : float
                Minimum mu value (typically 0.0).
            mumax_all : float
                Maximum mu value (typically 1.0).
            smin_mapping : float
                Minimum s value for mapping.
            smax_mapping : float
                Maximum s value for mapping.
            nthreads : int, optional
                Number of OpenMP threads. Default 1.
            
            Returns
            -------
            numpy.ndarray (sbin_sparse, mubin_sparse, element_size)
                Mapped data array.
          )pbdoc",
          py::arg("smutabstd"), py::arg("sbin_dense"), py::arg("mubin_dense"),
          py::arg("sbin_sparse"), py::arg("mubin_sparse"),
          py::arg("Hz_f"), py::arg("Hz_m"), py::arg("DA_f"), py::arg("DA_m"),
          py::arg("smin_all"), py::arg("smax_all"), py::arg("mumin_all"), py::arg("mumax_all"),
          py::arg("smin_mapping"), py::arg("smax_mapping"), py::arg("nthreads") = 1);
}
