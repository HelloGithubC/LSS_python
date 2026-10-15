#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "AP.hpp"

namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;

void validate_algorithm(const std::string& name) {
    if (name=="polygon") return;
    if (name=="curved_gauss8" || name=="curved_analytic") {
        PyErr_SetString(PyExc_NotImplementedError, "Requested curved AP algorithm is reserved and not implemented");
        throw py::error_already_set();
    }
    throw py::value_error("Unknown convert_algorithm; expected polygon, curved_gauss8 or curved_analytic");
}

py::array_t<double> map_counts(Array data, double a, double b, std::array<int,2> target_shape,
                              std::array<double,2> sb, std::array<double,2> mb,
                              std::array<double,4> window, int nthreads,
                              const std::string& algorithm) {
    validate_algorithm(algorithm);
    const auto input=data.request();
    if (input.ndim!=3 || input.shape[0]<1 || input.shape[1]<1 || input.shape[2]<1 ||
        input.shape[0]>std::numeric_limits<int>::max() || input.shape[1]>std::numeric_limits<int>::max())
        throw py::value_error("Input must have a nonempty (source_s, source_mu, channels) shape within native index limits");
    if (!std::isfinite(a) || !std::isfinite(b) || a<=0 || b<=0)
        throw std::runtime_error("Invalid AP mapping inputs: AP factors must be finite and positive");
    if (target_shape[0]<=0 || target_shape[1]<=0 || nthreads<=0)
        throw py::value_error("Target dimensions and nthreads must be positive");
    for (const double x:sb) if (!std::isfinite(x)) throw py::value_error("Grid bounds must be finite");
    for (const double x:mb) if (!std::isfinite(x)) throw py::value_error("Grid bounds must be finite");
    for (const double x:window) if (!std::isfinite(x)) throw py::value_error("Mapping bounds must be finite");
    if (sb[0]<0 || sb[1]<=sb[0] || mb[0]<0 || mb[1]>1 || mb[1]<=mb[0] ||
        window[1]<window[0] || window[2]<0 || window[3]>1 || window[3]<window[2])
        throw py::value_error("Invalid grid or mapping bounds");
    // Reject unrepresentable inverse geometry before entering the parallel region.
    const lss_ap::Geometry geometry(a,b,input.shape[0],input.shape[1],sb,mb);
    for (const double u:mb) {
        const auto point=geometry.inverse(sb[1],u);
        if (!std::isfinite(point.s) || !std::isfinite(point.mu))
            throw std::runtime_error("Invalid AP mapping inputs: non-finite inverse geometry");
    }
    const auto* source=static_cast<const double*>(input.ptr);
    for (py::ssize_t i=0; i<input.size; ++i)
        if (!std::isfinite(source[i])) throw py::value_error("Pair counts must be finite");
    py::array_t<double> result({py::ssize_t(target_shape[0]),py::ssize_t(target_shape[1]),input.shape[2]});
    auto* output=result.mutable_data();
    std::fill(output,output+result.size(),0.);
    {
        py::gil_scoped_release release;
        lss_ap::map_counts(source,output,input.shape[0],input.shape[1],target_shape[0],target_shape[1],
                           input.shape[2],a,b,sb,mb,window,nthreads);
    }
    return result;
}

py::array_t<double> mapping_smudata_dense(Array data, int ns, int nm, int nt_s, int nt_mu,
                                         double hf, double hm, double daf, double dam,
                                         double sl, double sh, double ml, double mh,
                                         double wsl, double wsh, int nthreads,
                                         const std::string& algorithm) {
    validate_algorithm(algorithm);
    if (data.ndim()!=3 || data.shape(0)!=ns || data.shape(1)!=nm)
        throw py::value_error("Input shape does not match the declared source grid");
    for (const double x:{hf,hm,daf,dam})
        if (!std::isfinite(x) || x<=0)
            throw std::runtime_error("Invalid AP mapping inputs: distances/rates must be finite and positive");
    return map_counts(data,dam/daf,hf/hm,{nt_s,nt_mu},{sl,sh},{ml,mh},{wsl,wsh,0.,1.},nthreads,algorithm);
}

PYBIND11_MODULE(AP_pybind,m) {
    m.doc()="Dense AP count conversion with polygon overlap weights";
    m.def("_map_counts",&map_counts,py::arg("data"),py::arg("alpha_perp"),py::arg("alpha_parallel"),
          py::arg("target_shape"),py::arg("s_bounds"),py::arg("mu_bounds"),py::arg("mapping_bounds"),
          py::arg("nthreads")=1,py::kw_only(),py::arg("convert_algorithm")="polygon");
    m.def("mapping_smudata_dense",&mapping_smudata_dense,
          "Compatibility entry point; the numerical kernel uses AP scale ratios only.",
          py::arg("smutabstd"),py::arg("sbin_dense"),py::arg("mubin_dense"),
          py::arg("sbin_sparse"),py::arg("mubin_sparse"),py::arg("Hz_f"),py::arg("Hz_m"),
          py::arg("DA_f"),py::arg("DA_m"),py::arg("smin_all"),py::arg("smax_all"),
          py::arg("mumin_all"),py::arg("mumax_all"),py::arg("smin_mapping"),py::arg("smax_mapping"),
          py::arg("nthreads")=1,py::kw_only(),py::arg("convert_algorithm")="polygon");
}
