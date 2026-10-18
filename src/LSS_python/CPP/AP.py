"""Compatibility adapter for the ratio-only C++ dense AP kernels."""

from .._ap_dense import validate_mapping


def mapping_smudata_dense_cpp(smutabstd, cosmos_tuple, smu_bin_tuple,
                              smu_all_bound_tuple, smu_mapping_bound_tuple,
                              nthreads=1, *, convert_algorithm="polygon"):
    """Map normalized channels using uniform mapped (s, mu) coordinate areas.

    The historical tuple formats and positional thread argument are preserved.
    All four mapping bounds are respected; target cells touching that window
    use AP geometry and cells outside it use exact identity rebinning.
    The algorithm selects polygon areas, adaptive Gauss8 curved areas, or
    analytic curved areas.
    """
    values = validate_mapping(smutabstd, cosmos_tuple, smu_bin_tuple,
                              smu_all_bound_tuple, smu_mapping_bound_tuple,
                              convert_algorithm, nthreads)
    from .lib.AP_pybind import _map_counts

    return _map_counts(*values, nthreads=int(nthreads), convert_algorithm=convert_algorithm)
