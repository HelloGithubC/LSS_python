"""Parity checks for the tuple-compatible C++ and Numba AP adapters."""

import numpy as np
import pytest

from LSS_python._AP_core import mapping_smudata_dense as mapping_jit


@pytest.fixture
def conversion_args():
    rng = np.random.default_rng(20261015)
    # Intentionally strided and non-float to check adapter normalization.
    backing = rng.random((12, 10, 3)).astype(np.float32)
    data = backing[::2, ::2]
    cosmos = (100., 95., 1000., 1050.)
    bins = (6, 7, 5, 6)
    full_bounds = (4., 16., .1, .9)
    mapping_bounds = (5., 14., .2, .8)
    return data, cosmos, bins, full_bounds, mapping_bounds


@pytest.mark.parametrize("algorithm", ["polygon", "curved_gauss8", "curved_analytic"])
def test_jit_tuple_api_accepts_keyword_only_algorithm(conversion_args, algorithm):
    data, cosmos, bins, full_bounds, mapping_bounds = conversion_args
    result = mapping_jit(
        data, cosmos, bins, full_bounds, mapping_bounds,
        convert_algorithm=algorithm,
    )
    assert result.shape == (bins[1], bins[3], data.shape[2])
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("algorithm", ["polygon", "curved_gauss8", "curved_analytic"])
def test_cpp_and_jit_tuple_adapters_match_and_keep_nthreads(conversion_args, algorithm):
    cpp_module = pytest.importorskip("LSS_python.CPP.AP")
    data, cosmos, bins, full_bounds, mapping_bounds = conversion_args
    jit = mapping_jit(data, cosmos, bins, full_bounds, mapping_bounds,
                      convert_algorithm=algorithm)
    cpp_one = cpp_module.mapping_smudata_dense_cpp(
        data, cosmos, bins, full_bounds, mapping_bounds,
        nthreads=1, convert_algorithm=algorithm,
    )
    cpp_many = cpp_module.mapping_smudata_dense_cpp(
        data, cosmos, bins, full_bounds, mapping_bounds,
        nthreads=2, convert_algorithm=algorithm,
    )
    np.testing.assert_allclose(cpp_one, jit, rtol=2e-11, atol=2e-12)
    np.testing.assert_array_equal(cpp_one, cpp_many)


@pytest.mark.parametrize("algorithm", ["curved_gauss8", "curved_analytic"])
@pytest.mark.parametrize("mapping_bounds", [(0., 3., 0., 1.), (.65, 2.4, .2, .8)])
def test_curved_cpp_numba_parity_on_small_full_and_mixed_windows(algorithm, mapping_bounds):
    cpp_module = pytest.importorskip("LSS_python.CPP.AP")
    rng = np.random.default_rng(20261018)
    data = rng.random((3, 4, 2))
    cosmos = (100., 95., 1000., 1050.)
    bins = (3, 5, 4, 6)
    full_bounds = (0., 3., 0., 1.)
    jit = mapping_jit(data, cosmos, bins, full_bounds, mapping_bounds,
                      convert_algorithm=algorithm)
    cpp_one = cpp_module.mapping_smudata_dense_cpp(
        data, cosmos, bins, full_bounds, mapping_bounds,
        nthreads=1, convert_algorithm=algorithm,
    )
    cpp_many = cpp_module.mapping_smudata_dense_cpp(
        data, cosmos, bins, full_bounds, mapping_bounds,
        nthreads=2, convert_algorithm=algorithm,
    )
    np.testing.assert_allclose(cpp_one, jit, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(cpp_one, cpp_many)


@pytest.mark.parametrize("algorithm,error", [
    ("unrecognized", ValueError),
])
def test_low_level_algorithm_validation(algorithm, error, conversion_args):
    data, cosmos, bins, full_bounds, mapping_bounds = conversion_args
    with pytest.raises(error):
        mapping_jit(
            data, cosmos, bins, full_bounds, mapping_bounds,
            convert_algorithm=algorithm,
        )


def test_native_unknown_algorithm_is_rejected(conversion_args):
    from LSS_python.CPP.lib.AP_pybind import _map_counts
    with pytest.raises(ValueError, match="convert_algorithm"):
        _map_counts(conversion_args[0], 1., 1., (2, 2), (0., 2.), (0., 1.),
                    (0., 2., 0., 1.), convert_algorithm="unknown")
