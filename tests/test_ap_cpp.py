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


def test_jit_tuple_api_accepts_keyword_only_polygon_algorithm(conversion_args):
    data, cosmos, bins, full_bounds, mapping_bounds = conversion_args
    result = mapping_jit(
        data, cosmos, bins, full_bounds, mapping_bounds,
        convert_algorithm="polygon",
    )
    assert result.shape == (bins[1], bins[3], data.shape[2])
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


def test_cpp_and_jit_tuple_adapters_match_and_keep_nthreads(conversion_args):
    cpp_module = pytest.importorskip("LSS_python.CPP.AP")
    data, cosmos, bins, full_bounds, mapping_bounds = conversion_args
    jit = mapping_jit(data, cosmos, bins, full_bounds, mapping_bounds,
                      convert_algorithm="polygon")
    cpp_one = cpp_module.mapping_smudata_dense_cpp(
        data, cosmos, bins, full_bounds, mapping_bounds,
        nthreads=1, convert_algorithm="polygon",
    )
    cpp_many = cpp_module.mapping_smudata_dense_cpp(
        data, cosmos, bins, full_bounds, mapping_bounds,
        nthreads=2, convert_algorithm="polygon",
    )
    np.testing.assert_allclose(cpp_one, jit, rtol=2e-11, atol=2e-12)
    np.testing.assert_array_equal(cpp_one, cpp_many)


@pytest.mark.parametrize("algorithm,error", [
    ("curved_gauss8", NotImplementedError),
    ("curved_analytic", NotImplementedError),
    ("unrecognized", ValueError),
])
def test_low_level_algorithm_validation(algorithm, error, conversion_args):
    data, cosmos, bins, full_bounds, mapping_bounds = conversion_args
    with pytest.raises(error):
        mapping_jit(
            data, cosmos, bins, full_bounds, mapping_bounds,
            convert_algorithm=algorithm,
        )
