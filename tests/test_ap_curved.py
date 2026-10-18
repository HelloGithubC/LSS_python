"""Independent quadrature checks for curved AP count mapping."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq


def _curved_reference(data, at, ap, target_shape, s_bounds, mu_bounds, window):
    """Integrate the exact curved image overlap in (s, mu) coordinates.

    The implementation is deliberately independent of production geometry:
    it integrates the mapped-coordinate radial overlap and locates
    every radial kink with a scalar root finder.
    """
    ns, nm, channels = data.shape
    ts, tm = target_shape
    s_edges = np.linspace(*s_bounds, ns + 1)
    mu_edges = np.linspace(*mu_bounds, nm + 1)
    out_s = np.linspace(*s_bounds, ts + 1)
    out_mu = np.linspace(*mu_bounds, tm + 1)
    result = np.zeros((ts, tm, channels), dtype=np.float64)

    def to_v(mu):
        return ap * mu / math.hypot(at * math.sqrt(max(0., 1. - mu * mu)), ap * mu)

    def radius(v):
        return 1. / math.hypot(math.sqrt(max(0., (1. - v) * (1. + v))) / at,
                               v / ap)

    for i in range(ns):
        slo, shi = s_edges[i:i + 2]
        for j in range(nm):
            v0, v1 = to_v(mu_edges[j]), to_v(mu_edges[j + 1])
            mapped_source_area = (shi - slo) * quad(radius, v0, v1,
                                                    epsabs=2e-14, epsrel=2e-13,
                                                    limit=80)[0]
            for oi in range(ts):
                for oj in range(tm):
                    # Dispatch is per complete target bin. A bin touching the
                    # AP window uses AP geometry; every other bin uses identity.
                    use_ap = (out_s[oi] <= window[1] and out_s[oi + 1] >= window[0]
                              and out_mu[oj] <= window[3] and out_mu[oj + 1] >= window[2])
                    if not use_ap:
                        overlap_s = max(0., min(shi, out_s[oi + 1]) - max(slo, out_s[oi]))
                        overlap_mu = max(0., min(mu_edges[j + 1], out_mu[oj + 1])
                                         - max(mu_edges[j], out_mu[oj]))
                        result[oi, oj] += (overlap_s * overlap_mu /
                                           ((shi - slo) * (mu_edges[j + 1] - mu_edges[j]))) * data[i, j]
                        continue
                    rlo, rhi = out_s[oi], out_s[oi + 1]
                    vlo = max(v0, out_mu[oj])
                    vhi = min(v1, out_mu[oj + 1])
                    if vhi <= vlo:
                        continue

                    def overlap(v):
                        r = radius(v)
                        return max(0., min(rhi, shi * r) - max(rlo, slo * r))

                    cuts = [vlo, vhi]
                    # W(v) changes formula where either radial edge crosses
                    # either edge of the source s interval.
                    for target_edge in (rlo, rhi):
                        for source_edge in (slo, shi):
                            if source_edge <= 0.:
                                continue
                            wanted = target_edge / source_edge
                            f0, f1 = radius(vlo) - wanted, radius(vhi) - wanted
                            if f0 * f1 < 0.:
                                cuts.append(brentq(lambda v: radius(v) - wanted,
                                                   vlo, vhi, xtol=5e-15))
                    cuts = sorted(set(cuts))
                    fraction = sum(
                        quad(overlap, a, b,
                             epsabs=2e-14, epsrel=2e-13, limit=80)[0]
                        for a, b in zip(cuts[:-1], cuts[1:])
                    ) / mapped_source_area
                    if fraction:
                        result[oi, oj] += fraction * data[i, j]
    return result


@pytest.mark.parametrize("algorithm", ["curved_gauss8", "curved_analytic"])
@pytest.mark.parametrize("ratios", [(1.0001, .9999), (.1, 10.), (10., .1),
                                     (.278, 1.54), (1.54, .278)])
@pytest.mark.parametrize("case", ["full", "partial"])
def test_curved_algorithms_match_independent_quadrature(algorithm, ratios, case):
    from LSS_python.CPP.lib.AP_pybind import _map_counts

    at, ap = ratios
    rng = np.random.default_rng(20261018)
    data = rng.random((3, 3, 2))
    s_bounds = (2., 8.)
    mu_bounds = (.12, .88)
    target_shape = (4, 5)
    window = ((2., 8., .12, .88) if case == "full"
              else (3.15, 6.7, .29, .73))
    expected = _curved_reference(data, at, ap, target_shape,
                                 s_bounds, mu_bounds, window)
    actual = _map_counts(data, at, ap, target_shape, s_bounds, mu_bounds,
                         window, convert_algorithm=algorithm)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("algorithm", ["curved_gauss8", "curved_analytic"])
@pytest.mark.parametrize("ratios", [(1., 1.), (1. + 1e-12, 1. - 1e-12),
                                     (.73, .91), (1.54, .278)])
def test_curved_unit_rates_match_reference_for_one_hot_small_grids(algorithm, ratios):
    from LSS_python.CPP.lib.AP_pybind import _map_counts

    at, ap = ratios
    rng = np.random.default_rng(8441)
    shape = (2, 3, 1)
    bounds = (0., 2.)
    mu_bounds = (0., 1.)
    target_shape = (4, 6)
    window = (0., 2., 0., 1.)
    for _ in range(5):
        data = np.zeros(shape, dtype=np.float64)
        i, j = (int(x) for x in rng.integers((shape[0], shape[1])))
        data[i, j, 0] = 1.
        expected = _curved_reference(data, at, ap, target_shape,
                                     bounds, mu_bounds, window)
        actual = _map_counts(data, at, ap, target_shape, bounds, mu_bounds,
                             window, convert_algorithm=algorithm)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-10)


@pytest.mark.parametrize("algorithm", ["curved_gauss8", "curved_analytic"])
def test_curved_cell_with_all_vertices_outside_can_cut_target_middle(algorithm):
    from LSS_python.CPP.lib.AP_pybind import _map_counts

    # Both source angular endpoints lie outside [.4, .5], so all four mapped
    # corners miss this target bin even though its middle intersects the cell.
    data = np.ones((1, 1, 1), dtype=np.float64)
    bounds, mu_bounds = (0., 2.), (0., 1.)
    target_shape = (4, 10)
    target_bin = (1, 4)
    at, ap = .55, 1.8
    window = (0., 2., 0., 1.)
    expected = _curved_reference(data, at, ap, target_shape,
                                 bounds, mu_bounds, window)
    actual = _map_counts(data, at, ap, target_shape, bounds, mu_bounds,
                         window, convert_algorithm=algorithm)
    assert expected[target_bin[0], target_bin[1], 0] > 0.
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-10)


@pytest.mark.parametrize("algorithm", ["curved_gauss8", "curved_analytic"])
def test_curved_identity_rebins_and_conserves_counts(algorithm):
    from LSS_python.CPP.lib.AP_pybind import _map_counts

    data = np.arange(12., dtype=np.float64).reshape(3, 4, 1) + 1.
    result = _map_counts(data, 1., 1., (7, 5), (4., 10.), (.2, .8),
                         (4., 10., .2, .8), convert_algorithm=algorithm)
    np.testing.assert_allclose(result.sum(axis=(0, 1)), data.sum(axis=(0, 1)),
                               rtol=0., atol=1e-10)


def test_curved_full_ap_mapping_conserves_counts_and_mixed_window_matches_reference():
    from LSS_python.CPP.lib.AP_pybind import _map_counts

    data = np.ones((2, 3, 1), dtype=np.float64)
    for algorithm in ("curved_gauss8", "curved_analytic"):
        complete = _map_counts(data, .73, .91, (5, 4), (0., 4.), (0., 1.),
                               (0., 4., 0., 1.), convert_algorithm=algorithm)
        window = (.7, 3.1, .18, .86)
        mixed = _map_counts(data, .73, .91, (5, 4), (0., 4.), (0., 1.),
                            window, convert_algorithm=algorithm)
        expected = _curved_reference(data, .73, .91, (5, 4), (0., 4.), (0., 1.), window)
        assert complete.sum() == pytest.approx(data.sum(), abs=1e-10)
        np.testing.assert_allclose(mixed, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("algorithm", ["curved_gauss8", "curved_analytic"])
@pytest.mark.parametrize("scale", [0.73, 1.54])
def test_isotropic_curved_mapping_is_exact_rectangle_overlap(algorithm, scale):
    """At strict isotropy, the mapped cells are rectangles with radius=scale."""
    from LSS_python.CPP.lib.AP_pybind import _map_counts
    from LSS_python._ap_polygon import map_counts

    data = np.arange(1., 5.).reshape(2, 2, 1)
    s_bounds, mu_bounds = (0., 4.), (0., 1.)
    target_shape = (3, 4)
    actual_cpp = _map_counts(data, scale, scale, target_shape, s_bounds, mu_bounds,
                             (0., 4., 0., 1.), convert_algorithm=algorithm)
    jit_algorithm = 1 if algorithm == "curved_gauss8" else 2
    actual_jit = map_counts(data, scale, scale, target_shape,
                            np.array(s_bounds), np.array(mu_bounds),
                            np.array((0., 4., 0., 1.)), jit_algorithm)

    source_s = np.linspace(*s_bounds, data.shape[0] + 1)
    source_mu = np.linspace(*mu_bounds, data.shape[1] + 1)
    target_s = np.linspace(*s_bounds, target_shape[0] + 1)
    target_mu = np.linspace(*mu_bounds, target_shape[1] + 1)
    expected = np.zeros_like(actual_cpp)
    for si in range(data.shape[0]):
        for mi in range(data.shape[1]):
            mapped_lo, mapped_hi = scale * source_s[si:si + 2]
            source_area = (mapped_hi - mapped_lo) * (source_mu[mi + 1] - source_mu[mi])
            for ti in range(target_shape[0]):
                overlap_s = max(0., min(mapped_hi, target_s[ti + 1]) - max(mapped_lo, target_s[ti]))
                for tj in range(target_shape[1]):
                    overlap_mu = max(0., min(source_mu[mi + 1], target_mu[tj + 1])
                                     - max(source_mu[mi], target_mu[tj]))
                    expected[ti, tj, 0] += (overlap_s * overlap_mu / source_area) * data[si, mi, 0]

    np.testing.assert_allclose(actual_cpp, expected, rtol=0., atol=2e-15)
    np.testing.assert_array_equal(actual_cpp, actual_jit)


@pytest.mark.parametrize("algorithm", ["curved_gauss8", "curved_analytic"])
@pytest.mark.parametrize("relative_delta", [
    np.nextafter(1., 0.) - 1.,
    np.nextafter(1., 2.) - 1.,
    -4.9e-9, -5.1e-9, 4.9e-9, 5.1e-9,
])
def test_near_isotropic_branch_matches_independent_quadrature(algorithm, relative_delta):
    """Strict equality stays distinct from neighboring values and the series boundary."""
    from LSS_python.CPP.lib.AP_pybind import _map_counts
    from LSS_python._ap_polygon import map_counts

    a = 1.3
    b = a * (1. + relative_delta)
    data = np.array([[[1.], [2.]], [[3.], [4.]]])
    s_bounds, mu_bounds = (0., 3.), (0., 1.)
    target_shape = (4, 5)
    window = (0., 3., 0., 1.)
    expected = _curved_reference(data, a, b, target_shape, s_bounds, mu_bounds, window)
    cpp = _map_counts(data, a, b, target_shape, s_bounds, mu_bounds,
                      window, convert_algorithm=algorithm)
    jit_algorithm = 1 if algorithm == "curved_gauss8" else 2
    jit = map_counts(data, a, b, target_shape, np.array(s_bounds), np.array(mu_bounds),
                     np.array(window), jit_algorithm)
    np.testing.assert_allclose(cpp, expected, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(jit, expected, rtol=2e-10, atol=2e-12)
    np.testing.assert_array_equal(cpp, jit)


@pytest.mark.parametrize("radial_bounds,expected_positive", [
    ((10., 11.), False),
    ((-1., 10.), True),
])
def test_adaptive_integral_handles_adjacent_endpoints(radial_bounds, expected_positive):
    """An interval with no interior float is accepted only from its actual scale."""
    from LSS_python._ap_curved import adaptive_R_or_width, radius

    lo = 0.5
    hi = np.nextafter(lo, np.inf)
    value = adaptive_R_or_width(lo, hi, radial_bounds[0], radial_bounds[1],
                                0., 1., 0.8, 1.2, False)
    assert np.isfinite(value)
    assert (value > 0.) is expected_positive
    if expected_positive:
        # The whole interval is covered, so its area is the integral of R.
        expected = (hi - lo) * radius(lo, 0.8, 1.2)
        assert value == pytest.approx(expected, rel=2e-15, abs=0.)
    else:
        assert value == 0.


@pytest.mark.parametrize("analytic", [False, True])
def test_repeated_root_cuts_preserve_overlap_area(analytic):
    """Two distinct radial-edge pairs meeting at the same root retain area."""
    from LSS_python._ap_curved import curved_overlap

    a, b = 0.8, 1.2
    # sl/s0 == sh/s1 == 1, so both radial pairs meet at the same v root.
    actual = curved_overlap(0., 1., 1., 2., 1., 2., a, b, analytic)

    def width(v):
        r = 1. / math.hypot(math.sqrt(max(0., 1. - v * v)) / a, v / b)
        return max(0., min(2., 2. * r) - max(1., r))

    expected = quad(width, 0., 1., epsabs=1e-14, epsrel=1e-13)[0]
    assert expected > 0.
    assert actual == pytest.approx(expected, rel=2e-12, abs=1e-14)


@pytest.mark.parametrize("algorithm", ["curved_gauss8", "curved_analytic"])
@pytest.mark.parametrize("a,b", [(0.1, 10.0), (10.0, 0.1)])
def test_coarse_cell_with_strong_anisotropy_matches_reference_and_backends(algorithm, a, b):
    """A broad source angular bin forces adaptive refinement near its endpoint."""
    from LSS_python.CPP.lib.AP_pybind import _map_counts
    from LSS_python._ap_polygon import map_counts

    data = np.ones((1, 1, 1))
    sb, mb, window = (0., 3.), (0., 1.), (0., 3., 0., 1.)
    shape = (5, 7)
    expected = _curved_reference(data, a, b, shape, sb, mb, window)
    cpp = _map_counts(data, a, b, shape, sb, mb, window, convert_algorithm=algorithm)
    jit = map_counts(data, a, b, shape, np.array(sb), np.array(mb),
                     np.array(window), 1 if algorithm == "curved_gauss8" else 2)
    np.testing.assert_allclose(cpp, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(jit, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cpp, jit, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_gauss8_reports_nonconvergence_without_fallback(backend):
    # A boundary layer narrower than the permitted subdivision depth cannot
    # satisfy the tolerance. Both backends must report it, never fall back.
    data = np.ones((1, 1, 1))
    with pytest.raises(RuntimeError, match="converge.*depth 24"):
        if backend == "cpp":
            from LSS_python.CPP.lib.AP_pybind import _map_counts
            _map_counts(data, 1., 1e8, (2, 2), (0., 2.), (0., 1.),
                        (0., 2., 0., 1.), nthreads=2,
                        convert_algorithm="curved_gauss8")
        else:
            from LSS_python._ap_polygon import map_counts
            map_counts(data, 1., 1e8, (2, 2), np.array((0., 2.)),
                       np.array((0., 1.)), np.array((0., 2., 0., 1.)), 1)
