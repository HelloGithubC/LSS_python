"""Regression tests for polygon-based dense AP count mapping."""

from __future__ import annotations

import numpy as np
import pytest

from LSS_python._ap_polygon import map_counts, polygon_rate
from LSS_python._ap_dense import ap_geometry, validate_mapping
from LSS_python.AP import _ap_convert_factors, tpcf_convert_main
from LSS_python.tpcf import xismu


def _area(points):
    """Return polygon area with the shoelace formula."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(poly, axis, bound, keep_greater):
    if not poly:
        return []
    result = []
    previous = poly[-1]
    prev_inside = previous[axis] >= bound if keep_greater else previous[axis] <= bound
    for current in poly:
        inside = current[axis] >= bound if keep_greater else current[axis] <= bound
        if inside != prev_inside:
            fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
            result.append(previous + fraction * (current - previous))
        if inside:
            result.append(current)
        previous, prev_inside = current, inside
    return result


def _reference_polygon_rate(corners, bounds):
    """Independent small-grid clipping reference; does not call production code."""
    corners = np.asarray(corners, dtype=np.float64)
    polygon = [corners[i].copy() for i in (0, 1, 3, 2)]
    area = _area(polygon)
    if area == 0.0:
        return 0.0
    for axis, bound, keep_greater in (
        (0, bounds[0], True), (0, bounds[1], False),
        (1, bounds[2], True), (1, bounds[3], False),
    ):
        polygon = _clip(polygon, axis, bound, keep_greater)
    return min(1.0, max(0.0, _area(polygon) / area))


def _reference_map_full_window(data, alpha_perp, alpha_parallel, target_shape,
                               s_bounds, mu_bounds):
    """Slow source-scatter reference for complete AP mapping windows."""
    ns, nm, channels = data.shape
    ts, tm = target_shape
    s_edges = np.linspace(*s_bounds, ns + 1)
    mu_edges = np.linspace(*mu_bounds, nm + 1)
    out_s = np.linspace(*s_bounds, ts + 1)
    out_mu = np.linspace(*mu_bounds, tm + 1)
    output = np.zeros((ts, tm, channels), dtype=np.float64)
    for i in range(ns):
        for j in range(nm):
            q0 = np.hypot(alpha_perp * np.sqrt(max(0.0, 1.0 - mu_edges[j] ** 2)),
                          alpha_parallel * mu_edges[j])
            q1 = np.hypot(alpha_perp * np.sqrt(max(0.0, 1.0 - mu_edges[j + 1] ** 2)),
                          alpha_parallel * mu_edges[j + 1])
            v0 = alpha_parallel * mu_edges[j] / q0
            v1 = alpha_parallel * mu_edges[j + 1] / q1
            corners = np.array([
                [s_edges[i] * q0, v0], [s_edges[i + 1] * q0, v0],
                [s_edges[i] * q1, v1], [s_edges[i + 1] * q1, v1],
            ])
            for oi in range(ts):
                for oj in range(tm):
                    rate = _reference_polygon_rate(
                        corners, (out_s[oi], out_s[oi + 1], out_mu[oj], out_mu[oj + 1])
                    )
                    if rate:
                        output[oi, oj] += rate * data[i, j]
    return output


def test_polygon_rate_matches_independent_clipping_and_handles_degenerate_shapes():
    square = np.array([[0., 0.], [2., 0.], [0., 2.], [2., 2.]])
    bounds = (1., 3., 0., 1.)
    assert polygon_rate(square, bounds) == pytest.approx(0.25)
    assert polygon_rate(square, (3., 4., 0., 1.)) == 0.0
    assert polygon_rate(square, (-1., 3., -1., 3.)) == pytest.approx(1.0)
    degenerate = np.array([[0., 0.], [1., 0.], [0., 0.], [1., 0.]])
    assert polygon_rate(degenerate, (-1., 2., -1., 1.)) == 0.0


def test_polygon_rate_matches_reference_for_clipped_and_outside_polygons():
    cases = [
        (np.array([[0., 0.], [2., 0.], [0., 2.], [2., 2.]]), (.5, 1.5, .25, 1.25)),
        (np.array([[3., 3.], [4., 3.], [3., 4.], [4., 4.]]), (0., 1., 0., 1.)),
        (np.array([[-1., .2], [1., .2], [-1., .8], [1., .8]]), (0., .5, .25, .75)),
    ]
    for corners, bounds in cases:
        assert polygon_rate(corners, bounds) == pytest.approx(
            _reference_polygon_rate(corners, bounds), rel=1e-13, abs=1e-14
        )


@pytest.mark.parametrize("alpha_perp,alpha_parallel", [(1.31, .72), (.69, 1.28)])
def test_map_counts_matches_independent_source_scatter(alpha_perp, alpha_parallel):
    data = np.random.default_rng(20261015).random((4, 3, 2))
    kwargs = dict(target_shape=(5, 4), s_bounds=(0., 6.), mu_bounds=(0., 1.),
                  mapping_bounds=(0., 6., 0., 1.))
    result = map_counts(data, alpha_perp, alpha_parallel, **kwargs)
    expected = _reference_map_full_window(
        data, alpha_perp, alpha_parallel, kwargs["target_shape"],
        kwargs["s_bounds"], kwargs["mu_bounds"],
    )
    np.testing.assert_allclose(result, expected, rtol=2e-11, atol=2e-12)


def test_all_vertices_outside_candidate_box_can_still_overlap_output():
    # A sheared quadrilateral crosses the target square despite every vertex
    # lying outside that square. This guards against vertex-only rejection.
    corners = np.array([[-2., .4], [2., -.6], [-2., 1.4], [2., .4]])
    rate = polygon_rate(corners, (0., 1., 0., 1.))
    assert rate > 0.0
    assert rate == pytest.approx(_reference_polygon_rate(corners, (0., 1., 0., 1.)))


def test_identity_conserves_counts_on_nonzero_bounds_and_crop_is_fractional():
    data = np.arange(4 * 3, dtype=np.float64).reshape(4, 3, 1) + 1.0
    result = map_counts(
        data, 1., 1., target_shape=(7, 5), s_bounds=(5., 13.),
        mu_bounds=(.2, .8), mapping_bounds=(5., 13., .2, .8),
    )
    np.testing.assert_allclose(result.sum(axis=(0, 1)), data.sum(axis=(0, 1)), atol=1e-11)

    cropped = map_counts(
        np.ones((1, 1, 1)), 2., 2., target_shape=(1, 1),
        s_bounds=(0., 1.), mu_bounds=(0., 1.), mapping_bounds=(0., 1., 0., 1.),
    )
    assert cropped[0, 0, 0] == pytest.approx(.5, abs=1e-12)


def test_mapping_window_partial_cut_and_zero_radius_are_finite():
    data = np.ones((3, 3, 1), dtype=np.float64)
    result = map_counts(
        data, 1.25, .8, target_shape=(4, 4), s_bounds=(0., 3.),
        mu_bounds=(0., 1.), mapping_bounds=(.75, 2.25, .25, .75),
    )
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.)
    assert 0. < result.sum() < data.sum()

    zero_radius = np.zeros((2, 4, 1))
    zero_radius[0, 2, 0] = 1.0
    mapped = map_counts(
        zero_radius, 2., 1., target_shape=(4, 8), s_bounds=(0., 2.),
        mu_bounds=(0., 1.), mapping_bounds=(0., 2., 0., 1.),
    )
    assert np.all(np.isfinite(mapped))
    assert mapped.sum() == pytest.approx(1.0, abs=1e-12)


def test_map_counts_accepts_strided_and_non_float_inputs_without_mutation():
    original = np.arange(8 * 6 * 2, dtype=np.int32).reshape(8, 6, 2)
    sliced = original[::2, ::2]
    before = sliced.copy()
    result = map_counts(
        sliced, 1., 1., target_shape=(4, 3), s_bounds=(0., 4.),
        mu_bounds=(0., 1.), mapping_bounds=(0., 4., 0., 1.),
    )
    assert result.dtype == np.float64
    np.testing.assert_array_equal(sliced, before)


@pytest.mark.parametrize("alpha_perp,alpha_parallel", [(0., 1.), (1., 0.), (-1., 1.), (np.nan, 1.)])
def test_ap_ratios_must_be_finite_and_positive(alpha_perp, alpha_parallel):
    data = np.ones((1, 1, 1))
    with pytest.raises((ValueError, RuntimeError)):
        validate_mapping(
            data, (alpha_parallel, 1., 1., alpha_perp),
            (1, 1, 1, 1), (0., 1., 0., 1.), (0., 1., 0., 1.), "polygon",
        )


def test_prepared_geometry_floors_redshift_and_falls_back_when_wa_is_none():
    geometry = ap_geometry(0., .3071, -1., .34, -.92, None, None)
    expected = _ap_convert_factors(1e-4, .3071, -1., .34, -.92, 0., 0.)
    assert geometry.redshift == pytest.approx(1e-4)
    assert geometry.alpha_perp == pytest.approx(expected[0], rel=1e-13)
    assert geometry.alpha_parallel == pytest.approx(expected[1], rel=1e-13)


def _make_xismu(shape, *, smax, seed, set_data=True):
    ns, nm = shape
    rng = np.random.default_rng(seed)
    norms = (11.0, 17.0, 23.0)
    s_edges = np.linspace(0.0, smax, ns + 1)
    mu_edges = np.linspace(0.0, 1.0, nm + 1)
    S, Mu = np.meshgrid((s_edges[:-1] + s_edges[1:]) / 2,
                        (mu_edges[:-1] + mu_edges[1:]) / 2, indexing="ij")
    return xismu(
        smax=smax, sbin=ns, mubin=nm,
        DDnorm=norms[0], DRnorm=norms[1], RRnorm=norms[2],
        S=S, Mu=Mu,
        DD=(.8 + rng.random(shape)) * norms[0],
        DR=(.7 + rng.random(shape)) * norms[1],
        RR=(.9 + rng.random(shape)) * norms[2],
        sum_wd=2.5, sum_wr=3.5, sum_wd2=4.5, sum_wr2=5.5,
        set_data=set_data,
    )


def _make_assistant(shape, *, smax):
    assistant = _make_xismu(shape, smax=smax, seed=1, set_data=False)
    ns, nm = shape
    s_edges = np.linspace(0., smax, ns + 1)
    mu_edges = np.linspace(0., 1., nm + 1)
    assistant.S, assistant.Mu = np.meshgrid(
        (s_edges[:-1] + s_edges[1:]) / 2,
        (mu_edges[:-1] + mu_edges[1:]) / 2,
        indexing="ij",
    )
    return assistant


def test_high_level_polygon_api_preserves_xismu_metadata_and_assistant_grid():
    # A non-150 smax checks that physical bounds follow the source object's grid.
    source = _make_xismu((300, 240), smax=30., seed=42)
    assistant = _make_assistant((6, 4), smax=30.)
    assistant.S[0, 0] = .123
    assistant.Mu[-1, -1] = .987
    original = {name: np.array(getattr(source, name), copy=True)
                for name in ("DD", "DR", "RR", "S", "Mu")}
    fid, model, z = (.3071, -1.), (.34, -.92), .8
    converted = tpcf_convert_main(
        source, *fid, *model, z, assis_xismu=assistant,
        smin_mapping=0., smax_mapping=30., c_api=False,
        convert_algorithm="polygon",
    )
    assert converted.sbin == assistant.sbin
    assert converted.mubin == assistant.mubin
    np.testing.assert_array_equal(converted.S, assistant.S)
    np.testing.assert_array_equal(converted.Mu, assistant.Mu)
    np.testing.assert_allclose(converted.s_array, np.nanmean(assistant.S, axis=1))
    np.testing.assert_allclose(converted.mu_array, np.nanmean(assistant.Mu, axis=0))
    for name in ("DDnorm", "DRnorm", "RRnorm", "sum_wd", "sum_wr", "sum_wd2", "sum_wr2"):
        assert getattr(converted, name) == getattr(source, name)
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(source, name), value)


def test_high_level_identity_returns_assistant_and_requires_grid_for_conversion():
    source = _make_xismu((8, 6), smax=30., seed=12)
    assistant = _make_assistant((3, 2), smax=30.)
    assert tpcf_convert_main(
        source, .3071, -1., .3071, -1., .7,
        assis_xismu=assistant, c_api=False, convert_algorithm="polygon",
    ) is assistant

    with pytest.raises(RuntimeError, match="assistant|Assis xismu"):
        tpcf_convert_main(
            source, .3071, -1., .34, -.92, .7,
            c_api=False, convert_algorithm="polygon",
        )


def test_direct_xismu_method_can_use_explicit_output_sizes_without_assistant():
    source = _make_xismu((8, 6), smax=12., seed=28)
    result = source.cosmo_conv_DenseToSparse(
        omstd=.3071, wstd=-1., omwrong=.34, wwrong=-.92, redshift=.7,
        sbin2=3, mubin2=2, smin_mapping=0., smax_mapping=12.,
        assistant_xismu=None, c_api=False, convert_algorithm="polygon",
    )
    assert result.DD.shape == (3, 2)
    assert result.S.shape == (3, 2)
    assert result.Mu.shape == (3, 2)


@pytest.mark.parametrize("algorithm,error", [
    ("curved_gauss8", NotImplementedError),
    ("curved_analytic", NotImplementedError),
    ("bogus", ValueError),
])
def test_algorithm_validation_precedes_identity_fast_path(algorithm, error):
    source = _make_xismu((2, 2), smax=2., seed=3)
    with pytest.raises(error):
        tpcf_convert_main(
            source, .3, -1., .3, -1., .5, assis_xismu=object(),
            c_api=False, convert_algorithm=algorithm,
        )


def test_high_level_ap_tolerance_zero_forces_nonidentity_conversion():
    source = _make_xismu((300, 240), smax=30., seed=14)
    assistant = _make_assistant((4, 3), smax=30.)
    result = tpcf_convert_main(
        source, .3071, -1., .3071, -1., .8, assis_xismu=assistant,
        smin_mapping=0., smax_mapping=30., ap_tol=0., c_api=False,
        convert_algorithm="polygon",
    )
    assert result is not assistant
    for name in ("DD", "DR", "RR"):
        expected = getattr(source, name).reshape(4, 75, 3, 80).sum(axis=(1, 3))
        np.testing.assert_allclose(getattr(result, name), expected, rtol=2e-12, atol=1e-10)


def test_background_is_evaluated_once_per_conversion(monkeypatch):
    from unittest.mock import Mock
    import LSS_python._ap_dense as dense

    da = Mock(wraps=dense.DA)
    hz = Mock(wraps=dense.Hz_w0wa)
    monkeypatch.setattr(dense, "DA", da)
    monkeypatch.setattr(dense, "Hz_w0wa", hz)
    source = _make_xismu((8, 6), smax=12., seed=78)
    assistant = _make_assistant((4, 3), smax=12.)
    tpcf_convert_main(source, .3071, -1., .34, -.92, .8,
                      assis_xismu=assistant, c_api=False, ap_tol=0.)
    assert da.call_count == hz.call_count == 2


def test_simple_dispatch_preserves_existing_interpolation():
    from unittest.mock import Mock

    class SimpleSource:
        xis = np.zeros((10, 8))
        cosmo_conv_simple = Mock(return_value=object())

    source = SimpleSource()
    result = tpcf_convert_main(source, .3071, -1., .34, -.92, .8,
                               convert_method="simple", wa_m=.2)
    assert result is source.cosmo_conv_simple.return_value
    source.cosmo_conv_simple.assert_called_once_with(
        omstd=.3071, wstd=-1., omwrong=.34, wwrong=-.92, redshift=.8,
        wastd=0., wawrong=.2, smin_mapping=3., smax_mapping=60.)


def test_real_four_corner_search_and_fractional_overlap_regression():
    from LSS_python.CPP.lib.AP_pybind import _map_counts

    data = np.zeros((750, 600, 1))
    data[450, 558, 0] = 1.
    result = _map_counts(data, .5791154768764223, .4058608239642733,
                         (150, 120), (0., 150.), (0., 1.), (0., 150., 0., 1.))
    assert result[38, 104, 0] == pytest.approx(.3173398046, abs=2e-6)
    assert result[39, 104, 0] == pytest.approx(.6826601954, abs=2e-6)
    assert result.sum() == pytest.approx(1., abs=1e-10)


def test_raw_binding_keeps_historical_positional_signature():
    from LSS_python.CPP.lib.AP_pybind import mapping_smudata_dense
    from LSS_python.CPP.AP import mapping_smudata_dense_cpp

    data = np.arange(24.).reshape(4, 3, 2)[::-1]
    raw = mapping_smudata_dense(data, 4, 3, 5, 4, 100., 95., 1000., 1050.,
                                0., 8., 0., 1., 0., 8., 1)
    wrapped = mapping_smudata_dense_cpp(data, (100., 95., 1000., 1050.),
                                        (4, 5, 3, 4), (0., 8., 0., 1.), (0., 8., 0., 1.))
    np.testing.assert_array_equal(raw, wrapped)
    assert raw.flags.owndata
