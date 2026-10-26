import numpy as np
import pytest

from LSS_python.marks import (
    create_rho,
    create_drho,
    cal_rho,
    cal_rho_array,
    cal_drho,
    cal_drho_array,
    w_kernel,
    dw_kernel,
    w_kernel_jit,
    dw_kernel_jit,
)


def manual_sph_density(pos, weights, k, h_fixed=None, ignore_self=False):
    """Reference manual computation of SPH kernel density."""
    n = pos.shape[0]
    rho = np.zeros(n, dtype=np.float64)
    for i in range(n):
        diff = pos - pos[i]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        order = np.argsort(dists)
        if ignore_self:
            sel = order[1:k + 1]
        else:
            sel = order[:k]
        sel_dists = dists[sel]
        sel_weights = weights[sel]
        h = np.max(sel_dists) / 2.0 if h_fixed is None else h_fixed

        sigma = 1.0 / (np.pi * h**3)
        q = sel_dists / h
        w = np.zeros_like(q)
        m1 = (q >= 0.0) & (q <= 1.0)
        m2 = (q > 1.0) & (q < 2.0)
        w[m1] = sigma * (0.25 * (2.0 - q[m1])**3 - (1.0 - q[m1])**3)
        w[m2] = sigma * 0.25 * (2.0 - q[m2])**3
        rho[i] = np.sum(w * sel_weights)
    return rho


def test_create_rho_unweighted_vs_ones():
    np.random.seed(42)
    pos = np.random.uniform(10, 90, size=(100, 3)).astype(np.float64)
    boxsize = 100.0

    rho_unweighted = create_rho(pos, boxsize=boxsize, k=25)
    weights_ones = np.ones(100)
    rho_weighted_ones = create_rho(pos, boxsize=boxsize, k=25, weights=weights_ones)

    np.testing.assert_allclose(rho_unweighted, rho_weighted_ones, rtol=1e-12)


def test_create_rho_scaled_weights():
    np.random.seed(42)
    pos = np.random.uniform(10, 90, size=(100, 3)).astype(np.float64)
    boxsize = 100.0
    scale = 3.75

    rho_unweighted = create_rho(pos, boxsize=boxsize, k=20)
    weights = scale * np.ones(100)
    rho_weighted = create_rho(pos, boxsize=boxsize, k=20, weights=weights)

    np.testing.assert_allclose(rho_weighted, scale * rho_unweighted, rtol=1e-12)


def test_create_rho_against_manual_reference():
    np.random.seed(123)
    pos = np.random.uniform(20, 80, size=(50, 3)).astype(np.float64)
    weights = np.random.uniform(0.5, 2.5, size=50).astype(np.float64)

    # Use boxsize=None so distances are purely Euclidean without PBC
    k = 15
    rho_tested = create_rho(pos, boxsize=None, k=k, weights=weights, ignore_self=False)
    rho_ref = manual_sph_density(pos, weights, k=k, ignore_self=False)

    np.testing.assert_allclose(rho_tested, rho_ref, rtol=1e-10)


def test_create_rho_with_ignore_self():
    np.random.seed(456)
    pos = np.random.uniform(20, 80, size=(40, 3)).astype(np.float64)
    weights = np.random.uniform(0.8, 1.8, size=40).astype(np.float64)
    k = 10

    rho_tested = create_rho(pos, boxsize=None, k=k, weights=weights, ignore_self=True)
    rho_ref = manual_sph_density(pos, weights, k=k, ignore_self=True)

    np.testing.assert_allclose(rho_tested, rho_ref, rtol=1e-10)


def test_create_rho_return_pos_and_rho():
    pos = np.random.uniform(0, 100, size=(30, 3)).astype(np.float64)
    weights = np.random.uniform(0.5, 1.5, size=30).astype(np.float64)

    out = create_rho(pos, boxsize=100.0, k=10, weights=weights, only_return_rho=False)
    assert out.shape == (30, 4)
    np.testing.assert_allclose(out[:, :3], pos)

    rho_only = create_rho(pos, boxsize=100.0, k=10, weights=weights, only_return_rho=True)
    np.testing.assert_allclose(out[:, 3], rho_only)


def test_create_rho_invalid_weights_shape():
    pos = np.zeros((20, 3))
    wrong_weights = np.ones(15)
    with pytest.raises(ValueError, match="weights must have shape"):
        create_rho(pos, boxsize=100.0, weights=wrong_weights)


def test_cal_rho_with_weights():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    h = 2.0
    r = cal_rho(d, h=h, weights=w)
    # Expected analytical value: 0.10568882939696175
    assert abs(r - 0.10568882939696175) < 1e-12


def test_kernel_derivative():
    h = 2.5
    eps = 1e-6
    test_dists = np.array([0.2, 0.8, 1.5, 2.4, 3.5, 4.8])
    dw_ana = dw_kernel_jit(test_dists, h)
    for d, dw in zip(test_dists, dw_ana):
        # Scalar version should match jit version
        assert abs(dw - dw_kernel(d, h)) < 1e-14
        # Finite difference check
        w_plus = w_kernel(d + eps, h)
        w_minus = w_kernel(d - eps, h)
        dw_num = (w_plus - w_minus) / (2.0 * eps)
        assert abs(dw - dw_num) < 1e-5


def test_cal_drho_analytical_vs_numerical():
    np.random.seed(42)
    center = np.array([50.0, 50.0, 50.0])
    neighbors = np.random.uniform(45.0, 55.0, size=(10, 3))
    weights = np.random.uniform(0.5, 2.0, size=10)

    diff = center - neighbors
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    h = 10.0

    grad_analytical = cal_drho(dists, diff, h=h, weights=weights)

    # Numerical gradient by finite difference
    eps = 1e-6
    grad_numerical = np.zeros(3)
    for dim in range(3):
        step = np.zeros(3)
        step[dim] = eps
        d_plus = np.sqrt(np.sum((center + step - neighbors)**2, axis=-1))
        rho_plus = cal_rho(d_plus, h=h, weights=weights)
        d_minus = np.sqrt(np.sum((center - step - neighbors)**2, axis=-1))
        rho_minus = cal_rho(d_minus, h=h, weights=weights)
        grad_numerical[dim] = (rho_plus - rho_minus) / (2.0 * eps)

    np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-5)


def test_create_drho_and_return_drho():
    np.random.seed(123)
    pos = np.random.uniform(10, 90, size=(60, 3)).astype(np.float64)
    weights = np.random.uniform(0.5, 2.0, size=60).astype(np.float64)
    boxsize = 100.0

    # Test return_drho=True with only_return_rho=True
    rho, drho = create_rho(pos, boxsize=boxsize, k=20, weights=weights, return_drho=True)
    assert rho.shape == (60,)
    assert drho.shape == (60, 3)

    # The helper retains both raw fields from the same neighbor query.
    rho_direct, drho_direct = create_drho(pos, boxsize=boxsize, k=20, weights=weights)
    np.testing.assert_array_equal(rho, rho_direct)
    np.testing.assert_allclose(drho, drho_direct, rtol=1e-12)

    # Test return_drho=True with only_return_rho=False (7 columns: x, y, z, rho, drhox, drhoy, drhoz)
    pos_all = create_rho(
        pos,
        boxsize=boxsize,
        k=20,
        weights=weights,
        only_return_rho=False,
        return_drho=True,
    )
    assert pos_all.shape == (60, 7)
    np.testing.assert_allclose(pos_all[:, :3], pos)
    np.testing.assert_allclose(pos_all[:, 3], rho)
    np.testing.assert_allclose(pos_all[:, 4:7], drho)


