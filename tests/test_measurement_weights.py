"""Shared measurement-weight semantics and non-AP CLI integration checks."""

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from LSS_python.measurement_weights import (
    apply_mark_power, attach_weight_metadata, prepare_box_catalog, prepare_mock_catalog,
)
from LSS_python.weight_token import add_measurement_weight_arguments, resolve_measurement_weights
from LSS_python.xismu import XismuCovarianceIO


ROOT = Path(__file__).resolve().parents[1]


def settings(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_weight", "-ww", action="store_true")
    parser.add_argument("--set_weight_to_one", action="store_true")
    add_measurement_weight_arguments(parser)
    return resolve_measurement_weights(parser.parse_args(arguments), parser)


@pytest.mark.parametrize("options,token", [
    (["-ww"], "n30_weipow1"),
    (["-ww", "--weight-method", "r2", "-weipow", "2.0"], "r2_weipow2"),
    (["-ww", "--weight-token", "gn6_weipow-0.5"], "gn6_weipow-0.5"),
    (["-ww", "--weight-token", "r2_weipow2", "--weight-method", "r2", "-weipow", "+2.0"], "r2_weipow2"),
])
def test_configuration(options, token):
    args = settings(options)
    assert args.weight_token == token
    assert float(args.weight_power_text) == args.weight_pow


@pytest.mark.parametrize("options", [
    ["--weight-method", "r2"], ["--weight-token", "n30_weipow1"],
    ["-ww", "--weight-token", "r2_weipow2", "--weight-method", "n30"],
    ["-ww", "--weight-token", "r2_weipow2", "-weipow", "3"],
    ["-ww", "--weight-token", "n30_weipow-1_1"],
    ["-ww", "--weight-method", "r2.0"], ["-ww", "-weipow", "nan"],
    ["-ww", "-weipow", "1e1000"], ["-ww", "-weipow", "1e-1000"],
    ["--no-divide-by-rho"], ["-ww", "--no-divide-by-rho"],
    ["-ww", "--weight-method", "r2", "--no-divide-by-rho"],
])
def test_configuration_rejects_conflicts_or_invalid_identity(options):
    with pytest.raises(SystemExit) as error:
        settings(options)
    assert error.value.code == 2


def test_legacy_unweighted_options_are_ignored():
    args = settings(["-weipow", "2", "--set_weight_to_one"])
    assert not args.set_weight_to_one and args.weight_pow == 1
    assert args.weight_token is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["n30", "gn6", "r2"])
def test_scalar_mark_power_once_and_random_unchanged(dtype, method):
    source = np.array([[1, 2, 3, 2], [4, 5, 6, 3]], dtype=dtype)
    original = source.copy()
    data = prepare_box_catalog(source, with_weight=True, method=method, power=2)
    random = prepare_box_catalog(source, with_weight=True, method=method, power=2, is_data=False)
    np.testing.assert_array_equal(data[:, 3], [4, 9])
    np.testing.assert_array_equal(random[:, 3], [2, 3])
    np.testing.assert_array_equal(source, original)
    assert data.dtype == dtype and random.dtype == dtype


def test_missing_marks_and_explicit_unit_weights():
    positions = np.ones((2, 3))
    np.testing.assert_array_equal(prepare_box_catalog(positions, with_weight=True)[:, 3], 1)
    with pytest.raises(ValueError, match="requires"):
        prepare_box_catalog(positions, with_weight=True, require_weight=True)
    result = prepare_box_catalog(positions, with_weight=True, require_weight=True, weight_set_one=True)
    np.testing.assert_array_equal(result[:, 3], 1)


@pytest.mark.parametrize("mark,power", [([0.0], -1), ([-2.0], 0.5), ([1e300], 2), ([np.nan], 1)])
def test_invalid_power_results_are_rejected(mark, power):
    with pytest.raises(ValueError):
        apply_mark_power(np.array(mark), power)


def test_mock_base_and_random_auxiliary_weights_are_not_exponentiated():
    data = np.array([[20., 10., 100., 3., 2., 5.], [30., 5., 110., 4., 3., 7.]])
    original = data.copy()
    effective = prepare_mock_catalog(data, with_weight=True, method="r2", power=2)
    np.testing.assert_array_equal(effective[:, 3], [3 * 4 * 5, 4 * 9 * 7])
    random = prepare_mock_catalog(data, with_weight=True, power=2, is_data=False)
    np.testing.assert_array_equal(random[:, 3], [3 * 2 * 5, 4 * 3 * 7])
    np.testing.assert_array_equal(data, original)


def test_mock_base_only_compatibility_and_missing_mark():
    base = np.array([[20., 10., 100., 3.]])
    assert prepare_mock_catalog(base, with_weight=True)[0, 3] == 3
    with pytest.raises(ValueError, match="separate raw-mark"):
        prepare_mock_catalog(base, with_weight=True, power=2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("power", [2, -0.5, 0])
@pytest.mark.parametrize("mock", [False, True])
@pytest.mark.parametrize("divide_by_rho", [True, False])
def test_gradient_norm_precedes_power_and_preserves_input(dtype, power, mock, divide_by_rho):
    positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    gradient = np.array([[0, -3, 4], [-2, 3, -6]], dtype=dtype)
    base = np.array([2, 3], dtype=dtype)
    rho = np.array([2, 4], dtype=dtype)
    source = np.column_stack((positions, base, rho, gradient) if mock else (positions, rho, gradient))
    original = source.copy()
    prepare = prepare_mock_catalog if mock else prepare_box_catalog
    result = prepare(source, with_weight=True, method="gn6", power=power, divide_by_rho=divide_by_rho)
    expected = np.array([5, 7], dtype=dtype)
    if divide_by_rho:
        expected /= rho
    expected = expected ** power
    if mock:
        expected *= base
    np.testing.assert_allclose(result[:, 3], expected, rtol=2e-7)
    np.testing.assert_array_equal(source, original)
    assert result.dtype == dtype and result.flags.c_contiguous
    np.testing.assert_array_equal(prepare(source, with_weight=True, method="gn6", power=power,
                                         weight_set_one=True)[:, 3], 1)


@pytest.mark.parametrize("dtype,scale", [(np.float32, 1e20), (np.float64, 1e200)])
def test_gradient_norm_avoids_unnecessary_square_overflow(dtype, scale):
    source = np.array([[1, 2, 3, 1, scale, -scale, scale]], dtype=dtype)
    result = prepare_box_catalog(source, with_weight=True, method="gn6")
    np.testing.assert_allclose(result[:, 3] / scale, np.sqrt(3), rtol=2e-7)


@pytest.mark.parametrize("mock", [False, True])
@pytest.mark.parametrize("gradient,power", [
    ([0, 0, 0], -1), ([np.nan, 1, 2], 1), ([np.inf, 1, 2], 1),
    ([1.5e308, 1.5e308, 0], 1),
])
def test_invalid_gradient_weights_are_rejected(mock, gradient, power):
    source = np.array([[1, 2, 3, 2, 1, *gradient] if mock else [1, 2, 3, 1, *gradient]], dtype=float)
    prepare = prepare_mock_catalog if mock else prepare_box_catalog
    with pytest.raises(ValueError):
        prepare(source, with_weight=True, method="gn6", power=power)


@pytest.mark.parametrize("mock", [False, True])
@pytest.mark.parametrize("columns", [2, 5])
def test_gradient_ambiguous_column_counts_are_rejected(mock, columns):
    source = np.ones((2, (4 if mock else 3) + columns))
    prepare = prepare_mock_catalog if mock else prepare_box_catalog
    with pytest.raises(ValueError, match="one prepared scalar column or rho plus three gradient columns"):
        prepare(source, with_weight=True, method="gn6")


@pytest.mark.parametrize("mock", [False, True])
def test_legacy_gradient_vectors_require_explicit_raw_mode(mock):
    source = np.array([[1, 2, 3, 2, 0, -3, 4] if mock else [1, 2, 3, 0, -3, 4]], dtype=float)
    prepare = prepare_mock_catalog if mock else prepare_box_catalog
    with pytest.raises(ValueError, match="requires rho"):
        prepare(source, with_weight=True, method="gn6")
    result = prepare(source, with_weight=True, method="gn6", divide_by_rho=False)
    assert result[0, 3] == (10 if mock else 5)


@pytest.mark.parametrize("rho", [0, -1, np.nan, np.inf])
@pytest.mark.parametrize("mock", [False, True])
def test_normalized_gradient_requires_positive_finite_density(rho, mock):
    source = np.array([[1, 2, 3, 2, rho, 0, -3, 4] if mock else [1, 2, 3, rho, 0, -3, 4]], dtype=float)
    prepare = prepare_mock_catalog if mock else prepare_box_catalog
    with pytest.raises(ValueError, match="strictly positive rho"):
        prepare(source, with_weight=True, method="gn6", power=0)
    result = prepare(source, with_weight=True, method="gn6", divide_by_rho=False)
    assert result[0, 3] == (10 if mock else 5)


def test_density_division_avoids_raw_norm_overflow():
    source = np.array([[1, 2, 3, 2, 1.5e308, 1.5e308, 0]])
    result = prepare_box_catalog(source, with_weight=True, method="gn6")
    np.testing.assert_allclose(result[0, 3] / 1e308, np.sqrt(2) * .75)
    with pytest.raises(ValueError, match="overflowing"):
        prepare_box_catalog(source, with_weight=True, method="gn6", divide_by_rho=False)


@pytest.mark.parametrize("divide_by_rho", [True, False])
def test_gradient_division_configuration_and_metadata(divide_by_rho):
    options = ["-ww", "--weight-token", "gn6_weipow2"]
    if not divide_by_rho:
        options.append("--no-divide-by-rho")
    args = settings(options)
    assert args.divide_by_rho is divide_by_rho
    result = {}
    attach_weight_metadata(result, args)
    assert result["gradient_divide_by_rho"] is divide_by_rho
    objects = [SimpleNamespace()]
    attach_weight_metadata(objects, args)
    assert objects[0].gradient_divide_by_rho is divide_by_rho


def test_metadata_matches_saved_effective_weight_identity():
    args = settings(["-ww", "--weight-token", "r2_weipow2"])
    result = {}
    attach_weight_metadata(result, args)
    assert result["weight_method"] == "r2" and result["weight_power"] == "2"
    objects = np.array([SimpleNamespace(), None], dtype=object)
    attach_weight_metadata(objects, args)
    assert objects[0].weight_token == "r2_weipow2" and objects[1] is None


def _run(tmp_path, script, args, mpi=False):
    command = [sys.executable, "-B", str(ROOT / "bin" / script), *map(str, args)]
    if mpi:
        launcher = shutil.which("mpiexec") or shutil.which("mpirun")
        if launcher is None:
            pytest.skip("MPI launcher unavailable")
        command = [launcher, "-n", "2", *command]
    env = os.environ.copy()
    env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    done = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90)
    (tmp_path / f"{script}.stdout.log").write_text(done.stdout)
    (tmp_path / f"{script}.stderr.log").write_text(done.stderr)
    assert done.returncode == 0, done.stdout + done.stderr
    return done


def _box_inputs(tmp_path, covariance=False, method="r2"):
    rng = np.random.default_rng(20261025)
    blocks = np.indices((2, 2, 2)).reshape(3, -1).T * 6
    data_pos = np.concatenate([origin + rng.uniform(0.3, 5.7, (8, 3)) for origin in blocks])
    marks = rng.uniform(1.2, 2.1, len(data_pos))
    data = np.column_stack((data_pos, marks))
    random = np.column_stack((rng.uniform(0.3, 5.7 if covariance else 11.7, (80, 3)), rng.uniform(0.5, 1.5, 80)))
    if method == "gn6":
        np.save(tmp_path / "positions.npy", data_pos)
        _run(tmp_path, "create_rho", ["positions.npy", "12", "data_gn6.npy", "-method", "gn6", "-n", "1"])
        data = np.load(tmp_path / "data_gn6.npy")
        assert data.shape == (len(data_pos), 7)
    else:
        np.save(tmp_path / f"data_{method}.npy", data)
    np.save(tmp_path / "random.npy", random)
    return data, random


@pytest.mark.parametrize("script", ["cal_MCF_box", "cal_MCF_box_mpi"])
@pytest.mark.parametrize("method,divide_by_rho", [("r2", True), ("gn6", True), ("gn6", False)])
def test_box_cli_effective_weights_and_metadata(tmp_path, script, method, divide_by_rho):
    data, random = _box_inputs(tmp_path, method=method)
    args = [f"data_{method}.npy", "random.npy", "1", "-smin", ".1", "-smax", "3", "-sbin", "2", "-mubin", "2",
            "-ww", "--weight-token", f"{method}_weipow2", "-tpcf", "result.pkl", "-rr", "rr.pkl"]
    if script.endswith("_mpi"):
        args += ["--boxsize", "12", "--ngrids", "2"]
    if not divide_by_rho:
        args.append("--no-divide-by-rho")
    _run(tmp_path, script, args, mpi=script.endswith("_mpi"))
    result = joblib.load(tmp_path / "result.pkl")
    assert result["weight_token"] == f"{method}_weipow2"
    mark = np.sqrt(np.sum(data[:, 4:] ** 2, axis=1)) if method == "gn6" else data[:, 3]
    if method == "gn6":
        assert result["gradient_divide_by_rho"] is divide_by_rho
        if divide_by_rho:
            mark /= data[:, 3]
    np.testing.assert_allclose(result["sum_wd"], np.sum(mark ** 2))
    np.testing.assert_allclose(result["sum_wr"], np.sum(random[:, 3]))


@pytest.mark.parametrize("script", ["cal_cov_source_tpcf", "cal_cov_source_tpcf_mpi"])
@pytest.mark.parametrize("method,divide_by_rho", [("r2", True), ("gn6", True), ("gn6", False)])
def test_covariance_cli_effective_weights_and_metadata(tmp_path, script, method, divide_by_rho):
    from LSS_python.cov import run_subsample_tpCF

    data, random = _box_inputs(tmp_path, covariance=True, method=method)
    args = [f"data_{method}.npy", "random.npy", "12", "2", "subsample", "-n", "1", "--s_max", "3",
            "--sbin", "2", "--mubin", "2", "-ww", "--weight-method", method, "-weipow", "2",
            "-tpcf", "result.pkl", "-rr", str(tmp_path / "rr.pkl")]
    if not divide_by_rho:
        args.append("--no-divide-by-rho")
    _run(tmp_path, script, args, mpi=script.endswith("_mpi"))
    objects = XismuCovarianceIO.load(tmp_path / "result.pkl")
    valid = [item for item in objects if item is not None]
    assert len(valid) == 8
    assert all(item.weight_token == f"{method}_weipow2" for item in valid)
    mark = np.sqrt(np.sum(data[:, 4:] ** 2, axis=1)) if method == "gn6" else data[:, 3]
    if method == "gn6":
        assert all(item.gradient_divide_by_rho is divide_by_rho for item in valid)
        if divide_by_rho:
            mark /= data[:, 3]
    expected_data = np.column_stack((data[:, :3], mark ** 2))
    reference = run_subsample_tpCF(
        expected_data, random, np.linspace(0, 3, 3), 2, with_weight=True,
        boxsize=12, ngrids=2, rr_filename=str(tmp_path / "reference_rr.pkl"),
        force_rr=True, nthreads=1, verbose=False, full_output=False,
    )
    for actual, expected in zip(objects, reference):
        assert (actual is None) == (expected is None)
        if actual is not None:
            for name in ("DDnorm", "DRnorm", "RRnorm"):
                np.testing.assert_allclose(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("method,divide_by_rho", [("r2", True), ("gn6", True), ("gn6", False)])
def test_mock_cli_powers_only_mark_and_normalizes_same_effective_weight(tmp_path, method, divide_by_rho):
    rng = np.random.default_rng(87)
    coords = np.column_stack((rng.uniform(20, 60, 40), rng.uniform(-10, 10, 40), rng.uniform(100, 110, 40)))
    base, marks = rng.uniform(0.7, 1.3, 40), rng.uniform(1.2, 2.1, 40)
    if method == "gn6":
        gradient = rng.normal(size=(len(coords), 3))
        rho = rng.uniform(.5, 2, len(coords))
        data = np.column_stack((coords, base, rho, gradient))
        marks = np.sqrt(np.sum(gradient ** 2, axis=1))
        if divide_by_rho:
            marks /= rho
    else:
        data = np.column_stack((coords, base, marks))
    random = np.column_stack((coords + [0.1, 0.1, 0.2], rng.uniform(0.8, 1.2, 40)))
    np.save(tmp_path / "data.npy", data)
    np.save(tmp_path / "random.npy", random)
    args = ["data.npy", "random.npy", "1", "-smin", ".1", "-smax", "15",
            "-sbin", "2", "-mubin", "2", "-ww", "--weight-token", f"{method}_weipow2", "-tpcf", "result.pkl", "-rr", "rr.pkl"]
    if not divide_by_rho:
        args.append("--no-divide-by-rho")
    _run(tmp_path, "cal_MCF_mock", args)
    result = joblib.load(tmp_path / "result.pkl")
    assert result["weight_token"] == f"{method}_weipow2"
    if method == "gn6":
        assert result["gradient_divide_by_rho"] is divide_by_rho
    np.testing.assert_allclose(result["sum_wd"], np.sum(base * marks ** 2))
    np.testing.assert_allclose(result["sum_wr"], np.sum(random[:, 3]))
