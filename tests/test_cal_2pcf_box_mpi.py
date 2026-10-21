"""Integration tests for the MPI simulation-box 2pCF command line."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "cal_2pCF_box_mpi"
BOX_SIZE = 20.0
NGRIDS = 2
SEDGES = np.linspace(0.5, 8.0, 4)
MUBIN = 2


def _mpi_launcher():
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None or importlib.util.find_spec("mpi4py") is None:
        pytest.skip("requires mpiexec/mpirun and mpi4py")
    return launcher


def _run_mpi(tmp_path, *arguments, timeout=90):
    command = [
        _mpi_launcher(), "-n", "2", sys.executable, str(SCRIPT),
        *map(str, arguments),
    ]
    environment = os.environ.copy()
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    return subprocess.run(
        command,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )


def _base_args(data_path, random_path):
    return [
        data_path, random_path, "1",
        "-smin", str(SEDGES[0]),
        "-smax", str(SEDGES[-1]),
        "-sbin", str(len(SEDGES) - 1),
        "-mubin", str(MUBIN),
        "--boxsize", str(BOX_SIZE),
        "--ngrids", str(NGRIDS),
    ]


def _assert_success(completed):
    assert completed.returncode == 0, (
        f"MPI command failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _raw_weight_sum(result, weighted):
    counts = np.asarray(result["npairs"], dtype=np.float64)
    if weighted:
        return counts * np.asarray(result["weightavg"], dtype=np.float64)
    return counts


def _serial_reference(data, random, weighted):
    from LSS_python.tpcf import cal_tpCF_from_pairs, run_tpCF

    pairs = run_tpCF(data, random, SEDGES, MUBIN, weighted, nthreads=1)
    xi = cal_tpCF_from_pairs(
        pairs["DD"], pairs["DR"], pairs["RR"],
        data, random, len(SEDGES) - 1, MUBIN, weighted,
    )
    return pairs, xi


def _compare_pair(actual, expected, weighted):
    np.testing.assert_array_equal(actual["npairs"], expected["npairs"])
    np.testing.assert_allclose(
        _raw_weight_sum(actual, weighted), _raw_weight_sum(expected, weighted),
        rtol=2e-12, atol=1e-10,
    )


def _compare_xi(actual, expected):
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_allclose(actual_value, expected_value, rtol=2e-12, atol=1e-10, equal_nan=True)
        elif isinstance(expected_value, (float, int, np.floating, np.integer)):
            np.testing.assert_allclose(actual_value, expected_value, rtol=2e-12, atol=1e-10, equal_nan=True)
        else:
            assert actual_value == expected_value


def test_no_arguments_and_help_do_not_initialize_mpi():
    for arguments in ([], ["--help"]):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert completed.returncode == 0
        assert "usage:" in completed.stdout.lower()
        assert "--boxsize" in completed.stdout
        assert "--ngrids" in completed.stdout


def test_weighted_full_run_cache_controls_and_unit_weights(tmp_path):
    rng = np.random.default_rng(2701)
    data = np.column_stack((rng.uniform(0.5, 19.5, size=(10, 3)), rng.uniform(0.4, 2.0, 10)))
    random = np.column_stack((rng.uniform(0.5, 19.5, size=(14, 3)), rng.uniform(0.3, 1.8, 14)))
    data_path, random_path = tmp_path / "data.npy", tmp_path / "random.npy"
    np.save(data_path, data)
    np.save(random_path, random)
    reference_pairs, reference_xi = _serial_reference(data, random, weighted=True)
    args = _base_args(data_path, random_path) + ["--with_weight", "--save_dd_dr"]

    _assert_success(_run_mpi(tmp_path, *args))
    for name in ("DD", "DR", "RR"):
        cached = joblib.load(tmp_path / f"{name}_temp.pkl")
        _compare_pair(cached, reference_pairs[name], weighted=True)
    saved_xi = joblib.load(tmp_path / "tpCF_temp.pkl")
    _compare_xi(saved_xi, reference_xi)
    assert saved_xi["with_weight"] is True

    cache_paths = [tmp_path / f"{name}_temp.pkl" for name in ("DD", "DR", "RR")] + [tmp_path / "tpCF_temp.pkl"]
    original_mtimes = {path: path.stat().st_mtime_ns for path in cache_paths}
    skipped = _run_mpi(tmp_path, *args)
    _assert_success(skipped)
    assert "skip" in skipped.stdout.lower()
    assert {path: path.stat().st_mtime_ns for path in cache_paths} == original_mtimes

    (tmp_path / "tpCF_temp.pkl").unlink()
    cached_only = _run_mpi(tmp_path, *args)
    _assert_success(cached_only)
    assert {path: path.stat().st_mtime_ns for path in cache_paths[:-1]} == {
        path: original_mtimes[path] for path in cache_paths[:-1]
    }
    _compare_xi(joblib.load(tmp_path / "tpCF_temp.pkl"), reference_xi)

    recompute_dd_dr = _run_mpi(tmp_path, *args, "-force")
    _assert_success(recompute_dd_dr)
    assert (tmp_path / "RR_temp.pkl").stat().st_mtime_ns == original_mtimes[tmp_path / "RR_temp.pkl"]
    assert (tmp_path / "DD_temp.pkl").stat().st_mtime_ns != original_mtimes[tmp_path / "DD_temp.pkl"]
    assert (tmp_path / "DR_temp.pkl").stat().st_mtime_ns != original_mtimes[tmp_path / "DR_temp.pkl"]

    recompute_all = _run_mpi(tmp_path, *args, "-force", "--force_rr")
    _assert_success(recompute_all)
    assert (tmp_path / "RR_temp.pkl").stat().st_mtime_ns != original_mtimes[tmp_path / "RR_temp.pkl"]

    unit_weight = _run_mpi(tmp_path, *args, "-force", "--force_rr", "--set_weight_to_one")
    _assert_success(unit_weight)
    ones_data = data.copy()
    ones_random = random.copy()
    ones_data[:, 3] = 1.0
    ones_random[:, 3] = 1.0
    unit_pairs, unit_xi = _serial_reference(ones_data, ones_random, weighted=True)
    for name in ("DD", "DR", "RR"):
        _compare_pair(joblib.load(tmp_path / f"{name}_temp.pkl"), unit_pairs[name], weighted=True)
    saved_unit_xi = joblib.load(tmp_path / "tpCF_temp.pkl")
    _compare_xi(saved_unit_xi, unit_xi)


def test_rr_only_does_not_read_data_catalog(tmp_path):
    rng = np.random.default_rng(2711)
    random = rng.uniform(0.5, 19.5, size=(12, 3))
    random_path = tmp_path / "random.npy"
    np.save(random_path, random)
    output_path = tmp_path / "rr-only.pkl"
    random_with_unit_weights = np.column_stack((random, np.ones(len(random))))
    from LSS_python.tpcf import run_tpCF

    expected = run_tpCF(
        None, random_with_unit_weights, SEDGES, MUBIN, True,
        run_parts=["RR"], nthreads=1,
    )["RR"]
    arguments = _base_args(tmp_path / "missing-data.npy", random_path) + [
        "--with_weight", "--only_run_rr", "--rr_filename", output_path,
    ]
    completed = _run_mpi(
        tmp_path,
        *arguments,
    )
    _assert_success(completed)
    assert output_path.exists()
    _compare_pair(joblib.load(output_path), expected, weighted=True)
    original_mtime = output_path.stat().st_mtime_ns

    skipped = _run_mpi(tmp_path, *arguments)
    _assert_success(skipped)
    assert "skip" in skipped.stdout.lower()
    assert output_path.stat().st_mtime_ns == original_mtime

    forced = _run_mpi(tmp_path, *arguments, "--force_rr")
    _assert_success(forced)
    assert output_path.stat().st_mtime_ns != original_mtime


@pytest.mark.parametrize("failure", ["read", "rr_output", "tpcf_output"])
def test_root_io_errors_exit_without_hanging(tmp_path, failure):
    rng = np.random.default_rng(2721)
    data = rng.uniform(0.5, 19.5, size=(8, 3))
    random = rng.uniform(0.5, 19.5, size=(10, 3))
    data_path, random_path = tmp_path / "data.npy", tmp_path / "random.npy"
    np.save(data_path, data)
    np.save(random_path, random)
    if failure == "read":
        args = _base_args(data_path, tmp_path / "missing-random.npy") + ["--only_run_rr"]
    elif failure == "rr_output":
        blocked_parent = tmp_path / "not-a-directory"
        blocked_parent.write_text("file", encoding="utf-8")
        args = _base_args(data_path, random_path) + [
            "--only_run_rr", "--force_rr", "--rr_filename", blocked_parent / "rr.pkl",
        ]
    else:
        blocked_parent = tmp_path / "not-a-directory"
        blocked_parent.write_text("file", encoding="utf-8")
        args = _base_args(data_path, random_path) + ["--tpcf_filename", blocked_parent / "tpcf.pkl"]
    completed = _run_mpi(tmp_path, *args, timeout=30)
    assert completed.returncode != 0
    assert "Traceback" in completed.stderr or "error" in completed.stderr.lower()
