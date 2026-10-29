"""Tests for safe in-place covariance archive migration."""

import os
from pathlib import Path
import runpy
import subprocess
import sys

import joblib
import numpy as np
import pytest

from LSS_python.xismu import XismuCovarianceIO, xismu


SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "migrate_xismu_covariance"


def make_samples(kind="jk"):
    s, mu = np.meshgrid(np.arange(1, 5, dtype=np.float32),
                        np.arange(1, 4, dtype=np.float32) / 4, indexing="ij")
    samples = []
    for i in range(2):
        sample = xismu(smax=4, sbin=4, mubin=3, DDnorm=2 + i,
                       DRnorm=3 + i, RRnorm=4, S=s.copy(), Mu=mu.copy(),
                       DD=np.full((4, 3), 10 + i, dtype=np.float32),
                       DR=np.full((4, 3), 5 + i, dtype=np.float32),
                       RR=np.full((4, 3), 4, dtype=np.float32))
        sample.custom_metadata = {"label": str(i)}
        if kind == "jk" and i:
            sample.RR *= 2
            sample.RRnorm *= 2
        samples.append(sample)
    samples[0].get_xis()
    return np.array([samples[0], None, samples[1]], dtype=object)


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, check=False)


def assert_equal(expected, actual):
    assert actual.shape == expected.shape
    for before, after in zip(expected, actual):
        if before is None:
            assert after is None
            continue
        assert vars(before).keys() == vars(after).keys()
        for name, value in vars(before).items():
            result = vars(after)[name]
            if isinstance(value, np.ndarray):
                assert result.dtype == value.dtype
                np.testing.assert_array_equal(result, value)
            else:
                assert result == value
        np.testing.assert_array_equal(before.get_xis(), after.get_xis())


def test_no_arguments_prints_help():
    result = run_cli()
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("kind", ["jk", "subsample"])
def test_migrates_in_place_and_preserves_state(tmp_path, kind):
    path = tmp_path / "legacy.pkl"
    expected = make_samples(kind)
    joblib.dump(expected, path)
    path.chmod(0o640)
    result = run_cli(path, "--covariance-type", kind)
    assert result.returncode == 0, result.stderr
    assert joblib.load(path)["format"] == XismuCovarianceIO._FORMAT
    assert_equal(expected, XismuCovarianceIO.load(path))
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_symlink_remains_and_target_is_replaced(tmp_path):
    target = tmp_path / "legacy.pkl"
    link = tmp_path / "link.pkl"
    expected = make_samples()
    joblib.dump(expected, target)
    link.symlink_to(target.name)
    result = run_cli(link, "--covariance-type", "jk")
    assert result.returncode == 0, result.stderr
    assert link.is_symlink() and os.readlink(link) == target.name
    assert_equal(expected, XismuCovarianceIO.load(target))
    assert set(tmp_path.iterdir()) == {target, link}


def test_compact_file_is_skipped_without_modification(tmp_path):
    path = tmp_path / "input.pkl"
    XismuCovarianceIO.save(path, make_samples(), "jk")
    initial = path.read_bytes()
    initial_stat = path.stat()
    result = run_cli(path, "--covariance-type", "jk")
    assert result.returncode == 0, result.stderr
    assert "skipped" in result.stdout and "already converted" in result.stdout
    assert path.read_bytes() == initial
    assert path.stat().st_ino == initial_stat.st_ino
    assert path.stat().st_mtime_ns == initial_stat.st_mtime_ns
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("case", ["wrong_type", "invalid", "invalid_kind", "unsupported_version", "malformed"])
def test_rejected_input_is_unchanged(tmp_path, case):
    path = tmp_path / "input.pkl"
    if case in ("unsupported_version", "malformed"):
        XismuCovarianceIO.save(path, make_samples(), "jk")
        payload = joblib.load(path)
        if case == "unsupported_version":
            payload["version"] = 999
        else:
            payload.pop("samples")
        joblib.dump(payload, path)
    elif case == "invalid":
        path.write_bytes(b"not a joblib file")
    else:
        joblib.dump(make_samples(), path)
    initial = path.read_bytes()
    kind = {"wrong_type": "subsample", "invalid_kind": "bad"}.get(case, "jk")
    result = run_cli(path, "--covariance-type", kind)
    assert result.returncode != 0
    assert path.read_bytes() == initial
    assert list(tmp_path.iterdir()) == [path]


def test_failure_cleans_partial_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "legacy.pkl"
    joblib.dump(make_samples(), path)
    initial = path.read_bytes()
    migrate = runpy.run_path(str(SCRIPT))["migrate"]

    def fail_save(filename, samples, covariance_type):
        Path(filename).write_bytes(b"incomplete")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(XismuCovarianceIO, "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        migrate(path, "jk")
    assert path.read_bytes() == initial
    assert list(tmp_path.iterdir()) == [path]


def test_verification_failure_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "legacy.pkl"
    joblib.dump(make_samples(), path)
    initial = path.read_bytes()
    migrate = runpy.run_path(str(SCRIPT))["migrate"]
    load = XismuCovarianceIO.load

    def changed_load(filename):
        result = load(filename)
        result[0].DD[0, 0] += 1
        return result

    monkeypatch.setattr(XismuCovarianceIO, "load", changed_load)
    with pytest.raises(ValueError, match="verification failed"):
        migrate(path, "jk")
    assert path.read_bytes() == initial
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_input_change_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "legacy.pkl"
    joblib.dump(make_samples(), path)
    initial = path.read_bytes()
    initial_stat = path.stat()
    migrate = runpy.run_path(str(SCRIPT))["migrate"]
    save = XismuCovarianceIO.save

    def change_source(filename, samples, covariance_type):
        save(filename, samples, covariance_type)
        os.utime(path, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns + 1_000_000))

    monkeypatch.setattr(XismuCovarianceIO, "save", change_source)
    with pytest.raises(RuntimeError, match="changed during conversion"):
        migrate(path, "jk")
    assert path.read_bytes() == initial
    assert list(tmp_path.iterdir()) == [path]
