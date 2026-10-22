"""Atomic migration tests for historical xismu pickle references."""

import os
import shutil
import subprocess
import sys
import types
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest

from LSS_python.xismu import xismu


REPOSITORY = Path(__file__).resolve().parents[1]
CLI = REPOSITORY / "bin" / "migrate_xismu"


def _make_xismu(values):
    """Build a stateful xismu object without invoking its constructor."""
    result = xismu.__new__(xismu)
    result.sbin = len(values)
    result.filename = None
    result.DD = np.asarray(values, dtype=np.float64)
    result.s_array = np.asarray([1.0, np.nan], dtype=np.float64)
    result.extra = {"labels": ("a", "b"), "enabled": True}
    return result


def _write_legacy_file(path, payload, monkeypatch, compress=0):
    """Serialize the canonical class under its historical module spelling."""
    legacy_module = types.ModuleType("LSS_python.tpcf")
    legacy_module.xismu = xismu
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "LSS_python.tpcf", legacy_module)
        original_module = xismu.__module__
        try:
            xismu.__module__ = "LSS_python.tpcf"
            joblib.dump(payload, path, compress=compress)
        finally:
            xismu.__module__ = original_module


def _temporary_files(directory, original_name):
    prefix = f".{original_name}.migrate-"
    return [path for path in directory.iterdir() if path.name.startswith(prefix)]


@pytest.mark.parametrize("compression", [0, 3, ("gzip", 3), ("bz2", 3), ("lzma", 3), ("xz", 3)])
def test_nested_graph_migrates_once_and_preserves_values_aliases_and_mode(
    tmp_path, monkeypatch, compression,
):
    from LSS_python.xismu import migrate_xismu_file

    first = _make_xismu([1.0, np.nan, -4.5])
    second = _make_xismu([8.0, 9.0])
    objects = np.empty(2, dtype=object)
    objects[0] = first
    objects[1] = second
    payload = {
        "nested": [first, (second, first)],
        "array": objects,
        "ordinary": np.array([[1, 2], [3, 4]], dtype=np.int16),
    }
    path = tmp_path / "legacy.joblib"
    _write_legacy_file(path, payload, monkeypatch, compress=compression)
    path.chmod(0o640)
    original_mode = path.stat().st_mode & 0o777

    # joblib materializes object-array entries separately from ordinary
    # containers, so count every unique instance in the loaded graph.
    assert migrate_xismu_file(path) == 4

    loaded = joblib.load(path)
    migrated_first = loaded["nested"][0]
    assert type(migrated_first) is xismu
    assert type(migrated_first).__module__ == "LSS_python.xismu"
    assert migrated_first is loaded["nested"][1][1]
    assert type(loaded["array"][0]) is xismu
    assert type(loaded["array"][1]) is xismu
    np.testing.assert_equal(migrated_first.DD, first.DD)
    np.testing.assert_equal(migrated_first.s_array, first.s_array)
    assert migrated_first.filename is None
    assert migrated_first.extra == first.extra
    assert loaded["ordinary"].dtype == np.dtype(np.int16)
    assert loaded["ordinary"].shape == (2, 2)
    np.testing.assert_array_equal(loaded["ordinary"], payload["ordinary"])
    assert path.stat().st_mode & 0o777 == original_mode

    # Repeating a migration must avoid rewriting and full data validation.
    from LSS_python import xismu as module

    migrated_bytes = path.read_bytes()
    migrated_stat = path.stat()

    def unexpected_rewrite(*args, **kwargs):
        raise AssertionError("An already migrated archive must be skipped")

    monkeypatch.setattr(module.joblib, "dump", unexpected_rewrite)
    monkeypatch.setattr(module, "_same_migration_value", unexpected_rewrite)
    assert migrate_xismu_file(path) == 0
    assert path.read_bytes() == migrated_bytes
    assert path.stat().st_mtime_ns == migrated_stat.st_mtime_ns
    assert path.stat().st_ino == migrated_stat.st_ino
    assert _temporary_files(tmp_path, path.name) == []


@pytest.mark.parametrize("object_array", [False, True])
def test_current_class_archive_skips_even_with_old_path_in_attributes(
    tmp_path, monkeypatch, object_array,
):
    from LSS_python import xismu as module

    obj = _make_xismu([1, 2])
    obj.old_module = "LSS_python.tpcf"
    payload = np.array([obj], dtype=object) if object_array else obj
    path = tmp_path / "current.joblib"
    joblib.dump(payload, path)
    original_bytes = path.read_bytes()
    original_stat = path.stat()

    def unexpected_rewrite(*args, **kwargs):
        raise AssertionError("A current archive must be skipped")

    monkeypatch.setattr(module.joblib, "dump", unexpected_rewrite)
    monkeypatch.setattr(module, "_same_migration_value", unexpected_rewrite)
    assert module.migrate_xismu_file(path) == 0
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert path.stat().st_ino == original_stat.st_ino
    assert _temporary_files(tmp_path, path.name) == []


def test_old_references_are_detected_after_warning_has_already_fired(
    tmp_path, monkeypatch,
):
    from LSS_python import tpcf
    from LSS_python.xismu import migrate_xismu_file

    paths = [tmp_path / name for name in ("first.joblib", "second.joblib")]
    for path in paths:
        _write_legacy_file(path, _make_xismu([1, 2]), monkeypatch)
    monkeypatch.setattr(tpcf, "_legacy_xismu_warned", False)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", FutureWarning)
        assert tpcf.xismu is xismu
        assert migrate_xismu_file(paths[0]) == 1
        assert migrate_xismu_file(paths[1]) == 1
        assert migrate_xismu_file(paths[0]) == 0
    assert len(seen) == 1


def test_mixed_old_and_current_classes_rewrites_only_once(tmp_path, monkeypatch):
    from LSS_python.xismu import migrate_xismu_file

    legacy_class = type("xismu", (), {"__module__": "LSS_python.tpcf"})
    legacy_object = legacy_class()
    legacy_object.__dict__.update(vars(_make_xismu([3, 4])))
    payload = {"old": legacy_object, "current": _make_xismu([5, 6])}
    legacy_module = types.ModuleType("LSS_python.tpcf")
    legacy_module.xismu = legacy_class
    path = tmp_path / "mixed.joblib"
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "LSS_python.tpcf", legacy_module)
        joblib.dump(payload, path)

    assert migrate_xismu_file(path) == 2
    loaded = joblib.load(path)
    assert type(loaded["old"]) is type(loaded["current"]) is xismu
    np.testing.assert_array_equal(loaded["old"].DD, [3, 4])
    np.testing.assert_array_equal(loaded["current"].DD, [5, 6])
    assert migrate_xismu_file(path) == 0


def test_uncompressed_joblib_is_migrated_to_canonical_class(tmp_path, monkeypatch):
    from LSS_python.xismu import migrate_xismu_file

    obj = _make_xismu([2, 3])
    path = tmp_path / "legacy.pkl"
    _write_legacy_file(path, obj, monkeypatch)

    assert migrate_xismu_file(path) == 1
    reloaded = joblib.load(path)
    assert type(reloaded) is xismu
    assert type(reloaded).__module__ == "LSS_python.xismu"


def test_file_without_xismu_is_skipped_byte_for_byte(tmp_path):
    from LSS_python.xismu import migrate_xismu_file

    path = tmp_path / "plain.joblib"
    joblib.dump({"array": np.arange(5), "missing": None}, path)
    before = path.read_bytes()

    assert migrate_xismu_file(path) == 0
    assert path.read_bytes() == before
    assert _temporary_files(tmp_path, path.name) == []


def test_symlink_input_replaces_real_source_and_keeps_link(tmp_path, monkeypatch):
    from LSS_python.xismu import migrate_xismu_file

    obj = _make_xismu([3, 4])
    target = tmp_path / "target.joblib"
    _write_legacy_file(target, obj, monkeypatch)
    link = tmp_path / "alias.joblib"
    link.symlink_to(target.name)

    assert migrate_xismu_file(link) == 1
    assert link.is_symlink()
    assert type(joblib.load(target)) is xismu


def test_dump_failure_keeps_original_bytes_and_removes_temp_files(
    tmp_path, monkeypatch,
):
    from LSS_python import xismu as module

    obj = _make_xismu([5, 6])
    path = tmp_path / "dump-failure.joblib"
    _write_legacy_file(path, obj, monkeypatch)
    original = path.read_bytes()

    def fail_dump(*args, **kwargs):
        raise OSError("simulated dump failure")

    monkeypatch.setattr(module.joblib, "dump", fail_dump)
    with pytest.raises(OSError, match="simulated dump failure"):
        module.migrate_xismu_file(path)
    assert path.read_bytes() == original
    assert _temporary_files(tmp_path, path.name) == []


def test_reload_failure_keeps_original_bytes_and_removes_temp_files(
    tmp_path, monkeypatch,
):
    from LSS_python import xismu as module

    obj = _make_xismu([7, 8])
    path = tmp_path / "reload-failure.joblib"
    _write_legacy_file(path, obj, monkeypatch)
    original = path.read_bytes()
    real_load = module.joblib.load
    calls = 0

    def fail_validation_load(filename, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated reload failure")
        return real_load(filename, *args, **kwargs)

    monkeypatch.setattr(module.joblib, "load", fail_validation_load)
    with pytest.raises(ValueError, match="simulated reload failure"):
        module.migrate_xismu_file(path)
    assert path.read_bytes() == original
    assert _temporary_files(tmp_path, path.name) == []


def test_validation_rejects_changed_array_values_without_replacing_source(
    tmp_path, monkeypatch,
):
    from LSS_python import xismu as module

    obj = _make_xismu([9, 10])
    path = tmp_path / "validation-failure.joblib"
    _write_legacy_file(path, obj, monkeypatch)
    original = path.read_bytes()
    real_dump = module.joblib.dump

    def dump_changed_payload(payload, filename, *args, **kwargs):
        old_value = payload.DD[0]
        payload.DD[0] = -100
        try:
            return real_dump(payload, filename, *args, **kwargs)
        finally:
            payload.DD[0] = old_value

    monkeypatch.setattr(module.joblib, "dump", dump_changed_payload)
    with pytest.raises(ValueError, match="changed object data"):
        module.migrate_xismu_file(path)
    assert path.read_bytes() == original
    assert _temporary_files(tmp_path, path.name) == []


def test_baseline_fixture_smoke_uses_copy_and_leaves_fixture_unchanged(tmp_path):
    from LSS_python.xismu import migrate_xismu_file

    fixture = (
        REPOSITORY / "validation" / "xismu_reorganization"
        / "20261021_kjo9l5p6" / "legacy_xismu_fixture.joblib"
    )
    if not fixture.is_file():
        pytest.skip("local legacy baseline fixture is not part of a clean checkout")
    source_bytes = fixture.read_bytes()
    smoke_copy = tmp_path / "legacy_xismu_fixture_copy.joblib"
    shutil.copy2(fixture, smoke_copy)

    count = migrate_xismu_file(smoke_copy)

    assert count >= 1
    migrated = joblib.load(smoke_copy)
    assert type(migrated["xismu_full"]) is xismu
    assert migrated["xismu_jk"].dtype.hasobject
    assert all(type(item) is xismu for item in migrated["xismu_jk"].flat)
    assert fixture.read_bytes() == source_bytes


def _run_cli(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, str(CLI), *map(str, args)],
        cwd=REPOSITORY,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_no_arguments_and_help_return_zero_with_usage():
    no_args = _run_cli()
    explicit_help = _run_cli("--help")

    assert no_args.returncode == 0
    assert "usage:" in no_args.stdout.lower()
    assert no_args.stderr == ""
    assert explicit_help.returncode == 0
    assert "PATH" in explicit_help.stdout


def test_cli_batches_files_continues_after_error_and_returns_failure(
    tmp_path, monkeypatch,
):
    valid = tmp_path / "valid.joblib"
    plain = tmp_path / "plain.joblib"
    invalid = tmp_path / "missing.joblib"
    _write_legacy_file(valid, _make_xismu([1, 2]), monkeypatch)
    joblib.dump({"plain": True}, plain)

    result = _run_cli(valid, invalid, plain)

    assert result.returncode == 1
    assert f"migrated: {valid}" in result.stdout
    assert f"skipped: {plain}" in result.stdout
    assert f"error: {invalid}:" in result.stderr
    assert type(joblib.load(valid)) is xismu

    migrated_bytes = valid.read_bytes()
    repeated = _run_cli(valid)
    assert repeated.returncode == 0
    assert f"skipped: {valid} (no legacy xismu references)" in repeated.stdout
    assert "migrated:" not in repeated.stdout
    assert repeated.stderr == ""
    assert valid.read_bytes() == migrated_bytes
