"""Exercise compact covariance output through both measurement entry points."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pytest

from LSS_python.xismu import XismuCovarianceIO


ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path, script, kind, *extra):
    command = [sys.executable, "-B", str(ROOT / "bin" / script),
               "data.npy", "random.npy", "12", "2", kind,
               "-n", "1", "--s_max", "3", "--sbin", "2", "--mubin", "2",
               "-tpcf", "result.pkl", "-rr", "rr.pkl", *extra]
    if script.endswith("_mpi"):
        launcher = shutil.which("mpiexec") or shutil.which("mpirun")
        if launcher is None or importlib.util.find_spec("mpi4py") is None:
            pytest.skip("requires mpiexec/mpirun and mpi4py")
        command = [launcher, "-n", "2", *command]
    env = os.environ.copy()
    env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=120)


@pytest.mark.parametrize("script", ["cal_cov_source_tpcf", "cal_cov_source_tpcf_mpi"])
@pytest.mark.parametrize("kind", ["jk", "subsample"])
def test_compact_output_skip_force_and_reject_legacy(tmp_path, script, kind):
    if importlib.util.find_spec("Corrfunc") is None:
        pytest.skip("requires Corrfunc")
    rng = np.random.default_rng(13)
    origins = np.indices((2, 2, 2)).reshape(3, -1).T * 6
    data = np.concatenate([origin + rng.uniform(.3, 5.7, (8, 3)) for origin in origins])
    # Subsample uses one random subbox; JK uses the whole box.
    random = rng.uniform(.3, 5.7 if kind == "subsample" else 11.7, (120, 3))
    np.save(tmp_path / "data.npy", data)
    np.save(tmp_path / "random.npy", random)
    filename = tmp_path / "result.pkl"

    first = run_cli(tmp_path, script, kind)
    assert first.returncode == 0, first.stdout + first.stderr
    payload = joblib.load(filename)
    assert payload["format"] == XismuCovarianceIO._FORMAT
    assert payload["covariance_type"] == kind
    samples = XismuCovarianceIO.load(filename)
    assert len(samples) == 8
    assert any(sample is not None for sample in samples)

    original = filename.read_bytes()
    skipped = run_cli(tmp_path, script, kind)
    assert skipped.returncode == 0, skipped.stdout + skipped.stderr
    assert "skip" in skipped.stdout.lower()
    assert filename.read_bytes() == original

    # A legacy object array must be migrated explicitly, never silently skipped.
    joblib.dump(samples, filename)
    legacy = filename.read_bytes()
    rejected = run_cli(tmp_path, script, kind)
    assert rejected.returncode != 0
    assert "migrate_xismu_covariance" in rejected.stdout + rejected.stderr
    assert filename.read_bytes() == legacy

    forced = run_cli(tmp_path, script, kind, "-f")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert joblib.load(filename)["format"] == XismuCovarianceIO._FORMAT
    assert len(XismuCovarianceIO.load(filename)) == len(samples)
