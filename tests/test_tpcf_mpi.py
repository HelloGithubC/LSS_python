"""Launch the MPI validation driver in a separate two-rank process."""

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest


def test_validation_driver_without_arguments_prints_help():
    driver = os.path.join(os.path.dirname(__file__), "validate_tpcf_mpi.py")
    completed = subprocess.run(
        [sys.executable, driver], check=False, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_tpcf_mpi_matches_serial(tmp_path):
    """Exercise collective counting without importing MPI in the pytest host."""
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None or importlib.util.find_spec("mpi4py") is None:
        pytest.skip("requires mpiexec/mpirun and mpi4py")

    driver = os.path.join(os.path.dirname(__file__), "validate_tpcf_mpi.py")
    command = [
        launcher, "-n", "2", sys.executable, driver,
        "--output-dir", str(tmp_path),
    ]
    environment = os.environ.copy()
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )
    assert completed.returncode == 0, (
        f"MPI validation failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
