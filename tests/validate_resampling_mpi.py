"""Compare one small Jackknife sample and one subsample sample with serial runs.

Run collectively with --output-dir. Root generates deterministic weighted
float64 catalogs, saves full serial/MPI outputs and separate RR caches, and
compares counts, normalizations and xi. The same samples are then reloaded from
RR cache to check root-only cache reuse and the default array return. No other
validation suite is invoked. Existing output filenames are overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import traceback

import joblib
import numpy as np


RTOL, ATOL = 2e-12, 1e-10


def _catalogs():
    rng = np.random.default_rng(20261024)

    def partitioned(populations):
        chunks = []
        for i, size in enumerate(populations):
            origin = np.asarray(np.unravel_index(i, (2, 2, 2))) * 10.
            chunks.append(np.column_stack((origin + rng.uniform(.05, 9.95, (size, 3)),
                                           rng.uniform(.5, 1.5, size))))
        return np.concatenate(chunks)

    data = partitioned([0, 1] + [16] * 6)
    random = partitioned([0, 1] + [64] * 6)
    sub_random = np.column_stack((rng.uniform(.05, 9.95, (384, 3)), rng.uniform(.5, 1.5, 384)))
    return data, random, sub_random


def _compare(actual, expected, metrics, category="pair_counts"):
    """Compare nested counts and result objects without reconstructing objects."""
    if actual is None or expected is None:
        assert actual is expected
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            _compare(actual[key], expected[key], metrics, category)
    elif isinstance(expected, np.ndarray) and expected.dtype.hasobject:
        assert actual.shape == expected.shape
        for a, b in zip(actual.flat, expected.flat):
            _compare(a, b, metrics, category)
    elif isinstance(expected, np.ndarray) and expected.dtype.names:
        assert actual.dtype == expected.dtype
        for field in expected.dtype.names:
            _compare(actual[field], expected[field], metrics, category)
    elif hasattr(expected, "DDnorm"):
        assert type(actual) is type(expected)
        for name in ("DDnorm", "DRnorm", "RRnorm"):
            _compare(getattr(actual, name), getattr(expected, name), metrics, "normalizations")
        for name in ("DD", "DR", "RR", "S", "Mu"):
            _compare(getattr(actual, name), getattr(expected, name), metrics, "normalized_counts")
        a = (actual.DD - 2 * actual.DR + actual.RR) / actual.RR
        b = (expected.DD - 2 * expected.DR + expected.RR) / expected.RR
        _compare(a, b, metrics, "xi")
    else:
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape and a.dtype == b.dtype
        if np.issubdtype(b.dtype, np.integer):
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
        if b.size:
            error = float(np.max(np.abs(a.astype(float) - b.astype(float))))
            metrics[category] = max(metrics.get(category, 0.), error)


def _root_check(comm, root, function):
    """Propagate validation failures before peers enter another measurement."""
    result, error = None, None
    if comm.rank == root:
        try:
            result = function()
        except Exception:
            error = traceback.format_exc()
    error = comm.bcast(error, root=root)
    if error is not None:
        raise AssertionError(error)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", default=1, type=int, help="Input/output rank (default: 1)")
    if (argv is None and len(sys.argv) == 1) or argv == []:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    from mpi4py import MPI
    from LSS_python import tpcf_mpi
    from LSS_python.cov import run_jackknife_tpCF, run_subsample_tpCF

    comm, root = MPI.COMM_WORLD, args.root
    if not 0 <= root < comm.size:
        parser.error("--root must be a rank in this communicator")
    output = args.output_dir.resolve()
    tpcf_mpi._TRANSFER_CHUNK_BYTES = 128
    inputs = None

    def prepare():
        nonlocal inputs
        output.mkdir(parents=True, exist_ok=True)
        inputs = _catalogs()
        np.savez(output / "catalogs.npz", data=inputs[0], random=inputs[1],
                 sub_random=inputs[2], sedges=np.linspace(.5, 5., 7))

    _root_check(comm, root, prepare)
    hosts = comm.allgather(socket.gethostname())
    summary = {"ranks": comm.size, "root": root, "hosts": hosts, "seed": 20261024,
               "data_count": 97, "jackknife_random_count": 385, "subsample_random_count": 384,
               "dtype": "float64", "with_weight": True, "boxsize": 20., "ngrids": 2,
               "sedges": np.linspace(.5, 5., 7).tolist(), "mubin": 4, "nthreads": 1,
               "validation_chunk_bytes": 128, "production_chunk_bytes": 256 * 1024**2,
               "rtol": RTOL, "atol": ATOL, "methods": {}}
    for method, serial, parallel, key, random_index in (
        ("jackknife", run_jackknife_tpCF, tpcf_mpi.run_jackknife_tpCF_mpi, "xismu_jk", 1),
        ("subsample", run_subsample_tpCF, tpcf_mpi.run_subsample_tpCF_mpi, "xismu_sub", 2),
    ):
        reference, settings = None, None

        def serial_run():
            nonlocal reference, settings
            settings = dict(data=inputs[0], random=inputs[random_index],
                            sedges=np.linspace(.5, 5., 7), mubin=4, with_weight=True,
                            boxsize=20., ngrids=2, nthreads=1, full_output=True)
            reference = serial(**settings, rr_filename=str(output / f"{method}_RR_serial.pkl"))
            joblib.dump(reference, output / f"{method}_serial.pkl")

        _root_check(comm, root, serial_run)
        # Only root supplies measurement settings as well as catalogs.
        mpi_settings = (dict(settings, rr_filename=str(output / f"{method}_RR_mpi.pkl"), verbose=True)
                        if comm.rank == root else
                        dict(data=None, random=None, sedges=None, mubin=None, with_weight=None,
                             boxsize=None, ngrids=None))
        actual = parallel(**mpi_settings, mpicomm=comm, root=root)
        assert (actual is None) == (comm.rank != root)
        metrics, cache_hash = {}, None

        def compare_fresh():
            nonlocal cache_hash
            _compare(actual, reference, metrics)
            _compare(joblib.load(output / f"{method}_RR_mpi.pkl"),
                     joblib.load(output / f"{method}_RR_serial.pkl"), metrics)
            joblib.dump(actual, output / f"{method}_mpi.pkl")
            cache_hash = hashlib.sha256((output / f"{method}_RR_mpi.pkl").read_bytes()).hexdigest()

        _root_check(comm, root, compare_fresh)
        if comm.rank == root:
            mpi_settings.update(full_output=False, verbose=False)
        cached = parallel(**mpi_settings, mpicomm=comm, root=root)
        assert (cached is None) == (comm.rank != root)

        def compare_cached():
            cached_metrics = {}
            _compare(cached, actual[key], cached_metrics)
            assert hashlib.sha256((output / f"{method}_RR_mpi.pkl").read_bytes()).hexdigest() == cache_hash
            summary["methods"][method] = {
                "passed": True, "sample_count": len(actual[key]),
                "none_sample_indices": [i for i, item in enumerate(actual[key]) if item is None],
                "max_absolute_errors": metrics, "cache_reload_max_absolute_errors": cached_metrics,
                "cache_unchanged": True, "nonroot_returns_none": True,
                "serial_output": str(output / f"{method}_serial.pkl"),
                "mpi_output": str(output / f"{method}_mpi.pkl"),
            }
            print(f"{method}: PASS; max absolute errors={metrics}", flush=True)

        _root_check(comm, root, compare_cached)

    def finish():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Results: {output}", flush=True)

    _root_check(comm, root, finish)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
