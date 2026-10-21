"""Compare distributed ``run_tpCF_mpi`` results with serial Corrfunc."""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ndata", type=positive_int, default=256)
    parser.add_argument("--nrandom", type=positive_int, default=512)
    parser.add_argument("--boxsize", type=float, default=900.0)
    parser.add_argument("--ngrids", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser


def collective_check(comm, condition, message):
    """Raise on every rank when any rank reports a failed check."""
    states = comm.allgather((bool(condition), str(message)))
    failures = [f"rank {rank}: {detail}" for rank, (ok, detail) in enumerate(states) if not ok]
    if failures:
        raise AssertionError("; ".join(failures))


def catalog(rng, size, boxsize, mode):
    positions = rng.uniform(0.0, boxsize, size=(size, 3))
    if mode == "weighted":
        return np.column_stack((positions, rng.uniform(0.2, 2.0, size=size)))
    return positions


def serial_raw_counts(data, random, sedges, mubin, weighted):
    from LSS_python.tpcf import run_tpCF

    if weighted:
        data = data if data.shape[1] == 4 else np.column_stack((data, np.ones(len(data))))
        random = random if random.shape[1] == 4 else np.column_stack((random, np.ones(len(random))))
    result = run_tpCF(data, random, sedges, mubin, weighted, nthreads=1, verbose=False)
    return result


def raw_weight_sum(result, weighted):
    counts = np.asarray(result["npairs"], dtype=np.float64)
    if weighted:
        return counts * np.asarray(result["weightavg"], dtype=np.float64)
    return counts


def normalizations(data, random, weighted):
    if not weighted:
        nd, nr = len(data), len(random)
        return {"DD": nd * (nd - 1), "DR": nd * nr, "RR": nr * (nr - 1)}

    def moments(values):
        weights = values[:, 3] if values.shape[1] == 4 else np.ones(len(values))
        return weights.sum(), np.square(weights).sum()

    sd, sd2 = moments(data)
    sr, sr2 = moments(random)
    return {"DD": sd * sd - sd2, "DR": sd * sr, "RR": sr * sr - sr2}


def xi_from(raw, norms):
    dd, dr, rr = (raw[name] / norms[name] for name in ("DD", "DR", "RR"))
    mask = rr > 0
    xi = np.full(rr.shape, np.nan, dtype=np.float64)
    xi[mask] = (dd[mask] - 2.0 * dr[mask] + rr[mask]) / rr[mask]
    return xi, mask


def compare_results(mpi_result, serial_result, weighted, sedges, mubin, norms):
    if mpi_result is None or serial_result is None:
        raise AssertionError("expected a result on every rank for broadcast validation")
    shape = (len(sedges) - 1, mubin)
    raw_mpi, raw_serial = {}, {}
    errors = {}
    for name in ("DD", "DR", "RR"):
        left, right = mpi_result[name], serial_result[name]
        if left is None or right is None:
            raise AssertionError(f"{name} result is missing")
        for field in ("smin", "smax", "savg", "mu_max", "npairs", "weightavg"):
            if field not in left.dtype.names:
                raise AssertionError(f"{name} is missing Corrfunc field {field}")
            if field in ("smin", "smax", "mu_max", "npairs"):
                np.testing.assert_array_equal(left[field], right[field])
            elif field == "weightavg":
                np.testing.assert_allclose(left[field], right[field], rtol=2e-12, atol=1e-10)
        lnp, rnp = np.asarray(left["npairs"]).reshape(shape), np.asarray(right["npairs"]).reshape(shape)
        np.testing.assert_array_equal(lnp, rnp)
        lw, rw = raw_weight_sum(left, weighted).reshape(shape), raw_weight_sum(right, weighted).reshape(shape)
        np.testing.assert_allclose(lw, rw, rtol=2e-12, atol=1e-10)
        raw_mpi[name], raw_serial[name] = lw, rw
        errors[name] = float(np.max(np.abs(lw - rw))) if lw.size else 0.0

    xi_mpi, mask_mpi = xi_from(raw_mpi, norms)
    xi_serial, mask_serial = xi_from(raw_serial, norms)
    np.testing.assert_array_equal(mask_mpi, mask_serial)
    np.testing.assert_allclose(xi_mpi[mask_mpi], xi_serial[mask_serial], rtol=2e-12, atol=1e-10)
    errors["xi"] = float(np.max(np.abs(xi_mpi[mask_mpi] - xi_serial[mask_serial]))) if np.any(mask_mpi) else 0.0
    return errors


def assert_selected_counts(actual, reference, names, weighted, sedges, mubin):
    """Compare raw count and weight histograms for only the requested parts."""
    shape = (len(sedges) - 1, mubin)
    for name in names:
        if actual[name] is None or reference[name] is None:
            raise AssertionError(f"{name} result is missing")
        np.testing.assert_array_equal(actual[name]["npairs"], reference[name]["npairs"])
        left = raw_weight_sum(actual[name], weighted).reshape(shape)
        right = raw_weight_sum(reference[name], weighted).reshape(shape)
        np.testing.assert_allclose(left, right, rtol=2e-12, atol=1e-10)


def structured_histograms(result, names, shape, weighted):
    values = {}
    for name in names:
        item = result[name]
        if item is None:
            continue
        values[f"{name}_npairs"] = np.asarray(item["npairs"]).reshape(shape)
        values[f"{name}_weightavg"] = np.asarray(item["weightavg"]).reshape(shape)
        values[f"{name}_raw_weight_sum"] = raw_weight_sum(item, weighted).reshape(shape)
    return values


def main(argv=None):
    parser = build_parser()
    if (argv is None and len(sys.argv) == 1) or argv == []:
        parser.print_help()
        return 0
    args, remaining = parser.parse_known_args(argv)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    # Keep MPI initialization after argument parsing so ``--help`` works without mpi4py.
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise SystemExit("mpi4py is required to run this validation") from exc
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    try:
        from LSS_python.tpcf_mpi import run_tpCF_mpi
        from LSS_python.tpcf import run_tpCF
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from LSS_python.tpcf_mpi import run_tpCF_mpi
        from LSS_python.tpcf import run_tpCF

    hostnames = comm.allgather(platform.node())
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    sedges = np.linspace(0.5, 200.0, 5)
    mubin = 4
    rng = np.random.default_rng(args.seed)
    report = {"passed": False, "cases": {}, "extra_checks": {}, "max_abs_error": 0.0}
    started = time.perf_counter()

    for case_index, mode in enumerate(("unweighted", "weighted", "unit_weight_missing_column")):
        data = catalog(rng, args.ndata, args.boxsize, "weighted" if mode == "weighted" else "plain")
        random = catalog(rng, args.nrandom, args.boxsize, "weighted" if mode == "weighted" else "plain")
        weighted = mode != "unweighted"
        root = 1 if size > 1 else 0
        serial, serial_error = None, ""
        if rank == 0:
            try:
                serial = serial_raw_counts(data, random, sedges, mubin, weighted)
            except Exception as exc:
                serial_error = str(exc)
        collective_check(comm, (rank != 0) or not serial_error,
                         f"{mode} serial reference failed: {serial_error}")
        result = run_tpCF_mpi(
            data if rank == root else None,
            random if rank == root else None,
            sedges, mubin, weighted,
            boxsize=args.boxsize, ngrids=args.ngrids,
            mpicomm=comm, root=root, broadcast_result=True,
        )

        if rank == 0:
            # The serial reference is computed on rank zero, then shared for the comparison.
            reference = serial
        else:
            reference = None
        reference = comm.bcast(reference, root=0)
        norms = normalizations(data, random, weighted)
        try:
            errors = compare_results(result, reference, weighted, sedges, mubin, norms)
            ok, detail = True, ""
        except Exception as exc:
            errors, ok, detail = {}, False, str(exc)
        collective_check(comm, ok, f"{mode} comparison: {detail}")
        case_result = {"passed": bool(ok), "max_abs_error": max(errors.values(), default=0.0), "errors": errors}
        report["cases"][mode] = case_result
        report["max_abs_error"] = max(report["max_abs_error"], case_result["max_abs_error"])
        if rank == root:
            np.savez_compressed(
                args.output_dir / f"case_{case_index}_{mode}.npz",
                data=data, random=random, sedges=sedges,
                muedges=np.linspace(0.0, 1.0, mubin + 1),
                **{f"mpi_{key}": value for key, value in structured_histograms(result, ("DD", "DR", "RR"), (len(sedges) - 1, mubin), weighted).items()},
                **{f"serial_{key}": value for key, value in structured_histograms(reference, ("DD", "DR", "RR"), (len(sedges) - 1, mubin), weighted).items()},
            )

    # Verify selected-count modes, root selection, root-only output, and broadcast.
    rng_extra = np.random.default_rng(args.seed + 1)
    data = catalog(rng_extra, 72, args.boxsize, "plain")
    random = catalog(rng_extra, 96, args.boxsize, "plain")
    root = size - 1
    checks = {}
    for run_parts, dcat, rcat in (("DD", data, None), ("RR", None, random)):
        got = run_tpCF_mpi(
            dcat if rank == root else None,
            rcat if rank == root else None,
            sedges, mubin, False, run_parts=(run_parts,),
            boxsize=args.boxsize, ngrids=args.ngrids,
            mpicomm=comm, root=root,
        )
        local_ok = (got is not None) if rank == root else (got is None)
        if local_ok and rank == root:
            local_ok = got[run_parts] is not None and all(
                got[name] is None for name in ("DD", "DR", "RR") if name != run_parts
            )
            try:
                if local_ok:
                    serial_selected = serial_raw_counts(data, random, sedges, mubin, False)
                    local_ok = np.array_equal(got[run_parts]["npairs"], serial_selected[run_parts]["npairs"])
                    local_ok = local_ok and np.allclose(
                        raw_weight_sum(got[run_parts], False),
                        raw_weight_sum(serial_selected[run_parts], False), rtol=2e-12, atol=1e-10,
                    )
            except Exception:
                local_ok = False
        collective_check(comm, local_ok, f"{run_parts}-only/root-only return contract failed")
        checks[f"{run_parts.lower()}_only_root_selection"] = True

    # Empty local blocks and zero-sized catalogs should still complete collectively.
    empty = np.empty((0, 3), dtype=np.float64)
    empty_result = run_tpCF_mpi(
        empty if rank == root else None, empty if rank == root else None,
        sedges, mubin, False, boxsize=args.boxsize, ngrids=args.ngrids,
        mpicomm=comm, root=root,
    )
    empty_ok = (empty_result is not None) if rank == root else (empty_result is None)
    if rank == root and empty_ok:
        empty_ok = all(np.count_nonzero(empty_result[name]["npairs"]) == 0 for name in ("DD", "DR", "RR"))
    collective_check(comm, empty_ok, "empty-catalog return contract failed or counts were nonzero")
    checks["empty_catalog"] = True

    # An s=0 bin and one-point local blocks exercise native zero-bin behavior.
    points = np.array([[299.9, 40.0, 60.0, 2.0], [300.2, 40.1, 60.2, 3.0]], dtype=np.float64)
    zero_edges = np.array([0.0, 1.0, 200.0])
    zero_result = run_tpCF_mpi(
        points if rank == root else None, points if rank == root else None,
        zero_edges, mubin, True, boxsize=900.0, ngrids=3,
        mpicomm=comm, root=root,
    )
    zero_serial, zero_serial_error = None, ""
    if rank == 0:
        try:
            zero_serial = serial_raw_counts(points, points, zero_edges, mubin, True)
        except Exception as exc:
            zero_serial_error = str(exc)
    collective_check(comm, (rank != 0) or not zero_serial_error,
                     f"weighted zero-bin serial reference failed: {zero_serial_error}")
    zero_serial = comm.bcast(zero_serial, root=0)
    zero_ok = (zero_result is not None) if rank == root else (zero_result is None)
    if rank == root and zero_ok:
        try:
            assert_selected_counts(zero_result, zero_serial, ("DD", "DR", "RR"), True, zero_edges, mubin)
            zero_ok = int(zero_result["DD"]["npairs"][0]) == 2
        except Exception:
            zero_ok = False
    collective_check(comm, zero_ok, "weighted cross-block zero-bin counts differ or self-pairs were changed")
    checks["weighted_zero_bin_cross_block"] = True

    # Rectangular box geometry and grid dimensions must be handled per axis.
    rectangular_box = (900.0, 780.0, 660.0)
    rectangular_grids = (2, 3, 2)
    rng_rect = np.random.default_rng(args.seed + 2)
    rect_data = rng_rect.random((80, 3)) * np.asarray(rectangular_box)
    rect_random = rng_rect.random((112, 3)) * np.asarray(rectangular_box)
    rect_result = run_tpCF_mpi(
        rect_data if rank == root else None,
        rect_random if rank == root else None,
        sedges, mubin, False,
        boxsize=rectangular_box, ngrids=rectangular_grids,
        mpicomm=comm, root=root, broadcast_result=True,
    )
    rect_reference, rect_reference_error = None, ""
    if rank == 0:
        try:
            rect_reference = serial_raw_counts(rect_data, rect_random, sedges, mubin, False)
        except Exception as exc:
            rect_reference_error = str(exc)
    collective_check(comm, (rank != 0) or not rect_reference_error,
                     f"rectangular serial reference failed: {rect_reference_error}")
    rect_reference = comm.bcast(rect_reference, root=0)
    rect_norms = normalizations(rect_data, rect_random, False)
    try:
        rect_errors = compare_results(rect_result, rect_reference, False, sedges, mubin, rect_norms)
        rect_ok, rect_detail = True, ""
    except Exception as exc:
        rect_errors, rect_ok, rect_detail = {}, False, str(exc)
    collective_check(comm, rect_ok, f"rectangular box comparison: {rect_detail}")
    checks["rectangular_box"] = True
    rect_case = {"passed": bool(rect_ok), "max_abs_error": max(rect_errors.values(), default=0.0), "errors": rect_errors}
    report["cases"]["rectangular_box"] = rect_case
    report["max_abs_error"] = max(report["max_abs_error"], rect_case["max_abs_error"])

    # Inject one worker-side task exception and verify collective propagation.
    from LSS_python import tpcf_mpi as mpi_module
    original_counter = mpi_module.count_pair_task
    if rank == 0:
        def fail_worker(*args, **kwargs):
            raise ValueError("injected worker counting failure")
        mpi_module.count_pair_task = fail_worker
    try:
        try:
            run_tpCF_mpi(
                data if rank == root else None, random if rank == root else None,
                sedges, mubin, False, boxsize=args.boxsize, ngrids=args.ngrids,
                mpicomm=comm, root=root,
            )
            worker_message = "no exception"
        except RuntimeError as exc:
            worker_message = str(exc)
        collective_check(comm, "pair counting" in worker_message,
                         f"worker error was not propagated collectively: {worker_message}")
        checks["worker_counting_error"] = True
    finally:
        mpi_module.count_pair_task = original_counter

    # A successful output must be written by root and contain its Corrfunc result.
    saved_path = args.output_dir / "root_written_dd.joblib"
    saved = run_tpCF_mpi(
        data if rank == root else None, random if rank == root else None,
        sedges, mubin, False, run_parts=("DD",),
        output_dict={"DD": str(saved_path)},
        boxsize=args.boxsize, ngrids=args.ngrids, mpicomm=comm, root=root,
    )
    saved_ok = (saved is not None) if rank == root else (saved is None)
    if rank == root and saved_ok:
        try:
            import joblib
            written = joblib.load(saved_path)
            saved_ok = np.array_equal(written["npairs"], saved["DD"]["npairs"])
        except Exception:
            saved_ok = False
    collective_check(comm, saved_ok, "root output_dict file was not written or did not match the result")
    checks["successful_root_output"] = True

    # Root creates a regular file where the output parent directory is expected.
    # The MPI implementation's mkdir(parent, parents=True) must then fail collectively.
    blocker = args.output_dir / "not_a_directory"
    if rank == root:
        blocker.write_text("file used to block directory creation\n")
    comm.Barrier()

    # Expected collective preparation/output failures must return on every rank.
    for label, kwargs, expected in (
        ("preparation_error", {"sedges": np.array([0.0, 1.0, 0.5])}, "preparation"),
        ("output_error", {"output_dict": {"DD": str(blocker / "dd.pkl")}}, "output"),
    ):
        options = dict(sedges=sedges, mubin=mubin, with_weight=False,
                       boxsize=args.boxsize, ngrids=args.ngrids, mpicomm=comm, root=root)
        options.update(kwargs)
        try:
            run_tpCF_mpi(data if rank == root else None, random if rank == root else None, **options)
            observed = "no exception"
        except RuntimeError as exc:
            observed = str(exc)
        collective_check(comm, expected in observed, f"{label} did not report {expected!r}: {observed}")
        checks[label] = True

    if rank == 0:
        try:
            import mpi4py
            mpi_version = mpi4py.__version__
        except Exception:
            mpi_version = "unknown"
        try:
            package_version = importlib.metadata.version("LSS_python")
        except importlib.metadata.PackageNotFoundError:
            package_version = "source checkout"
        try:
            corrfunc_version = importlib.metadata.version("Corrfunc")
        except importlib.metadata.PackageNotFoundError:
            corrfunc_version = "unknown"
        all_checks_passed = all(case.get("passed", False) for case in report["cases"].values())
        all_checks_passed = all_checks_passed and all(checks.values())
        report.update({
            "process_count": size,
            "hostnames_by_rank": hostnames,
            "mpi4py_version": mpi_version,
            "package_version": package_version,
            "corrfunc_version": corrfunc_version,
            "seed": args.seed,
            "ndata": args.ndata,
            "nrandom": args.nrandom,
            "binning": {"sedges": sedges.tolist(), "mubin": mubin},
            "elapsed_seconds": time.perf_counter() - started,
            "extra_checks": checks,
            "passed": bool(all_checks_passed),
        })
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "tpcf_mpi_report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            args.output_dir / "tpcf_mpi_report.npz",
            report_json=np.asarray(json.dumps(report)),
            sedges=sedges,
        )
        print(json.dumps(report, indent=2))
    return comm.bcast(0 if rank == 0 and report["passed"] else 1 if rank == 0 else None, root=0)


if __name__ == "__main__":
    raise SystemExit(main())
