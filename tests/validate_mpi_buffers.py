"""Check MPI buffer transport and one small weighted pair-count calculation.

Run with two local MPI ranks and --output-dir. Inputs are generated in memory;
128-byte messages force chunking without allocating large catalogs. This driver
performs exactly two checks and does not invoke the broader regression suite.
"""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import traceback

import numpy as np


def _sample_package(rank):
    """Return deterministic contiguous views, including an empty block."""
    first = np.arange(360, dtype=np.float32).reshape(120, 3) + rank
    second = np.arange(640, dtype=np.float64).reshape(160, 4) / 7 + rank
    return (
        (rank,),
        {3: first[7:110], 8: np.empty((0, 3), dtype=np.float32)},
        {2: second[5:140]},
    )


def _check_transfer(comm, root, module):
    """Compare received arrays elementwise and bytewise with their originals."""
    rank = comm.Get_rank()
    packages = [_sample_package(i) for i in range(comm.Get_size())] if rank == root else None
    actual = module._distribute_work(comm, packages, root)
    expected = _sample_package(rank)
    assert actual[0] == expected[0]
    details = []
    for label, received, reference in zip(("data", "random"), actual[1:], expected[1:]):
        assert received.keys() == reference.keys()
        for block_id, block in received.items():
            original = reference[block_id]
            assert block.dtype == original.dtype and block.shape == original.shape
            assert block.flags.c_contiguous
            np.testing.assert_array_equal(block, original)
            assert block.tobytes() == original.tobytes()
            if rank == root:
                assert block is packages[root][1 if label == "data" else 2][block_id]
            chunks = (block.nbytes + module._TRANSFER_CHUNK_BYTES - 1) // module._TRANSFER_CHUNK_BYTES
            details.append({"catalog": label, "block": block_id, "shape": list(block.shape),
                            "dtype": block.dtype.str, "bytes": block.nbytes,
                            "chunks": chunks,
                            "sha256": hashlib.sha256(block.tobytes()).hexdigest()})
    reports = comm.allgather({"rank": rank, "host": socket.gethostname(),
                              "root_reuses_original_arrays": True if rank == root else None,
                              "arrays": details})
    assert any(item["chunks"] > 1 for report in reports for item in report["arrays"])
    return {"passed": True, "elementwise_equal": True, "bytes_equal": True, "ranks": reports}


def _check_counts(comm, root, module, output_dir):
    """Compare a single weighted sample with whole-catalog serial Corrfunc."""
    from validate_tpcf_mpi import compare_results, normalizations, serial_raw_counts

    rank = comm.Get_rank()
    data, random, reference, norms = None, None, None, None
    sedges = np.linspace(0., 6., 7)
    if rank == root:
        rng = np.random.default_rng(20261022)
        data = np.column_stack((rng.uniform(0., 24., (256, 3)), rng.uniform(.25, 2., 256)))
        random = np.column_stack((rng.uniform(0., 24., (512, 3)), rng.uniform(.25, 2., 512)))
        reference = serial_raw_counts(data, random, sedges, 6, True)
        norms = normalizations(data, random, True)
    actual = module.run_tpCF_mpi(
        data, random, sedges, 6, True, nthreads=1,
        boxsize=24., ngrids=2, mpicomm=comm, root=root, broadcast_result=True,
    )
    digests = {name: hashlib.sha256(actual[name].tobytes()).hexdigest()
               for name in ("DD", "DR", "RR")}
    copies = comm.allgather(digests)
    assert all(item == copies[0] for item in copies)
    metrics = None
    if rank == root:
        errors = compare_results(actual, reference, True, sedges, 6, norms)
        for name in ("DD", "DR", "RR"):
            assert actual[name].dtype == reference[name].dtype
        np.savez(output_dir / "sample_counts.npz", data=data, random=random, sedges=sedges,
                 **{f"{prefix}_{name}": result[name]
                    for prefix, result in (("mpi", actual), ("serial", reference))
                    for name in ("DD", "DR", "RR")})
        metrics = {"passed": True, "ndata": len(data), "nrandom": len(random),
                   "dtype": data.dtype.str, "weighted": True, "boxsize": 24.,
                   "ngrids": 2, "mubin": 6, "seed": 20261022, "nthreads": 1,
                   "integer_counts_exact": True, "broadcast_results_bytewise_equal": True,
                   "rtol": 2e-12, "atol": 1e-10, "max_absolute_errors": errors}
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)

    from mpi4py import MPI
    from LSS_python import tpcf_mpi as module

    comm = MPI.COMM_WORLD
    root = 1
    try:
        if comm.Get_size() != 2:
            raise ValueError("This focused validation requires exactly two MPI ranks")
        # Apply the same small chunk limit on both ranks for these two checks.
        module._TRANSFER_CHUNK_BYTES = 128
        if comm.Get_rank() == root:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        transfer = _check_transfer(comm, root, module)
        if comm.Get_rank() == root:
            print("1/2 PASS: catalog dtype, shape, elements and bytes match; root reuses arrays", flush=True)
        counts = _check_counts(comm, root, module, args.output_dir)
        if comm.Get_rank() == root:
            report = {"passed": True, "checks": 2, "ranks": comm.Get_size(), "root": root,
                      "forced_chunk_bytes": 128, "production_chunk_bytes": 256 * 1024**2,
                      "mpi_library": MPI.Get_library_version(), "python": sys.executable,
                      "transfer": transfer, "counts": counts}
            (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print("2/2 PASS: DD/DR/RR integer counts match; weighted counts and xi meet tolerance", flush=True)
            print(json.dumps(counts["max_absolute_errors"], indent=2), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        comm.Abort(1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
