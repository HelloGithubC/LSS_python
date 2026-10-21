"""Validate the pair-counting workflow against whole-catalog Corrfunc results."""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

try:
    from Corrfunc.theory import DDsmu
except ImportError as exc:
    raise SystemExit("Corrfunc is required to run this validation") from exc

try:
    from LSS_python.pair_counting import count_pairs_in_blocks
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from LSS_python.pair_counting import count_pairs_in_blocks


BOX_SIZE = 900.0
NGRIDS = 3
MUBIN = 4
SEDGES = np.linspace(0.5, 200.0, 5)
RTOL = 2e-12
ATOL = 1e-10


def positive_int(value):
    """Parse a positive integer command-line value."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def make_catalog(rng, size, weighted):
    """Create uniform positions and optional positive, nonuniform weights."""
    positions = rng.uniform(0.0, BOX_SIZE, size=(size, 3))
    if not weighted:
        return positions
    weights = rng.uniform(0.2, 2.0, size=size)
    return np.column_stack((positions, weights))


def corrfunc_counts(data, random, with_weight, nthreads):
    """Compute direct whole-catalog DDsmu histograms for DD, DR, and RR."""
    if with_weight:
        if data.shape[1] == 3:
            data = np.column_stack((data, np.ones(len(data))))
        if random.shape[1] == 3:
            random = np.column_stack((random, np.ones(len(random))))

    results = {}
    for pair_type, first, second, autocorr in (
        ("DD", data, None, True),
        ("DR", data, random, False),
        ("RR", random, None, True),
    ):
        kwargs = {
            "nthreads": nthreads,
            "binfile": SEDGES,
            "mu_max": 1.0,
            "nmu_bins": MUBIN,
            "X1": first[:, 0],
            "Y1": first[:, 1],
            "Z1": first[:, 2],
            "verbose": False,
            "periodic": False,
            "boxsize": BOX_SIZE,
            "xbin_refine_factor": 2,
            "ybin_refine_factor": 2,
            "zbin_refine_factor": 1,
        }
        if with_weight:
            kwargs.update(
                weights1=first[:, 3],
                weight_type="pair_product",
            )
        if second is not None:
            kwargs.update(
                X2=second[:, 0],
                Y2=second[:, 1],
                Z2=second[:, 2],
            )
            if with_weight:
                kwargs["weights2"] = second[:, 3]
        result = DDsmu(autocorr, **kwargs)
        shape = (len(SEDGES) - 1, MUBIN)
        npairs = np.asarray(result["npairs"], dtype=np.uint64).reshape(shape).copy()
        wpairs = npairs.astype(np.float64)
        if with_weight:
            wpairs *= np.asarray(result["weightavg"], dtype=np.float64).reshape(shape)
            if not np.any(wpairs != 0):
                wpairs = npairs.astype(np.float64)
        results[pair_type] = {"npairs": npairs, "wpairs": wpairs}
    return results


def pair_normalizations(data, random, with_weight):
    """Compute ordered-pair normalizations matching Corrfunc counts."""
    if not with_weight:
        nd, nr = len(data), len(random)
        return {
            "DD": float(nd * (nd - 1)),
            "DR": float(nd * nr),
            "RR": float(nr * (nr - 1)),
        }

    def sums(catalog):
        weights = catalog[:, 3] if catalog.shape[1] == 4 else np.ones(len(catalog))
        return np.sum(weights), np.sum(weights * weights)

    sum_d, sum_d2 = sums(data)
    sum_r, sum_r2 = sums(random)
    return {
        "DD": float(sum_d * sum_d - sum_d2),
        "DR": float(sum_d * sum_r),
        "RR": float(sum_r * sum_r - sum_r2),
    }


def estimator(counts, normalizations):
    """Return Landy-Szalay xi and its valid RR-bin mask."""
    dd = counts["DD"]["wpairs"] / normalizations["DD"]
    dr = counts["DR"]["wpairs"] / normalizations["DR"]
    rr = counts["RR"]["wpairs"] / normalizations["RR"]
    mask = rr > 0
    xi = np.full(rr.shape, np.nan, dtype=np.float64)
    xi[mask] = (dd[mask] - 2.0 * dr[mask] + rr[mask]) / rr[mask]
    return xi, mask


def max_abs_difference(first, second):
    """Return the largest absolute elementwise difference."""
    difference = np.abs(np.asarray(first, dtype=np.float64) -
                        np.asarray(second, dtype=np.float64))
    return float(np.max(difference)) if difference.size else 0.0


def validate_case(name, data, random, with_weight, nthreads, output_dir):
    """Run a block-versus-whole comparison and save its reproducible inputs."""
    whole_start = time.perf_counter()
    whole = corrfunc_counts(data, random, with_weight, nthreads)
    whole_seconds = time.perf_counter() - whole_start

    block_start = time.perf_counter()
    blocked = count_pairs_in_blocks(
        data,
        random,
        SEDGES,
        MUBIN,
        BOX_SIZE,
        NGRIDS,
        with_weight=with_weight,
        nthreads=nthreads,
    )
    block_seconds = time.perf_counter() - block_start

    normalizations = pair_normalizations(data, random, with_weight)
    block_normalizations = {name: blocked[name]["normalization"] for name in ("DD", "DR", "RR")}
    whole_xi, whole_mask = estimator(whole, normalizations)
    block_xi, block_mask = estimator(blocked, block_normalizations)
    if not np.array_equal(whole_mask, block_mask):
        raise AssertionError(f"{name}: whole and block RR masks differ")

    comparisons = {}
    for pair_type in ("DD", "DR", "RR"):
        npairs_equal = np.array_equal(
            whole[pair_type]["npairs"], blocked[pair_type]["npairs"]
        )
        wpairs_close = bool(np.allclose(
            whole[pair_type]["wpairs"], blocked[pair_type]["wpairs"],
            rtol=RTOL, atol=ATOL,
        ))
        norm_close = bool(np.isclose(normalizations[pair_type], block_normalizations[pair_type],
                                     rtol=RTOL, atol=ATOL))
        comparisons[pair_type] = {
            "npairs_max_abs_error": max_abs_difference(
                whole[pair_type]["npairs"], blocked[pair_type]["npairs"]
            ),
            "npairs_equal": bool(npairs_equal),
            "wpairs_max_abs_error": max_abs_difference(
                whole[pair_type]["wpairs"], blocked[pair_type]["wpairs"]
            ),
            "wpairs_allclose": wpairs_close,
            "normalization_abs_error": abs(normalizations[pair_type] - block_normalizations[pair_type]),
            "normalization_allclose": norm_close,
            "passed": bool(npairs_equal and wpairs_close and norm_close),
        }

    xi_close = bool(np.allclose(
        whole_xi[whole_mask], block_xi[block_mask], rtol=RTOL, atol=ATOL
    ))
    xi_error = max_abs_difference(whole_xi[whole_mask], block_xi[block_mask])
    comparisons["xi"] = {
        "max_abs_error_on_rr_positive_bins": xi_error,
        "allclose": xi_close,
        "compared_bins": int(np.count_nonzero(whole_mask)),
        "tested": bool(np.any(whole_mask)),
        "passed": xi_close,
    }

    case_passed = all(item["passed"] for item in comparisons.values())
    npz_values = {
        "data": data,
        "random": random,
        "sedges": SEDGES,
        "muedges": np.linspace(0.0, 1.0, MUBIN + 1),
        "normalization_DD": np.asarray(normalizations["DD"]),
        "normalization_DR": np.asarray(normalizations["DR"]),
        "normalization_RR": np.asarray(normalizations["RR"]),
        "normalization_DD_blocks": np.asarray(block_normalizations["DD"]),
        "normalization_DR_blocks": np.asarray(block_normalizations["DR"]),
        "normalization_RR_blocks": np.asarray(block_normalizations["RR"]),
        "xi_whole": whole_xi,
        "xi_blocks": block_xi,
        "xi_mask": whole_mask,
    }
    for pair_type in ("DD", "DR", "RR"):
        for source, result in (("whole", whole[pair_type]), ("blocks", blocked[pair_type])):
            npz_values[f"{pair_type}_npairs_{source}"] = result["npairs"]
            npz_values[f"{pair_type}_wpairs_{source}"] = result["wpairs"]
        task_results = [
            (task, hist)
            for task, hist in blocked["task_counts"].items()
            if task.pair_type == pair_type
        ]
        npz_values[f"{pair_type}_task_first"] = np.asarray(
            [task.first for task, _ in task_results], dtype=np.int32
        )
        npz_values[f"{pair_type}_task_second"] = np.asarray(
            [task.second for task, _ in task_results], dtype=np.int32
        )
        npz_values[f"{pair_type}_task_autocorr"] = np.asarray(
            [task.autocorr for task, _ in task_results], dtype=np.bool_
        )
        npz_values[f"{pair_type}_task_multiplicity"] = np.asarray(
            [task.multiplicity for task, _ in task_results], dtype=np.uint8
        )
        npz_values[f"{pair_type}_task_npairs"] = np.stack(
            [hist.npairs for _, hist in task_results]
        ) if task_results else np.empty((0, len(SEDGES) - 1, MUBIN), dtype=np.uint64)
        npz_values[f"{pair_type}_task_wpairs"] = np.stack(
            [hist.wpairs for _, hist in task_results]
        ) if task_results else np.empty((0, len(SEDGES) - 1, MUBIN), dtype=np.float64)

    np.savez_compressed(output_dir / f"{name}.npz", **npz_values)
    return {
        "name": name,
        "with_weight": bool(with_weight),
        "data_size": int(len(data)),
        "random_size": int(len(random)),
        "task_count": int(len(blocked["tasks"])),
        "task_counts_by_pair_type": {
            pair_type: int(sum(task.pair_type == pair_type for task in blocked["tasks"]))
            for pair_type in ("DD", "DR", "RR")
        },
        "block_sizes": {
            "data": blocked["block_sizes"]["data"].astype(int).tolist(),
            "random": blocked["block_sizes"]["random"].astype(int).tolist(),
        },
        "normalizations": normalizations,
        "block_normalizations": block_normalizations,
        "timing_seconds": {
            "whole_corrfunc": whole_seconds,
            "blocks_including_task_calls": block_seconds,
            "note": "Small correctness sample only; timings are not an acceleration benchmark.",
        },
        "comparisons": comparisons,
        "passed": bool(case_passed),
    }


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate spatial-block pair counts against whole-catalog Corrfunc."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--ndata", type=positive_int, default=256)
    parser.add_argument("--nrandom", type=positive_int, default=512)
    parser.add_argument("--nthreads", type=positive_int, default=1)
    return parser


def main(argv=None):
    """Run all three requested validation cases."""
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.ndata < 2 or args.nrandom < 2:
        parser.error("--ndata and --nrandom must each be at least 2 for normalized xi")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for index, (name, weighted, with_weight) in enumerate((
        ("unweighted", False, False),
        ("weighted", True, True),
        ("unit_weights", False, True),
    )):
        rng = np.random.default_rng(args.seed + index)
        data = make_catalog(rng, args.ndata, weighted)
        random = make_catalog(rng, args.nrandom, weighted)
        cases.append(validate_case(
            name, data, random, with_weight, args.nthreads, args.output_dir
        ))

    report = {
        "input": {
            "seed": args.seed,
            "ndata": args.ndata,
            "nrandom": args.nrandom,
            "nthreads": args.nthreads,
            "boxsize": BOX_SIZE,
            "ngrids": NGRIDS,
            "sedges": SEDGES.tolist(),
            "mubin": MUBIN,
            "rtol": RTOL,
            "atol": ATOL,
            "periodic": False,
            "jackknife_correction": None,
            "weighting": {
                "unweighted": "No weight columns and with_weight=False.",
                "weighted": "Positive nonuniform weights in [0.2, 2.0] with pair_product.",
                "unit_weights": "No input weight columns; with_weight=True supplies unit weights.",
            },
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Corrfunc": importlib.metadata.version("Corrfunc"),
        },
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
