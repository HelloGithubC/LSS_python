#!/usr/bin/env python3
"""Representative science and performance comparison for dense AP algorithms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np

BASELINES = (
    "params2-tpcf-pca-ap-comparison",
    "params2-mcf-refit_fix-weipow-1-pca-ap-comparison",
    "params3-mcf-refit_fix-weipow-1-pca-ap-comparison",
)
ALGORITHMS = ("polygon", "curved_gauss8", "curved_analytic")
BACKENDS = (("cpp", True), ("numba", False))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def resolve_payload(raw: str) -> Path:
    original = Path(raw)
    candidates = [original]
    marker = "/outputs/"
    if marker in raw:
        candidates.append(Path(raw.replace(marker, "/outputs.bak/", 1)))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Payload path missing at original and outputs.bak locations: {candidates}")


def resolve_input(raw: str) -> Path:
    original = Path(raw)
    if original.is_file():
        return original
    text = str(original)
    marker = "/outputs/"
    if marker in text:
        relocated = Path(text.replace(marker, "/outputs.bak/", 1))
        if relocated.is_file():
            return relocated
    raise FileNotFoundError(f"AP input is missing at original and outputs.bak locations: {original}")


def load_baselines(project: Path):
    all_entries, fingerprints = [], {}
    payload_cache = {}
    for folder in BASELINES:
        root = project / "outputs.bak" / "diagnostics" / folder
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_path = resolve_payload(manifest["payload_path"])
        payload = payload_cache.setdefault(str(payload_path), None)
        if payload is None:
            payload = joblib.load(payload_path)
            payload_cache[str(payload_path)] = payload
        if int(manifest["n_params"]) not in (2, 3):
            raise ValueError(f"Unexpected parameter count in {manifest_path}")
        fingerprints[str(manifest_path)] = digest(manifest_path)
        fingerprints[str(payload_path)] = digest(payload_path)
        for entry in manifest["pairs"]:
            data_path = root / entry["output_file"]
            with np.load(data_path, allow_pickle=False) as archive:
                required = {"parameters", "new_vectors", "target", "precision", "ap_factors"}
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(f"{data_path} lacks keys: {sorted(missing)}")
                ref = {key: archive[key] for key in archive.files}
            all_entries.append({"root": root, "manifest_path": manifest_path,
                                "manifest": manifest, "entry": entry, "payload": payload,
                                "reference": ref, "data_path": data_path})
            fingerprints[str(data_path)] = digest(data_path)
    return all_entries, fingerprints


def selected_rows(entry):
    params = np.asarray(entry["reference"]["parameters"], dtype=float)
    factors = np.asarray(entry["reference"]["ap_factors"], dtype=float)
    if factors.shape != (len(params), 4):
        raise ValueError(f"Expected ap_factors with shape (N,4), got {factors.shape}")
    fid = np.asarray(entry["payload"]["fiducial_values"], dtype=float)[:params.shape[1]]
    picks = {}

    def add(label, index):
        picks.setdefault(int(index), []).append(label)

    add("nearest_fiducial", int(np.argmin(np.linalg.norm(params - fid, axis=1))))
    add("first", 0)
    add("middle", len(params) // 2)
    add("last", len(params) - 1)
    for col, label in enumerate(("alpha_perp", "alpha_parallel", "alpha_perp", "alpha_parallel")):
        snap = entry["entry"]["pair"][col // 2]
        add(f"min_{label}_snap{snap}", int(np.nanargmin(factors[:, col])))
        add(f"max_{label}_snap{snap}", int(np.nanargmax(factors[:, col])))
    for col in (0, 2):
        snap = entry["entry"]["pair"][col // 2]
        ratio = factors[:, col]/factors[:, col+1]
        add(f"min_a_over_b_snap{snap}", int(np.nanargmin(ratio)))
        add(f"max_a_over_b_snap{snap}", int(np.nanargmax(ratio)))
    if params.shape[1] == 3:
        for index in (1572, 4947, 5499):
            if index < len(params):
                add(f"known_{index}", index)
    return params, picks, factors


def prepare(project: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(project / "src"))
    from LSS_python.AP import tpcf_convert_main
    from LSS_python.xismu import xismu
    from model import pair_statistic
    return tpcf_convert_main, xismu, pair_statistic


class Evaluator:
    def __init__(self, entry, convert, xismu, pair_statistic):
        self.entry = entry
        self.convert = convert
        self.xismu = xismu
        self.pair_statistic = pair_statistic
        self.payload = entry["payload"]
        self.pair = tuple(entry["entry"]["pair"])
        self.sources, self.assistants = {}, {}
        for snap in self.pair:
            self.sources[snap] = xismu.load(resolve_input(self.payload["ap_source_paths"][snap]), sbin=750, mubin=600)
            self.assistants[snap] = xismu.load(resolve_input(self.payload["ap_assistant_paths"][snap]), sbin=150, mubin=120)

    def convert_pair(self, point, backend, algorithm):
        fid = self.payload["fiducial_values"]
        opts = dict(self.payload["ap_mapping_kwargs"])
        opts.update(c_api=(backend == "cpp"), convert_algorithm=algorithm)
        if int(self.payload["n_params"]) == 3:
            opts.update(wa_f=float(fid[2]), wa_m=float(point[2]))
        converted, elapsed = {}, 0.0
        for snap in self.pair:
            started = time.perf_counter()
            converted[snap] = self.convert(self.sources[snap], *fid[:2], *point[:2],
                                           self.payload["redshift_dict"][snap],
                                           assis_xismu=self.assistants[snap], **opts)
            elapsed += time.perf_counter() - started
        return converted, elapsed

    def call(self, point, backend, algorithm):
        converted, _ = self.convert_pair(point, backend, algorithm)
        vector = np.asarray(self.pair_statistic(converted, self.pair, 1), dtype=float).reshape(-1)
        counts = {f"{snap}_{name}": np.asarray(getattr(converted[snap], name), dtype=float)
                  for snap in self.pair for name in ("DD", "DR", "RR")}
        return vector, counts


def chi2(vector, target, precision):
    residual = np.asarray(vector) - np.asarray(target)
    return float(residual @ np.asarray(precision) @ residual)


def science(entries, convert, xismu, pair_statistic):
    records, proof = [], {}
    for entry in entries:
        print(f"{entry['root'].name} / {entry['entry']['name']}", flush=True)
        evaluator = Evaluator(entry, convert, xismu, pair_statistic)
        ref = entry["reference"]
        params, picks, factors = selected_rows(entry)
        saved = np.asarray(ref["new_vectors"], dtype=float)
        for index, labels in picks.items():
            point = params[index]
            reference_vector = saved[index]
            label = ",".join(labels)
            backend_outputs = {}
            sample_records = []
            for backend, _ in BACKENDS:
                outputs, count_outputs = {}, {}
                for algorithm in ALGORITHMS:
                    outputs[algorithm], count_outputs[algorithm] = evaluator.call(point, backend, algorithm)
                np.testing.assert_allclose(outputs["polygon"], reference_vector, rtol=0, atol=1e-12,
                                           err_msg=f"Saved polygon mismatch: {entry['entry']['name']} {label} {backend}")
                np.testing.assert_allclose(outputs["curved_gauss8"], outputs["curved_analytic"],
                                           rtol=1e-10, atol=1e-10,
                                           err_msg=f"Curved algorithm disagreement: {label} {backend}")
                record = {"comparison": entry["root"].name, "pair": entry["entry"]["name"],
                          "row": label, "index": index, "parameters": point,
                          "ap_factors": factors[index], "backend": backend,
                          "chi2_saved": chi2(reference_vector, ref["target"], ref["precision"])}
                for algorithm, vector in outputs.items():
                    record[f"chi2_{algorithm}"] = chi2(vector, ref["target"], ref["precision"])
                    record[f"max_abs_delta_saved_{algorithm}"] = float(np.max(np.abs(vector-reference_vector)))
                record["max_abs_curves"] = float(np.max(np.abs(outputs["curved_gauss8"]-outputs["curved_analytic"])))
                for algorithm in ALGORITHMS:
                    record[f"max_abs_counts_vs_polygon_{algorithm}"] = max(
                        float(np.max(np.abs(count_outputs[algorithm][key] - count_outputs["polygon"][key])))
                        for key in count_outputs[algorithm]
                    )
                    record[f"peak_relative_counts_vs_polygon_{algorithm}"] = max(
                        float(np.max(np.abs(count_outputs[algorithm][key] - count_outputs["polygon"][key])))
                        / max(float(np.max(np.abs(count_outputs["polygon"][key]))), np.finfo(float).tiny)
                        for key in count_outputs[algorithm]
                    )
                record["max_abs_counts_curves"] = max(
                    float(np.max(np.abs(count_outputs["curved_gauss8"][key] - count_outputs["curved_analytic"][key])))
                    for key in count_outputs["curved_gauss8"]
                )
                sample_records.append(record)
                backend_outputs[backend] = (outputs, count_outputs)
                proof[f"{entry['root'].name}:{entry['entry']['name']}:{index}:{backend}"] = {
                    algorithm: vector for algorithm, vector in outputs.items()
                }
            cpp_vectors, cpp_counts = backend_outputs["cpp"]
            numba_vectors, numba_counts = backend_outputs["numba"]
            vector_delta = max(float(np.max(np.abs(cpp_vectors[a] - numba_vectors[a]))) for a in ALGORITHMS)
            for algorithm in ALGORITHMS:
                np.testing.assert_allclose(cpp_vectors[algorithm], numba_vectors[algorithm], rtol=1e-10, atol=1e-12,
                                           err_msg=f"C++/Numba vector mismatch: {label} {algorithm}")
                for key in cpp_counts[algorithm]:
                    np.testing.assert_allclose(cpp_counts[algorithm][key], numba_counts[algorithm][key],
                                               rtol=1e-10, atol=1e-12,
                                               err_msg=f"C++/Numba count mismatch: {label} {algorithm} {key}")
            count_delta = max(float(np.max(np.abs(cpp_counts[a][key] - numba_counts[a][key])))
                              for a in ALGORITHMS for key in cpp_counts[a])
            delta_chi2 = {
                backend: chi2(vectors["curved_gauss8"], ref["target"], ref["precision"])
                          - chi2(vectors["curved_analytic"], ref["target"], ref["precision"])
                for backend, (vectors, _counts) in backend_outputs.items()
            }
            for record in sample_records:
                record["max_abs_vector_cpp_numba"] = vector_delta
                record["max_abs_counts_cpp_numba"] = count_delta
                record["delta_chi2_gauss8_analytic"] = delta_chi2[record["backend"]]
            records.extend(sample_records)
    return records, proof


def benchmark(entries, convert, xismu, pair_statistic, repeats):
    out = []
    for entry in entries:
        print(f"{entry['root'].name} / {entry['entry']['name']}", flush=True)
        evaluator = Evaluator(entry, convert, xismu, pair_statistic)
        params, picks, factors = selected_rows(entry)
        ordinary = params[len(params) // 2]
        factor_deviation = np.nanmax(np.abs(factors - 1.0), axis=1)
        extreme_index = int(np.nanargmax(factor_deviation))
        extreme = params[extreme_index]
        for case, point in (("ordinary", ordinary), ("extreme", extreme)):
            jobs = [(backend, algorithm) for backend, _ in BACKENDS for algorithm in ALGORITHMS]
            samples = {(b, a): [] for b, a in jobs}
            for cycle in range(repeats + 1):
                order = jobs[cycle % len(jobs):] + jobs[:cycle % len(jobs)]
                for backend, algorithm in order:
                    _objects, elapsed = evaluator.convert_pair(point, backend, algorithm)
                    if cycle:
                        samples[(backend, algorithm)].append(elapsed)
            for (backend, algorithm), times in samples.items():
                out.append({"comparison": entry["root"].name, "pair": entry["entry"]["name"],
                            "case": case, "backend": backend, "algorithm": algorithm,
                            "warmup_calls_excluded": 1, "timed_calls": repeats,
                            "seconds": times, "median_seconds": float(np.median(times)),
                            "point_index": (len(params) // 2 if case == "ordinary" else extreme_index),
                            "parameters": point,
                            "ap_factors": factors[len(params) // 2 if case == "ordinary" else extreme_index],
                            "timing_scope": "sum of tpcf_convert_main calls for both snapshots; excludes pair statistic and output packing"})
    return out


def write_outputs(output: Path, summary):
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for key, val in summary.pop("_proof").items():
        for algorithm, vector in val.items():
            arrays[f"{key}:{algorithm}"] = vector
    np.savez_compressed(output / "vectors.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=safe)+"\n", encoding="utf-8")
    (output / "REPORT.md").write_text(render_report(summary), encoding="utf-8")


def render_report(summary):
    rows = summary["science"]
    groups = {}
    for row in rows:
        groups.setdefault((row["comparison"], row["pair"]), []).append(row)
    lines = ["# Curved AP representative validation", "",
             f"Science checks: {len(rows)//2} distinct pair/parameter samples, both backends.",
             f"Timing cells: {len(summary['benchmarks'])}, each warmed and measured independently.", "",
             "Polygon vectors match saved results at absolute tolerance 1e-12. The curved vectors "
             "agree at rtol=atol=1e-10. C++/Numba vectors and normalized DD/DR/RR counts agree "
             "at rtol=1e-10, atol=1e-12. These are representative checks, not a full training/PCA scan.", "",
             "## Statistics and chi-square", "",
             "All maxima below refer only to the selected points. Chi-square uses each frozen target and precision; "
             "no PCA projection or covariance refit is performed. Curved/polygon columns use curved_analytic.", "",
             "| Baseline | Pair | Points | Max abs statistic change | Max abs chi-square change | Max abs Gauss8/analytic chi-square difference |",
             "|---|---|---:|---:|---:|---:|"]
    for (comparison, pair), values in groups.items():
        stat = max(v["max_abs_delta_saved_curved_analytic"] for v in values)
        chi = max(abs(v["chi2_curved_analytic"]-v["chi2_polygon"]) for v in values)
        curves = max(abs(v["delta_chi2_gauss8_analytic"]) for v in values)
        lines.append(f"| {comparison} | {pair} | {len(values)//2} | {stat:.8g} | {chi:.8g} | {curves:.8g} |")
    lines += ["", "## Counts", "",
              "Objects store normalized counts. These maxima are over both snapshots and DD/DR/RR channels. "
              "Comparisons use reconstructed xismu arrays, including the existing zero-RR floor.", "",
              "| Baseline | Pair | Max abs curved/polygon count difference | Max abs Gauss8/analytic count difference |",
              "|---|---|---:|---:|"]
    for (comparison, pair), values in groups.items():
        absolute = max(v["max_abs_counts_vs_polygon_curved_analytic"] for v in values)
        curves = max(v["max_abs_counts_curves"] for v in values)
        lines.append(f"| {comparison} | {pair} | {absolute:.8g} | {curves:.8g} |")
    lines += ["", "## Timings", "",
              "Each sample sums the two complete tpcf_convert_main calls (one per snapshot), including background "
              "preparation, mapping and object reconstruction. Data loading, statistics and chi-square are excluded. "
              "C++ uses one thread; Numba is serial. First calls are excluded and algorithms run in rotated order. "
              "Factors, coordinates and all timings are in summary.json.", "",
              "| Baseline | Pair | Case | Backend | Algorithm | Median seconds |",
              "|---|---|---|---|---|---:|"]
    for item in summary["benchmarks"]:
        lines.append(f"| {item['comparison']} | {item['pair']} | {item['case']} | {item['backend']} | {item['algorithm']} | {item['median_seconds']:.6g} |")
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run representative science and warmed benchmarks")
    parser.add_argument("--project", type=Path, required=True, help="cov_bias_test project directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--repeats", type=int, default=7, help="Timed calls per benchmark cell")
    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if not args.run:
        parser.error("pass --run to execute the comparison")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists():
        parser.error(f"output path already exists: {args.output}")
    entries, fingerprints = load_baselines(args.project.resolve())
    convert, xismu, pair_statistic = prepare(args.project.resolve())
    science_rows, proof = science(entries, convert, xismu, pair_statistic)
    benchmark_rows = benchmark(entries, convert, xismu, pair_statistic, args.repeats)
    groups = {}
    for row in science_rows:
        key = (row["comparison"], row["pair"])
        groups.setdefault(key, []).append(abs(row["delta_chi2_gauss8_analytic"]))
    delta_summary = [{"comparison": comparison, "pair": pair,
                      "max_abs_delta_chi2": float(max(values))}
                     for (comparison, pair), values in sorted(groups.items())]
    import numba
    import scipy
    import inspect
    summary = {"algorithms": ALGORITHMS, "backends": [name for name, _ in BACKENDS],
               "baseline_fingerprints_sha256": fingerprints,
               "libraries": {"numpy": {"version": np.__version__, "path": np.__file__},
                             "numba": {"version": numba.__version__, "path": numba.__file__},
                             "scipy": {"version": scipy.__version__, "path": scipy.__file__},
                             "ap_converter": inspect.getsourcefile(convert), "python": sys.version},
               "science": science_rows, "delta_chi2_summary": delta_summary,
               "benchmarks": benchmark_rows, "_proof": proof}
    write_outputs(args.output.resolve(), summary)
    print(f"Wrote representative AP validation to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
