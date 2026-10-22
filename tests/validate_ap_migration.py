#!/usr/bin/env python3
"""Regression and warmed benchmark runner for the production AP polygon migration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np

BASELINES = (
    "params2-tpcf-pca-ap-comparison",
    "params2-mcf-refit_fix-weipow-1-pca-ap-comparison",
    "params3-mcf-refit_fix-weipow-1-pca-ap-comparison",
)
POLICIES = ("default", "default_s", "variance99p5")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value).__name__}")


def atomic_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_baselines(project: Path, load_reference: bool = True):
    entries = []
    payloads = {}
    paths = []
    for folder in BASELINES:
        root = project / "outputs" / "diagnostics" / folder
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_path = Path(manifest["payload_path"])
        if str(payload_path) not in payloads:
            payloads[str(payload_path)] = joblib.load(payload_path)
        payload = payloads[str(payload_path)]
        if int(manifest["n_params"]) not in (2, 3):
            raise ValueError(f"Unsupported parameter count in {manifest_path}")
        for entry in manifest["pairs"]:
            data_path = root / entry["output_file"]
            required = {"parameters", "new_vectors", "target", "precision", "covariance", "noise"}
            if not load_reference:
                required = {"parameters"}
            with np.load(data_path, allow_pickle=False) as archive:
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(f"{data_path} lacks keys: {sorted(missing)}")
                keys = archive.files if load_reference else ("parameters",)
                data = {name: archive[name] for name in keys}
            if load_reference and manifest["n_params"] == 3 and "normalization_profiles" not in data:
                raise ValueError(f"Three-parameter baseline lacks raw profiles: {data_path}")
            entries.append({"root": root, "manifest_path": manifest_path, "manifest": manifest,
                            "entry": entry, "payload": payload, "reference": data, "data_path": data_path})
            paths.extend((manifest_path, data_path, payload_path))
    return entries, paths


class Converter:
    def __init__(self, payload, production_convert, pair_statistic, profile_diagnostics):
        self.payload = payload
        self.production_convert = production_convert
        self.pair_statistic = pair_statistic
        self.profile_diagnostics = profile_diagnostics
        from LSS_python.xismu import xismu
        self._xismu = xismu
        self.sources = {}
        self.assistants = {}

    def objects(self, point, pair):
        converted = {}
        for snap in pair:
            if snap not in self.sources:
                self.sources[snap] = self._xismu.load(self.payload["ap_source_paths"][snap], sbin=750, mubin=600)
                self.assistants[snap] = self._xismu.load(self.payload["ap_assistant_paths"][snap], sbin=150, mubin=120)
            options = dict(self.payload["ap_mapping_kwargs"])
            fiducial = self.payload["fiducial_values"]
            if int(self.payload["n_params"]) == 3:
                options.update(wa_f=float(fiducial[2]), wa_m=float(point[2]))
            converted[snap] = self.production_convert(
                self.sources[snap], *fiducial[:2], *point[:2], self.payload["redshift_dict"][snap],
                assis_xismu=self.assistants[snap], **options
            )
        return converted

    def evaluate(self, point, pair, audit_profiles):
        objects = self.objects(point, pair)
        vector = self.pair_statistic(objects, pair, 1)
        profiles = (np.asarray([self.profile_diagnostics(objects[snap])["raw_profile116"]
                                for snap in pair]) if audit_profiles else None)
        return vector, profiles



def basis_metrics(a, b):
    """Compare PCA row spaces without depending on component signs."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("PCA bases must be 2D with the same feature count")
    singular = np.clip(np.linalg.svd(a @ b.T, compute_uv=False), 0.0, 1.0)
    projector_sq = max(0.0, a.shape[0] + b.shape[0] - 2 * float(np.sum(singular ** 2)))
    return {"rank_reference": int(a.shape[0]), "rank_refit": int(b.shape[0]),
            "principal_cosines": singular.tolist(),
            "principal_angles_degrees": np.degrees(np.arccos(singular)).tolist(),
            "projector_frobenius": float(np.sqrt(projector_sq))}


def prepare_imports(project: Path):
    src = Path(__file__).resolve().parents[1] / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Production source is missing: {src}")
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(project / "src"))
    sys.path.insert(0, str(project / "tests"))
    sys.path.insert(0, str(project / "tests" / "ap_conversion"))
    from LSS_python.AP import tpcf_convert_main
    from LSS_python.xismu import xismu
    from model import pair_statistic
    from normalization import profile_diagnostics
    return tpcf_convert_main, xismu, pair_statistic, profile_diagnostics


def run_pca_checks(project: Path, entry_info, vectors):
    """Refit each frozen PCA policy and compare with its saved polygon model."""
    from LSS_python.compressor import Compressor
    from compare_pca import _fit_policy

    entry = entry_info["entry"]
    reference = entry_info["reference"]
    name = entry["name"]
    result = {}
    metadata = entry["training_metadata"]
    volume_factor = float(metadata["volume_factor"])
    noise = np.asarray(reference["noise"], dtype=float)
    for policy in POLICIES:
        model_path = entry_info["root"] / "models" / name / policy / "polygon.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing saved polygon PCA model: {model_path}")
        saved = Compressor.load(model_path)
        np.random.seed(0)
        refit = _fit_policy(vectors, noise, policy, volume_factor)
        saved_basis = np.asarray(saved._pca.components_, dtype=float)
        refit_basis = np.asarray(refit._pca.components_, dtype=float)
        if saved_basis.shape != refit_basis.shape:
            raise AssertionError(f"{name}/{policy}: PCA retained component count changed")
        projector_error = float(np.linalg.norm(saved_basis.T @ saved_basis - refit_basis.T @ refit_basis))
        if projector_error > 1e-6:
            raise AssertionError(f"{name}/{policy}: PCA projector error {projector_error}")
        scores = np.asarray(saved.transform(vectors), dtype=float)
        recon_saved = np.asarray(saved._pca.inverse_transform(scores), dtype=float)
        refit_scores = np.asarray(refit.transform(vectors), dtype=float)
        recon_refit = np.asarray(refit._pca.inverse_transform(refit_scores), dtype=float)
        result[policy] = {
            "saved_model": str(model_path),
            "saved_modes": int(saved.n_components), "refit_modes": int(refit.n_components),
            "subspace": basis_metrics(saved_basis, refit_basis),
            "projector_direct_frobenius": projector_error,
            "saved_reconstruction_rmse": float(np.sqrt(np.mean((vectors - recon_saved) ** 2))),
            "refit_reconstruction_rmse": float(np.sqrt(np.mean((vectors - recon_refit) ** 2))),
            "reconstruction_max_abs_difference": float(np.max(np.abs(recon_saved - recon_refit))),
            "projected_score_max_abs_difference": float(np.max(np.abs(scores - refit_scores)))
                if scores.shape == refit_scores.shape else None,
            "explained_variance_max_abs_difference": float(np.max(np.abs(
                np.asarray(saved._pca.explained_variance_) - np.asarray(refit._pca.explained_variance_))))
                if len(saved._pca.explained_variance_) == len(refit._pca.explained_variance_) else None,
        }
    return result


def benchmark(project: Path, entry_info, production_convert, xismu, pair_statistic):
    """Time warmed, root-only full conversions through production and test APIs."""
    payload, pair = entry_info["payload"], tuple(entry_info["entry"]["pair"])
    point = np.asarray(entry_info["reference"]["parameters"][len(entry_info["reference"]["parameters"]) // 2])
    snap = pair[0]
    source = xismu.load(payload["ap_source_paths"][snap], sbin=750, mubin=600)
    assistant = xismu.load(payload["ap_assistant_paths"][snap], sbin=150, mubin=120)
    fiducial = payload["fiducial_values"]
    options = dict(payload["ap_mapping_kwargs"])
    if int(payload["n_params"]) == 3:
        options.update(wa_f=float(fiducial[2]), wa_m=float(point[2]))
    from ap_conversion import tpcf_convert_main as test_convert

    def call(function, algorithm=None):
        kwargs = dict(options)
        if algorithm is not None:
            kwargs["convert_algorithm"] = algorithm
        if function is test_convert:
            kwargs["nthreads"] = 1
        converted = function(source, *fiducial[:2], *point[:2], payload["redshift_dict"][snap],
                             assis_xismu=assistant, **kwargs)
        return converted

    samples = {}
    outputs = {}
    for label, function, algorithm in (("production_default", production_convert, None),
                                       ("test_ap_conversion_api_polygon", test_convert, "polygon")):
        outputs[label] = call(function, algorithm)
        times = []
        for _ in range(3):
            started = time.perf_counter()
            outputs[label] = call(function, algorithm)
            times.append(time.perf_counter() - started)
        samples[label] = {"warmup_calls": 1, "timed_calls": len(times), "seconds": times,
                          "median_seconds": float(np.median(times))}
    outputs["production_explicit_polygon"] = call(production_convert, "polygon")
    samples["explicit_polygon_default_max_abs_DD_difference"] = float(np.max(np.abs(
        outputs["production_default"].DD - outputs["production_explicit_polygon"].DD)))
    samples["explicit_polygon_default_max_abs_DR_difference"] = float(np.max(np.abs(
        outputs["production_default"].DR - outputs["production_explicit_polygon"].DR)))
    samples["explicit_polygon_default_max_abs_RR_difference"] = float(np.max(np.abs(
        outputs["production_default"].RR - outputs["production_explicit_polygon"].RR)))
    samples["max_abs_DD_difference_vs_test_api"] = float(np.max(np.abs(
        outputs["production_default"].DD - outputs["test_ap_conversion_api_polygon"].DD)))
    samples["max_abs_DR_difference_vs_test_api"] = float(np.max(np.abs(
        outputs["production_default"].DR - outputs["test_ap_conversion_api_polygon"].DR)))
    samples["max_abs_RR_difference_vs_test_api"] = float(np.max(np.abs(
        outputs["production_default"].RR - outputs["test_ap_conversion_api_polygon"].RR)))
    samples["scope"] = "single warmed full 750x600 to 150x120 snapshot conversion; rank 0 only"
    return samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the full migration regression scan.")
    parser.add_argument("--project", type=Path, required=True, help="cov_bias_test project directory")
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent output directory")
    parser.add_argument("--mpi", action="store_true", help="Distribute rows over exactly 30 MPI ranks")
    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if not args.run:
        parser.error("--run is required")
    project, output = args.project.resolve(), args.output.resolve()
    if not project.is_dir():
        parser.error(f"project directory does not exist: {project}")

    comm = None
    rank, size = 0, 1
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
        if size != 30:
            if rank == 0:
                print(f"--mpi requires exactly 30 ranks, got {size}", file=sys.stderr)
            return 2

    setup = None
    if rank == 0:
        try:
            entries, source_paths = load_baselines(project)
            output.mkdir(parents=True, exist_ok=False)
            setup = {"ok": True}
        except Exception as error:
            setup = {"ok": False, "error": f"{type(error).__name__}: {error}",
                     "traceback": traceback.format_exc()}
    if comm:
        setup = comm.bcast(setup, root=0)
    if not setup["ok"]:
        if rank == 0:
            print(setup["traceback"], file=sys.stderr)
        return 1

    status = None
    entries, source_paths = [], []
    local_records, local_errors, local_fatal = [], [], None
    started = time.perf_counter()
    try:
        # Reload on each rank from the immutable reference folders; only small metadata is broadcast.
        entries, source_paths = load_baselines(project, load_reference=(rank == 0))
        production_convert, xismu, pair_statistic, profile_diagnostics = prepare_imports(project)
        converter_cache = {}
        jobs = [(pair_index, row) for pair_index, item in enumerate(entries)
                for row in range(len(item["reference"]["parameters"]))]
        for ordinal, job in enumerate(range(rank, len(jobs), size), 1):
            pair_index, row = jobs[job]
            item = entries[pair_index]
            manifest, baseline, pair = item["manifest"], item["reference"], tuple(item["entry"]["pair"])
            key = str(item["manifest"]["payload_path"])
            if key not in converter_cache:
                converter_cache[key] = Converter(item["payload"], production_convert, pair_statistic, profile_diagnostics)
            converter = converter_cache[key]
            try:
                vector, profiles = converter.evaluate(
                    baseline["parameters"][row], pair, audit_profiles=(manifest["n_params"] == 3))
                local_records.append((pair_index, row, vector, profiles, None))
            except Exception as error:
                local_errors.append({"baseline": item["root"].name, "pair": item["entry"]["name"],
                                     "row": row, "error": f"{type(error).__name__}: {error}"})
                local_records.append((pair_index, row, None, None, local_errors[-1]["error"]))
            if rank == 0 and ordinal % 100 == 0:
                print(f"rank {rank}: {ordinal}/{len(range(rank, len(jobs), size))} rows", flush=True)
    except Exception as error:
        local_fatal = {"rank": rank, "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()}

    package = {"records": local_records, "errors": local_errors, "fatal": local_fatal}
    gathered = comm.gather(package, root=0) if comm else [package]
    if rank == 0:
        fatal_errors = [item["fatal"] for item in gathered if item["fatal"]]
        try:
            if fatal_errors:
                raise RuntimeError("One or more MPI ranks failed before completing their assigned rows: " +
                                   "; ".join(item["error"] for item in fatal_errors))
            if entries is None:
                entries, source_paths = load_baselines(project)
            arrays = []
            for item in entries:
                ref = item["reference"]
                n = len(ref["parameters"])
                arrays.append({"vectors": np.full((n, 115), np.nan),
                               "profiles": np.full((n, 2, 116), np.nan) if item["manifest"]["n_params"] == 3 else None})
            for package in gathered:
                for pair_index, row, vector, profiles, error in package["records"]:
                    if error is None:
                        arrays[pair_index]["vectors"][row] = vector
                        if profiles is not None:
                            arrays[pair_index]["profiles"][row] = profiles
            errors = [error for package in gathered for error in package["errors"]]
            summaries, max_vector_error, max_chi_error, max_profile_error = [], 0.0, 0.0, 0.0
            for index, item in enumerate(entries):
                ref, current = item["reference"], arrays[index]
                if errors:
                    continue
                delta = current["vectors"] - ref["new_vectors"]
                max_vector = float(np.max(np.abs(delta)))
                residual_reference = ref["new_vectors"] - ref["target"]
                residual_current = current["vectors"] - ref["target"]
                chi_reference = np.einsum("ni,ij,nj->n", residual_reference, ref["precision"], residual_reference)
                chi_current = np.einsum("ni,ij,nj->n", residual_current, ref["precision"], residual_current)
                chi_error = float(np.max(np.abs(chi_current - chi_reference)))
                np.testing.assert_allclose(current["vectors"], ref["new_vectors"], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(chi_current, chi_reference, rtol=1e-10, atol=1e-8)
                profiles_error = None
                if current["profiles"] is not None:
                    profiles_ref = ref["normalization_profiles"][:, 1, :, :]
                    profiles_error = float(np.max(np.abs(current["profiles"] - profiles_ref)))
                    np.testing.assert_allclose(current["profiles"], profiles_ref, rtol=1e-10, atol=1e-12)
                    max_profile_error = max(max_profile_error, profiles_error)
                max_vector_error = max(max_vector_error, max_vector)
                max_chi_error = max(max_chi_error, chi_error)
                model_metrics = run_pca_checks(project, item, current["vectors"])
                np.savez_compressed(output / f"{item['root'].name}__{item['entry']['name']}.npz",
                    parameters=ref["parameters"], reference_polygon_vectors=ref["new_vectors"],
                    recomputed_vectors=current["vectors"], vector_delta=delta,
                    reference_chi2=chi_reference, recomputed_chi2=chi_current,
                    **({"reference_raw_profiles": profiles_ref, "recomputed_raw_profiles": current["profiles"]}
                       if current["profiles"] is not None else {}))
                summaries.append({"baseline": item["root"].name, "pair": item["entry"]["name"],
                    "rows": len(ref["parameters"]), "max_vector_abs_error": max_vector,
                    "max_chi2_abs_error": chi_error,
                    "raw_profile_max_abs_error": profiles_error, "pca_models": model_metrics})
            code_identity = {}
            for path in source_paths:
                candidate = Path(path)
                if candidate.is_file():
                    code_identity[str(candidate)] = digest(candidate)
            for path in (Path(__file__).resolve(), Path(production_convert.__code__.co_filename).resolve()):
                if path.is_file():
                    code_identity[str(path)] = digest(path)
            for module in list(sys.modules.values()):
                module_path = getattr(module, "__file__", None)
                if module_path:
                    resolved = Path(module_path).resolve()
                    if resolved.is_file() and (resolved.is_relative_to(Path(__file__).resolve().parents[1] / "src")
                                               or "ap_conversion" in resolved.parts):
                        code_identity[str(resolved)] = digest(resolved)
            status = {"complete": not errors, "all_rows": sum(len(item["reference"]["parameters"]) for item in entries),
                "successful_rows": sum(len(item["reference"]["parameters"]) for item in entries) - len(errors),
                "mpi_processes": size, "rank_wall_seconds": time.perf_counter() - started,
                "max_vector_abs_error": max_vector_error, "max_chi2_abs_error": max_chi_error,
                "max_raw_profile_abs_error": max_profile_error, "errors": errors,
                "baselines": summaries, "library_file_sha256": code_identity,
                "benchmark": benchmark(project, entries[0], production_convert, xismu, pair_statistic)}
            atomic_json(output / "summary.json", status)
            atomic_json(output / "completion.json", {"complete": status["complete"],
                "successful_rows": status["successful_rows"], "all_rows": status["all_rows"],
                "errors": errors})
            print(f"Completed {status['successful_rows']}/{status['all_rows']} rows; "
                  f"max vector error={max_vector_error:.6g}, max chi2 error={max_chi_error:.6g}", flush=True)
        except Exception as error:
            status = {"complete": False, "all_rows": sum(len(item["reference"]["parameters"]) for item in entries),
                      "successful_rows": 0, "errors": [{"error": f"{type(error).__name__}: {error}",
                      "traceback": traceback.format_exc(), "rank_failures": fatal_errors}]}
            atomic_json(output / "completion.json", {"complete": False, "failures": status["errors"]})
            print(status["errors"][0]["traceback"], file=sys.stderr)
    if comm:
        status = comm.bcast(status, root=0)
    return 0 if status and status.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
