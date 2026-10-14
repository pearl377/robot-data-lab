from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .io import fingerprint, read_json, safe_path, sha256, write_json


def code_fingerprint():
    return fingerprint({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def doctor():
    import torch
    packages = {}
    for package in ["numpy", "pandas", "pyarrow", "av", "torch", "Pillow", "lerobot"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.system(), "packages": packages,
            "cuda_available": torch.cuda.is_available(), "cuda_devices": torch.cuda.device_count(),
            "gpu_training_ready": platform.system() == "Linux" and torch.cuda.is_available() and packages["lerobot"] is not None}


def validate_config(config):
    if config["training"]["steps"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("steps and batch_size must be positive")
    if config["training"]["sample_stride"] < 1 or config["training"]["image_grid"] < 1:
        raise ValueError("sample_stride and image_grid must be positive")
    if config["audit"]["video_samples"] < 1:
        raise ValueError("At least one video sample per camera required")
    if not config["training"]["seeds"] or len(set(config["training"]["seeds"])) != len(config["training"]["seeds"]):
        raise ValueError("Provide unique training seeds")
    if not 0 < config["split"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1")


def run_local(config, project, offline=False, force=False):
    from .data import audit_dataset, build_features, fetch_dataset
    from .baseline import run_experiments
    from .report import render_report
    validate_config(config)
    project = Path(project).resolve()
    artifacts, reports = project / "artifacts", project / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    source_root = safe_path(project, config["dataset"]["root"])
    if not offline:
        source = fetch_dataset(config, project)
    else:
        source = read_json(source_root / "_source.json")
        if source["repo_id"] != config["dataset"]["repo_id"] or source["revision"] != config["dataset"]["revision"]:
            raise ValueError("Offline source revision mismatch")
        for relative, meta in source["files"].items():
            if sha256(safe_path(source_root, relative)) != meta["sha256"]:
                raise ValueError(f"Dataset integrity failure: {relative}")
    provenance = {"config": config, "config_fingerprint": fingerprint(config), "code_fingerprint": code_fingerprint(),
                  "source_manifest_hash": fingerprint(source), "environment": doctor(),
                  "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    cache_key = fingerprint({k: provenance[k] for k in ("config_fingerprint", "code_fingerprint", "source_manifest_hash", "environment")})
    state_path = artifacts / "pipeline.json"
    state = read_json(state_path) if state_path.exists() else {}
    if state.get("cache_key") != cache_key or force:
        state = {"cache_key": cache_key, "stages": {}}

    def stage(name, operation, outputs):
        previous = state["stages"].get(name, {})
        hashes = previous.get("outputs", {})
        if previous.get("status") == "complete" and hashes and all((project / p).exists() and sha256(project / p) == h for p, h in hashes.items()):
            print(f"[cached] {name}", flush=True)
            return
        started = time.perf_counter()
        state["stages"][name] = {"status": "running"}
        write_json(state_path, state)
        try:
            operation()
            completed_outputs = outputs() if callable(outputs) else outputs
            state["stages"][name] = {"status": "complete", "seconds": time.perf_counter() - started,
                "outputs": {str(Path(p).relative_to(project)).replace('\\', '/'): sha256(p) for p in completed_outputs}}
        except Exception as exc:
            state["stages"][name] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            write_json(state_path, state)
            raise
        write_json(state_path, state)

    audit_path, features_path = artifacts / "audit.json", artifacts / "features.npz"
    stage("audit", lambda: audit_dataset(config, project, audit_path), [audit_path])
    audit = read_json(audit_path)
    if audit["dataset_errors"]:
        raise ValueError(f"Dataset-level audit failed: {audit['dataset_errors']}")
    stage("features", lambda: build_features(config, project, audit, features_path), [features_path, features_path.with_suffix('.json')])
    run_dir = artifacts / "experiments"
    stage("experiments", lambda: run_experiments(config, audit, features_path, read_json(features_path.with_suffix('.json')), run_dir),
          lambda: [run_dir / "summary.json", run_dir / "splits.json"] + [p for p in run_dir.glob("*/*") if p.is_file()])
    summary = read_json(run_dir / "summary.json")
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "audit.json", audit)
    write_json(reports / "experiments.json", summary)
    write_json(reports / "provenance.json", provenance)
    write_json(reports / "source-manifest.json", source)
    render_report(audit, summary, provenance, reports / "index.html")
    print(f"Report: {reports / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Robot Data Lab: data quality and reproducible offline experiments")
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--project", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("download")
    sub.add_parser("audit")
    run = sub.add_parser("run-local")
    run.add_argument("--offline", action="store_true")
    run.add_argument("--force", action="store_true")
    sub.add_parser("report")
    annotation = sub.add_parser("annotations")
    annotation.add_argument("action", choices=["export", "validate"])
    annotation.add_argument("--file", default="artifacts/annotations.jsonl")
    server = sub.add_parser("serve")
    server.add_argument("--port", type=int, default=8765)
    cloud = sub.add_parser("cloud")
    cloud.add_argument("action", choices=["plan", "train", "eval"])
    cloud.add_argument("--checkpoint")
    cloud.add_argument("--steps", type=int, default=20000)
    cloud.add_argument("--batch-size", type=int, default=4)
    cloud.add_argument("--episodes", type=int, default=10)
    cloud.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    if args.command == "doctor":
        print(json.dumps(doctor(), ensure_ascii=False, indent=2))
        return
    if args.command == "serve":
        from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
        from functools import partial
        directory = project / "reports"
        if not (directory / "index.html").exists():
            parser.error("No report exists; run run-local first")
        handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
        print(f"Serving reports only at http://127.0.0.1:{args.port}", flush=True)
        ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()
        return
    if args.command == "cloud":
        from .cloud import cloud_command
        cloud_command(args, project)
        return
    config = read_json(project / args.config)
    validate_config(config)
    if args.command == "annotations":
        from .annotations import export_annotations, validate_annotations
        annotation_path = safe_path(project, args.file)
        source = safe_path(project, config["dataset"]["root"])
        if args.action == "export":
            export_annotations(source, annotation_path)
        else:
            result = validate_annotations(source, annotation_path)
            write_json(project / "reports/annotation-validation.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["errors"]:
                raise SystemExit(2)
    elif args.command == "download":
        from .data import fetch_dataset
        fetch_dataset(config, project)
    elif args.command == "audit":
        from .data import audit_dataset
        result = audit_dataset(config, project, project / "artifacts/audit.json")
        if result["dataset_errors"] or result["valid_episodes"] == 0:
            raise SystemExit(2)
    elif args.command == "run-local":
        run_local(config, project, args.offline, args.force)
    elif args.command == "report":
        from .report import render_report
        reports = project / "reports"
        render_report(read_json(reports / "audit.json"), read_json(reports / "experiments.json"),
                      read_json(reports / "provenance.json"), reports / "index.html")


if __name__ == "__main__":
    main()
