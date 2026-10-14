from __future__ import annotations

import concurrent.futures
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from .io import download_file, read_json, request_json, safe_path, sha256, write_json


def fetch_dataset(config, project):
    ds = config["dataset"]
    revision = ds["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be a full immutable commit SHA")
    root = safe_path(project, ds["root"])
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "_source.json"
    existing = read_json(source_path) if source_path.exists() else {}
    if existing and (existing["repo_id"], existing["revision"]) != (ds["repo_id"], revision):
        raise ValueError("Dataset root contains a different revision; use a new root")
    api = f"https://huggingface.co/api/datasets/{quote(ds['repo_id'], safe='/')}/revision/{revision}"
    metadata = request_json(api)
    if metadata["sha"] != revision:
        raise ValueError("Resolved dataset SHA differs from requested revision")
    files = [x["rfilename"] for x in metadata["siblings"]
             if x["rfilename"].startswith(("data/", "meta/", "videos/")) or x["rfilename"] == "README.md"]
    def get(relative):
        target = safe_path(root, relative)
        known = existing.get("files", {}).get(relative)
        if not (known and target.exists() and sha256(target) == known["sha256"]):
            url = f"https://huggingface.co/datasets/{ds['repo_id']}/resolve/{revision}/{quote(relative, safe='/')}"
            download_file(url, target)
        return relative, {"bytes": target.stat().st_size, "sha256": sha256(target)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        checksums = dict(pool.map(get, files))
    manifest = {"repo_id": ds["repo_id"], "revision": revision,
                "license": metadata.get("cardData", {}).get("license"), "files": checksums}
    write_json(source_path, manifest)
    print(f"Fetched and verified {len(files)} files ({sum(f['bytes'] for f in checksums.values()):,} bytes)", flush=True)
    return manifest


def load_dataset(root):
    root = Path(root)
    info = read_json(root / "meta/info.json")
    if info["codebase_version"] != "v3.0":
        raise ValueError("This reader currently supports LeRobot v3.0; convert other versions explicitly")
    frames = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))], ignore_index=True)
    if episodes.episode_index.duplicated().any():
        raise ValueError("Duplicate episode metadata")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if "task_index" not in tasks.columns:
        tasks = tasks.reset_index()
    if "task_index" not in tasks.columns:
        raise ValueError("Task metadata has no task_index")
    return info, frames, episodes.set_index("episode_index", drop=False), tasks


def matrix(series):
    a = np.stack(series.to_numpy()).astype(np.float32)
    if a.ndim != 2:
        raise ValueError("Expected a vector feature per frame")
    return a


def video_reference(root, info, episode, camera):
    prefix = f"videos/{camera}"
    relative = info["video_path"].format(video_key=camera,
        chunk_index=int(episode[f"{prefix}/chunk_index"]),
        file_index=int(episode[f"{prefix}/file_index"]))
    return safe_path(root, relative), float(episode[f"{prefix}/from_timestamp"])


def sample_video(path, timestamps, size=(64, 48)):
    """Decode once in timestamp order; return frames aligned to requested video times."""
    import av
    from PIL import Image
    times = np.asarray(timestamps, dtype=float)
    order = np.argsort(times)
    output = [None] * len(times)
    pointer = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if len(times) and times[order[0]] > 1:
            container.seek(max(0, int(times[order[0]] / stream.time_base)), stream=stream, backward=True)
        tolerance = 0.55 / float(stream.average_rate or 30)
        for frame in container.decode(stream):
            when = frame.time
            if when is None:
                continue
            while pointer < len(times) and when + tolerance >= times[order[pointer]]:
                if abs(when - times[order[pointer]]) <= max(0.1, 2 * tolerance):
                    rgb = frame.to_ndarray(format="rgb24")
                    output[order[pointer]] = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BOX))
                pointer += 1
            if pointer >= len(times):
                break
    return output


def audit_arrays(group, fps, rules, state_dim=None, action_dim=None):
    errors, warnings = [], []
    times = group.timestamp.to_numpy(dtype=float)
    if not np.isfinite(times).all():
        errors.append("nonfinite_timestamp")
    elif len(times) > 1:
        diff = np.diff(times)
        if (diff <= 0).any():
            errors.append("nonmonotonic_timestamp")
        if (abs(diff - 1 / fps) > 0.2 / fps).any():
            warnings.append("irregular_timestep")
    if len(group) < rules["min_frames"]:
        warnings.append("short_episode")
    if group.frame_index.duplicated().any():
        errors.append("duplicate_frame_index")
    if not np.array_equal(group.frame_index.to_numpy(), np.arange(len(group))):
        errors.append("noncontiguous_frame_index")
    metrics = {"idle_fraction": None, "max_state_step": None}
    try:
        states, actions = matrix(group["observation.state"]), matrix(group.action)
        if state_dim is not None and states.shape[1] != state_dim:
            errors.append("state_dimension_mismatch")
        if action_dim is not None and actions.shape[1] != action_dim:
            errors.append("action_dimension_mismatch")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            errors.append("nonfinite_state_or_action")
        elif len(states) > 1:
            step = abs(np.diff(states, axis=0)).max(axis=1)
            metrics["max_state_step"] = float(step.max())
            metrics["idle_fraction"] = float(np.mean(step < rules["idle_state_step"]))
            if metrics["max_state_step"] > rules["max_state_step"]:
                warnings.append("state_jump")
            if metrics["idle_fraction"] > rules["max_idle_fraction"]:
                warnings.append("long_idle")
    except (ValueError, TypeError):
        errors.append("malformed_vector_feature")
    return {"errors": errors, "warnings": warnings, **metrics}


def audit_dataset(config, project, output):
    root = safe_path(project, config["dataset"]["root"])
    info, frames, episodes, tasks = load_dataset(root)
    rules = config["audit"]
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    task_ids = set(tasks.task_index.astype(int))
    dataset_errors = []
    if tasks.task_index.duplicated().any():
        dataset_errors.append("duplicate_task_index")
    if len(frames) != info["total_frames"]:
        dataset_errors.append("total_frames_mismatch")
    if frames.episode_index.nunique() != info["total_episodes"]:
        dataset_errors.append("total_episodes_mismatch")
    if set(frames.episode_index.astype(int)) != set(episodes.index.astype(int)):
        dataset_errors.append("episode_metadata_mismatch")
    if frames["index"].duplicated().any():
        dataset_errors.append("duplicate_global_index")
    rows, requests, decoded = [], {}, {}
    for episode_id, group in frames.groupby("episode_index", sort=True):
        # Preserve stored order: sorting would hide a broken time sequence.
        result = audit_arrays(group, info["fps"], rules,
                              info["features"]["observation.state"]["shape"][0], info["features"]["action"]["shape"][0])
        result.update(episode_index=int(episode_id), frames=len(group), task_indices=sorted(set(map(int, group.task_index))),
                      video_samples_checked=0, blurry_samples=0)
        if not set(group.task_index).issubset(task_ids):
            result["errors"].append("missing_task_annotation")
        if episode_id not in episodes.index:
            result["errors"].append("missing_episode_metadata")
        else:
            ep = episodes.loc[episode_id]
            if int(ep["length"]) != len(group):
                result["errors"].append("episode_length_mismatch")
            expected = np.arange(int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
            if not np.array_equal(group["index"].to_numpy(), expected):
                result["errors"].append("episode_global_index_range_mismatch")
            for camera in cameras:
                try:
                    path, offset = video_reference(root, info, ep, camera)
                    if not path.exists():
                        result["errors"].append("missing_video:" + camera)
                        continue
                    indices = np.linspace(0, len(group) - 1, rules["video_samples"], dtype=int)
                    for idx in indices:
                        when = offset + float(group.timestamp.iloc[idx])
                        requests.setdefault(str(path), []).append((when, int(episode_id), camera))
                except (KeyError, ValueError):
                    result["errors"].append("invalid_video_metadata:" + camera)
        rows.append(result)
    by_id = {r["episode_index"]: r for r in rows}
    for path, reqs in requests.items():
        try:
            images = sample_video(path, [r[0] for r in reqs], size=(160, 120))
        except Exception as exc:
            images = [None] * len(reqs)
            decoded[path] = type(exc).__name__
        for (_, ep_id, camera), im in zip(reqs, images):
            row = by_id[ep_id]
            if im is None:
                row["errors"].append("video_decode_or_alignment_failure:" + camera)
                continue
            row["video_samples_checked"] += 1
            gray = im.astype(float).mean(axis=2)
            laplace = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
            if laplace.var() < rules["blur_variance_threshold"]:
                row["blurry_samples"] += 1
    for row in rows:
        if row["blurry_samples"]:
            row["warnings"].append("low_texture_or_blur")
        row["errors"] = sorted(set(row["errors"]))
        row["warnings"] = sorted(set(row["warnings"]))
        row["valid"] = not row["errors"]
        row["selected"] = row["valid"] and not any(x in row["warnings"] for x in ("state_jump", "long_idle", "short_episode"))
    report = {"dataset": config["dataset"], "format": info["codebase_version"], "rules": rules,
              "total_frames": len(frames), "total_episodes": len(rows), "cameras": cameras,
              "dataset_errors": dataset_errors, "decode_errors": decoded, "episodes": rows,
              "valid_episodes": sum(r["valid"] for r in rows), "selected_episodes": sum(r["selected"] for r in rows),
              "limitations": ["Sampled video checks do not certify every frame or physical synchronization.",
                               "Blur and idle metrics are heuristics, not task-success labels.",
                               "State step thresholds use the dataset's native units; configure per robot."]}
    write_json(output, report)
    print(f"Audit: {len(rows)} episodes; {report['valid_episodes']} structurally valid; {report['selected_episodes']} selected", flush=True)
    return report


def split_episodes(ids, seed, validation_fraction):
    ids = np.array(sorted(set(map(int, ids))))
    if len(ids) < 3 or not 0 < validation_fraction < 1:
        raise ValueError("Need at least 3 valid episodes and a fraction between 0 and 1")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    count = max(1, min(len(ids) - 1, round(len(ids) * validation_fraction)))
    return sorted(ids[count:].tolist()), sorted(ids[:count].tolist())


def build_features(config, project, audit, destination):
    root = safe_path(project, config["dataset"]["root"])
    info, frames, episodes, _ = load_dataset(root)
    valid = {r["episode_index"] for r in audit["episodes"] if r["valid"]}
    chosen = frames[frames.episode_index.isin(valid) & (frames.frame_index % config["training"]["sample_stride"] == 0)].copy()
    chosen = chosen.reset_index(drop=True)
    cameras = audit["cameras"]
    grid = config["training"]["image_grid"]
    blocks = [matrix(chosen["observation.state"])]
    for camera in cameras:
        rgb = np.zeros((len(chosen), 3 * grid * grid), dtype=np.float32)
        requests = {}
        for row_index, row in chosen.iterrows():
            path, offset = video_reference(root, info, episodes.loc[int(row.episode_index)], camera)
            requests.setdefault(str(path), []).append((row_index, offset + float(row.timestamp)))
        for path, reqs in requests.items():
            ims = sample_video(path, [x[1] for x in reqs], (grid, grid))
            for (idx, _), im in zip(reqs, ims):
                if im is None:
                    raise ValueError(f"Video frame missing at sample {idx}; refuse silent zero-image training")
                rgb[idx] = im.reshape(-1) / 255.0
        blocks.append(rgb)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, x=np.concatenate(blocks, axis=1), y=matrix(chosen.action),
                        episode=chosen.episode_index.to_numpy(dtype=np.int64), frame=chosen.frame_index.to_numpy(dtype=np.int64))
    write_json(destination.with_suffix(".json"), {"rows": len(chosen), "cameras": cameras,
        "state_dim": info["features"]["observation.state"]["shape"][0],
        "action_names": info["features"]["action"].get("names"), "config": config["training"],
        "feature_type": "native joint state + per-camera area-resized RGB grid; no language encoder"})
    print(f"Features: {len(chosen):,} samples, {sum(b.shape[1] for b in blocks)} dimensions", flush=True)
