"""Prepare or execute the upstream Linux/GPU recipe. Never provisions rented compute."""
from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path

from .io import read_json, write_json, sha256


def commands(config, project, args, model_path=None):
    output = project / "runs" / f"smolvla-libero-s{args.seed}-steps{args.steps}"
    model_path = model_path or project / "artifacts/models/smolvlm"
    train = ["lerobot-train", "--policy.type=smolvla", "--policy.load_vlm_weights=true",
        f"--policy.vlm_model_name={model_path}", "--policy.push_to_hub=false", "--policy.device=cuda",
        f"--dataset.repo_id={config['dataset_repo']}", f"--dataset.revision={config['dataset_revision']}",
        "--dataset.video_backend=pyav", f"--output_dir={output}", f"--steps={args.steps}",
        f"--batch_size={args.batch_size}", "--num_workers=2", "--wandb.enable=false",
        f"--seed={args.seed}", "--save_freq=1000", "--log_freq=50", "--env_eval_freq=0"]
    checkpoint = args.checkpoint or str(output / "checkpoints/last/pretrained_model")
    evaluate = ["lerobot-eval", f"--policy.path={checkpoint}", "--policy.device=cuda", "--env.type=libero",
        f"--env.task={config['task_suite']}", f"--eval.n_episodes={args.episodes}", "--eval.batch_size=1",
        f"--seed={args.seed}", f"--output_dir={project / 'runs' / f'libero-eval-s{args.seed}'}"]
    return train, evaluate


def cloud_command(args, project):
    config = read_json(project / "configs/cloud.json")
    if min(args.steps, args.batch_size, args.episodes) < 1:
        raise ValueError("steps, batch size and evaluation episodes must be positive")
    train, evaluate = commands(config, project, args)
    if args.action == "plan":
        print(json.dumps({"status": "prepared_not_executed", "requirements": "Linux, Python 3.12, CUDA GPU, requirements-cloud.txt",
            "initialization": "Pretrained VLM backbone with a newly trained SmolVLA action expert; not a claimed paper reproduction",
            "dataset": config, "train_argv": train, "eval_argv": evaluate,
            "train_shell": shlex.join(train), "eval_shell": shlex.join(evaluate)}, ensure_ascii=False, indent=2))
        return
    if platform.system() != "Linux":
        raise RuntimeError("LIBERO recipe requires Linux. Windows local pipeline remains usable; use cloud plan to inspect commands.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected; refusing to silently start expensive VLA training on CPU")
    if not shutil.which("lerobot-train") or not shutil.which("lerobot-eval"):
        raise RuntimeError("Install requirements-cloud.txt in the Linux venv first")
    # Verify the installed upstream source, rather than merely its version label.
    dist = importlib.metadata.distribution("lerobot")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != config["lerobot_commit"]:
        raise RuntimeError("LeRobot source commit differs from configs/cloud.json; install the pinned git requirement")
    if args.action == "train":
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(config["vlm_repo"], revision=config["vlm_revision"],
                                       local_dir=project / "artifacts/models/smolvlm")
        train, _ = commands(config, project, args, model_path)
        command = train
    else:
        if not args.checkpoint or not (Path(args.checkpoint)/"config.json").is_file():
            raise ValueError("Provide --checkpoint pointing to an actual trained pretrained_model directory")
        command = evaluate
    run_dir = project / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = run_dir / f"{args.action}-{stamp}.json"
    manifest = {"status": "running", "config": config, "argv": command,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "started_at_utc": stamp}
    write_json(manifest_path, manifest)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    with (run_dir / f"{args.action}-{stamp}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    manifest.update(status="complete" if code == 0 else "failed", exit_code=code)
    write_json(manifest_path, manifest)
    if code:
        raise RuntimeError(f"Upstream command failed ({code}); inspect {manifest_path.name}")
