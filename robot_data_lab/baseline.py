from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import split_episodes
from .io import write_json


class BehaviorCloningMLP(nn.Module):
    """Small offline engineering baseline. This is not ACT, SmolVLA or a VLA."""
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


def normalize_fit(x):
    return x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)


def metrics(prediction, target, scale, episode_ids):
    error = np.abs(prediction - target)
    episode_mae = [float(error[episode_ids == ep].mean()) for ep in np.unique(episode_ids)]
    return {"mae_native_units": float(error.mean()), "normalized_mae": float((error / scale).mean()),
            "per_joint_mae": error.mean(axis=0).tolist(), "episode_macro_mae": float(np.mean(episode_mae)),
            "validation_samples": len(target), "validation_episodes": len(episode_mae)}


def train_one(x, y, episode, train_ids, val_ids, config, seed, destination, metric_scale, state_dim):
    started = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    train_mask, val_mask = np.isin(episode, train_ids), np.isin(episode, val_ids)
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Empty training or validation selection")
    # All fitted statistics use this arm's training set only.
    xmean, xstd = normalize_fit(x[train_mask])
    ymean, ystd = normalize_fit(y[train_mask])
    xt = torch.from_numpy(((x[train_mask] - xmean) / xstd).astype(np.float32))
    yt = torch.from_numpy(((y[train_mask] - ymean) / ystd).astype(np.float32))
    xv = torch.from_numpy(((x[val_mask] - xmean) / xstd).astype(np.float32))
    model = BehaviorCloningMLP(x.shape[1], y.shape[1], config["hidden_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(seed)
    history = []
    for step in range(1, config["steps"] + 1):
        indices = torch.from_numpy(rng.integers(0, len(xt), size=config["batch_size"]))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(xt[indices]), yt[indices])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == config["steps"]:
            history.append({"step": step, "train_mse_normalized": float(loss.item())})
    model.eval()
    with torch.inference_mode():
        pred = model(xv).numpy() * ystd + ymean
    result = metrics(pred, y[val_mask], metric_scale, episode[val_mask])
    result.update(seed=seed, train_episodes=len(train_ids), train_samples=int(train_mask.sum()),
                  steps=config["steps"], seconds=time.perf_counter() - started, history=history,
                  validation_mean_action_mae=float(abs(ymean - y[val_mask]).mean()))
    if state_dim == y.shape[1]:
        result["validation_copy_state_mae"] = float(abs(x[val_mask, :state_dim] - y[val_mask]).mean())
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Only load checkpoints that you created or trust. Store normalization as tensors.
    torch.save({"state_dict": model.state_dict(), "input_dim": x.shape[1], "output_dim": y.shape[1],
                "hidden_dim": config["hidden_dim"], "xmean": torch.from_numpy(xmean),
                "xstd": torch.from_numpy(xstd), "ymean": torch.from_numpy(ymean), "ystd": torch.from_numpy(ystd)},
               destination / "model.pt")
    np.savez_compressed(destination / "predictions.npz", prediction=pred, target=y[val_mask], episode=episode[val_mask])
    write_json(destination / "metrics.json", result)
    return result


def run_experiments(config, audit, feature_path, feature_metadata, output_dir):
    features = np.load(feature_path, allow_pickle=False)
    x, y, episode = features["x"], features["y"], features["episode"]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite feature matrix")
    train_ids, val_ids = split_episodes(np.unique(episode), **config["split"])
    selected = {r["episode_index"] for r in audit["episodes"] if r["selected"]}
    filtered = sorted(selected.intersection(train_ids))
    if not filtered:
        raise ValueError("Quality rules removed every training episode; inspect audit and adjust rules")
    _, metric_scale = normalize_fit(y[np.isin(episode, train_ids)])
    splits = {"train": train_ids, "validation": val_ids, "filtered_train": filtered}
    write_json(Path(output_dir) / "splits.json", splits)
    results = []
    filtering_changed = len(filtered) < len(train_ids)
    for seed in config["training"]["seeds"]:
        arms = {"raw": train_ids}
        if filtering_changed:
            arms["filtered"] = filtered
            arms["random_matched"] = sorted(np.random.default_rng(seed).choice(train_ids, size=len(filtered), replace=False).tolist())
        for arm, ids in arms.items():
            print(f"Training {arm} seed={seed}, {len(ids)} episodes", flush=True)
            metrics_one = train_one(x, y, episode, ids, val_ids, config["training"], seed,
                Path(output_dir) / f"{arm}-{seed}", metric_scale, feature_metadata["state_dim"])
            metrics_one.update(arm=arm, train_episode_ids=ids)
            results.append(metrics_one)
    summary = {"model": "RGB-grid + state MLP behavior cloning (CPU)", "is_vla": False,
               "evaluation": "offline held-out episodes; no simulator or robot execution", "results": results,
               "split": splits, "filtering_changed_training_set": filtering_changed,
               "comparison_note": "Identical filtered/random arms were skipped; no filtering benefit can be claimed."
               if not filtering_changed else "Same validation episodes, optimizer steps and batch size across arms; episode count matched for random control.",
               "aggregate": {}, "closed_loop_success_rate": None}
    for arm in sorted({r["arm"] for r in results}):
        values = np.array([r["mae_native_units"] for r in results if r["arm"] == arm])
        summary["aggregate"][arm] = {"mean_mae": float(values.mean()), "std_mae": float(values.std(ddof=1)) if len(values)>1 else 0.0,
                                      "seeds": len(values)}
    write_json(Path(output_dir) / "summary.json", summary)
    return summary
