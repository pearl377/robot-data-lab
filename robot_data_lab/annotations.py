"""Human-reviewable task annotations; preserve upstream files and label provenance."""
import json
from pathlib import Path

from .data import load_dataset
from .io import read_json


def export_annotations(root, destination):
    _, _, episodes, _ = load_dataset(root)
    rows = []
    for ep_id, episode in episodes.iterrows():
        rows.append({"episode_index": int(ep_id), "task": " / ".join(episode["tasks"]),
                     "source": "upstream_dataset", "review_status": "unreviewed", "reviewer_note": ""})
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Annotation file already exists; use a new --file to preserve manual edits")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)+"\n", encoding="utf-8")
    print(f"Exported {len(rows)} annotations to {destination}")


def check_annotations(rows, episode_ids):
    errors, seen, reviewed = [], set(), 0
    for number, row in enumerate(rows, 1):
        ep = row.get("episode_index")
        if type(ep) is not int or ep not in episode_ids:
            errors.append(f"line {number}: unknown episode")
        if ep in seen:
            errors.append(f"line {number}: duplicate episode")
        seen.add(ep)
        if not isinstance(row.get("task"), str) or not row["task"].strip():
            errors.append(f"line {number}: empty task description")
        if row.get("review_status") not in ("unreviewed", "accepted", "rejected", "corrected"):
            errors.append(f"line {number}: invalid review status")
        if row.get("source") not in ("upstream_dataset", "human", "vlm_suggestion"):
            errors.append(f"line {number}: annotation source must be recorded")
        reviewed += row.get("review_status") in ("accepted", "rejected", "corrected")
    missing = sorted(episode_ids - seen)
    if missing:
        errors.append(f"missing episode annotations: {missing}")
    return {"rows": len(rows), "reviewed_rows": reviewed, "errors": errors,
            "semantic_accuracy": "not_measured", "note": "Schema/provenance validation is not semantic ground truth."}


def validate_annotations(root, annotation_file):
    _, _, episodes, _ = load_dataset(root)
    rows = [json.loads(line) for line in Path(annotation_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return check_annotations(rows, set(map(int, episodes.index)))
