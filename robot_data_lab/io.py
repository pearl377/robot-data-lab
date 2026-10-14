from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Unsafe dataset path: {relative}")
    return path


def request_json(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "robot-data-lab/0.1"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "robot-data-lab/0.1"})
            with urllib.request.urlopen(req, timeout=90) as response, temp.open("wb") as output:
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
                    count += len(chunk)
                expected = response.headers.get("Content-Length")
                if expected is not None and count != int(expected):
                    raise IOError("Incomplete download")
            os.replace(temp, destination)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
