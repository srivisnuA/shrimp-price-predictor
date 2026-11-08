"""Immutable trusted local artifacts, checksum verification and atomic promotion."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

import joblib
from filelock import FileLock

from shrimp_forecast import __version__
from shrimp_forecast.data import ROOT, atomic_bytes, json_bytes

ARTIFACT_SCHEMA = 1


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def environment(include_lightgbm: bool = False) -> dict:
    versions = {
        name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib"]
    }
    if include_lightgbm:
        versions["lightgbm"] = importlib.metadata.version("lightgbm")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "code_version": __version__,
        "git_revision": revision,
        "git_dirty": dirty,
        "source_sha256": source_sha256(),
    }


def publish(bundle: dict, root: Path, promote: bool = False) -> str:
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    stream = io.BytesIO()
    joblib.dump(bundle, stream, compress=3)
    payload = stream.getvalue()
    metadata = {
        "schema": ARTIFACT_SCHEMA,
        "run_id": run_id,
        "dataset": bundle["dataset"],
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "environment": environment(include_lightgbm=bundle["name"] == "lightgbm"),
        "report": bundle["report"],
    }
    run_dir = root / bundle["dataset"] / run_id
    with FileLock(str(root / ".registry.lock"), timeout=30):
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_bytes(run_dir / "model.joblib", payload)
        atomic_bytes(run_dir / "manifest.json", json_bytes(metadata))
        if promote:
            atomic_bytes(root / bundle["dataset"] / "CURRENT.json", json_bytes({"run_id": run_id}))
    return run_id


class Registry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[str, tuple[str, dict]] = {}
        self._lock = RLock()

    def load(self, dataset: str) -> dict:
        if dataset not in {"farmgate", "benchmark"}:
            raise KeyError("Unknown dataset")
        with self._lock:
            try:
                pointer = json.loads((self.root / dataset / "CURRENT.json").read_text())
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"No promoted {dataset} model. Run: shrimp train --dataset {dataset} --promote"
                ) from exc
            run_id = pointer["run_id"]
            if not isinstance(run_id, str) or not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}", run_id):
                raise ValueError("Invalid artifact identifier")
            if dataset in self._cache and self._cache[dataset][0] == run_id:
                return self._cache[dataset][1]
            run_dir = self.root / dataset / run_id
            metadata = json.loads((run_dir / "manifest.json").read_text())
            if metadata.get("schema") != ARTIFACT_SCHEMA or metadata.get("dataset") != dataset:
                raise ValueError("Incompatible artifact schema or dataset")
            versions = metadata["environment"]
            if versions["code_version"] != __version__:
                raise ValueError("Artifact code version mismatch; retrain with the serving environment")
            if versions["source_sha256"] != source_sha256():
                raise ValueError("Artifact source mismatch; retrain with the serving code")
            if platform.python_version_tuple()[:2] != tuple(versions["python"].split(".")[:2]):
                raise ValueError("Artifact Python version mismatch; retrain with the serving environment")
            for package, version in versions["packages"].items():
                try:
                    installed = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError as exc:
                    raise ValueError(f"Missing artifact dependency: {package}") from exc
                if installed != version:
                    raise ValueError(
                        f"Artifact dependency mismatch: {package}; retrain using the same lock file"
                    )
            raw = (run_dir / "model.joblib").read_bytes()
            if hashlib.sha256(raw).hexdigest() != metadata["artifact_sha256"]:
                raise ValueError("Artifact checksum mismatch")
            # Never load artifacts from an HTTP upload. This directory must be writable only by trusted trainers.
            bundle = joblib.load(io.BytesIO(raw))
            bundle["run_id"] = run_id
            self._cache[dataset] = (run_id, bundle)
            return bundle
