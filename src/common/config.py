"""Config loading and provenance.

AGENTS.md convention: every run is driven by a config file, and outputs record the
config hash and git commit. Use :func:`provenance` for that record and
:func:`write_parquet` to attach it to parquet outputs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of a config (key order does not matter)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def git_commit() -> str:
    """Current commit, suffixed with -dirty if the tree has uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance(cfg: dict[str, Any], config_path: str | Path | None = None) -> dict[str, str]:
    return {
        "config_path": str(config_path) if config_path is not None else "",
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_parquet(df, path: str | Path, prov: dict[str, str]) -> Path:
    """Write a DataFrame to parquet with provenance stored in the file metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"provenance"] = json.dumps(prov).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


def read_provenance(path: str | Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(b"provenance")
    return json.loads(raw) if raw else {}
