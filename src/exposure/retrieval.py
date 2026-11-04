"""Candidate retrieval from the Pile via the infini-gram API.

The request format below follows the infini-gram API as we understand it
(POST JSON with ``index``, ``query_type`` and ``query``). VERIFY against the
current infini-gram docs before the first large run, including the Pile index
name in configs/exposure.yaml. Responses are cached on disk so reruns are free.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from common.config import REPO_ROOT


class InfiniGramClient:
    def __init__(self, api_url: str, index: str, cache_dir: str | Path, max_requests_per_second: float = 5.0):
        self.api_url = api_url
        self.index = index
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir if cache_dir.is_absolute() else REPO_ROOT / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max_requests_per_second
        self._last = 0.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "InfiniGramClient":
        ig = cfg["infini_gram"]
        return cls(ig["api_url"], ig["index"], ig["cache_dir"], ig.get("max_requests_per_second", 5))

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def request(self, query_type: str, **kwargs: Any) -> dict[str, Any]:
        import requests

        payload = {"index": self.index, "query_type": query_type, **kwargs}
        cache = self._cache_path(payload)
        if cache.exists():
            return json.loads(cache.read_text())
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = requests.post(self.api_url, json=payload, timeout=60)
        self._last = time.monotonic()
        resp.raise_for_status()
        out = resp.json()
        if "error" in out:
            raise RuntimeError(f"infini-gram error for {payload}: {out['error']}")
        cache.write_text(json.dumps(out))
        return out

    def count(self, query: str) -> int:
        return int(self.request("count", query=query).get("count", 0))

    def find(self, query: str) -> dict[str, Any]:
        """Locations of ``query`` in the index (per-shard suffix-array ranges)."""
        return self.request("find", query=query)

    def get_doc_by_rank(self, shard: int, rank: int, max_disp_len: int = 1000) -> dict[str, Any]:
        return self.request("get_doc_by_rank", s=shard, rank=rank, max_disp_len=max_disp_len)
