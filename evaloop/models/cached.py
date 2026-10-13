"""Disk-backed caching wrapper for model clients.

Three things in this experiment depend on a call cache:

* **Cycle 1 is shared across arms.** All three anchors begin from the same
  dataset prompt with the same decoding settings, so cycle 1's generation and
  summarization are identical calls. Caching them makes pass@1 equal across arms
  *by construction* rather than by assumption, and removes a third of the spend.
* **Resumability.** A run interrupted at cycle 7 of 10 resumes from the cache
  instead of paying for the first six cycles again.
* **Free re-analysis.** Re-deriving metrics, or fixing a bug in the results
  writer, costs nothing.

The cache is keyed by the full request - model id, prompt and every decoding
setting that can change the output - so it can never serve a response generated
under different settings. It is keyed per model, not per configuration, which is
what lets the arms share cycle 1.

Records are appended as JSON lines, so a crash truncates at most one record and
the file stays greppable:

    {"key": "...", "model_id": "...", "prompt": "...", "response": "...",
     "usage": {...}, "created_at": 1760000000.0}
"""

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)


class CachingLLM(BaseLLM):
    """Wraps a model client and memoizes completions on disk.

    Delegates everything it does not implement (``usage_report``, custom
    attributes) to the wrapped client.
    """

    def __init__(self, inner: BaseLLM, cache_path: str, settings: Dict[str, Any] = None):
        """
        Args:
            inner: The model client to wrap.
            cache_path: JSONL file to read and append to; parents are created.
            settings: Decoding settings that affect the output and must therefore
                be part of the cache key. Defaults to the inner client's.
        """
        super().__init__(getattr(inner, "model_config", {}) or {})
        self.inner = inner
        self.model_name = getattr(inner, "model_name", "unnamed_model")
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._settings = settings if settings is not None else self._infer_settings(inner)
        self._lock = threading.Lock()
        self._entries: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0

        self._load()

    @staticmethod
    def _infer_settings(inner: BaseLLM) -> Dict[str, Any]:
        """Collect the decoding settings that change a completion."""
        report = getattr(inner, "usage_report", None)
        if callable(report):
            info = report()
            return {"model_id": info.get("model_id"), **(info.get("settings") or {})}
        return {
            "model_id": getattr(inner, "model_id", None),
            "temperature": getattr(inner, "temperature", None),
            "top_p": getattr(inner, "top_p", None),
            "max_output_tokens": getattr(inner, "max_tokens", None),
        }

    def _load(self) -> None:
        """Read existing cache records, tolerating a truncated final line."""
        if not self.cache_path.exists():
            logger.info("Starting a fresh call cache at %s", self.cache_path)
            return
        loaded = skipped = 0
        with self.cache_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._entries[record["key"]] = record["response"]
                    loaded += 1
                except (json.JSONDecodeError, KeyError):
                    skipped += 1
        logger.info("Loaded %d cached call(s) from %s%s", loaded, self.cache_path,
                    f" ({skipped} unreadable line(s) skipped)" if skipped else "")

    def _key(self, prompt: str, overrides: Dict[str, Any]) -> str:
        """Hash the full request: identical prompt + identical settings."""
        payload = json.dumps(
            {"prompt": prompt, "settings": {**self._settings, **overrides}},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _append(self, key: str, prompt: str, response: str) -> None:
        """Record a completion, under the lock so threads cannot interleave lines."""
        import time
        record = {
            "key": key,
            "model_id": self._settings.get("model_id"),
            "prompt": prompt,
            "response": response,
            "created_at": time.time(),
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._entries[key] = response
            with self.cache_path.open("a") as handle:
                handle.write(line + "\n")

    def generate(self, prompt: str, **kwargs) -> str:
        """Return a cached completion if present, otherwise call the inner model."""
        key = self._key(prompt, kwargs)
        with self._lock:
            cached = self._entries.get(key)
        if cached is not None:
            with self._lock:
                self.hits += 1
            return cached

        response = self.inner.generate(prompt, **kwargs)
        with self._lock:
            self.misses += 1
        # Never cache a failure: it would be replayed forever on resume.
        if not response.startswith("Error generating response"):
            self._append(key, prompt, response)
        return response

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Serve cached prompts locally and fan only the misses out to the model."""
        if not prompts:
            return []

        keys = [self._key(p, kwargs) for p in prompts]
        with self._lock:
            results: List[Optional[str]] = [self._entries.get(k) for k in keys]

        pending = [i for i, value in enumerate(results) if value is None]
        with self._lock:
            self.hits += len(prompts) - len(pending)
        if not pending:
            return [r for r in results]  # type: ignore[misc]

        inner_batch = getattr(self.inner, "batch_generate", None)
        missing = [prompts[i] for i in pending]
        fresh = inner_batch(missing, **kwargs) if callable(inner_batch) else [
            self.inner.generate(p, **kwargs) for p in missing
        ]

        with self._lock:
            self.misses += len(pending)
        for idx, response in zip(pending, fresh):
            results[idx] = response
            if not response.startswith("Error generating response"):
                self._append(keys[idx], prompts[idx], response)

        return [r for r in results]  # type: ignore[misc]

    def cache_report(self) -> Dict[str, Any]:
        """Hit/miss counts, used to assert that cycle 1 was shared across arms."""
        total = self.hits + self.misses
        return {
            "path": str(self.cache_path),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "entries": len(self._entries),
        }

    def usage_report(self) -> Dict[str, Any]:
        """Forward the inner client's usage report, annotated with cache stats."""
        report = getattr(self.inner, "usage_report", None)
        info = report() if callable(report) else {}
        return {**info, "cache": self.cache_report()}

    def __getattr__(self, item):
        """Delegate anything not defined here to the wrapped client."""
        return getattr(self.inner, item)

    def __str__(self) -> str:
        return str(self.inner)
