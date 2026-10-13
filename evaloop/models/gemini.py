"""Gemini client for EvaLoop, via Google AI Studio or Vertex AI.

Uses the ``google-genai`` SDK rather than the OpenAI-compatibility shim because
the experiment needs three things the shim does not give cleanly:

* **Deterministic settings.** Greedy decoding plus an explicit thinking budget.
  Gemini 2.5 models think by default, which changes both the output distribution
  and the cost, and makes a run incomparable to the published records.
* **Token accounting.** The assignment requires reporting per-example and total
  cost, so usage metadata is captured per call.
* **The resolved model version.** Recorded with the run, since an alias such as
  ``gemini-2.5-flash`` moves over time.

Batch generation is implemented as bounded-concurrency fan-out over the API. The
loop runner picks the batch path whenever a model exposes ``batch_generate``,
which turns 150 per-cycle requests into a handful of round trips.

Config::

    {"name": "gemini-2.0-flash-lite",
     "type": "gemini",
     "model_id": "gemini-2.0-flash-lite",
     "api_key": "...",          # omit to use Vertex AI ADC
     "use_vertex": false,
     "project": "...", "location": "us-central1",   # Vertex AI only
     "temperature": 0.0, "top_p": 1.0,
     "max_tokens": 1024,
     "thinking_budget": 0,
     "concurrency": 16}
"""

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)

#: Returned instead of raising, mirroring OpenAILLM, so one bad task cannot end a run.
ERROR_PREFIX = "Error generating response"


class GeminiLLM(BaseLLM):
    """Implementation for Google Gemini models."""

    DEFAULT_MODEL_ID = "gemini-2.0-flash-lite"
    MAX_RETRIES = 8
    BASE_RETRY_DELAY = 2.0
    MAX_RETRY_DELAY = 60.0

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_id = model_config.get("model_id", self.DEFAULT_MODEL_ID)
        self.max_tokens = model_config.get("max_tokens", 1024)
        # Greedy by default: the protocol the published records were produced with.
        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p", 1.0)
        # 0 disables thinking on the 2.5 series; None leaves the model default.
        self.thinking_budget = model_config.get("thinking_budget", 0)
        self.concurrency = max(1, int(model_config.get("concurrency", 16)))

        self._usage_lock = threading.Lock()
        #: Cumulative token counts, for the cost table.
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0}
        #: Resolved model version strings seen in responses.
        self.model_versions = set()

        self._client = self._build_client(model_config)
        self._config_cls, self._thinking_cls = self._load_types()

    def _build_client(self, model_config: Dict[str, Any]):
        """Create the genai client for either AI Studio or Vertex AI."""
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The Gemini backend needs the google-genai SDK: pip install google-genai"
            ) from exc

        if model_config.get("use_vertex"):
            project = model_config.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = model_config.get("location") or os.environ.get(
                "GOOGLE_CLOUD_LOCATION", "us-central1"
            )
            if not project:
                raise ValueError(
                    "Vertex AI needs a project: set 'project' in the model config "
                    "or the GOOGLE_CLOUD_PROJECT environment variable"
                )
            logger.info("Using Vertex AI (project=%s, location=%s)", project, location)
            return genai.Client(vertexai=True, project=project, location=location)

        api_key = model_config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini credentials: set 'api_key' in the model config, or the "
                "GEMINI_API_KEY environment variable, or use_vertex: true with ADC"
            )
        return genai.Client(api_key=api_key)

    @staticmethod
    def _load_types():
        """Return the SDK's config and thinking-config classes."""
        from google.genai import types
        return types.GenerateContentConfig, types.ThinkingConfig

    def _request_config(self, max_tokens: int, temperature: float, top_p: float):
        """Build the per-request generation config."""
        kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "candidate_count": 1,
        }
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = self._thinking_cls(thinking_budget=self.thinking_budget)
        return self._config_cls(**kwargs)

    def _record_usage(self, response) -> None:
        """Accumulate token counts and the resolved model version."""
        meta = getattr(response, "usage_metadata", None)
        version = getattr(response, "model_version", None)
        with self._usage_lock:
            self.usage["calls"] += 1
            if meta is not None:
                self.usage["prompt_tokens"] += getattr(meta, "prompt_token_count", 0) or 0
                self.usage["output_tokens"] += getattr(meta, "candidates_token_count", 0) or 0
                self.usage["thoughts_tokens"] += getattr(meta, "thoughts_token_count", 0) or 0
            if version:
                self.model_versions.add(version)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """True for transient failures: rate limits, overload, 5xx, timeouts."""
        text = str(exc).lower()
        markers = ("429", "resource_exhausted", "rate limit", "quota",
                   "500", "502", "503", "504", "unavailable", "overloaded",
                   "deadline", "timeout", "internal error")
        return any(marker in text for marker in markers)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text for one prompt, retrying transient failures.

        Returns the text, or a string starting with ``Error generating response``
        when the call cannot be completed - the loop then records the cycle as a
        failure rather than aborting the run.
        """
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        config = self._request_config(max_tokens, temperature, top_p)

        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_id, contents=prompt, config=config
                )
                self._record_usage(response)

                text = (getattr(response, "text", None) or "").strip()
                if text:
                    return text

                # An empty body is usually a truncation or a safety stop; both are
                # real outcomes, so only retry once in case it was transient.
                reason = self._finish_reason(response)
                if attempt == 0:
                    logger.warning("Empty response (finish_reason=%s), retrying once", reason)
                    time.sleep(self.BASE_RETRY_DELAY)
                    continue
                return f"{ERROR_PREFIX}: empty response (finish_reason={reason})"

            except Exception as exc:  # noqa: BLE001 - SDK raises provider-specific types
                last_error = exc
                if attempt < self.MAX_RETRIES and self._is_retryable(exc):
                    delay = min(self.BASE_RETRY_DELAY * 2 ** attempt, self.MAX_RETRY_DELAY)
                    delay *= 0.5 + random.random()  # jitter, to decorrelate workers
                    logger.warning(
                        "Retryable error (attempt %d/%d), sleeping %.1fs: %s",
                        attempt + 1, self.MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Gemini call failed: %s", exc)
                return f"{ERROR_PREFIX}: {exc}"

        return f"{ERROR_PREFIX} after {self.MAX_RETRIES} retries: {last_error}"

    @staticmethod
    def _finish_reason(response) -> str:
        """Best-effort finish reason, for diagnosing empty responses."""
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            return str(getattr(candidates[0], "finish_reason", "unknown"))
        return "no_candidates"

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate for many prompts with bounded concurrency, order preserved.

        The loop runner prefers this over per-prompt calls, which is what keeps a
        150-task cycle to a few seconds of wall clock instead of minutes.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], **kwargs)]

        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(prompts))) as pool:
            return list(pool.map(lambda p: self.generate(p, **kwargs), prompts))

    def usage_report(self) -> Dict[str, Any]:
        """Return cumulative token usage and the resolved model versions."""
        with self._usage_lock:
            report = dict(self.usage)
        report["model_id"] = self.model_id
        report["model_versions"] = sorted(self.model_versions)
        report["settings"] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_tokens,
            "thinking_budget": self.thinking_budget,
        }
        return report

    def __str__(self) -> str:
        return f"GeminiLLM(model_name={self.model_name})"
