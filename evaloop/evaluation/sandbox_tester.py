"""Batched functional testing of untrusted model output inside a container.

Model-generated code is untrusted. EvalPlus already guards each execution with
a subprocess, resource limits and a syscall-restricting ``reliability_guard``,
but the assignment requires the stronger property that generated code never runs
on the host at all. So the pipeline is split:

* **generation** - talks to the model API, never executes a line of model output;
* **testing** - executes model output inside a container started with
  ``--network none``, a non-root user, a read-only mount of the repo and
  memory/CPU caps. No API key is present in that container's environment.

Testing is batched per cycle rather than per task: one container launch tests
every live task in the cycle, so a 10-cycle run costs 10 launches instead of
1500, and the MBPP+ ground truth is computed once per launch rather than once
per task.

``LocalTester`` is the same interface without the container, for use *inside*
the container itself or on a machine where running untrusted code is already
acceptable. It is not the default and the runner warns when it is selected.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Shape returned for every candidate, matching the data package's field names.
FAILED_RESULT = {"passed": False, "base_passed": False, "plus_passed": False}


class SandboxUnavailable(RuntimeError):
    """Raised when the container runtime is missing or not running."""


class DockerSandboxTester:
    """Tests batches of candidate solutions in a locked-down container."""

    def __init__(
        self,
        image: str = "evaloop-test:latest",
        repo_root: Path = None,
        cache_dir: Path = None,
        memory: str = "2g",
        cpus: str = "2",
        timeout: int = 1800,
        docker_bin: str = "docker",
    ):
        """
        Args:
            image: Container image carrying evalplus and this repo's deps.
            repo_root: Repo to mount read-only at /app.
            cache_dir: Writable host dir for the EvalPlus dataset/ground-truth
                cache, so the 378-task oracle is computed once overall rather
                than once per launch.
            memory, cpus: Container resource caps.
            timeout: Seconds before a whole batch is abandoned.
            docker_bin: Container CLI to invoke.
        """
        self.image = image
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
        self.cache_dir = Path(cache_dir or (self.repo_root / ".evalplus-cache"))
        self.memory = memory
        self.cpus = cpus
        self.timeout = timeout
        self.docker_bin = docker_bin
        self.launches = 0

    def preflight(self) -> None:
        """Check the runtime is present, running, and the image exists.

        Raises:
            SandboxUnavailable: With an actionable message.
        """
        if shutil.which(self.docker_bin) is None:
            raise SandboxUnavailable(
                f"'{self.docker_bin}' is not on PATH. Install Docker, or pass "
                "--tester local to run untrusted code without a container."
            )
        probe = subprocess.run([self.docker_bin, "info"], capture_output=True, text=True)
        if probe.returncode != 0:
            # Take the tail: the CLI prints unrelated plugin warnings first.
            detail = " | ".join((probe.stderr or "").strip().splitlines()[-2:])
            raise SandboxUnavailable(
                "The container runtime is not running (start Docker Desktop). "
                f"'{self.docker_bin} info' said: {detail[:300]}"
            )
        images = subprocess.run(
            [self.docker_bin, "image", "inspect", self.image],
            capture_output=True, text=True,
        )
        if images.returncode != 0:
            raise SandboxUnavailable(
                f"Image '{self.image}' not found. Build it with:\n"
                f"  docker build -f docker/Dockerfile.test -t {self.image} ."
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _command(self) -> List[str]:
        """Build the hardened ``docker run`` invocation."""
        return [
            self.docker_bin, "run", "--rm", "-i",
            # No network: generated code cannot phone home, and the testing
            # stage has no need to reach the model API.
            "--network", "none",
            # Drop privileges and block privilege escalation.
            "--user", "65534:65534",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            # Resource caps, so a runaway solution cannot take the host down.
            "--memory", self.memory,
            "--memory-swap", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", "512",
            # The repo is read-only; only the dataset cache and /tmp are writable.
            "--mount", f"type=bind,source={self.repo_root},target=/app,readonly",
            "--mount", f"type=bind,source={self.cache_dir},target=/cache",
            "--tmpfs", "/tmp:rw,size=256m,exec",
            "--env", "XDG_CACHE_HOME=/cache",
            "--env", "HOME=/tmp",
            "--workdir", "/app",
            self.image,
            "python", "scripts/test_worker.py",
        ]

    def test_batch(self, items: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Test ``(task_id, code)`` pairs, returning one result per input.

        A crashed or timed-out batch yields a failed result for every item, with
        the reason attached, rather than raising - a cycle that cannot be tested
        is a cycle the model did not sustain.
        """
        if not items:
            return []

        payload = json.dumps({"items": [{"task_id": t, "code": c} for t, c in items]})
        self.launches += 1
        logger.info("Sandbox launch %d: testing %d candidate(s)", self.launches, len(items))

        try:
            completed = subprocess.run(
                self._command(), input=payload, capture_output=True,
                text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("Sandbox batch timed out after %ds", self.timeout)
            return self._all_failed(items, f"sandbox timeout after {self.timeout}s")

        if completed.returncode != 0:
            tail = (completed.stderr or "").strip().splitlines()[-5:]
            logger.error("Sandbox exited %d: %s", completed.returncode, " | ".join(tail))
            return self._all_failed(items, f"sandbox exit {completed.returncode}: {' | '.join(tail)}")

        return self._parse(completed.stdout, items, completed.stderr)

    def _parse(self, stdout: str, items, stderr: str) -> List[Dict[str, Any]]:
        """Read the worker's JSON payload and realign it with the inputs."""
        line = next((ln for ln in reversed((stdout or "").splitlines()) if ln.strip()), "")
        try:
            by_task = {r["task_id"]: r for r in json.loads(line)["results"]}
        except (json.JSONDecodeError, KeyError, StopIteration) as exc:
            tail = (stderr or "").strip().splitlines()[-5:]
            logger.error("Unreadable sandbox output (%s): %s", exc, " | ".join(tail))
            return self._all_failed(items, f"unreadable sandbox output: {exc}")

        # Realign by task_id: never trust the worker's ordering.
        results = []
        for task_id, _code in items:
            result = by_task.get(task_id)
            if result is None:
                results.append({"task_id": task_id, **FAILED_RESULT,
                                "error": "missing from sandbox output"})
            else:
                results.append(result)
        return results

    @staticmethod
    def _all_failed(items, reason: str) -> List[Dict[str, Any]]:
        """Mark every candidate in a broken batch as failed, with the reason."""
        return [{"task_id": t, **FAILED_RESULT, "error": reason} for t, _ in items]

    def __str__(self) -> str:
        return f"DockerSandboxTester(image={self.image})"


class LocalTester:
    """Same interface, no container. Executes untrusted code in this process tree."""

    def __init__(self):
        self._tester = None
        self._problems = None
        self.launches = 0

    def preflight(self) -> None:
        """Load MBPP+ once and warn that isolation has been given up."""
        logger.warning(
            "Running untrusted model output WITHOUT a container. Only do this "
            "inside a disposable environment."
        )
        from .py_testing import EvalPlusCodeTester, load_mbpp_dataset
        self._problems, ground_truth = load_mbpp_dataset()
        self._tester = EvalPlusCodeTester(self._problems, ground_truth)

    def test_batch(self, items: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Test ``(task_id, code)`` pairs in-process."""
        if self._tester is None:
            self.preflight()
        self.launches += 1

        results = []
        for task_id, code in items:
            problem = self._problems.get(task_id)
            if problem is None:
                results.append({"task_id": task_id, **FAILED_RESULT,
                                "error": f"unknown task_id {task_id}"})
                continue
            try:
                outcome = self._tester.test_code(code, task_id, problem["entry_point"])
            except Exception as exc:  # noqa: BLE001
                results.append({"task_id": task_id, **FAILED_RESULT,
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            results.append({
                "task_id": task_id,
                "passed": bool(outcome.get("success", False)),
                "base_passed": bool(outcome.get("base_success", False)),
                "plus_passed": bool(outcome.get("plus_success", False)),
            })
        return results

    def __str__(self) -> str:
        return "LocalTester(no sandbox)"
