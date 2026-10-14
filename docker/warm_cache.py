"""Precompute the MBPP+ dataset and ground-truth cache.

Runs once, inside the container, with network access so EvalPlus can download
MbppPlus.jsonl.gz. Computing the ground truth means executing the benchmark's
*own* canonical solutions, which is why it is done here rather than on the host.

After this, every testing launch runs with --network none and reads the warmed
cache from the bind mount.

    docker run --rm -v "$PWD/.evalplus-cache:/cache" \
        -v "$PWD:/app:ro" -w /app evaloop-test:latest python docker/warm_cache.py
"""

import sys
import time


def main() -> int:
    sys.path.insert(0, "/app")
    from evaloop.evaluation.py_testing import load_mbpp_dataset

    started = time.time()
    problems, ground_truth = load_mbpp_dataset()
    print(f"Warmed {len(problems)} problems and {len(ground_truth)} oracles "
          f"in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
