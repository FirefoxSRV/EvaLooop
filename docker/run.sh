#!/usr/bin/env bash
# Build the test image and warm the EvalPlus cache, then report readiness.
#
#   ./docker/run.sh build     build the image
#   ./docker/run.sh warm      download MBPP+ and precompute the oracles
#   ./docker/run.sh check     verify the sandbox is locked down
#   ./docker/run.sh setup     all three, in order
set -euo pipefail

IMAGE="${IMAGE:-evaloop-test:latest}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="$REPO_ROOT/.evalplus-cache"

build() {
  echo "==> Building $IMAGE"
  docker build -f "$REPO_ROOT/docker/Dockerfile.test" -t "$IMAGE" "$REPO_ROOT"
}

warm() {
  echo "==> Warming the EvalPlus cache (needs network, runs canonical solutions)"
  mkdir -p "$CACHE_DIR"
  docker run --rm \
    --mount "type=bind,source=$REPO_ROOT,target=/app,readonly" \
    --mount "type=bind,source=$CACHE_DIR,target=/cache" \
    --workdir /app "$IMAGE" python docker/warm_cache.py
}

check() {
  echo "==> Verifying sandbox isolation"
  echo -n "    no network       : "
  if docker run --rm --network none "$IMAGE" \
       python -c "import socket;socket.create_connection(('8.8.8.8',53),2)" 2>/dev/null; then
    echo "FAIL (the container reached the network)"; exit 1
  else
    echo "ok (connection refused)"
  fi

  echo -n "    non-root user    : "
  uid="$(docker run --rm --user 65534:65534 "$IMAGE" python -c "import os;print(os.getuid())")"
  [ "$uid" = "65534" ] && echo "ok (uid $uid)" || { echo "FAIL (uid $uid)"; exit 1; }

  echo -n "    repo read-only   : "
  if docker run --rm --mount "type=bind,source=$REPO_ROOT,target=/app,readonly" \
       --user 65534:65534 "$IMAGE" \
       python -c "open('/app/__probe','w')" 2>/dev/null; then
    echo "FAIL (wrote into the mounted repo)"; exit 1
  else
    echo "ok (read-only mount)"
  fi

  echo -n "    host \$HOME hidden: "
  if docker run --rm "$IMAGE" python -c "import os,sys;sys.exit(0 if os.path.exists('$HOME/.ssh') else 1)" 2>/dev/null; then
    echo "FAIL (host home visible)"; exit 1
  else
    echo "ok"
  fi

  echo -n "    resource limits  : "
  docker run --rm --memory 2g --cpus 2 --network none "$IMAGE" \
    python -c "import resource;resource.setrlimit(resource.RLIMIT_AS,(2**31,2**31));print('ok (RLIMIT_AS settable)')"
}

case "${1:-setup}" in
  build) build ;;
  warm)  warm ;;
  check) check ;;
  setup) build && warm && check && echo "
==> Sandbox ready. Run the ablation with:
    python scripts/run_anchored_experiment.py --model <id> --anchor assert,masked_assert,signature" ;;
  *) echo "usage: $0 {build|warm|check|setup}"; exit 2 ;;
esac
