#!/usr/bin/env bash
# cmdc-support.sh — one-shot support container for Command Code (cmdc)
# Usage:  export COMMAND_CODE_API_KEY="sk-..." && ./cmdc-support.sh
# On exit (Ctrl+D / 'exit'), the container is nuked and secrets wiped.
# One shot cli : curl -fsSL https://raw.githubusercontent.com/Kotali-2019/codeneedle-bench/main/cmdc-support.sh -o cmdc-support.sh && chmod +x cmdc-support.sh
set -euo pipefail

IMAGE_NAME="cmdc-support:latest"
CONTAINER_NAME="cmdc-support"
DOCKERFILE_DIR="$(mktemp -d)"
COMPOSE_FILE="${DOCKERFILE_DIR}/docker-compose.yml"

# --- preflight ---
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
if [[ -z "${COMMAND_CODE_API_KEY:-}" ]]; then
  echo "ERROR: COMMAND_CODE_API_KEY is not set."
  echo "       export COMMAND_CODE_API_KEY=\"sk-...\" then re-run."
  exit 1
fi

# --- build image (skip if already present) ---
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "==> Building $IMAGE_NAME (first run, may take a minute)..."
  cat > "${DOCKERFILE_DIR}/Dockerfile" <<'EOF'
FROM node:22-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      docker.io systemd \
    && rm -rf /var/lib/apt/lists/* \
    && npm i -g command-code
WORKDIR /workspace
EOF
  docker build -t "$IMAGE_NAME" -f "${DOCKERFILE_DIR}/Dockerfile" "$DOCKERFILE_DIR"
else
  echo "==> Image $IMAGE_NAME already present, skipping build."
fi

# --- write compose file ---
cat > "$COMPOSE_FILE" <<EOF
services:
  cmdc-support:
    image: ${IMAGE_NAME}
    container_name: ${CONTAINER_NAME}
    stdin_open: true
    tty: true
    network_mode: host
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /var/log/journal:/var/log/journal:ro
      - /run/log/journal:/run/log/journal:ro
      - /etc/machine-id:/etc/machine-id:ro
    tmpfs:
      - /root/.commandcode
    environment:
      - TERM=xterm-256color
      - COMMAND_CODE_API_KEY=${COMMAND_CODE_API_KEY}
    command: sleep infinity
EOF

# --- nuke on exit ---
cleanup() {
  echo ""
  echo "==> Nuking container ${CONTAINER_NAME} and wiping secrets..."
  docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  rm -rf "$DOCKERFILE_DIR"
  echo "==> Done. Secrets wiped from RAM."
}
trap cleanup EXIT INT TERM

# --- start ---
echo "==> Starting support container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker compose -f "$COMPOSE_FILE" up -d

# --- enter; when user exits, trap nukes everything ---
echo "==> Entering container. Type 'exit' or Ctrl+D to leave and nuke."
docker exec -it "$CONTAINER_NAME" cmdc
EOF
