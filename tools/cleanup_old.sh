#!/usr/bin/env bash
set -euo pipefail

echo "==> Listing containers"
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'

echo "==> Compose down (this directory)"
docker compose down --remove-orphans || true

# Known container names we used earlier
CANDIDATES=(
  "shobo-perception"
  "shobo-engine"
  "jon-engine"
)

echo "==> Removing known old containers"
for c in "${CANDIDATES[@]}"; do
  if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
    echo " - removing $c"
    docker rm -f "$c" || true
  fi
done

echo "==> Removing old images (if present)"
# Remove by repo name; ignore errors if none exist
docker rmi -f jon-engine 2>/dev/null || true
docker rmi -f shobo-engine 2>/dev/null || true

# Optionally remove unused L4T bases that are not in use
echo "==> Pruning dangling/unused layers, networks, volumes, builder cache"
docker system prune -f || true
docker network prune -f || true
docker volume prune -f || true
docker builder prune -f || true

echo "==> Final state"
docker ps -a
docker images
