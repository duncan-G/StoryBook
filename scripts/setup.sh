#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."

print_info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
print_warning() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
print_error()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

# ---------------------------------------------------------------------------
# Build local images
# ---------------------------------------------------------------------------

build_postgres_image() {
  local dockerfile="$ROOT_DIR/infra/db/Dockerfile"
  local image_tag="umt/postgres:latest"

  if [ ! -f "$dockerfile" ]; then
    print_error "Dockerfile not found at $dockerfile"
    return 1
  fi

  print_info "Building image $image_tag from $dockerfile"
  docker build -t "$image_tag" -f "$dockerfile" "$ROOT_DIR/infra/db"
}

# ---------------------------------------------------------------------------
# Pull external images used by infra
# ---------------------------------------------------------------------------

pull_infra_images() {
  local images=(
    "grafana/grafana:latest"
    "prom/prometheus:latest"
    "mcr.microsoft.com/dotnet/aspire-dashboard:latest"
  )

  for img in "${images[@]}"; do
    print_info "Pulling image $img"
    docker pull "$img"
  done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  print_info "Setting up Docker images for infra"

  build_postgres_image
  pull_infra_images

  print_info "Docker images ready for infra stacks."
}

main "$@"

