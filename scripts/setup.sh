#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."

print_info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
print_warning() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
print_error()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

usage() {
  echo "Usage: $0 [-a] [-h] [-s] [-t]"
  echo "  -a: Build all (infra + screenplay-parser + higgs-tts)"
  echo "  -s: Build only screenplay-parser"
  echo "  -t: Build only higgs-tts"
  echo "  -h: Show this help message"
  exit 0
}

build_all=false
screenplay_only=false
higgs_only=false

while getopts "ahst" opt; do
  case "$opt" in
    a) build_all=true ;;
    s) screenplay_only=true ;;
    t) higgs_only=true ;;
    h) usage ;;
    *) usage ;;
  esac
done

shift $((OPTIND - 1))

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


build_higgs_tts_image() {
  local dockerfile="$ROOT_DIR/apps/higgs-tts/Dockerfile"
  local image_tag="umt/higgs-tts:latest"

  if [ ! -f "$dockerfile" ]; then
    print_error "Dockerfile not found at $dockerfile"
    return 1
  fi

  print_info "Building image $image_tag from $dockerfile"
  docker build -t "$image_tag" -f "$dockerfile" "$ROOT_DIR/apps/higgs-tts"
}


build_screenplay_parser_image() {
  local dockerfile="$ROOT_DIR/apps/screenplay-parser/Dockerfile"
  local image_tag="umt/screenplay-parser:latest"

  if [ ! -f "$dockerfile" ]; then
    print_error "Dockerfile not found at $dockerfile"
    return 1
  fi

  print_info "Building image $image_tag from $dockerfile"
  docker build -t "$image_tag" -f "$dockerfile" "$ROOT_DIR/apps/screenplay-parser"
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
  if [[ "$build_all" == true ]]; then
    print_info "Setting up Docker images (infra + all apps)"
    build_postgres_image
    pull_infra_images
    build_screenplay_parser_image
    build_higgs_tts_image
    print_info "Docker images ready."
  elif [[ "$screenplay_only" == true ]]; then
    print_info "Building only screenplay-parser"
    build_screenplay_parser_image
    print_info "Done."
  elif [[ "$higgs_only" == true ]]; then
    print_info "Building only higgs-tts"
    build_higgs_tts_image
    print_info "Done."
  else
    print_info "Setting up Docker images (infra only)"
    build_postgres_image
    pull_infra_images
    print_info "Docker images ready for infra stacks."
  fi
}

main

