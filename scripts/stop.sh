#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

stop_backend=false
stop_screenplay=false

print_info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
print_warning() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
print_error()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

usage() {
  echo "Usage: $0 [-b] [-s] [-h]"
  echo "  -b  Stop backend Docker stacks (db, telemetry) and network"
  echo "  -s  Stop screenplay-viewer processes"
  echo "  -h  Show this help message"
  exit 0
}

while getopts "bsh" opt; do
  case "$opt" in
    b) stop_backend=true ;;
    s) stop_screenplay=true ;;
    h) usage ;;
    *) usage ;;
  esac
done

shift "$((OPTIND-1))"

kill_from_file() {
  local file="$1"
  local label="$2"

  if [ ! -f "$file" ]; then
    return 0
  fi

  local pid
  pid="$(cat "$file" 2>/dev/null || true)"
  if [ -z "${pid:-}" ]; then
    rm -f "$file"
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    print_info "Stopping $label (PID $pid)"
    kill "$pid" 2>/dev/null || true
  else
    print_warning "$label (PID $pid) is not running"
  fi

  rm -f "$file"
}

stop_screenplay_viewer() {
  if [ ! -d "$PID_DIR" ]; then
    print_info "No .pids directory found for screenplay-viewer; skipping process stop."
    return 0
  fi

  kill_from_file "$PID_DIR/backend-terminal.pid" "backend terminal"
  kill_from_file "$PID_DIR/backend.pid" "backend"
  kill_from_file "$PID_DIR/frontend-terminal.pid" "frontend terminal"
  kill_from_file "$PID_DIR/frontend.pid" "frontend"

  rmdir "$PID_DIR" 2>/dev/null || true
}

stop_stacks() {
  print_info "Removing Docker stacks 'telemetry' and 'db' (if present)"

  if command -v docker >/dev/null 2>&1; then
    docker stack rm grafana >/dev/null 2>&1 || print_warning "Stack 'grafana' not found or already removed."
    docker stack rm prometheus >/dev/null 2>&1 || print_warning "Stack 'prometheus' not found or already removed."
    docker stack rm aspire >/dev/null 2>&1 || print_warning "Stack 'aspire' not found or already removed."
    docker stack rm db >/dev/null 2>&1 || print_warning "Stack 'db' not found or already removed."
  else
    print_warning "Docker is not installed or not in PATH; skipping stack removal."
  fi
}

remove_network() {
  print_info "Attempting to remove Docker network 'net'"

  if command -v docker >/dev/null 2>&1; then
    docker network rm net >/dev/null 2>&1 || print_warning "Network 'net' not removed (may not exist or is still in use)."
  else
    print_warning "Docker is not installed or not in PATH; skipping network removal."
  fi
}

main() {
  print_info "Stopping development environment"

  # Default: if no specific target provided, stop everything.
  if [ "$stop_backend" = false ] && [ "$stop_screenplay" = false ]; then
    stop_backend=true
    stop_screenplay=true
  fi

  if [ "$stop_screenplay" = true ]; then
    stop_screenplay_viewer
  fi

  if [ "$stop_backend" = true ]; then
    stop_stacks
    remove_network
  fi

  print_info "Stop sequence completed."
}

main "$@"

