#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/apps/screenplay-parser"
FRONTEND_DIR="$REPO_ROOT/apps/screenplay-viewer"
PID_DIR="$SCRIPT_DIR/.pids"

print_info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
print_warning() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
print_error()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

if [ ! -d "$BACKEND_DIR" ]; then
  print_error "Backend directory not found: $BACKEND_DIR"
  exit 1
fi
if [ ! -d "$FRONTEND_DIR" ]; then
  print_error "Frontend directory not found: $FRONTEND_DIR"
  exit 1
fi

ensure_backend_env() {
  print_info "Ensuring backend virtualenv and dependencies (using uv)..."

  if ! command -v uv >/dev/null 2>&1; then
    print_error "uv is required but not installed. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi

  if [ ! -d "$BACKEND_DIR/.venv" ]; then
    print_info "Creating virtualenv in $BACKEND_DIR/.venv"
    uv venv "$BACKEND_DIR/.venv"
  fi

  local venv_python="$BACKEND_DIR/.venv/bin/python"
  if ! "$venv_python" -c "import uvicorn" 2>/dev/null; then
    print_info "Installing backend dependencies with uv..."
    uv pip install -e "$BACKEND_DIR" --python "$venv_python"
  fi
}

mkdir -p "$PID_DIR"

# ---------------------------------------------------------------------------
# Backend: FastAPI via uvicorn
# ---------------------------------------------------------------------------

start_backend_macos() {
  print_info "Starting backend (macOS)..."
  osascript >/dev/null <<OSA
tell application "Terminal"
  do script "cd '$BACKEND_DIR' && .venv/bin/python -m uvicorn main:app --reload"
end tell
OSA
  print_info "Backend start command issued in Terminal."
}

start_backend_linux() {
  print_info "Starting backend (Linux)..."
  local uv_run="cd '$BACKEND_DIR' && .venv/bin/python -m uvicorn main:app --reload"
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="Backend (screenplay-parser)" -- bash -c \
      "$uv_run & echo \$! > '$PID_DIR/backend.pid'; wait || { echo 'Backend failed. Press Enter to close.'; read -r; }" &
    local term_pid=$!
    echo "$term_pid" > "$PID_DIR/backend-terminal.pid"
    print_info "Backend start command issued. Terminal PID: $term_pid"
  else
    print_warning "gnome-terminal not found. Falling back to background mode (nohup)."
    (
      cd "$BACKEND_DIR"
      nohup .venv/bin/python -m uvicorn main:app --reload >/dev/null 2>&1 &
      echo $! > "$PID_DIR/backend.pid"
    )
    print_info "Backend started in background. PID saved to $PID_DIR/backend.pid"
  fi
}

# ---------------------------------------------------------------------------
# Frontend: Next.js dev server
# ---------------------------------------------------------------------------

start_frontend_macos() {
  print_info "Starting frontend (macOS)..."
  osascript >/dev/null <<OSA
tell application "Terminal"
  do script "cd '$FRONTEND_DIR' && npm run dev"
end tell
OSA
  print_info "Frontend start command issued in Terminal."
}

start_frontend_linux() {
  print_info "Starting frontend (Linux)..."
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="Frontend (screenplay-viewer)" -- bash -lc \
      "cd '$FRONTEND_DIR' && npm run dev & echo \$! > '$PID_DIR/frontend.pid'; wait || { echo 'Frontend failed. Press Enter to close.'; read -r; }" &
    local term_pid=$!
    echo "$term_pid" > "$PID_DIR/frontend-terminal.pid"
    print_info "Frontend start command issued. Terminal PID: $term_pid"
  else
    print_warning "gnome-terminal not found. Falling back to background mode (nohup)."
    (
      cd "$FRONTEND_DIR"
      nohup npm run dev >/dev/null 2>&1 &
      echo $! > "$PID_DIR/frontend.pid"
    )
    print_info "Frontend started in background. PID saved to $PID_DIR/frontend.pid"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OS="$(uname -s)"

case "$OS" in
  Darwin)
    ensure_backend_env
    start_backend_macos
    start_frontend_macos
    ;;
  Linux)
    ensure_backend_env
    start_backend_linux
    start_frontend_linux
    ;;
  *)
    print_error "Unsupported OS: $OS"
    exit 1
    ;;
esac

print_info "All services started."
