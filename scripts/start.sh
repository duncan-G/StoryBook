#!/usr/bin/env bash

set -e

usage() {
    echo "Usage: $0 [-f] [-s] [-d] [-h]"
    echo "  -f: Force re-initialization of Docker Swarm"
    echo "  -s: Start screenplay-viewer application"
    echo "  -d: Start only the database stack"
    echo "  -t: Start higgs-tts application"
    echo "  -h: Show this help message"
    echo "  -S: Start screenplay-viewer application in containerized mode"
    exit 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/apps/screenplay-viewer"
PID_DIR="$SCRIPT_DIR/.pids"

force=false
screenplay_viewer=false
sp_viewer_containerized=false
db_only=false
higgs_tts=false

while getopts "fsStdh" opt; do
    case "$opt" in
        f) force=true ;;
        s) screenplay_viewer=true ;;
        S) screenplay_viewer=true; sp_viewer_containerized=true ;;
        d) db_only=true ;;
        t) higgs_tts=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

shift $((OPTIND-1))

start_swarm() {
    if [ "$(docker info --format '{{.Swarm.LocalNodeState}}')" == "active" ]; then
        if [ "$force" == true ]; then
            echo "Leaving existing Docker Swarm"
            docker swarm leave --force
        else
            echo "Docker Swarm is active. Skipping initialization."
            return 0
        fi
    fi

    echo "Initializing new Docker Swarm"
    docker swarm init
}

create_network() {
    if docker network inspect net &>/dev/null; then
        if [ "$force" == true ]; then
            docker network rm net
        else
            echo "Network 'net' already exists. Skipping recreation."
            return 0
        fi
    fi

    echo "Creating Docker network 'net' for swarm"
    docker network create -d overlay --attachable --driver overlay net
}

start_db() {
    echo "Starting db"
    deploy_stack_if_needed db infra/db/postgres.stack.dev.yaml \
        DATABASE_NAME=dev \
        DATABASE_USER=dev \
        DATABASE_PASSWORD=dev
}

deploy_stack_if_needed() {
    local name=$1
    local file=$2
    shift 2
    if docker stack ls --format '{{.Name}}' 2>/dev/null | grep -qx "$name"; then
        if [ "$force" == true ]; then
            echo "Removing existing stack '$name'"
            docker stack rm "$name"
            sleep 2
        else
            echo "Stack '$name' already running. Skipping."
            return 0
        fi
    fi
    env "$@" docker stack deploy -c "$file" "$name"
}

start_telemetry() {
    echo "Starting telemetry"
    deploy_stack_if_needed grafana infra/telemetry/grafana.stack.dev.yaml
    deploy_stack_if_needed prometheus infra/telemetry/prometheus.stack.dev.yaml
    deploy_stack_if_needed aspire infra/telemetry/aspire.stack.dev.yaml ASPIRE_BROWSER_TOKEN=aspire
}


start_screenplay_viewer() {
    if [ "$sp_viewer_containerized" == true ]; then
        echo "Starting screenplay-viewer in containerized mode"
        docker stack deploy -c apps/screenplay-parser/stack.dev.yaml screenplay-parser

        mkdir -p "$PID_DIR"
        if command -v gnome-terminal >/dev/null 2>&1; then
            gnome-terminal --title="Frontend (screenplay-viewer)" -- bash -lc \
            "cd '$FRONTEND_DIR' && npm run dev & echo \$! > '$PID_DIR/frontend.pid'; wait || { echo 'Frontend failed. Press Enter to close.'; read -r; }" &
            local term_pid=$!
            echo "$term_pid" > "$PID_DIR/frontend-terminal.pid"
            echo "Frontend start command issued. Terminal PID: $term_pid"
        else
            echo "gnome-terminal not found. Falling back to background mode (nohup)."
            (
            cd "$FRONTEND_DIR"
            nohup npm run dev >/dev/null 2>&1 &
            echo $! > "$PID_DIR/frontend.pid"
            )
        fi

        return 0
    fi

    local viewer_script="$SCRIPT_DIR/start_screenplay_viewer.sh"

    if [ ! -f "$viewer_script" ]; then
        echo "start_screenplay_viewer.sh not found at $viewer_script"
        return 1
    fi

    echo "Starting screenplay-viewer via $viewer_script"
    bash "$viewer_script"
}

start_higgs_tts() {
    echo "Starting higgs-tts"
    deploy_stack_if_needed higgs-tts apps/higgs-tts/stack.dev.yaml
}

start_swarm
create_network
start_db

if [ "$db_only" != true ]; then
    start_telemetry

    if [ "$screenplay_viewer" == true ]; then
        start_screenplay_viewer
    fi

    if [ "$higgs_tts" == true ]; then
        start_higgs_tts
    fi
fi