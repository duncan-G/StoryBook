#!/bin/bash

# ------------------------------------------------------------------------------
# Root Convenience Script: stop.sh
# ------------------------------------------------------------------------------
# Purpose:
#   This script is a simple entry point for stopping the development environment.
#   It delegates execution to the main script located at:
#     scripts/stop.sh
#
# Usage:
#   ./stop.sh [options]
#   (All arguments are passed through to the underlying script.)
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/scripts/stop.sh" "$@"
