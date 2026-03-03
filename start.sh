#!/bin/bash

# ------------------------------------------------------------------------------
# Root Convenience Script: start.sh
# ------------------------------------------------------------------------------
# Purpose:
#   This script is a simple entry point for starting the development environment.
#   It delegates execution to the main script located at:
#     scripts/start.sh
#
# Usage:
#   ./start.sh [options]
#   (All arguments are passed through to the underlying script.)
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/scripts/start.sh" "$@"
