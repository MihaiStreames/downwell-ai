#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") [-a|--all] [-h|--help]"
  echo ""
  echo "options:"
  echo "  -a, --all   also remove training runs, venv and build metadata"
  echo "  -h, --help  show this message"
}

ALL=false

for arg in "$@"; do
  case "$arg" in
    -a | --all) ALL=true ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$arg'"
      usage
      exit 1
      ;;
  esac
done

cd "$(dirname "$0")/.."

find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" \) -not -path "./.venv/*" -print -exec rm -rf {} + 2> /dev/null || true

if [[ "$ALL" == true ]]; then
  [[ -d "runs" ]] && echo "runs" && rm -rf "runs"
  [[ -d ".venv" ]] && echo ".venv" && rm -rf ".venv"
  find . -type d -name "*.egg-info" -not -path "./.venv/*" -print -exec rm -rf {} + 2> /dev/null || true
fi

echo "clean complete"
