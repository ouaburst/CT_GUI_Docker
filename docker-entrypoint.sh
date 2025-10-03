#!/usr/bin/env bash
set -e

# Usage:
#   docker run ... serve [--host 0.0.0.0 --port 8000]
#   docker run ... reconstruction.py --tree_ID ... --disk_ID ...
#   docker run ... python your_script.py

if [[ "${1:-}" == "serve" ]]; then
  shift
  # Assumes /app/odl_stream_server.py is present or bind-mounted
  exec uvicorn odl_stream_server:app --host 0.0.0.0 --port 8000 "$@"
else
  if [[ "${1:-}" =~ \.py$ ]]; then
    exec python "$@"
  else
    exec "$@"
  fi
fi