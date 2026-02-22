#!/bin/bash
set -e

# ── Start VectorAI DB ─────────────────────────────────────────────────────────
# The base image (williamimoh/actian-vectorai-db:1.0b) ships the VectorAI DB
# server binary. Try known locations / entrypoints from the parent image.
echo "[startup] Starting VectorAI DB..."

if command -v vde &>/dev/null; then
    vde &
elif [ -f /docker-entrypoint.sh ]; then
    /docker-entrypoint.sh &
elif [ -f /entrypoint.sh ]; then
    /entrypoint.sh &
else
    echo "[startup] WARNING: VectorAI DB binary not found – semantic search will fall back to keyword mode."
fi

# ── Wait until VectorAI DB is accepting connections on port 50051 ─────────────
echo "[startup] Waiting for VectorAI DB on port 50051..."
for i in $(seq 1 30); do
    if nc -z localhost 50051 2>/dev/null; then
        echo "[startup] VectorAI DB is ready (${i}s)."
        break
    fi
    sleep 1
done

# ── Start FastAPI web server on port 80 ───────────────────────────────────────
echo "[startup] Starting MockMD web server on port 80..."
exec python -m uvicorn app.server:app --host 0.0.0.0 --port 80
