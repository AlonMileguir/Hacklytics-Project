#!/bin/bash
set -e

# ── Check for required environment variables ──────────────────────────────────
if [ -z "$GEMINI_API_KEY" ]; then
    echo "[startup] WARNING: GEMINI_API_KEY environment variable is not set."
    echo "[startup]          AI-powered features will not work without it."
fi

# ── Start VectorAI DB ─────────────────────────────────────────────────────────
# The base image (williamimoh/actian-vectorai-db:1.0b) ships the VectorAI DB
# server binary at /usr/local/actianvectorai/bin/vdss-grpc-server
echo "[startup] Starting VectorAI DB..."

if [ -f /usr/local/actianvectorai/bin/vdss-grpc-server ]; then
    /usr/local/actianvectorai/bin/vdss-grpc-server &
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
