FROM williamimoh/actian-vectorai-db:1.0b

# Install Python 3 and system utilities
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 → python (needed by some tooling)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ── Working directory matches the repo root so ROOT in server.py resolves correctly
WORKDIR /app

# ── Install VectorAI DB Python client (bundled wheel) ────────────────────────
COPY actiancortex-0.1.0b1-py3-none-any.whl ./
RUN pip install --no-cache-dir actiancortex-0.1.0b1-py3-none-any.whl --break-system-packages

# ── Install app Python dependencies ──────────────────────────────────────────
COPY app/requirements_medical.txt ./requirements_medical.txt
RUN pip install --no-cache-dir -r requirements_medical.txt --break-system-packages

# ── Copy application code and static assets ──────────────────────────────────
COPY app/ app/

# ── Copy case data and images ─────────────────────────────────────────────────
# Note: for large deployments, mount data/ as a Docker volume instead of baking it in.
COPY data/ data/

# Ensure the uploads directory exists for image-search file uploads
RUN mkdir -p uploads

# ── Expose web server port ────────────────────────────────────────────────────
EXPOSE 80

# ── Startup script ────────────────────────────────────────────────────────────
COPY docker-startup.sh /docker-startup.sh
RUN chmod +x /docker-startup.sh

ENTRYPOINT ["/docker-startup.sh"]
