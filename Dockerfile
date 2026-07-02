# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_USE_MOCK_API=false
ENV VITE_API_URL=
RUN npm run build

# ── Stage 2: Python runtime serving API + built frontend ────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# CPU-only torch wheel keeps the image well under the CUDA build's size.
# pymupdf (JD-PDF parsing, unused by rank.py) and streamlit (the separate HF Space UI,
# not present in this image) are dropped here only — requirements.txt stays untouched
# since local dev and the HF Space both still install from it.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -v -E "^(pymupdf|streamlit)([><=~ #]|$)" requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir -r requirements.docker.txt \
    && python -m spacy download en_core_web_sm

# Ranking code + fixed challenge JD (candidates are supplied per-request via upload).
COPY pipeline/ ./pipeline/
COPY data/preprocessing/pipeline_1.py ./data/preprocessing/pipeline_1.py
COPY data/jd.json ./data/jd.json
COPY rank.py ./rank.py
COPY backend/ ./backend/

# Decompress the bundled models at build time (offline at runtime).
COPY models/compressed/ ./models/compressed/
RUN python -c "\
import tarfile; \
from pathlib import Path; \
comp = Path('models/compressed'); \
decomp = Path('models/decompressed'); \
decomp.mkdir(parents=True, exist_ok=True); \
[tarfile.open(a, 'r:gz').extractall(decomp) for a in sorted(comp.glob('*.tar.gz'))]"

COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONIOENCODING=utf-8 \
    API_REQUIRE_AUTH=false

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
