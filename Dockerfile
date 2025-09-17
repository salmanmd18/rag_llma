# Dockerfile tailored for deploying the FastAPI backend on Render.
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Recommended Python env flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal system dependencies needed by numerical wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install backend requirements (kept separate for layer caching)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source code
COPY backend ./backend

# Ensure vector store directory exists for runtime writes (Render disks can mount here)
RUN mkdir -p /app/chroma_db

# Render sets PORT at runtime; default to 8000 for local runs
ENV PORT=8000
EXPOSE 8000

# Launch FastAPI app via uvicorn honoring Render's PORT assignment
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
