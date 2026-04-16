FROM python:3.11-slim

# ── System dependencies (FFmpeg + fonts for ASS subtitles)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# ── App directory
WORKDIR /app

# ── Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app source
COPY . .

# ── Create uploads directory
RUN mkdir -p uploads

# ── Expose port
EXPOSE 8000

# ── Start server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
