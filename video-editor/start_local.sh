#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# SaykinEdit — Local Development Startup Script
# ─────────────────────────────────────────────────────────────────
set -e

echo ""
echo "  ██████  █████  ██    ██ ██   ██ ██ ███    ██ ███████ ██████  ██ ████████ "
echo " ██      ██   ██  ██  ██  ██  ██  ██ ████   ██ ██      ██   ██ ██    ██    "
echo "  ██████ ███████   ████   █████   ██ ██ ██  ██ █████   ██   ██ ██    ██    "
echo "       ██ ██   ██   ██    ██  ██  ██ ██  ██ ██ ██      ██   ██ ██    ██    "
echo "  ██████  ██   ██   ██    ██   ██ ██ ██   ████ ███████ ██████  ██    ██    "
echo ""
echo "  AI Video Editor — Saykin Style"
echo "─────────────────────────────────────────────────────────────────"

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
  echo "❌  FFmpeg not found. Please install it:"
  echo "    macOS:  brew install ffmpeg"
  echo "    Ubuntu: sudo apt install ffmpeg"
  exit 1
fi
echo "✅  FFmpeg found"

# Install Python deps
echo "📦  Installing Python dependencies..."
pip install -r requirements.txt -q

# Copy .env if it doesn't exist
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "⚠️   Created .env file."
  echo "     ► Open it and set OPENAI_API_KEY=sk-proj-YOUR_KEY"
  echo "     OR paste your key directly in the web UI."
  echo ""
fi

echo ""
echo "🚀  Starting SaykinEdit on http://localhost:8000"
echo "     Press Ctrl+C to stop."
echo "─────────────────────────────────────────────────────────────────"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
