#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════╗"
echo "║    Startup Ranker — Cloudflare Deploy        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env and fill in your keys."
    exit 1
fi

export $(grep -v '^#' .env | xargs)

LLM_PROVIDER="${LLM_PROVIDER:-gemini}"

case "$LLM_PROVIDER" in
  anthropic)
    if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
      echo "ERROR: ANTHROPIC_API_KEY not set in .env (required when LLM_PROVIDER=anthropic)"
      exit 1
    fi
    echo "Using Anthropic (MODEL_NAME=${MODEL_NAME:-claude-haiku-4-5-20251001})"
    ;;
  gemini)
    if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
      echo "ERROR: GOOGLE_API_KEY not set in .env (required when LLM_PROVIDER=gemini)"
      exit 1
    fi
    echo "Using Gemini (MODEL_NAME=${MODEL_NAME:-gemini-3.1-flash-lite-preview})"
    ;;
  *)
    echo "Using LLM provider: $LLM_PROVIDER"
    ;;
esac

if [ -z "$PPLX_API_KEY" ] || [ "$PPLX_API_KEY" = "your_perplexity_api_key_here" ]; then
  echo "WARNING: PPLX_API_KEY not set — web search will be disabled"
else
  echo "Perplexity web search enabled"
fi

echo "[1/3] Installing dependencies..."
pip3 install -q fastapi uvicorn python-multipart 2>/dev/null

echo "[2/3] Starting web server on port 8000..."
python3 web/app.py &
SERVER_PID=$!
sleep 3

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

echo "[3/3] Creating Cloudflare Tunnel..."
echo ""
echo "  Your app will be available at the URL printed below."
echo "  Password: ${APP_PASSWORD:-9876}"
echo "  Press Ctrl+C to stop."
echo ""

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SERVER_PID 2>/dev/null
    exit 0
}
trap cleanup INT TERM

cloudflared tunnel --url http://localhost:8000 2>&1 &
CF_PID=$!

wait $CF_PID
cleanup
