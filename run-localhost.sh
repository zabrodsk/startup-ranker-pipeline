#!/bin/bash
# Run Rockaway Deal Intelligence on localhost:8000
# Double-click or run: ./run-localhost.sh

cd "$(dirname "$0")"

echo "=== Rockaway Deal Intelligence — Localhost ==="
echo ""

# Check .env
if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example to .env and add your API keys."
    exit 1
fi

# Kill any existing server on 8000
echo "Stopping any existing server on port 8000..."
lsof -ti :8000 | xargs kill -9 2>/dev/null && echo "  Stopped." || echo "  (none running)"
sleep 2

# Ensure dependencies
echo ""
echo "Starting server..."
echo ""
echo "  Open: http://localhost:8000"
echo "  Password: \${APP_PASSWORD:-9876}"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

# Run server (errors will show in terminal)
exec python3 -m uvicorn web.app:app --reload --host 0.0.0.0 --port 8000
