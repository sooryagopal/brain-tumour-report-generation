#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# start.sh — Start both FastAPI backend and React frontend
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "📁 Project: $PROJECT_DIR"

# ── Backend ───────────────────────────────────────────────────────
echo ""
echo "🚀 Starting FastAPI backend on port 8000..."
cd "$PROJECT_DIR"

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "📦 Installing Python dependencies..."
  pip3 install -r requirements.txt
fi

# Start backend in background
python3 -m uvicorn backend.api:app --reload --port 8000 &
BACKEND_PID=$!
echo "✅ Backend PID: $BACKEND_PID"

sleep 2

# ── Frontend ──────────────────────────────────────────────────────
echo ""
echo "🎨 Starting React frontend on port 5173..."
cd "$PROJECT_DIR/frontend"

if [ ! -d "node_modules" ]; then
  echo "📦 Installing npm packages..."
  npm install
fi

npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend PID: $FRONTEND_PID"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🧠 NeuroScan AI is running!"
echo "  Frontend:  http://localhost:5173"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait and clean up on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Servers stopped.'" EXIT
wait
