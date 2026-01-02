#!/bin/bash

echo "======================================"
echo "  CareerBridge - Starting Application"
echo "======================================"
echo ""

# Start FastAPI backend
echo "Starting backend server..."
cd backend
python3 main.py &
BACKEND_PID=$!

echo "Backend started (PID: $BACKEND_PID)"
echo ""
echo "======================================"
echo "  Application Ready!"
echo "======================================"
echo ""
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "To view the frontend:"
echo "  Open frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for Ctrl+C
trap "echo 'Stopping backend...'; kill $BACKEND_PID; exit" INT
wait
