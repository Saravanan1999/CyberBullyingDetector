#!/bin/bash

# Cyberbullying Detection Meta Model - FastAPI Startup Script

echo "🚀 Starting Cyberbullying Detection API..."
echo "=========================================="

# Activate virtual environment
source cyberbullying_env/bin/activate

# Check if meta model exists (optional - API will auto-download if missing)
if [ ! -f "models/trained_models/xgb_cyberbullying_model.pkl" ]; then
    echo "⚠️  Models not found locally - will auto-download from Hugging Face on startup"
else
    echo "✅ Models found locally"
fi

echo "✅ Virtual environment activated"

# Start the FastAPI server
echo "🌐 Starting FastAPI server on http://localhost:7899"
echo "📖 API documentation available at http://localhost:7899/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Change to src directory and run the FastAPI server
cd src
python fastapi_app.py 