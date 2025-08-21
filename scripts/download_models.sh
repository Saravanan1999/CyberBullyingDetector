#!/bin/bash

# Model Download Script
# Downloads pre-trained models from external storage

echo "🤖 Downloading cyberbullying detection models..."
echo "================================================"

# Create models directory if it doesn't exist
mkdir -p models/trained_models

# Hugging Face repository
HF_REPO="Saravanan1999/Cyberbullying"

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing Hugging Face Hub..."
    pip install huggingface_hub
fi

# Download models from Hugging Face Hub
echo "📥 Downloading models from Hugging Face Hub..."
echo "Repository: $HF_REPO"
echo ""

if [ "$HF_REPO" = "YOUR_USERNAME/cyberbullying-models" ]; then
    echo "⚠️  Please update the HF_REPO variable in this script!"
    echo "   1. Upload your models to Hugging Face Hub"
    echo "   2. Update HF_REPO with your repository name"
    echo "   3. Run this script again"
    echo ""
    echo "   See scripts/setup_huggingface.sh for detailed instructions"
    exit 1
fi

# Download using huggingface_hub
python3 << EOF
from huggingface_hub import snapshot_download
import os

try:
    print("📥 Downloading models from Hugging Face...")
    snapshot_download(
        repo_id="$HF_REPO",
        repo_type="model",
        local_dir="models/trained_models",
        local_dir_use_symlinks=False
    )
    print("✅ Models downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading models: {e}")
    print("Please check:")
    print("1. Repository name is correct")
    print("2. Repository is public or you're logged in")
    print("3. Run 'huggingface-cli login' if needed")
EOF

echo ""
echo "🎯 Model download complete!"
echo "   Run './scripts/start_api.sh' to start the API server" 