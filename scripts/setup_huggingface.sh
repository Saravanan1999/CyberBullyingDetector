#!/bin/bash

# Hugging Face Model Setup Script
echo "🤗 Setting up Hugging Face Hub for model storage..."
echo "=================================================="

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing Hugging Face Hub..."
    pip install huggingface_hub
fi

echo ""
echo "🔑 Please follow these steps:"
echo ""
echo "1. Create a Hugging Face account at: https://huggingface.co/join"
echo "2. Go to: https://huggingface.co/settings/tokens"
echo "3. Create a new token with 'Write' permissions"
echo "4. Run: huggingface-cli login"
echo "5. Enter your token when prompted"
echo ""
echo "6. Create a model repository:"
echo "   huggingface-cli repo create YOUR_USERNAME/cyberbullying-models --type model"
echo ""
echo "7. Upload your models:"
echo "   huggingface-cli upload YOUR_USERNAME/cyberbullying-models models/trained_models/ --repo-type model"
echo ""
echo "📝 After uploading, update the download script with your repository name!"
echo ""
echo "Example repository URL: https://huggingface.co/YOUR_USERNAME/cyberbullying-models" 