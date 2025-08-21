# 🚀 Deployment Guide

## Overview

This guide covers multiple deployment options for your cyberbullying detection API and frontend.

## 🤗 Option 1: Hugging Face Spaces (Recommended - FREE)

**Best for:** Academic projects, demos, portfolios

### Setup Steps:
1. **Create new Space**: Go to https://huggingface.co/new-space
2. **Configure Space**:
   - Name: `cyberbullying-detection`
   - SDK: `Gradio`
   - Make it Public
3. **Upload files**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/cyberbullying-detection
   cd cyberbullying-detection
   
   # Copy your project files
   cp ../cyberbullyingDetection/app.py .
   cp ../cyberbullyingDetection/requirements_spaces.txt requirements.txt
   cp ../cyberbullyingDetection/README_SPACES.md README.md
   cp -r ../cyberbullyingDetection/src .
   
   # Commit and push
   git add .
   git commit -m "Deploy cyberbullying detection"
   git push
   ```

### Benefits:
- ✅ **FREE** hosting
- ✅ **Automatic HTTPS**
- ✅ **GPU support** (if needed)
- ✅ **Easy sharing** via URL

---

## ☁️ Option 2: Railway (Simple & Affordable)

**Best for:** Production APIs, custom domains

### Setup:
1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 7899
   
   CMD ["python", "src/fastapi_app.py"]
   ```

2. **Deploy**:
   - Connect GitHub repo to Railway
   - Set environment variables if needed
   - Deploy automatically

### Benefits:
- ✅ **$5/month** starter plan
- ✅ **Custom domains**
- ✅ **Automatic deployments**

---

## 🌊 Option 3: Render (Free Tier Available)

**Best for:** Full-stack deployment

### Backend (API):
```yaml
# render.yaml
services:
  - type: web
    name: cyberbullying-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python src/fastapi_app.py
    envVars:
      - key: PORT
        value: 7899
```

### Frontend:
- Upload your `frontend_demo.html` as static site
- Update API_BASE URL to your Render backend

---

## 🐳 Option 4: Docker + Cloud Platforms

### Create Dockerfile:
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7899

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7899/health || exit 1

# Run the application
CMD ["python", "src/fastapi_app.py"]
```

### Deploy to:
- **Google Cloud Run**: Serverless, pay-per-use
- **AWS ECS/Fargate**: Enterprise-grade
- **Azure Container Instances**: Simple containers

---

## 🔧 Option 5: Traditional VPS (DigitalOcean, Linode)

### Setup Script:
```bash
#!/bin/bash
# setup.sh

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3-pip nginx -y

# Clone your repository
git clone YOUR_REPO_URL
cd cyberbullyingDetection

# Setup virtual environment
python3.10 -m venv cyberbullying_env
source cyberbullying_env/bin/activate
pip install -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/cyberbullying-api.service > /dev/null <<EOF
[Unit]
Description=Cyberbullying Detection API
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/cyberbullying_env/bin
ExecStart=$(pwd)/cyberbullying_env/bin/python src/fastapi_app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable cyberbullying-api
sudo systemctl start cyberbullying-api

# Setup Nginx reverse proxy
sudo tee /etc/nginx/sites-available/cyberbullying > /dev/null <<EOF
server {
    listen 80;
    server_name YOUR_DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:7899;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/cyberbullying /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

---

## 📊 Comparison Table

| Platform | Cost | Difficulty | Features |
|----------|------|------------|----------|
| **Hugging Face Spaces** | FREE | ⭐ Easy | Auto-scaling, GPU |
| **Railway** | $5/month | ⭐⭐ Medium | Custom domains |
| **Render** | FREE tier | ⭐⭐ Medium | Full-stack |
| **Cloud Run** | Pay-per-use | ⭐⭐⭐ Hard | Serverless |
| **VPS** | $5-20/month | ⭐⭐⭐⭐ Expert | Full control |

---

## 🎯 Recommended Deployment Path

### For Academic/Demo Use:
1. **Start with Hugging Face Spaces** (free, easy)
2. **Use Railway** if you need API endpoints

### For Production:
1. **Railway or Render** for MVPs
2. **Google Cloud Run** for scale
3. **VPS** for maximum control

---

## 🔒 Security Considerations

- **Environment Variables**: Never commit API keys
- **HTTPS**: Always use SSL certificates
- **Rate Limiting**: Prevent abuse
- **Input Validation**: Sanitize user inputs
- **Monitoring**: Set up logging and alerts

---

## 📈 Monitoring & Analytics

Add these to your deployment:
- **Health checks**: `/health` endpoint
- **Metrics**: Request count, response time
- **Logging**: Error tracking
- **Alerts**: Downtime notifications 