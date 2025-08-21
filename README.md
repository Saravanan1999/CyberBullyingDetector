# Cyberbullying Detection with Explainable AI

A comprehensive cyberbullying detection system using ensemble machine learning with SHAP and LIME explanations.

## 🏗️ Project Structure

```
cyberbullyingDetection/
├── src/                           # Source code
│   ├── meta_model_final.py       # Main meta-model implementation
│   ├── fastapi_app.py            # FastAPI web service
│   └── meta_model_deployment.py  # Legacy deployment script
├── models/                        # Model artifacts
│   └── trained_models/           # Pre-trained models
│       ├── xgb_cyberbullying_model.pkl
│       ├── trained_bert_cyberbullying_mendeley/
│       ├── trained_bert_cyberbullying/
│       └── hate_speech_recall_optimized_model/
├── notebooks/                     # Jupyter notebooks
│   ├── Stage1.ipynb             # Main development notebook
│   ├── DSPP_Project_CNN_and_LSTMS.ipynb
│   ├── BERTForContext.ipynb
│   ├── ExplicitHateBERT.ipynb
│   ├── ExplicitHateClassicML.ipynb
│   └── NERTarget.ipynb
├── frontend/                      # Web interface
│   └── frontend_demo.html        # SHAP visualization demo
├── tests/                         # Test files
│   ├── test_meta_model.py
│   ├── test_api.py
│   └── test_word_analysis.py
├── scripts/                       # Utility scripts
│   └── start_api.sh              # API startup script
├── docs/                          # Documentation
│   ├── README_DEPLOYMENT.md
│   └── SHAP_WATERFALL_GUIDE.md
├── cyberbullying_env/            # Virtual environment
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source cyberbullying_env/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
# Using the startup script
chmod +x scripts/start_api.sh
./scripts/start_api.sh

# Or manually
cd src
python fastapi_app.py
```

### 3. Access the Demo
- **API Documentation**: http://localhost:7899/docs
- **Frontend Demo**: Open `frontend/frontend_demo.html` in your browser

## 🤖 Automatic Model Management

The system automatically downloads models from **Hugging Face Hub** when needed:

- **Repository**: https://huggingface.co/Saravanan1999/Cyberbullying
- **Auto-download**: Models download automatically when missing
- **No manual setup**: Just run the API server and go!

### Manual Model Download (Optional)
```bash
# Pre-download models if preferred
./scripts/download_models.sh

# Or use Hugging Face CLI directly
hf download Saravanan1999/Cyberbullying --repo-type=model --local-dir=models/trained_models
```

## 🧠 Model Architecture

### Meta-Learning Approach
- **XGBoost Meta-Model**: Combines predictions from multiple base models
- **BERT Models**: Fine-tuned for cyberbullying and hate speech detection
- **Sentiment Analysis**: RoBERTa and BERTweet models
- **Sarcasm Detection**: Specialized transformer model
- **NER Features**: Target-directed language detection

### Feature Pipeline
1. **Text Preprocessing**: Tokenization and normalization
2. **BERT Inference**: Cyberbullying and hate speech confidence
3. **Sentiment Analysis**: Negative, neutral, positive scores
4. **Sarcasm Detection**: Sarcasm confidence score
5. **NER Analysis**: Target detection and entity recognition
6. **Text Statistics**: Word count, character count, punctuation analysis
7. **Meta-Model Prediction**: Final ensemble prediction

## 🔍 Explainable AI Features

### SHAP (SHapley Additive exPlanations)
- **Word-level importance**: Token contribution analysis
- **Feature importance**: Model component contributions
- **Waterfall visualization**: Prediction flow from base to final value

### LIME (Local Interpretable Model-agnostic Explanations)
- **Local explanations**: Text perturbation analysis
- **Feature attribution**: Word-level importance scores

## 📊 API Endpoints

- `POST /predict` - Get cyberbullying prediction
- `POST /explain/shap` - Get SHAP explanations with visualizations
- `POST /explain/lime` - Get LIME explanations
- `GET /health` - API health check
- `GET /model/info` - Model information

## 🎯 Usage Example

### Python API Client
```python
import requests

# Predict cyberbullying
response = requests.post("http://localhost:7899/predict", 
                        json={"text": "You are the worst person ever"})
print(response.json())

# Get SHAP explanations
response = requests.post("http://localhost:7899/explain/shap", 
                        json={"text": "You are the worst person ever"})
explanations = response.json()
```

### Frontend Demo
The `frontend/frontend_demo.html` provides an interactive interface with:
- Text input and analysis
- SHAP waterfall visualization
- Word-level importance display
- Token attribution tables

## 🔧 Development

### Running Tests
```bash
cd tests
python test_meta_model.py
python test_api.py
python test_word_analysis.py
```

### Model Training
See notebooks in `notebooks/` directory for model training pipelines:
- `Stage1.ipynb` - Main development notebook
- `BERTForContext.ipynb` - BERT fine-tuning
- `ExplicitHateBERT.ipynb` - Hate speech detection

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- FastAPI
- SHAP
- spaCy
- XGBoost
- See `requirements.txt` for complete list

## 🏆 Performance

The ensemble approach combines multiple specialized models for robust cyberbullying detection with explainable predictions.

## 📄 License

This project is for academic and research purposes. 