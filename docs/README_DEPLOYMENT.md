# Cyberbullying Detection Meta Model - Deployment Guide

## 🎯 Quick Fix for Red Lines in IDE

The red lines you're seeing are just IDE configuration issues. All packages are correctly installed!

### Fix for VS Code/Cursor:
1. **Open Command Palette**: `Cmd + Shift + P` (Mac) or `Ctrl + Shift + P` (Windows/Linux)
2. **Type**: `Python: Select Interpreter`
3. **Select**: `cyberbullying_env/bin/python` (the virtual environment we created)

### Fix for PyCharm:
1. **File** → **Settings** → **Project** → **Python Interpreter**
2. **Add Interpreter** → **Existing Environment**
3. **Browse to**: `cyberbullying_env/bin/python`

### Alternative: Status Bar Method
- Click the Python version in the bottom-left status bar
- Choose the `cyberbullying_env` environment

---

## 🚀 Deployment Instructions

### 1. Verify Installation
```bash
# Test the meta model works
source cyberbullying_env/bin/activate
python test_meta_model.py
```

### 2. Start the API Server
```bash
# Easy way - use the startup script
./start_api.sh

# Manual way
source cyberbullying_env/bin/activate
python fastapi_app.py
```

### 3. Access the API
- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📊 Meta Model Architecture

### Core Components:
1. **XGBoost Meta Model** (`xgb_cyberbullying_model.pkl`) - 112KB ✅
2. **BERT Models** (Local):
   - `trained_bert_cyberbullying_mendeley/` - Cyberbullying detection ✅
   - `hate_speech_recall_optimized_model/` - Hate speech detection ✅
3. **External Models** (Auto-downloaded):
   - RoBERTa & BERTweet for sentiment analysis
   - Sarcasm detection model
   - spaCy NER model

### Feature Pipeline:
The meta model combines **11 features**:
1. `cyber_conf` - Cyberbullying confidence (82.8% importance)
2. `hate_prob` - Hate speech probability  
3. `target_directed` - Target-directed language detection
4. `sent_neg`, `sent_neu`, `sent_pos` - Sentiment scores
5. `sarcasm_conf` - Sarcasm detection
6. `word_count`, `char_count`, `exclamation_count`, `question_count` - Text metadata

---

## 🔧 API Endpoints

### Prediction:
```bash
# Single text prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "You are so stupid!"}'

# Batch predictions
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello friend!", "You are awful"]}'
```

### Explanations:
```bash
# LIME explanation
curl -X POST "http://localhost:8000/explain/lime" \
  -H "Content-Type: application/json" \
  -d '{"text": "Nobody likes you"}'

# SHAP explanation
curl -X POST "http://localhost:8000/explain/shap" \
  -H "Content-Type: application/json" \
  -d '{"text": "Nobody likes you"}'
```

### Model Information:
```bash
# Feature importance
curl "http://localhost:8000/model/features"

# Model info
curl "http://localhost:8000/model/info"

# Health check
curl "http://localhost:8000/health"
```

---

## 📁 Project Structure

```
cyberbullying_detection/
├── 🔧 Core Files
│   ├── meta_model_final.py           # Production meta model
│   ├── fastapi_app.py               # API application
│   ├── test_meta_model.py           # Test script
│   └── requirements.txt             # Dependencies
│
├── 🤖 Models
│   ├── xgb_cyberbullying_model.pkl  # XGBoost meta model
│   ├── trained_bert_cyberbullying_mendeley/  # BERT cyberbullying
│   └── hate_speech_recall_optimized_model/   # BERT hate speech
│
├── 🚀 Deployment
│   ├── start_api.sh                 # Startup script
│   ├── cyberbullying_env/           # Virtual environment
│   └── README_DEPLOYMENT.md         # This guide
│
└── 📊 Analysis (Optional)
    ├── Stage1.ipynb                 # Original research notebook
    └── README.md                    # Research documentation
```

---

## ✅ Verification Checklist

- [ ] Virtual environment created (`cyberbullying_env/`)
- [ ] All packages installed (`pip install -r requirements.txt`)
- [ ] spaCy model downloaded (`python -m spacy download en_core_web_lg`)
- [ ] Meta model test passes (`python test_meta_model.py`)
- [ ] IDE interpreter set to `cyberbullying_env/bin/python`
- [ ] No red lines in code editor
- [ ] API starts successfully (`./start_api.sh`)
- [ ] API docs accessible (http://localhost:8000/docs)

---

## 🎯 Success Metrics

When everything is working correctly:
- ✅ No import errors or red lines
- ✅ Meta model loads successfully
- ✅ API server starts on port 8000
- ✅ LIME and SHAP explanations work
- ✅ All 11 features are extracted properly
- ✅ Predictions are consistent with training

---

## 🆘 Troubleshooting

### Red Lines Still Showing?
1. Restart your IDE after selecting the interpreter
2. Check the status bar shows `cyberbullying_env`
3. Try reloading the Python extension

### API Won't Start?
1. Check the meta model file exists: `ls -la xgb_cyberbullying_model.pkl`
2. Verify virtual environment: `source cyberbullying_env/bin/activate`
3. Test basic import: `python -c "import torch, transformers, shap, lime"`

### Segmentation Fault?
- This can happen with some transformer models
- The meta model includes fallback features for when models fail
- The core XGBoost model will still work even if BERT models fail

---

## 🎉 You're Ready for Production!

Your cyberbullying detection meta model is now properly deployed with:
- ✅ Full LIME & SHAP explainability
- ✅ RESTful API with FastAPI
- ✅ Robust error handling
- ✅ Multiple model integration
- ✅ Production-ready architecture 