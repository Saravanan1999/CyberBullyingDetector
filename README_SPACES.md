---
title: Cyberbullying Detection with Explainable AI
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
models:
  - Saravanan1999/Cyberbullying
---

# 🛡️ Cyberbullying Detection with Explainable AI

A comprehensive cyberbullying detection system using ensemble machine learning with SHAP explanations.

## 🧠 Model Architecture

### Meta-Learning Approach
- **XGBoost Meta-Model**: Combines predictions from multiple base models
- **BERT Models**: Fine-tuned for cyberbullying and hate speech detection
- **Sentiment Analysis**: RoBERTa and BERTweet models
- **Sarcasm Detection**: Specialized transformer model
- **NER Features**: Target-directed language detection

## 🔍 Features

- **Real-time Detection**: Instant cyberbullying classification
- **SHAP Explanations**: Feature-level importance analysis
- **Word-Level Analysis**: Token contribution breakdown
- **Interactive Interface**: Easy-to-use web interface

## 📊 How to Use

1. **Enter Text**: Type or paste text in the input box
2. **Analyze**: Click "Analyze Text" to get predictions
3. **Understand**: View SHAP explanations and word-level analysis

## 🎯 Example Inputs

Try these examples to see how the model works:
- "You are such a great friend!" (Positive)
- "You are literally the worst human being ever" (Cyberbullying)
- "Nobody likes you. Just leave already." (Cyberbullying)

## 🏆 Performance

The ensemble approach combines multiple specialized models for robust cyberbullying detection with explainable predictions.

## 📄 Academic Use

This project is designed for academic and research purposes in the field of NLP and social media safety. 