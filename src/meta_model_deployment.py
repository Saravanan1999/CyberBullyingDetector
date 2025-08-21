#!/usr/bin/env python3
"""
Complete Meta Model for Cyberbullying Detection
Combines multiple transformer models through XGBoost meta learner
Supports LIME and SHAP explanations
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap
import warnings
warnings.filterwarnings('ignore')

class CyberbullyingMetaModel:
    def __init__(self, model_dir="/Users/stygianphantom/Documents/Columbia University - Masters/cyberbullyingDetection"):
        """Initialize the meta model with all component models"""
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature columns in the expected order
        self.feature_columns = [
            'cyber_conf', 'hate_prob', 'target_directed', 
            'sent_neg', 'sent_neu', 'sent_pos', 'sarcasm_conf',
            'word_count', 'char_count', 'exclamation_count', 'question_count'
        ]
        
        self._load_models()
        
    def _load_models(self):
        """Load all the component models"""
        print("Loading meta model components...")
        
        # Load XGBoost meta model
        meta_model_path = os.path.join(self.model_dir, "xgb_cyberbullying_model.pkl")
        if os.path.exists(meta_model_path):
            self.meta_model = joblib.load(meta_model_path)
            print("✓ Loaded XGBoost meta model")
        else:
            raise FileNotFoundError(f"Meta model not found at {meta_model_path}")
        
        # Load cyberbullying BERT model (local) - try both possible paths
        cyber_paths = [
            os.path.join(self.model_dir, "trained_bert_cyberbullying_mendeley"),
            os.path.join(self.model_dir, "trained_bert_cyberbullying")
        ]
        
        self.cyber_model = None
        self.cyber_tokenizer = None
        
        for cyber_path in cyber_paths:
            if os.path.exists(cyber_path):
                try:
                    self.cyber_model = AutoModelForSequenceClassification.from_pretrained(cyber_path)
                    self.cyber_tokenizer = AutoTokenizer.from_pretrained(cyber_path)
                    self.cyber_model.to(self.device).eval()
                    print(f"✓ Loaded cyberbullying BERT model from {cyber_path}")
                    break
                except Exception as e:
                    print(f"⚠ Failed to load cyberbullying model from {cyber_path}: {e}")
                    continue
        
        if self.cyber_model is None:
            print("⚠ Cyberbullying model not found, will use fallback predictions")
            
        # Load hate speech BERT model (local)
        hate_path = os.path.join(self.model_dir, "hate_speech_recall_optimized_model")
        if os.path.exists(hate_path):
            self.hate_model = AutoModelForSequenceClassification.from_pretrained(hate_path)
            self.hate_tokenizer = AutoTokenizer.from_pretrained(hate_path)
            self.hate_model.to(self.device).eval()
            print("✓ Loaded hate speech BERT model")
        else:
            print("⚠ Hate speech model not found, will skip this feature")
            self.hate_model = None
        
        # Load sentiment models (from HuggingFace)
        try:
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
            self.roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
            self.roberta_model.to(self.device).eval()
            
            self.bertweet_model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
            self.bertweet_tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis", use_fast=False)
            self.bertweet_model.to(self.device).eval()
            print("✓ Loaded sentiment analysis models")
        except Exception as e:
            print(f"⚠ Could not load sentiment models: {e}")
            self.roberta_model = None
            self.bertweet_model = None
            
        # Load sarcasm model (from HuggingFace)
        try:
            self.sarcasm_model = AutoModelForSequenceClassification.from_pretrained("helinivan/english-sarcasm-detector")
            self.sarcasm_tokenizer = AutoTokenizer.from_pretrained("helinivan/english-sarcasm-detector")
            self.sarcasm_model.to(self.device).eval()
            print("✓ Loaded sarcasm detection model")
        except Exception as e:
            print(f"⚠ Could not load sarcasm model: {e}")
            self.sarcasm_model = None
            
        # Load spaCy NER model
        try:
            self.nlp = spacy.load("en_core_web_lg")
            print("✓ Loaded spaCy NER model")
        except Exception as e:
            print(f"⚠ Could not load spaCy model: {e}")
            self.nlp = None
            
    def get_cyberbullying_confidence(self, text):
        """Get cyberbullying confidence from BERT model"""
        if self.cyber_model is None:
            return 0.5  # neutral fallback
            
        inputs = self.cyber_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.cyber_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
        if predicted_class == 0:
            return confidence
        else:
            return 1 - confidence

    def get_hate_speech_prob(self, text):
        """Get hate speech probability from BERT model"""
        if self.hate_model is None:
            return 0.5  # neutral fallback
            
        inputs = self.hate_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.hate_model(**inputs)
        logits = outputs.logits
        hate_prob = 1 / (1 + np.exp(-logits.cpu().numpy()))
        return float(hate_prob[0][0])

    def get_target_detection(self, text):
        """Detect if text contains target-directed language"""
        PERSON_PRONOUNS = {"you", "he", "she", "they", "him", "her", "your", "yours"}
        
        # Simple keyword-based approach if spaCy is not available
        if self.nlp is None:
            words = text.lower().split()
            for word in words:
                if word in PERSON_PRONOUNS:
                    return 1
            return 0
            
        # Full spaCy-based approach
        doc = self.nlp(text)
        found = False
        for token in doc:
            if token.text.lower() in PERSON_PRONOUNS:
                found = True
                break
        if not found:
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    found = True
                    break
        return int(found)

    def get_sentiment(self, text):
        """Get sentiment scores from ensemble of RoBERTa and BERTweet"""
        if self.roberta_model is None or self.bertweet_model is None:
            return {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33}  # neutral fallback
            
        roberta_inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        roberta_inputs = {k: v.to(self.device) for k, v in roberta_inputs.items()}
        with torch.no_grad():
            roberta_probs = F.softmax(self.roberta_model(**roberta_inputs).logits, dim=1).cpu().numpy()[0]

        bertweet_inputs = self.bertweet_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        bertweet_inputs = {k: v.to(self.device) for k, v in bertweet_inputs.items()}
        with torch.no_grad():
            bertweet_probs = F.softmax(self.bertweet_model(**bertweet_inputs).logits, dim=1).cpu().numpy()[0]

        final_probs = np.array([
            0.6 * roberta_probs[0] + 0.4 * bertweet_probs[0],
            0.4 * roberta_probs[1] + 0.6 * bertweet_probs[1],
            0.6 * roberta_probs[2] + 0.4 * bertweet_probs[2]
        ])
        return {"Negative": final_probs[0], "Neutral": final_probs[1], "Positive": final_probs[2]}

    def get_sarcasm_confidence(self, text):
        """Get sarcasm detection confidence"""
        if self.sarcasm_model is None:
            return 0.5  # neutral fallback
            
        inputs = self.sarcasm_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = F.softmax(self.sarcasm_model(**inputs).logits, dim=1).cpu().numpy()[0]
        return probs[1]

    def get_text_metadata(self, text):
        """Extract basic text metadata features"""
        metadata = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?")
        }
        return metadata

    def extract_features(self, text):
        """Extract all features for a single text"""
        features = {}
        features["cyber_conf"] = self.get_cyberbullying_confidence(text)
        features["hate_prob"] = self.get_hate_speech_prob(text)
        features["target_directed"] = self.get_target_detection(text)

        sentiment = self.get_sentiment(text)
        features["sent_neg"] = sentiment["Negative"]
        features["sent_neu"] = sentiment["Neutral"]
        features["sent_pos"] = sentiment["Positive"]

        features["sarcasm_conf"] = self.get_sarcasm_confidence(text)
        features.update(self.get_text_metadata(text))
        return features

    def predict(self, text):
        """Make prediction for a single text"""
        features = self.extract_features(text)
        df = pd.DataFrame([features]).reindex(columns=self.feature_columns, fill_value=0)
        prediction = self.meta_model.predict(df)[0]
        probability = self.meta_model.predict_proba(df)[0, 1]  # probability of cyberbullying
        return prediction, probability

    def predict_batch(self, texts):
        """Make predictions for multiple texts"""
        feature_list = []
        for text in texts:
            features = self.extract_features(text)
            feature_list.append(features)
        
        df = pd.DataFrame(feature_list).reindex(columns=self.feature_columns, fill_value=0)
        predictions = self.meta_model.predict(df)
        probabilities = self.meta_model.predict_proba(df)[:, 1]
        return predictions, probabilities

    def predict_proba_for_lime(self, texts):
        """LIME-compatible prediction function"""
        feature_rows = []
        for text in texts:
            try:
                features = self.extract_features(text)
            except Exception as e:
                print(f"Error extracting features: {e}")
                features = {col: 0 for col in self.feature_columns}
            feature_rows.append(features)

        df = pd.DataFrame(feature_rows).reindex(columns=self.feature_columns, fill_value=0)
        probs = self.meta_model.predict_proba(df)
        
        if probs.shape[1] != 2:
            raise ValueError("predict_proba must return probabilities for both classes")
        return probs

    def explain_with_lime(self, text, num_features=10):
        """Generate LIME explanation for a text"""
        class_names = ['not cyberbullying', 'cyberbullying']
        explainer = LimeTextExplainer(class_names=class_names)
        
        exp = explainer.explain_instance(
            text, 
            self.predict_proba_for_lime, 
            num_features=num_features
        )
        
        return exp

    def explain_with_shap(self, text):
        """Generate SHAP explanation for a text"""
        # Extract features for the input text
        features = self.extract_features(text)
        input_df = pd.DataFrame([features])[self.feature_columns]
        
        # Generate SHAP values
        explainer = shap.Explainer(self.meta_model)
        shap_values = explainer(input_df)
        
        return shap_values

    def get_feature_importance(self):
        """Get feature importance from the meta model"""
        if hasattr(self.meta_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.meta_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None


def main():
    """Test the meta model"""
    # Initialize the meta model
    model = CyberbullyingMetaModel()
    
    # Test texts
    test_texts = [
        "You're such a great friend, always supportive!",         # likely not cyberbullying
        "You are literally the worst human being ever",           # likely cyberbullying
        "Nobody likes you. Just leave already.",                  # harsh - borderline case
        "Hope you have a good day!"                               # clearly positive
    ]
    
    print("\n" + "="*60)
    print("CYBERBULLYING META MODEL PREDICTIONS")
    print("="*60)
    
    for text in test_texts:
        prediction, probability = model.predict(text)
        label = "Cyberbullying" if prediction == 1 else "Not Cyberbullying"
        print(f"\nText: {text}")
        print(f"Prediction: {label} ({probability:.2%} confidence)")
    
    # Generate LIME explanation for first text
    print("\n" + "="*60)
    print("LIME EXPLANATION")
    print("="*60)
    
    text_to_explain = test_texts[1]  # cyberbullying example
    lime_exp = model.explain_with_lime(text_to_explain)
    print(f"\nExplaining: {text_to_explain}")
    
    # Print top features
    for feature, weight in lime_exp.as_list():
        print(f"'{feature}': {weight:.3f}")
    
    # Save LIME explanation
    lime_exp.save_to_file("lime_explanation.html")
    print("LIME explanation saved to lime_explanation.html")
    
    # Generate SHAP explanation
    print("\n" + "="*60)
    print("SHAP EXPLANATION")
    print("="*60)
    
    shap_values = model.explain_with_shap(text_to_explain)
    print(f"\nSHAP values for: {text_to_explain}")
    
    for i, (feature, value) in enumerate(zip(model.feature_columns, shap_values.values[0])):
        print(f"{feature}: {value:.3f}")
    
    # Show feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        print(importance_df)
    
    print("\n✅ Meta model test completed successfully!")


if __name__ == "__main__":
    main() 