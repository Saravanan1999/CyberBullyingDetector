#!/usr/bin/env python3
"""
Production-Ready Cyberbullying Meta Model
Combines multiple transformer models through XGBoost meta learner
Supports LIME and SHAP explanations with graceful fallbacks
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import spacy
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from lime.lime_text import LimeTextExplainer
import shap
import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import snapshot_download

class CyberbullyingMetaModel:
    def __init__(self, model_dir="../models/trained_models"):
        """Initialize the meta model with all component models"""
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_repo = "Saravanan1999/Cyberbullying"
        
        # Feature columns in the expected order
        self.feature_columns = [
            'cyber_conf', 'hate_prob', 'target_directed', 
            'sent_neg', 'sent_neu', 'sent_pos', 'sarcasm_conf',
            'word_count', 'char_count', 'exclamation_count', 'question_count'
        ]
        
        # Ensure models are available
        self._ensure_models_available()
        self._load_models()
        
    def _ensure_models_available(self):
        """Check if models exist locally, download from HuggingFace if missing"""
        required_files = [
            "xgb_cyberbullying_model.pkl",
            "trained_bert_cyberbullying_mendeley",
            "hate_speech_recall_optimized_model"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = os.path.join(self.model_dir, file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"🔍 Missing model files: {missing_files}")
            print(f"📥 Downloading models from Hugging Face Hub: {self.hf_repo}")
            
            try:
                # Create model directory if it doesn't exist
                os.makedirs(self.model_dir, exist_ok=True)
                
                # Download models from Hugging Face
                snapshot_download(
                    repo_id=self.hf_repo,
                    repo_type="model",
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
                print("✅ Models downloaded successfully!")
                
            except Exception as e:
                print(f"❌ Error downloading models: {e}")
                print("Please ensure you have internet connection and the repository is accessible.")
                print(f"Repository: https://huggingface.co/{self.hf_repo}")
                raise RuntimeError(f"Failed to download required models: {e}")
        else:
            print("✅ All required models found locally")
        
    def _load_models(self):
        """Load all the component models with graceful fallbacks"""
        print("Loading meta model components...")
        
        # Load XGBoost meta model (REQUIRED)
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
            print("⚠ Cyberbullying model not found, will use text-based fallback")
            
        # Load hate speech BERT model (local)
        hate_path = os.path.join(self.model_dir, "hate_speech_recall_optimized_model")
        if os.path.exists(hate_path):
            try:
                self.hate_model = AutoModelForSequenceClassification.from_pretrained(hate_path)
                self.hate_tokenizer = AutoTokenizer.from_pretrained(hate_path)
                self.hate_model.to(self.device).eval()
                print("✓ Loaded hate speech BERT model")
            except Exception as e:
                print(f"⚠ Failed to load hate speech model: {e}")
                self.hate_model = None
        else:
            print("⚠ Hate speech model not found, will use text-based fallback")
            self.hate_model = None
        
        # Load sentiment models (from HuggingFace) - optional
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
            
        # Load sarcasm model (from HuggingFace) - optional
        try:
            self.sarcasm_model = AutoModelForSequenceClassification.from_pretrained("helinivan/english-sarcasm-detector")
            self.sarcasm_tokenizer = AutoTokenizer.from_pretrained("helinivan/english-sarcasm-detector")
            self.sarcasm_model.to(self.device).eval()
            print("✓ Loaded sarcasm detection model")
        except Exception as e:
            print(f"⚠ Could not load sarcasm model: {e}")
            self.sarcasm_model = None
            
        # Load spaCy NER model - optional
        try:
            self.nlp = spacy.load("en_core_web_lg")
            print("✓ Loaded spaCy NER model")
        except Exception as e:
            print(f"⚠ Could not load spaCy model: {e}")
            self.nlp = None
            
        print(f"🚀 Meta model loaded successfully on {self.device}")
            
    def get_cyberbullying_confidence(self, text):
        """Get cyberbullying confidence from BERT model or text-based fallback"""
        if self.cyber_model is not None and self.cyber_tokenizer is not None:
            try:
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
            except Exception as e:
                print(f"Error in cyberbullying model: {e}")
                
        # Fallback: simple text-based scoring
        negative_words = ["hate", "stupid", "dumb", "ugly", "loser", "kill", "die", "worst", "horrible", "pathetic"]
        words = text.lower().split()
        negative_count = sum(1 for word in words if word in negative_words)
        return min(negative_count / len(words) * 2, 1.0) if words else 0.5

    def get_hate_speech_prob(self, text):
        """Get hate speech probability from BERT model or text-based fallback"""
        if self.hate_model is not None and self.hate_tokenizer is not None:
            try:
                inputs = self.hate_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.hate_model(**inputs)
                logits = outputs.logits
                hate_prob = 1 / (1 + np.exp(-logits.cpu().numpy()))
                return float(hate_prob[0][0])
            except Exception as e:
                print(f"Error in hate speech model: {e}")
                
        # Fallback: simple text-based scoring
        hate_words = ["hate", "stupid", "idiot", "moron", "retard", "gay", "fag"]
        words = text.lower().split()
        hate_count = sum(1 for word in words if word in hate_words)
        return min(hate_count / len(words) * 3, 1.0) if words else 0.2

    def get_target_detection(self, text):
        """Detect if text contains target-directed language"""
        # Ensure text is a proper Python string
        if not isinstance(text, str):
            text = str(text)
        
        PERSON_PRONOUNS = {"you", "he", "she", "they", "him", "her", "your", "yours", "ur"}
        
        # Simple keyword-based approach (works with or without spaCy)
        words = text.lower().split()
        for word in words:
            if word in PERSON_PRONOUNS:
                return 1
                
        # Enhanced spaCy-based approach if available
        if self.nlp is not None:
            try:
                # Ensure text is a native Python string for spaCy
                text_str = str(text) if hasattr(text, 'item') else text
                doc = self.nlp(text_str)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        return 1
            except Exception as e:
                # Silently handle spaCy errors and fall back to keyword detection
                pass
                
        return 0

    def get_sentiment(self, text):
        """Get sentiment scores from ensemble of RoBERTa and BERTweet or fallback"""
        if self.roberta_model is not None and self.bertweet_model is not None:
            try:
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
            except Exception as e:
                print(f"Error in sentiment models: {e}")
                
        # Fallback: simple text-based sentiment
        positive_words = ["good", "great", "awesome", "nice", "love", "happy", "wonderful", "amazing"]
        negative_words = ["bad", "terrible", "awful", "hate", "sad", "angry", "horrible", "disgusting"]
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        total = len(words)
        
        if total == 0:
            return {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33}
            
        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = max(0, 1 - pos_score - neg_score)
        
        # Normalize
        total_score = pos_score + neg_score + neu_score
        if total_score > 0:
            return {
                "Negative": neg_score / total_score,
                "Neutral": neu_score / total_score,
                "Positive": pos_score / total_score
            }
        else:
            return {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33}

    def get_sarcasm_confidence(self, text):
        """Get sarcasm detection confidence or fallback"""
        if self.sarcasm_model is not None and self.sarcasm_tokenizer is not None:
            try:
                inputs = self.sarcasm_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    probs = F.softmax(self.sarcasm_model(**inputs).logits, dim=1).cpu().numpy()[0]
                return float(probs[1])
            except Exception as e:
                print(f"Error in sarcasm model: {e}")
                
        # Fallback: simple sarcasm indicators
        sarcasm_indicators = ["sure", "right", "yeah right", "oh really", "great job", "brilliant"]
        text_lower = text.lower()
        sarcasm_score = sum(1 for indicator in sarcasm_indicators if indicator in text_lower)
        return min(sarcasm_score * 0.3, 1.0)

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
        if not isinstance(text, str):
            text = str(text)
            
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
                # Ensure text is a proper Python string
                if hasattr(text, 'item'):  # numpy string
                    text = text.item()
                elif not isinstance(text, str):
                    text = str(text)
                    
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
        # class_names = ['not cyberbullying', 'cyberbullying']
        # explainer = LimeTextExplainer(class_names=class_names)
        
        # exp = explainer.explain_instance(
        #     text, 
        #     self.predict_proba_for_lime, 
        #     num_features=num_features
        # )
        
        # return exp
        print("LIME functionality is currently disabled")
        return None

    def explain_with_shap(self, text):
        """Generate SHAP explanation for a text"""
        # Extract features for the input text
        features = self.extract_features(text)
        input_df = pd.DataFrame([features])[self.feature_columns]
        
        # Generate SHAP values
        explainer = shap.Explainer(self.meta_model)
        shap_values = explainer(input_df)
        
        return shap_values

    def explain_text_tokens(self, text):
        """Generate word-wise SHAP explanation like in Stage1.ipynb"""
        if not isinstance(text, str):
            text = str(text)
        
        try:
            # Create a text masker for token-level explanations
            masker = shap.maskers.Text()
            
            # Create explainer for token-level analysis
            explainer = shap.Explainer(
                self.predict_proba_for_lime,  # Reuse the LIME-compatible function
                masker,
                output_names=["Not Cyberbullying", "Cyberbullying"]
            )
            
            # Get SHAP values for the text
            shap_values = explainer([text])
            
            # Get the predicted class
            pred_class = shap_values[0].values.sum(axis=1).argmax()
            
            # Extract tokens and their SHAP scores for the predicted class
            tokens = shap_values[0].data
            scores = shap_values[0].values[:, pred_class]
            
            # Create token contributions
            token_contributions = []
            for i, (token, score) in enumerate(zip(tokens, scores)):
                token_contributions.append({
                    'token': str(token),
                    'shap_value': float(score),
                    'contribution_type': 'positive' if score > 0 else 'negative',
                    'importance_rank': i
                })
            
            # Sort by absolute importance
            token_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            # Update importance ranks
            for i, contrib in enumerate(token_contributions):
                contrib['importance_rank'] = i + 1
            
            # Generate HTML visualization
            try:
                html_str = shap.plots.text(shap_values[0], display=False)
            except Exception as e:
                print(f"HTML generation failed: {e}")
                html_str = f"<p>Text: {text}</p><p>HTML visualization not available</p>"
            
            return {
                'token_contributions': token_contributions,
                'html_visualization': html_str,
                'predicted_class': pred_class,
                'shap_values': shap_values
            }
            
        except Exception as e:
            print(f"Token-level SHAP explanation failed: {e}")
            # Enhanced fallback: use feature-level SHAP to estimate token importance
            words = text.split()
            token_contributions = []
            
            # Get overall feature importance for this text
            features = self.extract_features(text)
            cyber_conf = features.get('cyber_conf', 0.5)
            hate_prob = features.get('hate_prob', 0.5)
            sent_neg = features.get('sent_neg', 0.3)
            
            for i, word in enumerate(words):
                # Enhanced heuristic based on cyberbullying keywords and feature values
                negative_words = ["hate", "stupid", "ugly", "loser", "worst", "trash", "worthless", "pathetic", "dumb", "idiot"]
                personal_pronouns = ["you", "your", "yours", "ur"]
                
                if word.lower() in negative_words:
                    # Scale by actual model confidence
                    score = -(cyber_conf * 0.3 + hate_prob * 0.2)
                elif word.lower() in personal_pronouns:
                    # Personal targeting
                    score = -(cyber_conf * 0.15)
                elif word.lower() in ["no", "never", "not"]:
                    # Negation - can reduce cyberbullying
                    score = 0.05
                else:
                    # Neutral words - small contribution
                    score = 0.01 * (1 - cyber_conf)
                
                token_contributions.append({
                    'token': word,
                    'shap_value': float(score),
                    'contribution_type': 'negative' if score < 0 else 'positive',
                    'importance_rank': i + 1
                })
            
            html_str = f"<p><strong>Text:</strong> {text}</p><p><em>Using enhanced heuristic analysis</em></p>"
            
            return {
                'token_contributions': token_contributions,
                'html_visualization': html_str,
                'predicted_class': 1 if cyber_conf > 0.5 else 0,
                'shap_values': None
            }

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
    
    # Generate LIME explanation (DISABLED)
    print("\n" + "="*60)
    print("LIME EXPLANATION (DISABLED)")
    print("="*60)
    
    text_to_explain = test_texts[1]  # cyberbullying example
    # try:
    #     lime_exp = model.explain_with_lime(text_to_explain)
    #     print(f"\nExplaining: {text_to_explain}")
        
    #     # Print top features
    #     for feature, weight in lime_exp.as_list():
    #         print(f"'{feature}': {weight:.3f}")
        
    #     # Save LIME explanation
    #     lime_exp.save_to_file("lime_explanation.html")
    #     print("LIME explanation saved to lime_explanation.html")
    # except Exception as e:
    #     print(f"LIME explanation failed: {e}")
    print("LIME explanations are currently disabled")
    
    # Generate SHAP explanation
    print("\n" + "="*60)
    print("SHAP EXPLANATION")
    print("="*60)
    
    try:
        shap_values = model.explain_with_shap(text_to_explain)
        print(f"\nSHAP values for: {text_to_explain}")
        
        for i, (feature, value) in enumerate(zip(model.feature_columns, shap_values.values[0])):
            print(f"{feature}: {value:.3f}")
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
    
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