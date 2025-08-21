#!/usr/bin/env python3
"""
Simple test for the cyberbullying meta model
Tests basic functionality without all the heavy models
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def test_meta_model():
    """Test loading and using the XGBoost meta model"""
    
    model_dir = "/Users/stygianphantom/Documents/Columbia University - Masters/cyberbullyingDetection"
    
    # Load XGBoost meta model
    meta_model_path = os.path.join(model_dir, "xgb_cyberbullying_model.pkl")
    if not os.path.exists(meta_model_path):
        print(f"❌ Meta model not found at {meta_model_path}")
        return False
        
    try:
        meta_model = joblib.load(meta_model_path)
        print("✅ Successfully loaded XGBoost meta model")
    except Exception as e:
        print(f"❌ Failed to load meta model: {e}")
        return False
    
    # Test with dummy features (the 11 features expected by the model)
    feature_columns = [
        'cyber_conf', 'hate_prob', 'target_directed', 
        'sent_neg', 'sent_neu', 'sent_pos', 'sarcasm_conf',
        'word_count', 'char_count', 'exclamation_count', 'question_count'
    ]
    
    # Create test features
    test_features = {
        'cyber_conf': 0.7,     # High cyberbullying confidence
        'hate_prob': 0.6,      # Moderate hate speech probability
        'target_directed': 1,   # Contains target-directed language
        'sent_neg': 0.8,       # High negative sentiment
        'sent_neu': 0.1,       # Low neutral sentiment
        'sent_pos': 0.1,       # Low positive sentiment
        'sarcasm_conf': 0.3,   # Low sarcasm
        'word_count': 8,       # 8 words
        'char_count': 45,      # 45 characters
        'exclamation_count': 1, # 1 exclamation mark
        'question_count': 0    # 0 question marks
    }
    
    try:
        # Create DataFrame
        df = pd.DataFrame([test_features]).reindex(columns=feature_columns, fill_value=0)
        print(f"Test features shape: {df.shape}")
        print(f"Feature columns: {list(df.columns)}")
        
        # Make prediction
        prediction = meta_model.predict(df)[0]
        probability = meta_model.predict_proba(df)[0, 1]
        
        print(f"\n🔮 Test Prediction:")
        print(f"   Input: Mock cyberbullying features")
        print(f"   Prediction: {'Cyberbullying' if prediction == 1 else 'Not Cyberbullying'}")
        print(f"   Confidence: {probability:.2%}")
        
        # Test with benign features
        benign_features = {
            'cyber_conf': 0.2,     # Low cyberbullying confidence
            'hate_prob': 0.1,      # Low hate speech probability
            'target_directed': 0,   # No target-directed language
            'sent_neg': 0.1,       # Low negative sentiment
            'sent_neu': 0.3,       # Moderate neutral sentiment
            'sent_pos': 0.6,       # High positive sentiment
            'sarcasm_conf': 0.1,   # Low sarcasm
            'word_count': 6,       # 6 words
            'char_count': 30,      # 30 characters
            'exclamation_count': 0, # 0 exclamation marks
            'question_count': 0    # 0 question marks
        }
        
        df_benign = pd.DataFrame([benign_features]).reindex(columns=feature_columns, fill_value=0)
        prediction_benign = meta_model.predict(df_benign)[0]
        probability_benign = meta_model.predict_proba(df_benign)[0, 1]
        
        print(f"\n🔮 Benign Test Prediction:")
        print(f"   Input: Mock benign features")
        print(f"   Prediction: {'Cyberbullying' if prediction_benign == 1 else 'Not Cyberbullying'}")
        print(f"   Confidence: {probability_benign:.2%}")
        
        # Check if model has feature importance
        if hasattr(meta_model, 'feature_importances_'):
            print(f"\n📊 Feature Importance:")
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': meta_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for idx, row in importance_df.head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
        
        print("\n✅ Meta model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Cyberbullying Meta Model...")
    print("=" * 50)
    
    success = test_meta_model()
    
    if success:
        print("\n🎉 All tests passed! The meta model is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the model files and dependencies.") 