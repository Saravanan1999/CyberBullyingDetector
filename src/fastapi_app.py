#!/usr/bin/env python3
"""
FastAPI application for Cyberbullying Detection Meta Model
Provides REST API endpoints for predictions and explanations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import json
import base64
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import our meta model
from meta_model_final import CyberbullyingMetaModel

# Initialize FastAPI app
app = FastAPI(
    title="Cyberbullying Detection API",
    description="Meta model combining multiple transformer models for cyberbullying detection with LIME and SHAP explanations",
    version="1.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global model
    print("Loading cyberbullying meta model...")
    model = CyberbullyingMetaModel()
    print("✅ Model loaded successfully!")

# Request/Response models
class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    text: str
    prediction: int
    label: str
    probability: float

class FeatureContribution(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    contribution_type: str  # "positive" or "negative"

class TokenContribution(BaseModel):
    token: str
    shap_value: float
    importance_rank: int
    contribution_type: str  # "positive" or "negative"

class ShapExplanationResponse(BaseModel):
    text: str
    prediction: int
    label: str
    probability: float
    base_value: float
    expected_value: float
    feature_contributions: List[FeatureContribution]
    waterfall_data: Dict[str, Any]
    token_contributions: List[TokenContribution]
    text_highlight_html: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class LimeExplanationResponse(BaseModel):
    text: str
    prediction: int
    probability: float
    explanations: List[Dict[str, float]]
    html_file: str

# Removed duplicate - using the enhanced version above

class FeatureImportanceResponse(BaseModel):
    feature_importance: List[Dict[str, Any]]

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Cyberbullying Detection API",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Predict cyberbullying for a single text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction, probability = model.predict(input_data.text)
        label = "cyberbullying" if prediction == 1 else "not_cyberbullying"
        
        return PredictionResponse(
            text=input_data.text,
            prediction=int(prediction),
            label=label,
            probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_texts(input_data: BatchTextInput):
    """Predict cyberbullying for multiple texts"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions, probabilities = model.predict_batch(input_data.texts)
        
        results = []
        for text, pred, prob in zip(input_data.texts, predictions, probabilities):
            label = "cyberbullying" if pred == 1 else "not_cyberbullying"
            results.append(PredictionResponse(
                text=text,
                prediction=int(pred),
                label=label,
                probability=float(prob)
            ))
        
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/explain/lime", response_model=LimeExplanationResponse)
async def explain_with_lime(input_data: TextInput, num_features: int = 10):
    """Generate LIME explanation for a text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get prediction
        prediction, probability = model.predict(input_data.text)
        
        # Generate LIME explanation
        lime_exp = model.explain_with_lime(input_data.text, num_features=num_features)
        
        # Extract explanation data
        explanations = []
        for feature, weight in lime_exp.as_list():
            explanations.append({feature: weight})
        
        # Save HTML explanation
        html_filename = f"lime_explanation_{hash(input_data.text) % 10000}.html"
        lime_exp.save_to_file(html_filename)
        
        return LimeExplanationResponse(
            text=input_data.text,
            prediction=prediction,
            probability=probability,
            explanations=explanations,
            html_file=html_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LIME explanation failed: {str(e)}")

@app.post("/explain/shap", response_model=ShapExplanationResponse)
async def explain_with_shap(input_data: TextInput):
    """Generate SHAP explanation with waterfall visualization data"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get prediction first
        prediction, probability = model.predict(input_data.text)
        label = "Cyberbullying" if prediction == 1 else "Not Cyberbullying"
        
        # Get SHAP explanation
        shap_values = model.explain_with_shap(input_data.text)
        
        # Get token-level SHAP explanation
        token_explanation = model.explain_text_tokens(input_data.text)
        
        # Extract feature contributions
        feature_contributions = []
        shap_vals = shap_values.values[0]
        feature_vals = shap_values.data[0]
        base_value = float(shap_values.base_values[0])
        
        # Sort by absolute SHAP value for better visualization
        sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
        
        running_total = base_value
        waterfall_contributions = []
        
        for i in sorted_indices:
            feature_name = model.feature_columns[i]
            shap_val = float(shap_vals[i])
            feature_val = float(feature_vals[i])
            contribution_type = "positive" if shap_val > 0 else "negative"
            
            feature_contributions.append(FeatureContribution(
                feature_name=feature_name,
                feature_value=feature_val,
                shap_value=shap_val,
                contribution_type=contribution_type
            ))
            
            # For waterfall chart
            waterfall_contributions.append({
                "feature": feature_name,
                "feature_display_name": feature_name.replace('_', ' ').title(),
                "shap_value": shap_val,
                "feature_value": feature_val,
                "running_total": running_total + shap_val,
                "color": "#4CAF50" if shap_val > 0 else "#F44336",  # Green for positive, red for negative
                "abs_value": abs(shap_val)
            })
            running_total += shap_val
        
        # Create waterfall chart data structure
        waterfall_data = {
            "base_value": float(base_value),
            "final_value": float(base_value + sum(shap_vals)),
            "prediction_label": label,
            "prediction_probability": float(probability),
            "contributions": waterfall_contributions,
            "text_tokens": input_data.text.split(),
            "summary": {
                "total_positive_impact": float(sum([c["shap_value"] for c in waterfall_contributions if c["shap_value"] > 0])),
                "total_negative_impact": float(sum([c["shap_value"] for c in waterfall_contributions if c["shap_value"] < 0])),
                "net_impact": float(sum([c["shap_value"] for c in waterfall_contributions]))
            },
            "chart_config": {
                "title": f"SHAP Waterfall: {input_data.text[:50]}{'...' if len(input_data.text) > 50 else ''}",
                "subtitle": f"Prediction: {label} ({probability:.1%} confidence)",
                "x_axis_label": "Features",
                "y_axis_label": "Impact on Prediction",
                "positive_color": "#4CAF50",
                "negative_color": "#F44336",
                "base_color": "#9E9E9E",
                "height": 400,
                "width": 800
            }
        }
        
        # Process token contributions
        token_contributions = []
        for contrib in token_explanation['token_contributions']:
            token_contributions.append(TokenContribution(
                token=contrib['token'],
                shap_value=contrib['shap_value'],
                importance_rank=contrib['importance_rank'],
                contribution_type=contrib['contribution_type']
            ))
        
        return ShapExplanationResponse(
            text=input_data.text,
            prediction=int(prediction),
            label=label,
            probability=float(probability),
            base_value=float(base_value),
            expected_value=float(base_value + sum(shap_vals)),
            feature_contributions=feature_contributions,
            waterfall_data=waterfall_data,
            token_contributions=token_contributions,
            text_highlight_html=token_explanation['html_visualization']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")

@app.get("/model/features", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """Get feature importance from the meta model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        importance_df = model.get_feature_importance()
        if importance_df is None:
            raise HTTPException(status_code=500, detail="Feature importance not available")
        
        feature_importance = importance_df.to_dict('records')
        
        return FeatureImportanceResponse(
            feature_importance=feature_importance
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance retrieval failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Meta Model",
        "feature_columns": model.feature_columns,
        "num_features": len(model.feature_columns),
        "component_models": {
            "cyberbullying_bert": model.cyber_model is not None,
            "hate_speech_bert": model.hate_model is not None,
            "sentiment_roberta": model.roberta_model is not None,
            "sentiment_bertweet": model.bertweet_model is not None,
            "sarcasm_detector": model.sarcasm_model is not None,
            "spacy_ner": model.nlp is not None
        },
        "device": str(model.device)
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    
    # Test with a simple prediction
    try:
        test_prediction, test_probability = model.predict("Hello world")
        return {
            "status": "healthy",
            "model_loaded": True,
            "test_prediction_successful": True,
            "test_probability": float(test_probability)  # Convert numpy to Python float
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": True,
            "test_prediction_successful": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=7899,
        reload=True,
        log_level="info"
    ) 