#!/usr/bin/env python3
"""
Test script for the Cyberbullying Detection API
Demonstrates the SHAP waterfall endpoint functionality
"""

import requests
import json
from pprint import pprint

# API endpoint
API_BASE = "http://localhost:7899"

def test_prediction_and_shap(text):
    """Test prediction and SHAP explanation for a text"""
    print(f"\n{'='*80}")
    print(f"Testing: '{text}'")
    print('='*80)
    
    # Test basic prediction
    print("\n1. Basic Prediction:")
    prediction_response = requests.post(
        f"{API_BASE}/predict",
        json={"text": text}
    )
    
    if prediction_response.status_code == 200:
        pred_data = prediction_response.json()
        print(f"   Prediction: {pred_data['label']}")
        print(f"   Probability: {pred_data['probability']:.2%}")
    else:
        print(f"   Error: {prediction_response.status_code}")
        return
    
    # Test SHAP explanation
    print("\n2. SHAP Explanation:")
    shap_response = requests.post(
        f"{API_BASE}/explain/shap",
        json={"text": text}
    )
    
    if shap_response.status_code == 200:
        shap_data = shap_response.json()
        
        print(f"   Base Value: {shap_data['base_value']:.4f}")
        print(f"   Final Value: {shap_data['expected_value']:.4f}")
        print(f"   Prediction: {shap_data['label']} ({shap_data['probability']:.2%})")
        
        print("\n   Top Contributing Features:")
        for contrib in shap_data['feature_contributions'][:5]:  # Top 5
            feature_name = contrib['feature_name']
            shap_value = contrib['shap_value']
            feature_value = contrib['feature_value']
            impact = "↑" if shap_value > 0 else "↓"
            print(f"     {feature_name}: {shap_value:+.4f} {impact} (value: {feature_value:.3f})")
        
        print("\n   Waterfall Data Summary:")
        waterfall = shap_data['waterfall_data']
        summary = waterfall['summary']
        print(f"     Positive Impact: {summary['total_positive_impact']:+.4f}")
        print(f"     Negative Impact: {summary['total_negative_impact']:+.4f}")
        print(f"     Net Impact: {summary['net_impact']:+.4f}")
        
        # Show the structured data for frontend
        print("\n   Frontend Waterfall Data Structure:")
        print("   {")
        print(f"     \"base_value\": {waterfall['base_value']:.4f},")
        print(f"     \"final_value\": {waterfall['final_value']:.4f},")
        print(f"     \"prediction_label\": \"{waterfall['prediction_label']}\",")
        print(f"     \"contributions\": [")
        for i, contrib in enumerate(waterfall['contributions'][:3]):  # Show first 3
            print(f"       {{")
            print(f"         \"feature\": \"{contrib['feature']}\",")
            print(f"         \"feature_display_name\": \"{contrib['feature_display_name']}\",")
            print(f"         \"shap_value\": {contrib['shap_value']:.4f},")
            print(f"         \"running_total\": {contrib['running_total']:.4f},")
            print(f"         \"color\": \"{contrib['color']}\"")
            print(f"       }}{'' if i == 2 else ','}")
        if len(waterfall['contributions']) > 3:
            print(f"       ... and {len(waterfall['contributions']) - 3} more")
        print("     ],")
        print(f"     \"chart_config\": {{")
        print(f"       \"title\": \"{waterfall['chart_config']['title']}\",")
        print(f"       \"subtitle\": \"{waterfall['chart_config']['subtitle']}\",")
        print(f"       \"positive_color\": \"{waterfall['chart_config']['positive_color']}\",")
        print(f"       \"negative_color\": \"{waterfall['chart_config']['negative_color']}\"")
        print(f"     }}")
        print("   }")
        
    else:
        print(f"   Error: {shap_response.status_code}")
        print(f"   Response: {shap_response.text}")

def main():
    """Test the API with various examples"""
    
    # Check if API is running
    try:
        health_response = requests.get(f"{API_BASE}/health")
        if health_response.status_code != 200:
            print("❌ API is not healthy!")
            return
        print("✅ API is running and healthy!")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start the server first:")
        print("   ./start_api.sh")
        return
    
    # Test cases
    test_cases = [
        "You're such a great friend, always supportive!",           # Positive
        "You are literally the worst human being ever",             # Strong cyberbullying
        "Nobody likes you. Just leave already.",                    # Moderate cyberbullying
        "Hope you have a wonderful day!",                           # Very positive
        "I hate you so much, you're stupid and ugly"               # Strong negative with multiple indicators
    ]
    
    for text in test_cases:
        test_prediction_and_shap(text)
    
    print(f"\n{'='*80}")
    print("🎉 API Testing Complete!")
    print("💡 Use this waterfall_data structure in your frontend to build the SHAP waterfall chart!")
    print("   - base_value: starting point for the prediction")
    print("   - contributions: array of feature impacts (sorted by importance)")
    print("   - colors: ready-to-use hex colors for positive/negative impacts")
    print("   - chart_config: title, labels, and styling information")
    print('='*80)

if __name__ == "__main__":
    main() 