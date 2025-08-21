#!/usr/bin/env python3
"""
Comprehensive test for the enhanced SHAP API with word-wise analysis
"""

import requests
import json
from pprint import pprint

API_BASE = "http://localhost:7899"

def test_comprehensive_analysis(text):
    """Test both feature-level and word-level SHAP analysis"""
    print(f"\n{'='*100}")
    print(f"🔍 COMPREHENSIVE ANALYSIS: '{text}'")
    print('='*100)
    
    response = requests.post(
        f"{API_BASE}/explain/shap",
        json={"text": text}
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return
    
    data = response.json()
    
    # 1. Basic Prediction
    print(f"\n🎯 PREDICTION RESULT:")
    print(f"   Classification: {data['label']}")
    print(f"   Confidence: {data['probability']:.1%}")
    print(f"   Base Value: {data['base_value']:.4f}")
    print(f"   Final Value: {data['expected_value']:.4f}")
    
    # 2. Feature-Level Analysis
    print(f"\n⚙️ TOP FEATURE CONTRIBUTIONS:")
    for i, contrib in enumerate(data['feature_contributions'][:5]):
        impact = "↗️" if contrib['shap_value'] > 0 else "↘️"
        print(f"   {i+1}. {contrib['feature_name']}: {contrib['shap_value']:+.4f} {impact}")
    
    # 3. Word-Level Analysis  
    print(f"\n📝 TOP WORD CONTRIBUTIONS:")
    for i, token in enumerate(data['token_contributions'][:8]):
        impact = "↗️" if token['shap_value'] > 0 else "↘️"
        print(f"   {i+1}. '{token['token']}': {token['shap_value']:+.4f} {impact}")
    
    # 4. Waterfall Summary
    waterfall = data['waterfall_data']
    summary = waterfall['summary']
    print(f"\n🌊 WATERFALL SUMMARY:")
    print(f"   Positive Impact: {summary['total_positive_impact']:+.4f}")
    print(f"   Negative Impact: {summary['total_negative_impact']:+.4f}")
    print(f"   Net Impact: {summary['net_impact']:+.4f}")
    
    # 5. Most Important Elements
    print(f"\n🎯 KEY INSIGHTS:")
    top_feature = data['feature_contributions'][0]
    top_word = data['token_contributions'][0]
    print(f"   Most Important Feature: {top_feature['feature_name']} ({top_feature['shap_value']:+.4f})")
    print(f"   Most Important Word: '{top_word['token']}' ({top_word['shap_value']:+.4f})")
    
    return data

def main():
    """Test various types of text"""
    
    # Check API health
    try:
        health = requests.get(f"{API_BASE}/health")
        if health.status_code != 200:
            print("❌ API is not healthy!")
            return
        print("✅ API is running and healthy!")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure server is running on port 7899")
        return
    
    # Test cases with different characteristics
    test_cases = [
        {
            "text": "You're such a great friend, always supportive!",
            "description": "Positive, supportive message"
        },
        {
            "text": "You are literally the worst human being ever",
            "description": "Strong cyberbullying with personal attack"
        },
        {
            "text": "You're worthless trash, nobody likes you",
            "description": "Multiple negative words and personal targeting"
        },
        {
            "text": "That's stupid but whatever",
            "description": "Mild negative language"
        },
        {
            "text": "I hate you so much, ugly loser",
            "description": "Strong negative emotions with name-calling"
        }
    ]
    
    all_results = []
    
    for case in test_cases:
        print(f"\n\n📋 Testing: {case['description']}")
        result = test_comprehensive_analysis(case['text'])
        if result:
            all_results.append({
                'description': case['description'],
                'text': case['text'],
                'result': result
            })
    
    # Summary analysis
    print(f"\n\n{'='*100}")
    print("📊 SUMMARY ANALYSIS")
    print('='*100)
    
    for i, case in enumerate(all_results):
        result = case['result']
        print(f"\n{i+1}. {case['description']}")
        print(f"   Text: \"{case['text']}\"")
        print(f"   → {result['label']} ({result['probability']:.1%})")
        
        # Top contributing word
        if result['token_contributions']:
            top_word = result['token_contributions'][0]
            print(f"   → Key word: '{top_word['token']}' ({top_word['shap_value']:+.4f})")
        
        # Top contributing feature  
        if result['feature_contributions']:
            top_feature = result['feature_contributions'][0]
            print(f"   → Key feature: {top_feature['feature_name']} ({top_feature['shap_value']:+.4f})")
    
    print(f"\n🎉 Analysis complete! Your API now provides:")
    print("   ✅ Feature-level SHAP waterfall charts")
    print("   ✅ Word-level token attribution analysis")
    print("   ✅ Complete explanations for model decisions")
    print("   ✅ Ready for frontend integration")

if __name__ == "__main__":
    main() 