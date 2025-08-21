# 🌊 SHAP Waterfall API - Complete Guide

## 🎯 What You Have Now

Your FastAPI now returns **structured SHAP waterfall data** that can be used to build the exact visualization you showed me! Here's everything you need to know:

## 🚀 Quick Start

### 1. Start the API Server
```bash
./start_api.sh
```

### 2. Test the API
```bash
source cyberbullying_env/bin/activate
python test_api.py
```

### 3. View Frontend Demo
Open `frontend_demo.html` in your browser to see a working waterfall chart!

## 📊 API Endpoint Details

### **POST `/explain/shap`**

**Request:**
```json
{
  "text": "You are literally the worst human being ever"
}
```

**Response Structure:**
```json
{
  "text": "You are literally the worst human being ever",
  "prediction": 1,
  "label": "Cyberbullying",
  "probability": 0.85,
  "base_value": 0.1851,
  "expected_value": 0.8896,
  "feature_contributions": [
    {
      "feature_name": "cyber_conf",
      "feature_value": 0.95,
      "shap_value": 0.3421,
      "contribution_type": "positive"
    },
    // ... more features
  ],
  "waterfall_data": {
    "base_value": 0.1851,
    "final_value": 0.8896,
    "prediction_label": "Cyberbullying",
    "prediction_probability": 0.85,
    "contributions": [
      {
        "feature": "cyber_conf",
        "feature_display_name": "Cyber Conf",
        "shap_value": 0.3421,
        "feature_value": 0.95,
        "running_total": 0.5272,
        "color": "#4CAF50",
        "abs_value": 0.3421
      }
      // ... sorted by importance
    ],
    "summary": {
      "total_positive_impact": 0.7045,
      "total_negative_impact": 0.0000,
      "net_impact": 0.7045
    },
    "chart_config": {
      "title": "SHAP Waterfall: You are literally the worst...",
      "subtitle": "Prediction: Cyberbullying (85.0% confidence)",
      "positive_color": "#4CAF50",
      "negative_color": "#F44336",
      "base_color": "#9E9E9E"
    }
  }
}
```

## 🎨 Frontend Integration

### Key Data Points for Your Waterfall Chart:

1. **`base_value`** - Starting point (like 0.185 in your diagram)
2. **`contributions`** - Array of feature impacts (sorted by importance)
3. **`final_value`** - End result (like 0.889 in your diagram)
4. **`colors`** - Ready-to-use hex colors for each bar
5. **`chart_config`** - Title, labels, and styling info

### JavaScript Example:
```javascript
// Make API call
const response = await fetch('http://localhost:7899/explain/shap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: userInput })
});

const data = await response.json();

// Use waterfall_data to build your chart
const waterfallData = data.waterfall_data;
console.log('Base value:', waterfallData.base_value);
console.log('Contributions:', waterfallData.contributions);
console.log('Colors ready:', waterfallData.contributions.map(c => c.color));
```

## 🔍 What Each Feature Means

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `cyber_conf` | Cyberbullying confidence from BERT | 0.95 (high) |
| `hate_prob` | Hate speech probability | 0.78 |
| `target_directed` | Contains personal pronouns/names | 1 (yes) |
| `sent_neg/neu/pos` | Sentiment scores | 0.8, 0.1, 0.1 |
| `sarcasm_conf` | Sarcasm detection confidence | 0.15 |
| `word_count` | Number of words | 8 |
| `char_count` | Number of characters | 45 |
| `exclamation_count` | Number of ! marks | 0 |
| `question_count` | Number of ? marks | 0 |

## 🌈 Color Coding

- **🟢 Green (`#4CAF50`)** - Features that **increase** cyberbullying likelihood
- **🔴 Red (`#F44336`)** - Features that **decrease** cyberbullying likelihood  
- **⚫ Gray (`#9E9E9E`)** - Base value and neutral elements

## 📈 Building the Exact Chart You Showed

Your waterfall data has everything needed:

1. **Base bar** - `waterfallData.base_value`
2. **Feature bars** - `waterfallData.contributions` (sorted by importance)
3. **Colors** - Each contribution has a `color` field
4. **Labels** - Use `feature_display_name` for clean labels
5. **Values** - `shap_value` for bar heights
6. **Running totals** - `running_total` for positioning

## 🚦 Testing Different Text Types

```python
# Test cases from test_api.py
test_cases = [
    "You're such a great friend!",           # → Not Cyberbullying
    "You are literally the worst ever",      # → Cyberbullying  
    "Nobody likes you. Just leave.",         # → Moderate
    "I hate you so much, ugly loser"        # → Strong Cyberbullying
]
```

## 💡 Pro Tips

1. **Sorted by Importance** - Features are automatically sorted by `abs(shap_value)`
2. **Ready-to-Use Colors** - No need to calculate colors, they're provided
3. **Display Names** - Use `feature_display_name` for prettier labels
4. **Running Totals** - Pre-calculated for waterfall positioning
5. **Responsive Design** - The frontend demo shows mobile-friendly implementation

## 🛠️ Customization

You can modify:
- **Colors** in `fastapi_app.py` (line ~200)
- **Feature sorting** in `fastapi_app.py` (line ~185)  
- **Chart dimensions** in `chart_config`
- **Feature display names** in the contribution loop

## ✅ Production Ready

- ✅ Handles missing models gracefully
- ✅ Provides fallback features when transformers fail  
- ✅ CORS enabled for frontend integration
- ✅ Structured error handling
- ✅ Comprehensive logging
- ✅ Virtual environment isolated

## 🎉 You're All Set!

Your API now returns the **exact data structure** needed to recreate the SHAP waterfall diagram you showed me. The `waterfall_data` object contains:

- All the values
- All the colors  
- All the labels
- Chart configuration
- Summary statistics

Just send that data to your frontend chart library (Chart.js, D3.js, Plotly, etc.) and you'll have a beautiful, interactive waterfall chart! 🌊

---

**Questions? Check the working demo at `frontend_demo.html` or run `python test_api.py` to see the data structure in action!** 