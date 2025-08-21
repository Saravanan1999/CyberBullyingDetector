import gradio as gr
import sys
import os
sys.path.append('./src')

from src.meta_model_final import CyberbullyingMetaModel
import json

# Initialize the model
print("Loading cyberbullying detection model...")
model = CyberbullyingMetaModel(model_dir="./models/trained_models")

def analyze_text(text):
    """Analyze text for cyberbullying and return SHAP explanation"""
    if not text.strip():
        return "Please enter some text to analyze.", "", ""
    
    try:
        # Get prediction
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0]
        
        # Get SHAP explanation
        shap_explanation = model.explain_with_shap(text)
        token_explanation = model.explain_text_tokens(text)
        
        # Format results
        label = "Cyberbullying" if prediction == 1 else "Not Cyberbullying"
        confidence = f"{probability * 100:.1f}%"
        
        # Create result summary
        result = f"**Prediction:** {label}\n**Confidence:** {confidence}"
        
        # Format SHAP explanation
        shap_text = "## SHAP Feature Importance:\n"
        for contrib in shap_explanation['feature_contributions'][:5]:
            direction = "↗️" if contrib['shap_value'] > 0 else "↘️"
            shap_text += f"- **{contrib['feature_name']}**: {direction} {contrib['shap_value']:.4f}\n"
        
        # Format token explanation
        token_text = "## Word-Level Analysis:\n"
        sorted_tokens = sorted(token_explanation['token_contributions'], 
                             key=lambda x: abs(x['shap_value']), reverse=True)
        for token in sorted_tokens[:10]:
            direction = "🔴" if token['shap_value'] < 0 else "🔵"
            token_text += f"- **{token['token']}**: {direction} {token['shap_value']:.4f}\n"
        
        return result, shap_text, token_text
        
    except Exception as e:
        return f"Error analyzing text: {str(e)}", "", ""

# Create Gradio interface
with gr.Blocks(title="Cyberbullying Detection with Explainable AI") as demo:
    gr.Markdown("# 🛡️ Cyberbullying Detection with Explainable AI")
    gr.Markdown("This system uses ensemble machine learning with SHAP explanations to detect cyberbullying in text.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter text to analyze",
                placeholder="Type or paste text here...",
                lines=3
            )
            analyze_btn = gr.Button("🔍 Analyze Text", variant="primary")
            
            # Example buttons
            gr.Markdown("### Quick Examples:")
            examples = [
                "You are such a great friend!",
                "You are literally the worst human being ever",
                "Nobody likes you. Just leave already.",
                "I hate you so much, you are stupid and ugly"
            ]
            for example in examples:
                gr.Button(example, size="sm").click(
                    lambda x=example: x, outputs=text_input
                )
    
    with gr.Column():
        result_output = gr.Markdown(label="Prediction Result")
        shap_output = gr.Markdown(label="SHAP Explanation")
        token_output = gr.Markdown(label="Word Analysis")
    
    analyze_btn.click(
        analyze_text,
        inputs=text_input,
        outputs=[result_output, shap_output, token_output]
    )

if __name__ == "__main__":
    demo.launch() 