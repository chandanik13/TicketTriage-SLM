import gradio as gr
import pandas as pd
import logging
from src.inference import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the lazy-loading model manager
manager = ModelManager()

MODEL_CHOICES = {
    "TF-IDF + Logistic Regression": "baseline",
    "Qwen2.5-0.5B-Instruct Zero-Shot": "qwen_zero_shot",
    "Qwen2.5-0.5B-Instruct + LoRA": "qwen_lora"
}

def analyze_ticket(ticket_text, model_display_name):
    """Handles single ticket analysis using the selected model."""
    if not ticket_text or not ticket_text.strip():
        return {"Error": "Please enter a ticket text."}, "N/A"
        
    model_id = MODEL_CHOICES.get(model_display_name)
    if not model_id:
        return {"Error": "Invalid model selection."}, "N/A"
        
    logger.info(f"UI requested inference for {model_id}")
    result = manager.predict(model_id, ticket_text)
    
    if result["status"] == "unavailable":
        return {"Status": "Model not available — training/integration pending"}, "N/A"
    elif result["status"] == "error":
        return {"Status": result["error_message"]}, "N/A"
        
    output_dict = {
        "Category": result.get("category"),
        "Priority": result.get("priority"),
        "Sentiment": result.get("sentiment"),
        "Language": result.get("language"),
        "Resolution": result.get("resolution")
    }
    
    # Include JSON Validity for generative models
    if result.get("json_valid") != "N/A":
        output_dict["JSON Valid"] = result.get("json_valid")
        
    latency_str = f"{result.get('latency_seconds', 'N/A')} seconds"
    
    return output_dict, latency_str

def compare_all_models(ticket_text):
    """Runs all models and formats the results into a comparison dataframe."""
    if not ticket_text or not ticket_text.strip():
        return pd.DataFrame([{"Error": "Please enter a ticket text."}])
        
    results = []
    for display_name, model_id in MODEL_CHOICES.items():
        res = manager.predict(model_id, ticket_text)
        
        row = {"Model": display_name}
        if res["status"] == "unavailable":
            row["Category"] = "Model not available — training/integration pending"
        elif res["status"] == "error":
            row["Category"] = "Inference failed."
        else:
            row["Category"] = res.get("category")
            row["Priority"] = res.get("priority")
            row["Sentiment"] = res.get("sentiment")
            row["Language"] = res.get("language")
            row["Resolution"] = res.get("resolution")
            row["Latency (s)"] = res.get("latency_seconds")
            row["JSON Valid"] = res.get("json_valid")
            
        results.append(row)
        
    return pd.DataFrame(results)

def build_ui():
    """Builds the Gradio interface."""
    with gr.Blocks(title="TicketTriage-SLM", theme=gr.themes.Default(primary_hue="blue")) as app:
        gr.Markdown(
            """
            # TicketTriage-SLM
            ### Customer Support Ticket Triage using Traditional ML, Zero-Shot LLMs and LoRA Fine-Tuning
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                ticket_input = gr.Textbox(
                    label="Customer Support Ticket", 
                    placeholder="I was charged twice for my subscription and I haven't received the refund yet.",
                    lines=5
                )
                
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_CHOICES.keys()),
                    value="TF-IDF + Logistic Regression",
                    label="Model"
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Ticket", variant="primary")
                    compare_btn = gr.Button("Compare All Models")
                    
            with gr.Column(scale=3):
                gr.Markdown("### Prediction")
                prediction_json = gr.JSON(label="Structured Output")
                latency_label = gr.Label(label="Inference Latency")
                
        with gr.Row():
            comparison_df = gr.Dataframe(label="Model Comparison / Technical Details", interactive=False)
            
        gr.Markdown(
            """
            ---
            ### Research Transparency
            **Dataset**: Dataset V2  
            **Models**: TF-IDF + Logistic Regression, Qwen2.5-0.5B-Instruct Zero-Shot, Qwen2.5-0.5B-Instruct + LoRA  
            **Evaluation**: Accuracy, Macro F1, Weighted F1, Exact Match, Average Field Accuracy, JSON Validity and Latency
            """
        )
        
        # Wire up event handlers
        analyze_btn.click(
            fn=analyze_ticket,
            inputs=[ticket_input, model_dropdown],
            outputs=[prediction_json, latency_label]
        )
        
        compare_btn.click(
            fn=compare_all_models,
            inputs=[ticket_input],
            outputs=[comparison_df]
        )
        
    return app

if __name__ == "__main__":
    print("==================================================")
    print("TicketTriage-SLM Demonstration Interface")
    print("Available Models:")
    for m in MODEL_CHOICES.keys():
        print(f" - {m}")
    print("Models are loaded lazily upon first request.")
    print("==================================================")
    
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=False)
