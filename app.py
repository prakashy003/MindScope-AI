import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

# Request huggingface token if not already given.
# This model requires agreement to terms in huggingface.
token = os.environ.get("HF_TOKEN")
if not token:
    print("Warning: The 'mental/mental-bert-base-uncased' model is gated.")
    print("Please set the HF_TOKEN environment variable with your Hugging Face access token before starting the server.")
    print("Example: export HF_TOKEN='your_token'")
    exit(1)

login(token=token)

app = Flask(__name__)

# Constants and Setup
MODEL_NAME = "mental/mental-bert-base-uncased"
MODEL_PATH = "models/Mentalbert Model.pt"
LABEL_ORDER = ["Normal", "Depression", "Suicidal", "Anxiety", "Bipolar", "Stress", "Personality disorder"]
NUM_LABELS = len(LABEL_ORDER)

id2label = {i: label for i, label in enumerate(LABEL_ORDER)}
label2id = {label: i for i, label in enumerate(LABEL_ORDER)}

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading '{MODEL_NAME}' tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading '{MODEL_NAME}' base model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS, 
    id2label=id2label, 
    label2id=label2id
)

print(f"Loading pre-trained weights from {MODEL_PATH}...")
state_dict = torch.load(MODEL_PATH, map_location=device)

# MentalBERT weights often have mismatched keys depending on the save method. Let's fix missing "classifier.weight" vs "classifier.out_proj.weight" if needed.
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

print("Model loaded successfully!")

@app.route("/")
def home():
    """Render the main user interface."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict mental health category from user input text."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data.get("text")
    if not text.strip():
         return jsonify({"error": "Text cannot be empty"}), 400
    
    try:
        # Tokenize the input text
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get the predicted class index
            prediction_idx = torch.argmax(logits, dim=1).cpu().item()
            predicted_label = id2label[prediction_idx]

        # Format confidence scores for all classes
        confidence_scores = {
            LABEL_ORDER[i]: float(probabilities[i]) for i in range(NUM_LABELS)
        }

        # Sort the scores
        sorted_scores = sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True)

        return jsonify({
            "prediction": predicted_label,
            "confidence": float(probabilities[prediction_idx]),
            "all_scores": sorted_scores
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001, host='127.0.0.1')
